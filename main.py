"""
Action Balance Equation Solver
This algorithm for loading will be same required of Action Balance Equation
    du/dt + \/.cu = f
Main script that reads in i/o and calls all CFx routines
"""

import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import sparse as sp
from petsc4py import PETSc
from dolfinx import fem,mesh,io
import ufl
import time
import CFx.utils
import CFx.assemble
import CFx.transforms
import CFx.boundary
import CFx.wave
import CFx.io
import sys


time_start = time.time()


#read in i/o
#input text file path should be added in the 1st command line argument 
Model_Params = CFx.io.read_input_file(sys.argv[1])



#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#unpack input params and set up problem

#set initial time
t = Model_Params["Start Time"]
#set final time
t_f = Model_Params["End Time"]
#set time step
dt = Model_Params["DT"]
#calculate nt
nt = int(np.ceil(t_f/dt))
#plot every n time steps
nplot = Model_Params["Plot Every"]
#weak form (note if wet/dry is on, only strong form works for now)
method = Model_Params["Weak Form"]
#time stepping param (0 for explicit 1 for implicit, 0.5 for trapezoid)
theta_param = Model_Params["Theta Param"]
out_dir = Model_Params["Output Folder"]
fname = Model_Params["Run Name"]
####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2

#either the mesh will be structured or unstructured:
if Model_Params["Mesh Type"] == "Structured":

    #specify bouns in geographic mesh
    x_min = Model_Params["Geographic Bounds"][0]
    x_max = Model_Params["Geographic Bounds"][1]
    y_min = Model_Params["Geographic Bounds"][2]
    y_max = Model_Params["Geographic Bounds"][3]
    # Create cartesian mesh of two 2D and define function spaces
    nx = Model_Params["Geographic Cells"][0]
    ny = Model_Params["Geographic Cells"][1]

    domain1 = mesh.create_rectangle(comm, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny], mesh.CellType.triangle)
elif Model_Params["Mesh Type"] == "Unstructured":
    filename = Model_Params["Mesh Address"]
    encoding= io.XDMFFile.Encoding.HDF5
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        domain1 = file.read_mesh()
else:
    raise Exception("Mesh not properly defined")

V1 = fem.FunctionSpace(domain1, ("CG", 1))
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)
####################################################################
####################################################################
#wetting drying portion
elementwise = fem.FunctionSpace(domain1,("DG",0))
is_wet = fem.Function(elementwise)
depth_func = fem.Function(V1)
u_func = fem.Function(V1)
v_func = fem.Function(V1)
#from the function spaces and ownership ranges, generate global degrees of freedom
#this gives ghost and owned dof coords
dof_coords1 = V1.tabulate_dof_coordinates()
#suggested in forum, gives index of dofs I want
local_range1 = V1.dofmap.index_map.local_range
#vector of indexes that we want
dofs1 = np.arange(*local_range1,dtype=np.int32)
#gets number of dofs owned
N_dof_1 = V1.dofmap.index_map.size_local
#hopefully the dof coordinates owned by the process
local_dof_coords1 = dof_coords1[0:N_dof_1,:domain1.topology.dim]

#setting depth,u,v
#here are some presets built in for some test cases
if Model_Params["Bathymetry"] == "Uniform Slope":
    depth_func.x.array[:] = 20 - dof_coords1[:,1]/200
elif Model_Params["Bathymetry"] == "Deep":
    depth_func.x.array[:] = 10000*np.ones(dof_coords1[:,0].shape)
else:
    raise Exception("Bathymetry not defined")


#here are some presets for u,v
#setting none will default to 0
if Model_Params["Currents"] == "A31":
    v_func.x.array[:] = dof_coords1[:,1]/2000
elif Model_Params["Currents"] == "A32":
    v_func.x.array[:] = -dof_coords1[:,1]/2000
elif Model_Params["Currents"] == "A33":
    u_func.x.array[:] = dof_coords1[:,1]/2000
elif Model_Params["Currents"] == "A34":
    u_func.x.array[:] = dof_coords1[:,1]/2000

#need to include a wetting/drying variable in domain 1
CFx.wave.calculate_wetdry(domain1, V1, depth_func,is_wet,min_depth=0.05)
####################################################################
####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
# define spectral domain
#specify bounds in spectral mesh
omega_min = Model_Params["Spectral Bounds"][0]*2*np.pi
omega_max = Model_Params["Spectral Bounds"][1]*2*np.pi
theta_min = Model_Params["Spectral Bounds"][2]*np.pi/180
theta_max = Model_Params["Spectral Bounds"][3]*np.pi/180
n_sigma = Model_Params["Spectral Cells"][0]
n_theta = Model_Params["Spectral Cells"][1]

#MPI.COMM_SELF to not partition mesh
domain2 = mesh.create_rectangle(MPI.COMM_SELF, [np.array([omega_min, theta_min]), np.array([omega_max, theta_max])], [n_sigma, n_theta], mesh.CellType.triangle)
#domain2 = mesh.create_rectangle(MPI.COMM_SELF, [np.array([omega_min, theta_min]), np.array([omega_max, theta_max])], [n_sigma, n_theta], mesh.CellType.quadrilateral)
V2 = fem.FunctionSpace(domain2, ("CG", 1))
u2 = ufl.TrialFunction(V2)
v2 = ufl.TestFunction(V2)
###################################################################
###################################################################
#need local mass matrices to build global mass matrix
#mass of subdomain 1
m1 = is_wet*u1*v1*ufl.dx
m1_form = fem.form(m1)
M1 = fem.petsc.assemble_matrix(m1_form)
M1.assemble()
#mass of subdomain 2
m2 = u2*v2*ufl.dx
m2_form = fem.form(m2)
M2 = fem.petsc.assemble_matrix(m2_form)
M2.assemble()
#################################################################
#################################################################
###################################################################
###################################################################
#save sizes of subdomain matrices
M1_sizes = M1.getLocalSize()
M1_global_size = M1.getSize() 
M2_sizes = M2.getLocalSize()
#calculate sizes of matrix for global domain
#number of rows/cols on each processor
local_rows = int(M1_sizes[0]*M2_sizes[0])
global_rows = int(M1_global_size[0]*M2_sizes[0])
local_cols = int(M1_sizes[1]*M2_sizes[1])
global_cols = int(M1_global_size[1]*M2_sizes[1])
###################################################################
###################################################################
#Allocate global mass matrix
#need to generate global mass matrix to get global matrix layout and sparsity patterns
#global matrices are product of each subdomain

M=CFx.assemble.create_cartesian_mass_matrix(local_rows,global_rows,local_cols,global_cols)

#also need global stiffness matrix
#same exact structure as M
A = M.duplicate()
#Adjust RHS for SUPG
M_SUPG = M.duplicate()
#get ownership range
local_range = M.getOwnershipRange()
#vector of row numbers
rows = np.arange(local_range[0],local_range[1],dtype=np.int32)
####################################################################
####################################################################
#from the function spaces and ownership ranges, generate global degrees of freedom
#this gives ghost and owned dof coords
dof_coords2 = V2.tabulate_dof_coordinates()
#suggested in forum, gives index of dofs I want
local_range2 = V2.dofmap.index_map.local_range
#vector of indexes that we want
dofs2 = np.arange(*local_range2,dtype=np.int32)
#gets number of dofs owned
N_dof_2 = V2.dofmap.index_map.size_local
#hopefully the dof coordinates owned by the process
local_dof_coords2 = dof_coords2[0:N_dof_2,:domain2.topology.dim]

local_dof=CFx.transforms.cartesian_product_coords(local_dof_coords1,local_dof_coords2)

x = local_dof[:,0]
y = local_dof[:,1]
sigma = local_dof[:,2]
theta = local_dof[:,3]

#get global equation number of any node on entire global boundary
local_boundary_dofs = CFx.boundary.fetch_boundary_dofs(domain1,domain2,V1,V2,N_dof_1,N_dof_2)
tol= 1e-11


##############################################################################################################
#now only want subset that is the inflow, need to automate later
#this is assuming a rectangular shaped mesh with waves coming in from bottom side
y_min =0
x_max = 20000
x_min = 0
dum1 = local_boundary_dofs[y[local_boundary_dofs]<=(y_min+tol)]
dum2 = local_boundary_dofs[np.logical_and(x[local_boundary_dofs]>=(x_max-tol),theta[local_boundary_dofs]>np.pi/2)]
dum3 = local_boundary_dofs[np.logical_and(x[local_boundary_dofs]<=(x_min+tol),theta[local_boundary_dofs]<np.pi/2)]
dum4 = local_boundary_dofs[theta[local_boundary_dofs]<=(theta_min+tol)]
dum5 = local_boundary_dofs[theta[local_boundary_dofs]>=(theta_max-tol)]

local_boundary_dofs = np.unique(np.concatenate((dum1,dum2,dum3,dum4,dum5),0))
#local_boundary_dofs = dum2
global_boundary_dofs = local_boundary_dofs + local_range[0]
####################################################################
####################################################################
c,dry_dofs_local = CFx.wave.compute_wave_speeds(x,y,sigma,theta,depth_func,u_func,v_func,N_dof_2)
#exact solution and dirichlet boundary
dry_dofs = dry_dofs_local+local_range[0]


if Model_Params["Boundary Type"] == "Gaussian": 
    #also warning, hard coded for inflow boundary of y=0
    #will change later
    def u_func(x,y,sigma,theta,c,t):
        #takes in dof and paramters
        HS = Model_Params["Gaussian Params"][0]
        F_std = Model_Params["Gaussian Params"][1]
        F_peak = Model_Params["Gaussian Params"][2]
        Dir_mean = Model_Params["Gaussian Params"][3]
        Dir_rad = Dir_mean*np.pi/(180)
        Dir_exp = 500
        #returns vector with initial condition values at global DOF
        aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
        aux3 = 2*F_std**2
        tol=1e-11
        aux2 = (sigma - ( np.pi*2*F_peak ) )**2
        E = (y<tol)*aux1*np.exp(-aux2/aux3)/sigma
        CTOT = np.sqrt(0.5*Dir_exp/np.pi)/(1.0 - 0.25/Dir_exp)
        A_COS = np.cos(theta - Dir_rad)
        CDIR = (A_COS>0)*CTOT*np.maximum(A_COS**Dir_exp, 1.0e-10)
        return E*CDIR
####################################################################
####################################################################


#####################################################################
#####################################################################
#Preallocate and load/assemble cartesian mass matrix!
#now need to mass matrixes for stiffness and RHS, also optionally can out put the nnz
M_NNZ = CFx.assemble.build_cartesian_mass_matrix(M1,M2,M1_sizes,M1_global_size,M2_sizes,M)
A.setPreallocationNNZ(M_NNZ)
M_SUPG.setPreallocationNNZ(M_NNZ)
##################################################################
##################################################################
#Loading A matrix routine

if method == 'SUPG' or method == 'SUPG_strong':
    tau_pointwise = CFx.wave.compute_tau(domain1,domain2,c,N_dof_1,N_dof_2)
    CFx.assemble.build_action_balance_stiffness(domain1,domain2,V1,V2,c,dt,A,method=method,is_wet=is_wet,tau_vals=tau_pointwise)
    CFx.assemble.build_RHS(domain1,domain2,V1,V2,c,M_SUPG,is_wet=is_wet,tau_vals=tau_pointwise)
if method == 'CG' or method == 'CG_strong':
    CFx.assemble.build_action_balance_stiffness(domain1,domain2,V1,V2,c,dt,A,method=method,is_wet=is_wet)
    CFx.assemble.build_RHS(domain1,domain2,V1,V2,c,M_SUPG,is_wet=is_wet)

time_2 = time.time()

if method == 'SUPG' or method == 'SUPG_strong':
    M_SUPG = M+M_SUPG-(1-theta_param)*A
    A=A+M_SUPG
if method == 'CG' or method == 'CG_strong':
    M_SUPG = M-(1-theta_param)*A
    A = A + M_SUPG

#fixing 0 on diagonal due to wetting drying
dry_dofs = CFx.utils.fix_diag(A,local_range[0],rank)
A.zeroRowsColumns(dry_dofs,diag=1)

##################################################################
##################################################################
#initialize vectors
#holds dirichlet boundary values
u_D = PETSc.Vec()
u_D.create(comm=comm)
u_D.setSizes((local_rows,global_rows),bsize=1)
u_D.setFromOptions()

#holds temporary values to contribute to RHS
Temp = u_D.duplicate()
#RHS of linear system of equations
B = u_D.duplicate()
#Post Processiong
E = u_D.duplicate()
L2_E = u_D.duplicate()
u_exact = u_D.duplicate()
#solution vector
u_cart = u_D.duplicate()


Temp.setFromOptions()
E.setFromOptions()
B.setFromOptions()
L2_E.setFromOptions()
u_cart.setFromOptions()
u_exact.setFromOptions()
###################################################################
###################################################################
#Set initial condition and set solution for dirichlet
#u_cart will hold solution in time loop, this is also the initial condition
u_cart.setValues(rows,u_func(x,y,sigma,theta,c,t))
u_cart.assemble()

#this matrix will help apply b.c. efficiently
C = A.duplicate(copy=True)
#need C to be A but 0 when columns are not dirichlet
C.transpose()
mask = np.ones(rows.size, dtype=bool)
mask[local_boundary_dofs] = False
global_non_boundary = rows[mask]
C.zeroRows(global_non_boundary,diag=0)
C.transpose()
#now zero out rows/cols containing boundary
#set solution as the boundary condition, helps initial guess
#all_global_boundary_dofs = np.concatenate(MPI.COMM_WORLD.allgather(global_boundary_dofs))
A.zeroRowsColumns(global_boundary_dofs,diag=1,x=u_cart)
###################################################################
###################################################################
#Define solver/preconditioner
#create a direct linear solver
#pc2 = PETSc.PC().create()
#this is a direct solve with lu
#pc2.setType('none')
#pc2.setOperators(A)

ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
ksp2.setType('gmres')
#ksp2.setPC(pc2)
ksp2.setInitialGuessNonzero(True)


HS = fem.Function(V1)
#try to fix units maybe is issue?
#Temp.setValues(rows,sigma)
#PETSc.Vec.pointwiseMult(u_exact,Temp,u_cart)
HS_vec = CFx.wave.calculate_HS_actionbalance(u_cart,V2,N_dof_1,N_dof_2,local_range2)
HS.vector.setValues(dofs1,np.array(HS_vec))
HS.vector.ghostUpdate()

xdmf = io.XDMFFile(domain1.comm, out_dir+'Paraview/'+fname+"/solution.xdmf", "w")
xdmf.write_mesh(domain1)
xdmf.write_function(HS,t)


#########################################################
#######################################################
#Time Step
u = fem.Function(V1)
for i in range(nt):
    t+=dt
    #B will hold RHS of system of equations
    M_SUPG.mult(u_cart,B)

    #setting dirichlet BC
    u_2 = u_func(x,y,sigma,theta,c,t)
    u_d_vals = u_2[local_boundary_dofs]
    u_D.setValues(global_boundary_dofs,u_d_vals)
    C.mult(u_D,Temp)
    B = B - Temp
    B.setValues(global_boundary_dofs,u_d_vals)
    B.assemble()
    #solve for time t
    ksp2.solve(B, u_cart)
    B.zeroEntries()
    # Save solution to file in VTK format
    if (i%nplot==0):
        #u.vector.setValues(dofs1, np.array(u_cart.getArray()[4::N_dof_2]))
        #xdmf.write_function(u, t)
        HS_vec = CFx.wave.calculate_HS_actionbalance(u_cart,V2,N_dof_1,N_dof_2,local_range2)
        HS.vector.setValues(dofs1,np.array(HS_vec))
        HS.vector.ghostUpdate()
        xdmf.write_function(HS,t)
        
        #hdf5_file.write(u,"solution",t)
#print final iterations
HS_vec = CFx.wave.calculate_HS_actionbalance(u_cart,V2,N_dof_1,N_dof_2,local_range2)
HS.vector.setValues(dofs1,np.array(HS_vec))
HS.vector.ghostUpdate()
xdmf.write_function(HS,t)
xdmf.close()
time_end = time.time()
############################################################################
###########################################################################
#Post Processing section
#Logging some info
ksp2.view()
PETSc.Sys.Print('Niter',ksp2.getIterationNumber())
PETSc.Sys.Print('convergence code',ksp2.getConvergedReason())
PETSc.Sys.Print("dof",global_cols)
buildTime = time_2-time_start
solveTime = time_end-time_2
PETSc.Sys.Print(f'The build time is {buildTime} seconds')
PETSc.Sys.Print(f'The solve time is {solveTime} seconds')
PETSc.Sys.Print('Final solution on boundary')
#print(u_cart.getValues(global_boundary_dofs))

#try to extract HS at stations
#for now just assuming stations are a line in the y direction
numpoints = Model_Params["Station Params"][0]
y_stats = np.linspace(Model_Params["Station Params"][1],Model_Params["Station Params"][2],numpoints)

#also assuming its at this point
x_coord = Model_Params["Station Params"][3]
stations = x_coord*np.ones((numpoints,2))
stations[:,1] = y_stats


#now get data for each QOI
for QOI in Model_Params["QoI"]:
    if QOI == "HS":
        points_on_proc, vals_on_proc = CFx.utils.station_data(stations,domain1,HS)
        stats,HS_vals = CFx.utils.gather_station(comm,0,points_on_proc,vals_on_proc)

    if QOI == "Mean Dir":
        Dir = fem.Function(V1)
        Dir_vec = CFx.wave.calculate_mean_dir(u_cart,V2,N_dof_1,N_dof_2,local_range2)
        Dir.vector.setValues(dofs1,np.array(Dir_vec))
        Dir.vector.ghostUpdate()
        points_on_proc, Dirs_on_proc = CFx.utils.station_data(stations,domain1,Dir)
        stats,Dir_vals = CFx.utils.gather_station(comm,0,points_on_proc,Dirs_on_proc)


#save to text file
if rank ==0:
    #recast as column vector
    vals_out = np.zeros((numpoints,len(Model_Params["QoI"])))
    a=0
    for QOI in Model_Params["QoI"]:
        if QOI == "HS":
            if Model_Params["Mesh Type"] == "Unstructured":
                vals_out[:,a] = HS_vals[:]
            else:
                vals_out[:,a] = HS_vals[:,0]
        if QOI == "Mean Dir":
            if Model_Params["Mesh Type"] == "Unstructured":
                vals_out[:,a] = Dir_vals[:]
            else:
                vals_out[:,a] = Dir_vals[:,0]
        a+=1
    np.savetxt(out_dir+'Stations/'+fname+".csv", np.append(stats, vals_out, axis=1), delimiter=",")