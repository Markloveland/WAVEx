"""
Action Balance Equation Solver
This algorithm for loading will be same required of Action Balance Equation
    du/dt + \/.cu = f
Case from ONR Testbed case A21
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
time_start = time.time()



#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#specify bouns in geographic mesh
x_min = 0.0
x_max = 10000
y_min = 0.0
y_max = 4000
# Create cartesian mesh of two 2D and define function spaces
nx = 100
ny = 100
# define spectral domain
omega_min = np.pi*2*0.05 
omega_max = .25*np.pi*2
theta_min = np.pi/2 - 10/180*np.pi
theta_max = np.pi/2 + 10/180*np.pi
n_sigma = 40
n_theta = 24
#set initial time
t = 0
#set final time
t_f =750
#set time step
dt = 2.0
#calculate nt
nt = int(np.ceil(t_f/dt))
PETSc.Sys.Print('nt',nt)
#plot every n time steps
#nplot = 1
nplot = 10
#note, wetting/drying only works with "strong" forms
method = 'SUPG_strong'
out_dir = 'Outputs/A31/'
####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
domain1 = mesh.create_rectangle(comm, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny], mesh.CellType.triangle)
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
#for now lets set depth as x coordinate itself
#eventually this maybe read in from txt file or shallow water model
depth_func.x.array[:] = 10000*np.ones(dof_coords1[:,0].shape)# + dof_coords1[:,1]*1.8/4000
v_func.x.array[:] = dof_coords1[:,1]/2000
#need to include a wetting/drying variable in domain 1
CFx.wave.calculate_wetdry(domain1, V1, depth_func,is_wet,min_depth=0.05)
####################################################################
####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh
domain2 = mesh.create_rectangle(MPI.COMM_SELF, [np.array([omega_min, theta_min]), np.array([omega_max, theta_max])], [n_sigma, n_theta], mesh.CellType.triangle)
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
tol=1e-11
#now only want subset that is the inflow, need to automate later
dum1 = local_boundary_dofs[y[local_boundary_dofs]<=(y_min+tol)]
#dum2 = local_boundary_dofs[np.logical_and(x[local_boundary_dofs]>=(x_max-tol),theta[local_boundary_dofs]>np.pi/2)]
#dum3 = local_boundary_dofs[np.logical_and(x[local_boundary_dofs]<=(x_min+tol),theta[local_boundary_dofs]<np.pi/2)]
#dum4 = local_boundary_dofs[theta[local_boundary_dofs]<=(theta_min+tol)]
#dum5 = local_boundary_dofs[theta[local_boundary_dofs]>=(theta_max-tol)]

#local_boundary_dofs = np.unique(np.concatenate((dum1,dum2,dum3,dum4,dum5),0))
local_boundary_dofs = dum1
global_boundary_dofs = local_boundary_dofs + local_range[0]
####################################################################
####################################################################
#generate any coefficients that depend on the degrees of freedom
#depth = 20 - x/200
#zero rows and columns below a minumum depth
#min_depth = 0.05 
#min_depth = 0.05
#dry_dofs_local = np.array(np.where(depth<min_depth)[0],dtype=np.int32)
#dry_dofs = dry_dofs_local + local_range[0]
#wet_dofs_local = np.where(depth>=min_depth)[0]

#u = np.zeros(local_dof.shape[0])
#v = np.zeros(local_dof.shape[0])
#c1 = np.ones(local_dof.shape)
#c2 = np.ones(local_dof.shape)
#c1[wet_dofs_local,:] = CFx.wave.compute_wave_speeds_pointwise(x[wet_dofs_local],y[wet_dofs_local],sigma[wet_dofs_local],theta[wet_dofs_local],depth[wet_dofs_local],u[wet_dofs_local],v[wet_dofs_local])
c,dry_dofs_local = CFx.wave.compute_wave_speeds(x,y,sigma,theta,depth_func,u_func,v_func,N_dof_2)
if rank==0:
    np.savetxt(out_dir+"Stations/"+"currents_velocities.csv", np.append(local_dof, c, axis=1), delimiter=",")
#print('difference in velocities at rank',rank,np.sum(np.absolute(c-c1)))

#exact solution and dirichlet boundary
dry_dofs = dry_dofs_local+local_range[0]



def u_func(x,y,sigma,theta,c,t):
    #takes in dof and paramters
    HS = 1
    F_std = 0.04
    F_peak = 0.1
    Dir_mean = 90.0 #mean direction in degrees
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

#if rank==0:
#    print(local_dof_coords1)
#    print(A.getValues(30,30))
if method == 'SUPG' or method == 'SUPG_strong':
    M_SUPG = M+M_SUPG
    A=A+M_SUPG
if method == 'CG' or method == 'CG_strong':
    M_SUPG = M
    A = A + M_SUPG


dry_dofs = CFx.utils.fix_diag(A,local_range[0],rank)
#print('global rows with 0 in diagonal',dry_dofs)
#A.zeroRows(dry_dofs,diag=1)
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
#all_u_d = np.concatenate(MPI.COMM_WORLD.allgather(u_d))

A.zeroRowsColumns(global_boundary_dofs,diag=1,x=u_cart,b=u_cart)
###################################################################
###################################################################
#Define solver/preconditioner
#create a direct linear solver
pc2 = PETSc.PC().create()
#this is a direct solve with lu
pc2.setType('none')
pc2.setOperators(A)

ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
ksp2.setType('gmres')
ksp2.setPC(pc2)
ksp2.setInitialGuessNonzero(True)

#fname = 'ActionBalance_Currents_A31/solution'
#xdmf = io.XDMFFile(domain1.comm, out_dir+'Paraview/'+fname+".xdmf", "w")
#xdmf.write_mesh(domain1)
#########################################################
#######################################################
#Time Step
u = fem.Function(V1)


#compute significant wave height
HS = fem.Function(V1)
#try to fix units maybe is issue?
#Temp.setValues(rows,sigma)
#PETSc.Vec.pointwiseMult(u_exact,Temp,u_cart)
HS_vec = CFx.wave.calculate_HS_actionbalance(u_cart,V2,N_dof_1,N_dof_2,local_range2)
HS.vector.setValues(dofs1,np.array(HS_vec))
HS.vector.ghostUpdate()
fname = 'Currents_HS_structured/solution'
xdmf = io.XDMFFile(domain1.comm, out_dir+'Paraview/'+fname+".xdmf", "w")
xdmf.write_mesh(domain1)
xdmf.write_function(HS,t)


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
        HS_vec = CFx.wave.calculate_HS_actionbalance(u_cart,V2,N_dof_1,N_dof_2,local_range2)
        HS.vector.setValues(dofs1,np.array(HS_vec))
        HS.vector.ghostUpdate()
        xdmf.write_function(HS, t)
        #hdf5_file.write(u,"solution",t)
#print final iterations
ksp2.view()
PETSc.Sys.Print('Niter',ksp2.getIterationNumber())
PETSc.Sys.Print('convergence code',ksp2.getConvergedReason())

HS_vec = CFx.wave.calculate_HS_actionbalance(u_cart,V2,N_dof_1,N_dof_2,local_range2)
HS.vector.setValues(dofs1,np.array(HS_vec))
HS.vector.ghostUpdate()
xdmf.write_function(HS, t)
xdmf.close()
time_end = time.time()
############################################################################
###########################################################################
#Post Processing section

#print whole solution
#print('Cartesian solution')
#print(u_cart.getArray()[:])
#print('Exact')
#print(u_true[:])

u_true = u_func(x,y,sigma,theta,c,t)
u_exact.setValues(rows,u_true)

PETSc.Sys.Print("Final t",t)
#need function to evaluate L2 error
e1 = u_cart-u_exact
PETSc.Vec.pointwiseMult(E,e1,e1)
M.mult(E,L2_E)
#L2
PETSc.Sys.Print("L2 error",np.sqrt(L2_E.sum()))
#Linf
PETSc.Sys.Print("L inf error",e1.norm(PETSc.NormType.NORM_INFINITY))
#min/max
PETSc.Sys.Print("min in error",e1.min())
PETSc.Sys.Print("max error",e1.max())
#h
PETSc.Sys.Print("h",1/nx)
#dof
PETSc.Sys.Print("dof",(nx+1)**2*(ny+1)**2)
buildTime = time_2-time_start
solveTime = time_end-time_2
PETSc.Sys.Print(f'The build time is {buildTime} seconds')
PETSc.Sys.Print(f'The solve time is {solveTime} seconds')
PETSc.Sys.Print('Final solution on boundary')
#print(u_cart.getValues(global_boundary_dofs))


#try to extract HS at stations
numpoints = 150
y_stats = np.linspace(y_min,y_max,numpoints)

x_coord = x_max/2 
stations = x_coord*np.ones((numpoints,2))
stations[:,1] = y_stats


points_on_proc, vals_on_proc = CFx.utils.station_data(stations,domain1,HS)
stats,vals = CFx.utils.gather_station(comm,0,points_on_proc,vals_on_proc)

if rank ==0:
    #PETSc.Sys.Print('Station locs:')
    #PETSc.Sys.Print(stats)
    #PETSc.Sys.Print(stats.shape)
    #PETSc.Sys.Print('Station vals:')
    #PETSc.Sys.Print(vals)
    #PETSc.Sys.Print(vals.shape)
    np.savetxt(out_dir+"Stations/"+"Currents_stations_SUPG_structured.csv", np.append(stats, vals, axis=1), delimiter=",")

