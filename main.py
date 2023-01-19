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
import CFx.timestep
import CFx.source
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

elif Model_Params["Mesh Type"] == "L11":
    filename = Model_Params["Mesh Address"]
    encoding= io.XDMFFile.Encoding.HDF5
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
        domain1 = file.read_mesh()
    domain1.geometry.x[:,:] = domain1.geometry.x[:,:]*((30-7.4)/4000)
    domain1.geometry.x[:,1] = domain1.geometry.x[:,1]+7.4

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
    depth =  np.array(20 - local_dof_coords1[:,1]/200)
    depth_func.x.array[:] = np.array(20 - dof_coords1[:,1]/200)
elif Model_Params["Bathymetry"] == "Deep":
    depth = np.array(10000*np.ones(local_dof_coords1[:,0].shape))
    depth_func.x.array[:] = np.array(10000*np.ones(dof_coords1[:,0].shape))
elif Model_Params["Bathymetry"] == "Uniform Constant":
    depth = np.array(np.ones(local_dof_coords1[:,0].shape))
    depth_func.x.array[:] = np.array(np.ones(dof_coords1[:,0].shape))
elif Model_Params["Bathymetry"] == "L11":
    depth = np.array(np.zeros(local_dof_coords1[:,0].shape))
    bath_locs = np.linspace(-4.4,4.4*7,9)
    bath_vals = np.array([0.7,0.7,0.7,0.64,0.424,0.208,0.315,0.124,-0.06])
    for a in range(1,bath_locs.shape[0]):
        seg = np.logical_and(local_dof_coords1[:,1]>bath_locs[a-1],local_dof_coords1[:,1]<=bath_locs[a])
        depth[seg] = (bath_vals[a] - bath_vals[a-1])/(bath_locs[a]-bath_locs[a-1])*(local_dof_coords1[seg,1]-bath_locs[a-1]) + bath_vals[a-1]
    
    #repeat for water depth and add
    wlev_locs = np.linspace(-1,31,33)
    wlev_vals = np.zeros(wlev_locs.shape)
    wlev_vals[:14] = 0.062
    wlev_vals[14:17] = 0.061
    wlev_vals[17:19] = 0.060
    wlev_vals[19] = 0.061
    wlev_vals[20] = 0.062
    wlev_vals[21] = 0.061
    wlev_vals[22:24] = 0.062
    wlev_vals[24:26] = 0.061
    wlev_vals[26] = 0.060
    wlev_vals[27] = 0.061
    wlev_vals[28] = 0.066
    wlev_vals[29:] = 0.067
    for a in range(1,wlev_locs.shape[0]):
        seg = np.logical_and(local_dof_coords1[:,1]>wlev_locs[a-1],local_dof_coords1[:,1]<=wlev_locs[a])
        depth[seg] = depth[seg] + (wlev_vals[a] - wlev_vals[a-1])/(wlev_locs[a]-wlev_locs[a-1])*(local_dof_coords1[seg,1]-wlev_locs[a-1]) + wlev_vals[a-1]
    
    depth_func.vector.setValues(dofs1,np.array(depth))
    depth_func.vector.ghostUpdate()


else:
    raise Exception("Bathymetry not defined")


#xdmf = io.XDMFFile(domain1.comm, out_dir+'Paraview/Bath/'+fname+"/solution.xdmf", "w")
#xdmf.write_mesh(domain1)
#xdmf.write_function(depth_func)
#xdmf.close()



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

if Model_Params["Spectral Mesh Type"] == "logarithmic":
    PETSc.Sys.Print("Switching to logarithmic spacing in frequency")
    old_coords,inverted = np.unique(domain2.geometry.x[:,0],return_inverse=True)
    #modify x coords to be logarithmic
    gamma_space = (omega_max/omega_min)**(1/n_sigma)
    new_coords = gamma_space**(np.arange(n_sigma+1))*omega_min
    domain2.geometry.x[:,0] = new_coords[inverted]



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
#RHS for implicit time step that multiplies a vector (RHSb = b)
RHS = M.duplicate()
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
tol= 1e-9


##############################################################################################################
#now only want subset that is the inflow, need to automate later
#this is assuming a rectangular shaped mesh with waves coming in from bottom side
if Model_Params["Mesh Type"]=="L11":
    x_max = 113
    y_min = 7.4
else:
    x_max = 20000
    y_min =0

x_min = 0
dum1 = local_boundary_dofs[y[local_boundary_dofs]<=(y_min+tol)]
dum2 = local_boundary_dofs[np.logical_and(x[local_boundary_dofs]>=(x_max-tol),theta[local_boundary_dofs]>=(np.pi/2+tol))]
dum3 = local_boundary_dofs[np.logical_and(x[local_boundary_dofs]<=(x_min+tol),theta[local_boundary_dofs]<=(np.pi/2-tol))]
dum4 = local_boundary_dofs[theta[local_boundary_dofs]<=(theta_min+tol)]
dum5 = local_boundary_dofs[theta[local_boundary_dofs]>=(theta_max-tol)]

if Model_Params["Currents"]=="A31": 
    local_boundary_dofs = dum1
elif Model_Params["Currents"] == "A32":
    local_boundary_dofs=np.unique(np.concatenate((dum1,dum4,dum5),0))
elif Model_Params["Currents"]=="A33":
    local_boundary_dofs=np.unique(np.concatenate((dum1,dum4,dum5),0))
elif Model_Params["Currents"] == "A34":
    local_boundary_dofs=np.unique(np.concatenate((dum1,dum2,dum3,dum4,dum5),0))
elif Model_Params["Currents"] == "None":
    #print("currents=none")
    local_boundary_dofs = np.unique(np.concatenate((dum1,dum2,dum3),0))
else:
    local_boundary_dofs = np.unique(np.concatenate((dum1,dum2,dum3,dum4,dum5),0))
global_boundary_dofs = local_boundary_dofs + local_range[0]
####################################################################
####################################################################
c,dry_dofs_local,cph,k = CFx.wave.compute_wave_speeds(x,y,sigma,theta,depth_func,u_func,v_func,N_dof_2)
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
        return (1-np.exp(-0.02*t))*E*CDIR

elif Model_Params["Boundary Type"] == "JONSWAP":
    def u_func(x,y,sigma,theta,c,t,JONgamma=3.3):
        HS = Model_Params["JONSWAP Params"][0]
        PKPER = Model_Params["JONSWAP Params"][1]
        MDIR = Model_Params["JONSWAP Params"][2]
        MSINPUT = Model_Params["JONSWAP Params"][3]
        
        S_alpha = HS**2*((1/PKPER)**4)/((0.06533*(JONgamma**0.8015)+0.13467)*16)
        CPSHAP = 1.25*((1/PKPER)**4)/((sigma/(2*np.pi))**4)
        RA = np.zeros(CPSHAP.shape)
        RA[CPSHAP<=10.0] = (S_alpha/((sigma[CPSHAP<=10.0]/(2*np.pi))**5))*np.exp(-CPSHAP[CPSHAP<=10.0])
        
        coeff = 0.07*np.ones(CPSHAP.shape)
        coeff[sigma>=(1/PKPER*2*np.pi)] = 0.09

        APSHAP = 0.5*((sigma/(2*np.pi)-1/PKPER)/(coeff*(1/PKPER)))**2
        
        SYF = np.ones(CPSHAP.shape)
        SYF[APSHAP<=10.0] = 3.3**(np.exp(-APSHAP[APSHAP<=10.0]))

        N = SYF*RA/(sigma*2*np.pi)

        if MSINPUT <12:
            CTOT = 2**MSINPUT/(2*np.pi)
        else:
            CTOT = np.sqrt(0.5*MSINPUT/np.pi)/(1-0.25/MSINPUT)

        A_COS = np.cos(theta-MDIR*np.pi/180)

        CDIR = np.zeros(A_COS.shape)

        CDIR[A_COS>0.0] = CTOT*np.maximum(A_COS[A_COS>0.0]**MSINPUT,1e-10)
        tol =1e-11
        #return (1-np.exp(-0.02*t))*(y<tol)*CDIR*N
        return np.exp(-0.002*133*(y-7.4))*CDIR*N
        #return np.exp(-0.002*y*(np.exp(-0.02*t)))*CDIR*N
        #return CDIR*N

####################################################################
####################################################################


#####################################################################
#####################################################################
#Preallocate and load/assemble cartesian mass matrix!
#now need to mass matrixes for stiffness and RHS, also optionally can out put the nnz
M_NNZ = CFx.assemble.build_cartesian_mass_matrix(M1,M2,M1_sizes,M1_global_size,M2_sizes,M)
A.setPreallocationNNZ(M_NNZ)
M_SUPG.setPreallocationNNZ(M_NNZ)
RHS.setPreallocationNNZ(M_NNZ)
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
    #mass matrix for SUPG
    M_SUPG = M+M_SUPG
    RHS = M_SUPG-(1-theta_param)*A
    A=A+RHS
if method == 'CG' or method == 'CG_strong':
    RHS = M-(1-theta_param)*A
    A = A + RHS

#fixing 0 on diagonal due to wetting drying
dry_dofs = CFx.utils.fix_diag(A,local_range[0],rank)
A.zeroRowsColumns(dry_dofs,diag=1)

##################################################################
##################################################################
#initialize vectors
#holds dirichlet boundary values
u_cart = PETSc.Vec()
u_cart.create(comm=comm)
u_cart.setSizes((local_rows,global_rows),bsize=1)
u_cart.setFromOptions()

#holds temporary values to contribute to RHS
Temp = u_cart.duplicate()
#RHS of linear system of equations
B = u_cart.duplicate()
#Post Processiong
#E = u_D.duplicate()
#L2_E = u_D.duplicate()
#u_exact = u_D.duplicate()
#solution vector
u_D = u_cart.duplicate()


Temp.setFromOptions()
#E.setFromOptions()
B.setFromOptions()
#L2_E.setFromOptions()
u_D.setFromOptions()
#u_exact.setFromOptions()
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

if Model_Params["Source Terms"]=="off":
    u_cart,xdmf=CFx.timestep.no_source(t,nt,dt,u_cart,ksp2,RHS,C,x,y,sigma,theta,c,u_func,local_boundary_dofs,global_boundary_dofs,nplot,xdmf,HS,dofs1,V2,N_dof_1,N_dof_2,local_range2)
elif Model_Params["Source Terms"]=="Wind":
    #############
    #Preprocessing grid for DIA
    #Only works for logarithmic frequency spacing
    #save map for unique thetas
    thets_unique,inverse_thets = np.unique(domain2.geometry.x[:,1],return_inverse=True)
    #the map is side by side, append this
    map_to_matrix= np.array([inverted,inverse_thets])
    map_to_matrix = np.kron(np.ones(N_dof_1),map_to_matrix)
    map_to_matrix = map_to_matrix.T
    map_to_matrix = np.column_stack((map_to_matrix, np.kron(np.arange(N_dof_1),np.ones(N_dof_2)) ) )
    #need to flatten the map
    flat_map = np.array(map_to_matrix[:,0]*((n_theta+1)*N_dof_1) + map_to_matrix[:,1]*(N_dof_1) + map_to_matrix[:,2],dtype=np.int32)
    #need the inverse
    inverse_map = np.argsort(flat_map)
    WWINT,WWAWG,WWSWG,DIA_PARAMS = CFx.utils.DIA_weights(new_coords,thets_unique,g=9.81)
    
    
    #S_nl=CFx.source.Snl_DIA(WWINT,WWAWG,WWSWG,1,DIA_PARAMS,new_coords,thets_unique,dum.vector,sigma_vec,inverse_map,flat_map)
    
    U10 = Model_Params["U10"]
    theta_wind = Model_Params["Wind Direction"]*np.pi/180
    #u_cart,xdmf = CFx.timestep.strang_split(t,nt,dt,u_cart,ksp2,RHS,C,CFx.source.Gen3,x,y,sigma,theta,c,cph,k,depth,u_func,local_boundary_dofs,global_boundary_dofs,nplot,xdmf,HS,dofs1,V2,N_dof_1,N_dof_2,local_range2,U10,theta_wind,rows)
    u_cart,xdmf = CFx.timestep.strang_split(t,nt,dt,u_cart,ksp2,RHS,C,CFx.source.Gen3,x,y,sigma,theta,c,cph,k,depth,u_func,local_boundary_dofs,global_boundary_dofs,nplot,xdmf,HS,dofs1,V2,N_dof_1,N_dof_2,local_range2,U10,theta_wind,rows,\
            WWINT,WWAWG,WWSWG,DIA_PARAMS,new_coords,thets_unique,inverse_map,flat_map,dry_dofs)



#print final iterations
#HS_vec = CFx.wave.calculate_HS_actionbalance(u_cart,V2,N_dof_1,N_dof_2,local_range2)
#HS.vector.setValues(dofs1,np.array(HS_vec))
#HS.vector.ghostUpdate()
#xdmf.write_function(HS,t)
#xdmf.close()
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
numpoints = int(Model_Params["Station Params"][0])
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
            vals_out[:,a] = HS_vals.flatten()[:]
        if QOI == "Mean Dir":
            vals_out[:,a] = Dir_vals.flatten()[:]
        a+=1
    np.savetxt(out_dir+'Stations/'+fname+".csv", np.append(stats, vals_out, axis=1), delimiter=",")

