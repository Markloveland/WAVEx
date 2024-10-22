"""
Action Balance Equation Solver
This algorithm for loading will be same required of Action Balance Equation
    du/dt + \/.cu = f
"""

import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import sparse as sp
from petsc4py import PETSc
from dolfinx import fem,mesh,io
import ufl
import time
import CFx.wave
import CFx.utils
import CFx.assemble
import CFx.transforms
import CFx.boundary
time_start = time.time()



#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#soecify domain size
L = 10
# Create cartesian mesh of two 2D and define function spaces
nx = 16
ny = 16
#set initial time
t = 0
#set final time
t_f = 5
#set time step
#dt = 1.0
dt = 0.005
#calculate nt
nt = int(np.ceil(t_f/dt))
PETSc.Sys.Print('nt',nt)
#plot every n time steps
#nplot = 1
nplot = 100

method = 'SUPG_strong'
PETSc.Sys.Print('Method chosen:', method)
####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
domain1 = mesh.create_rectangle(comm, [np.array([0, 0]), np.array([L, L])], [nx, nx], mesh.CellType.triangle)
V1 = fem.FunctionSpace(domain1, ("CG", 1))
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)
####################################################################
###################################################################
#Special section for wetting/drying
elementwise = fem.FunctionSpace(domain1,("DG",0))
is_wet = fem.Function(elementwise)
depth_func = fem.Function(V1)
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
depth_func.x.array[:] = L + dof_coords1[:,0]
#need to include a wetting/drying variable in domain 1
CFx.wave.calculate_wetdry(domain1, V1, depth_func,is_wet,min_depth=0.05)
####################################################################
####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh
domain2 = mesh.create_rectangle(MPI.COMM_SELF, [np.array([0, 0]), np.array([L, L])], [ny, ny], mesh.CellType.triangle)
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

#now only want subset that is the inflow, need to automate later
x_min = 0
y_min = 0
sigma_min = 0
theta_min = 0
dum1 = local_boundary_dofs[x[local_boundary_dofs]<=(x_min+1e-14)]
dum2 = local_boundary_dofs[y[local_boundary_dofs]<=(y_min+1e-14)]
dum3 = local_boundary_dofs[sigma[local_boundary_dofs]<=(sigma_min+1e-14)]
dum4 = local_boundary_dofs[theta[local_boundary_dofs]<=(theta_min+1e-14)]
local_boundary_dofs = np.unique(np.concatenate((dum1,dum2,dum3,dum4),0))
#local_boundary_dofs = dum1
global_boundary_dofs = local_boundary_dofs + local_range[0]
####################################################################
####################################################################
#generate any coefficients that depend on the degrees of freedom
c = 2*np.ones(local_dof.shape)
#c[:,1:] = 0
#c[:,1] = 1
#exact solution and dirichlet boundary
def u_func(x,y,sigma,theta,c,t):
    return np.sin(x-c[:,0]*t) + np.cos(y-c[:,1]*t) + np.sin(sigma-c[:,2]*t) + np.cos(theta-c[:,3]*t)
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
    tau_vals = CFx.wave.compute_tau(domain1,domain2,c,N_dof_1,N_dof_2)
    tau_old = CFx.wave.compute_tau_old(domain1,domain2,c,subdomain=0) 
    #print('old tau',tau_old[0:10])
    #print('new tau',tau_vals[0:10])
    #print('Tau vals',tau_vals.shape)
    #print('c vals',c.shape)
    CFx.assemble.build_action_balance_stiffness(domain1,domain2,V1,V2,c,dt,A,method=method,is_wet=is_wet,tau_vals=tau_vals)
    CFx.assemble.build_RHS(domain1,domain2,V1,V2,c,M_SUPG,is_wet=is_wet,tau_vals=tau_vals)
if method == 'CG' or method == 'CG_strong':
    CFx.assemble.build_action_balance_stiffness(domain1,domain2,V1,V2,c,dt,A,method=method,is_wet=is_wet)

time_2 = time.time()

if method == 'SUPG' or method == 'SUPG_strong':
    M_SUPG = M+M_SUPG
    A=A+M_SUPG
if method == 'CG' or method == 'CG_strong':
    M_SUPG = M
    A = A + M_SUPG

dry_dofs = CFx.utils.fix_diag(A,local_range[0],rank)

A.zeroRows(dry_dofs,diag=1)
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
A.zeroRowsColumns(global_boundary_dofs,diag=1,x=u_exact,b=u_exact)
#all_global_boundary_dofs = np.concatenate(MPI.COMM_WORLD.allgather(global_boundary_dofs))
#all_u_d = np.concatenate(MPI.COMM_WORLD.allgather(u_d))
###################################################################
###################################################################
#Time step
#u_cart will hold solution
u_dry = np.ones(dry_dofs.shape)
#create a direct linear solver
#pc2 = PETSc.PC().create()
#this is a direct solve with lu
#pc2.setType('none')
#pc2.setOperators(A)

ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
ksp2.setType('gmres')
#options are cgne(worked well), gmres, cgs, bcgs, bicg
#ksp2.setPC(pc2)
#ksp2.setUp()

ksp2.setInitialGuessNonzero(True)

fname = 'Propagation/solution'
xdmf = io.XDMFFile(domain1.comm, fname+".xdmf", "w")
xdmf.write_mesh(domain1)
#B.setValues(dry_dofs,np.ones(dry_dofs.shape))

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
    B.setValues(dry_dofs,u_dry)
    B.setValues(global_boundary_dofs,u_d_vals)
    B.assemble()
    #solve for time t
    ksp2.solve(B, u_cart)
    B.zeroEntries()
    # Save solution to file in VTK format
    if (i%nplot==0):
        u.vector.setValues(dofs1, np.array(u_cart.getArray()[4::N_dof_2]))
        xdmf.write_function(u, t)
        #hdf5_file.write(u,"solution",t)
ksp2.view()	
PETSc.Sys.Print('Niter',ksp2.getIterationNumber())
PETSc.Sys.Print('convergence code',ksp2.getConvergedReason())
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



#compute significant wave height
HS = fem.Function(V1)
HS_vec = CFx.wave.calculate_HS(u_cart,V2,N_dof_1,N_dof_2,local_range2)
HS.vector.setValues(dofs1,np.array(HS_vec))
HS.vector.ghostUpdate()
fname = 'Test_HS/solution'
xdmf = io.XDMFFile(domain1.comm, fname+".xdmf", "w")
xdmf.write_mesh(domain1)
xdmf.write_function(HS)
xdmf.close()

#try to extract HS at stations
numpoints = 150
x_stats = np.linspace(0.01,L-0.01,numpoints)

y_coord = L/2
stations = y_coord*np.ones((numpoints,2))
stations[:,0] = x_stats
points_on_proc, vals_on_proc = CFx.utils.station_data(stations,domain1,HS)

#PETSc.Sys.Print("Printing put HS along with coords as found on each process")
#print(points_on_proc,vals_on_proc)

PETSc.Sys.Print("Trying to mpi gather")
gathered_coords = comm.gather(points_on_proc,root=0)
gathered_vals = comm.gather(vals_on_proc,root=0)
coords=[]
if rank ==0:
    for a in gathered_coords:
        if a.shape[0] !=0:
            for row in a:
                coords.append(row)
    coords =np.array(coords)
    print(coords.shape)
    coords = np.unique(coords)
    print(coords.shape)
#PETSc.Sys.Print(coords)
#PETSc.Sys.Print()

