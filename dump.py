import CFx.io
import CFx.wave
import CFx.source
import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem,mesh,io
from scipy import sparse as sp
from petsc4py import PETSc

def u_func(x,y,sigma,theta,c,t,JONgamma=3.3):
    HS = 3.5#Model_Params["JONSWAP Params"][0]
    PKPER = 10#Model_Params["JONSWAP Params"][1]
    MDIR = 90#Model_Params["JONSWAP Params"][2]
    MSINPUT = 500#Model_Params["JONSWAP Params"][3]

    S_alpha = HS**2*((1/PKPER)**4)/((0.06533*(JONgamma**0.8015)+0.13467)*16)
    CPSHAP = 1.25*((1/PKPER)**4)/((sigma/(2*np.pi))**4)
    
    RA = np.zeros(CPSHAP.shape)
    SF5 = (sigma/(2*np.pi))**5
    RA[CPSHAP<=10.0] = (S_alpha/SF5[CPSHAP<=10.0])*np.exp(-CPSHAP[CPSHAP<=10.0])
    
    
    coeff = 0.07*np.ones(CPSHAP.shape)
    coeff[sigma>=(1/PKPER*2*np.pi)] = 0.09

    APSHAP = 0.5*((sigma/(2*np.pi)-1/PKPER)/(coeff*(1/PKPER)))**2

    SYF = np.ones(CPSHAP.shape)
    SYF[APSHAP<=10.0] = 3.3**(np.exp(-APSHAP[APSHAP<=10.0]))

    N = SYF*RA/(sigma*2*np.pi)
    print('Sum Ef',np.sum(N)/25)
    if MSINPUT <12:
        CTOT = 2**MSINPUT/(2*np.pi)
    else:
        CTOT = np.sqrt(0.5*MSINPUT/np.pi)/(1-0.25/MSINPUT)

    A_COS = np.cos(theta-MDIR*np.pi/180)

    CDIR = np.zeros(A_COS.shape)

    CDIR[A_COS>0.0] = CTOT*np.maximum(A_COS[A_COS>0.0]**MSINPUT,1e-10)
    tol =1e-11
    return (y<tol)*CDIR*N

x = 0
y = 0

f_min = 0.01
f_max = .5
deg_min = 80
deg_max = 100



omega_min = f_min*2*np.pi
omega_max = f_max*2*np.pi
theta_min = deg_min*np.pi/180
theta_max = deg_max*np.pi/180
n_sigma = 40
n_theta = 24

#MPI.COMM_SELF to not partition mesh

domain2 = mesh.create_rectangle(MPI.COMM_SELF, [np.array([omega_min, theta_min]), np.array([omega_max, theta_max])], [n_sigma, n_theta], mesh.CellType.triangle)

PETSc.Sys.Print("Switching to logarithmic spacing in frequency")
#print(np.unique(domain2.geometry.x[:,0]))
old_coords,inverted = np.unique(domain2.geometry.x[:,0],return_inverse=True)
#modify x coords to be logarithmic
gamma_space = (omega_max/omega_min)**(1/n_sigma)
new_coords = gamma_space**(np.arange(n_sigma+1))*omega_min
domain2.geometry.x[:,0] = new_coords[inverted]


V2 = fem.FunctionSpace(domain2, ("CG", 1))
u2 = ufl.TrialFunction(V2)
v2 = ufl.TestFunction(V2)


dof_coords2 = V2.tabulate_dof_coordinates()
#suggested in forum, gives index of dofs I want
local_range2 = V2.dofmap.index_map.local_range
#vector of indexes that we want
dofs2 = np.arange(*local_range2,dtype=np.int32)
#gets number of dofs owned
N_dof_2 = V2.dofmap.index_map.size_local
#hopefully the dof coordinates owned by the process
local_dof_coords2 = dof_coords2[0:N_dof_2,:domain2.topology.dim]


#attempt computing u and then HS to see if I get same result
N_bc_pointwise = u_func(x,y,domain2.geometry.x[:,0],domain2.geometry.x[:,1],0,0)
print('Sum of N',np.sum(N_bc_pointwise))

dum = fem.Function(V2)
sigma = fem.Function(V2)
sigma.interpolate(lambda x: x[0])


jit_parameters = {"cffi_extra_compile_args": ["-Ofast", "-march=native"],
     "cffi_libraries": ["m"], "timeout":900}
intf = fem.form(sigma*dum*ufl.dx,jit_params=jit_parameters)

#vector of global indexes that we want
dofs = np.arange(*local_range2,dtype=np.int32)


dum.x.array[:] =N_bc_pointwise
local_intf = fem.assemble_scalar(intf)
HS = 4*np.sqrt(abs(local_intf))

print("HS = ",HS)
#print("Coords",np.unique(local_dof_coords2[:,0]),np.unique(local_dof_coords2[:,1]))
print(np.amax(dum.vector.getValues(dofs)))


print("now computing s_in")
sigma_vec = dof_coords2[:,0]
theta_vec = dof_coords2[:,1]
x = np.zeros(sigma_vec.shape)
y = np.zeros(sigma_vec.shape)
u = np.zeros(sigma_vec.shape)
v = np.zeros(sigma_vec.shape)
dHdx = np.zeros(sigma_vec.shape)
dHdy = np.zeros(sigma_vec.shape)
dudy = np.zeros(sigma_vec.shape)
dudx = np.zeros(sigma_vec.shape)
dvdy = np.zeros(sigma_vec.shape)
dvdx = np.zeros(sigma_vec.shape)
depth = np.ones(sigma_vec.shape)

c,cph,k = CFx.wave.compute_wave_speeds_pointwise(x,y,sigma_vec,theta_vec,depth,u,v,dHdx=dHdx,dHdy=dHdy,dudx=dudx,dudy=dudy,dvdx=dvdx,dvdy=dvdy,g=9.81)
U_mag = 25
theta_wind = 90
Sin = CFx.source.S_in(sigma_vec,theta_vec,dum.vector,U_mag,theta_wind,cph,g=9.81)
print(np.amax(Sin))
print(np.amin(Sin))



#calculate swc
local_size1 = 1
local_size2 = len(Sin)
#int int E dsigma dtheta = int int N*sigma dsigma dtheta
Etot = CFx.wave.calculate_Etot(dum.vector,V2,local_size1,local_size2,local_range2)
#int int E/sigma dsigma dtheta = int int N dsgima dtheta
sigma_factor = CFx.wave.calculate_sigma_tilde(dum.vector,V2,local_size1,local_size2,local_range2)
#int int E*sigma dsigma dtheta = int int N*sigma*sigma dsigma dtheta
sigma_factor2 = CFx.wave.calculate_sigma_tilde2(dum.vector,V2,local_size1,local_size2,local_range2)
#int int E/sqrt(k) dsigma dtheta= int int N*sigma/sqrt(k) dsigma dtheta
k_factor=CFx.wave.calculate_k_tilde(k,dum.vector,V2,1,local_size2,local_range2)

Sbrk = CFx.source.S_brk(dum.vector,10.0,local_size2,Etot,sigma_factor2)


print("Etot",Etot)
print("int int E/sigma",sigma_factor)
print("int int E/sqrt(k)",k_factor)
print("mean sigma",Etot/sigma_factor)
print("mean sigma version 2",sigma_factor2/Etot)
print("mean k",Etot**2/k_factor**2)
Swc = CFx.source.S_wc(sigma_vec,theta_vec,k,dum.vector,local_size2,Etot,sigma_factor,k_factor)
print("min Swc",np.amax(Swc))
print("max Swc",np.amin(Swc))
Swc = CFx.source.S_wc(sigma_vec,theta_vec,k,dum.vector,local_size2,Etot,sigma_factor2,k_factor,opt=2)
print("min SWc option 2",np.amax(Swc))
print("max SWc option2",np.amin(Swc))

print("min Sbrk option 2",np.amax(Sbrk))
print("max Sbrk option2",np.amin(Sbrk))
dum.x.array[:] = Swc

xdmf = io.XDMFFile(domain2.comm, "Outputs/JONSWAP_TEST/output.xdmf", "w")
xdmf.write_mesh(domain2)
xdmf.write_function(dum)
xdmf.close()
