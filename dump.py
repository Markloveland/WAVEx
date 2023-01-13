import CFx.io
import CFx.wave
import CFx.source
import CFx.utils
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
deg_min = 50
deg_max = 130



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


#save map for unique thetas
thets_unique,inverse_thets = np.unique(domain2.geometry.x[:,1],return_inverse=True)


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
#try to make this multiple nodes in geographic space just for testing purposes
NG = 5
local_size1 =NG
sigma_vec = np.kron(np.ones(NG),dof_coords2[:,0])
theta_vec = np.kron(np.ones(NG),dof_coords2[:,1])
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
depth_short = 10000*np.ones(NG)
depth = np.kron(depth_short,np.ones(N_dof_2))




#the map is side by side, append this
map_to_matrix= np.array([inverted,inverse_thets])
print('map_to_matrix shape',map_to_matrix.shape)
map_to_matrix = np.kron(np.ones(local_size1),map_to_matrix)
print('map_to_matrix shape',map_to_matrix.shape)
map_to_matrix = map_to_matrix.T
map_to_matrix = np.column_stack((map_to_matrix, np.kron(np.arange(local_size1),np.ones(N_dof_2)) ) )
print('map to matrix shape',map_to_matrix.shape)
#need to flatten the map
flat_map = np.array(map_to_matrix[:,0]*((n_theta+1)*local_size1) + map_to_matrix[:,1]*(local_size1) + map_to_matrix[:,2],dtype=np.int32)
#flat_map = np.array(map_to_matrix[:,0]*(n_theta+1)+map_to_matrix[:,1]*(local_size1)+map_to_matrix[:,2],dtype=np.int32)
#need the inverse
inverse_map = np.argsort(flat_map)


print('unique coords',new_coords,thets_unique)
print('top coords',sigma_vec[:5],theta_vec[:5])
print('potenital map',map_to_matrix[:5,:])
print('flat map',flat_map[:5])
print("inverse map",inverse_map[:5])
print("corresponnding dof",sigma_vec[inverse_map[:5]],theta_vec[inverse_map[:5]])






#this needs to be put inside a petsc vector
Nvals = np.kron(np.ones(NG),dum.vector.getArray())

u_cart = PETSc.Vec()
u_cart.create(comm=MPI.COMM_WORLD)
local_rows = NG*N_dof_2
global_rows = NG*N_dof_2
u_cart.setSizes((local_rows,global_rows),bsize=1)
u_cart.setFromOptions()
rows = np.arange(global_rows,dtype=np.int32)
u_cart.setValues(rows,Nvals)


c,cph,k = CFx.wave.compute_wave_speeds_pointwise(x,y,sigma_vec,theta_vec,depth,u,v,dHdx=dHdx,dHdy=dHdy,dudx=dudx,dudy=dudy,dvdx=dvdx,dvdy=dvdy,g=9.81)
U_mag = 25
theta_wind = 90
Sin = CFx.source.S_in(sigma_vec,theta_vec,u_cart,U_mag,theta_wind,cph,g=9.81)
print(np.amax(Sin))
print(np.amin(Sin))



#calculate swc
local_size2 = N_dof_2
#int int E dsigma dtheta = int int N*sigma dsigma dtheta
Etot = CFx.wave.calculate_Etot(u_cart,V2,local_size1,local_size2,local_range2)
#int int E/sigma dsigma dtheta = int int N dsgima dtheta
sigma_factor = CFx.wave.calculate_sigma_tilde(u_cart,V2,local_size1,local_size2,local_range2)
#int int E*sigma dsigma dtheta = int int N*sigma*sigma dsigma dtheta
sigma_factor2 = CFx.wave.calculate_sigma_tilde2(u_cart,V2,local_size1,local_size2,local_range2)
#int int E/sqrt(k) dsigma dtheta= int int N*sigma/sqrt(k) dsigma dtheta
k_factor=CFx.wave.calculate_k_tilde(k,u_cart,V2,local_size1,local_size2,local_range2)
k_factor2=CFx.wave.calculate_k_tilde2(k,u_cart,V2,local_size1,local_size2,local_range2)


Sbrk = CFx.source.S_brk(u_cart,depth_short,local_size2,Etot,sigma_factor2)


print("Etot",Etot)
print("int int E/sigma",sigma_factor)
print("int int E/sqrt(k)",k_factor)
print("mean sigma",Etot/sigma_factor)
print("mean sigma version 2",sigma_factor2/Etot)
print("mean k",Etot**2/k_factor**2)
Swc = CFx.source.S_wc(sigma_vec,theta_vec,k,u_cart,local_size2,Etot,sigma_factor,k_factor)
print("min Swc",np.amax(Swc))
print("max Swc",np.amin(Swc))
Swc = CFx.source.S_wc(sigma_vec,theta_vec,k,u_cart,local_size2,Etot,sigma_factor2,k_factor2,opt=2)
print("min SWc option 2",np.amax(Swc))
print("max SWc option2",np.amin(Swc))

print("min Sbrk option 2",np.amax(Sbrk))
print("max Sbrk option2",np.amin(Sbrk))

print('sigs',new_coords)

#thetlist = np.linspace(theta_min,theta_max,n_theta+1)
WWINT,WWAWG,WWSWG,DIA_PARAMS = CFx.utils.DIA_weights(new_coords,thets_unique,g=9.81)
S_nl=CFx.source.Snl_DIA(WWINT,WWAWG,WWSWG,local_size1,DIA_PARAMS,new_coords,thets_unique,u_cart,sigma_vec,inverse_map,flat_map)
print('max/min Snl',np.amax(S_nl),np.amin(S_nl))
dum.x.array[:] = S_nl[:N_dof_2]

xdmf = io.XDMFFile(domain2.comm, "Outputs/JONSWAP_TEST/output.xdmf", "w")
xdmf.write_mesh(domain2)
xdmf.write_function(dum)
xdmf.close()
