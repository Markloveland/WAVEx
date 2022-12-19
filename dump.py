import CFx.io
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
    RA[CPSHAP<=10.0] = (S_alpha/(sigma[CPSHAP<=10.0]/(2*np.pi)**5))*np.exp(-CPSHAP[CPSHAP<=10.0])

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

    CDIR[A_COS>0.0] = CTOT*np.maximum(A_COS[A_COS>0.0]**MSINPUT,0.1)
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
N_bc_pointwise = u_func(x,y,local_dof_coords2[:,0],local_dof_coords2[:,1],0,0)

dum = fem.Function(V2)
sigma = fem.Function(V2)
sigma.interpolate(lambda x: x[0])


jit_parameters = {"cffi_extra_compile_args": ["-Ofast", "-march=native"],
     "cffi_libraries": ["m"], "timeout":900}
intf = fem.form(sigma*dum*ufl.dx,jit_params=jit_parameters)

#vector of global indexes that we want
dofs = np.arange(*local_range2,dtype=np.int32)


dum.vector.setValues(dofs,N_bc_pointwise)
local_intf = fem.assemble_scalar(intf)
HS = 4*np.sqrt(abs(local_intf))

print("HS = ",HS)


xdmf = io.XDMFFile(domain2.comm, "Outputs/JONSWAP_TEST/output.xdmf", "w")
xdmf.write_mesh(domain2)
xdmf.write_function(dum)
xdmf.close()
