#any functions that deal with time stepping
import CFx.wave
import numpy as np
import time
from petsc4py import PETSc

def no_source(t,nt,dt,u,ksp,M,C,x,y,sigma,theta,c,u_func,local_boundary_dofs,global_boundary_dofs,nplot,xdmf,HS,dofs1,V2,N_dof_1,N_dof_2,local_range2):

    #holds temporary values to contribute to RHS
    d = u.duplicate()
    #RHS of linear system of equations
    b = u.duplicate()
    #pointwise value of dirichlet boundary vector
    u_D = u.duplicate()


    d.setFromOptions()
    b.setFromOptions()
    u_D.setFromOptions()

    for i in range(nt):
        t+=dt
        #B will hold RHS of system of equations
        M.mult(u,b)

        #setting dirichlet BC
        #u_d_vals = u_func(x,y,sigma,theta,c,t)[local_boundary_dofs]
        u_d_vals = u_func(x[local_boundary_dofs],y[local_boundary_dofs],sigma[local_boundary_dofs],theta[local_boundary_dofs],c[local_boundaey_dofs,:],t)
        u_D.setValues(global_boundary_dofs,u_d_vals)
        C.mult(u_D,d)
        b = b - d
        b.setValues(global_boundary_dofs,u_d_vals)
        b.assemble()


        #solve for time t
        ksp.solve(b, u)

        #is this necessary?
        b.zeroEntries()


        # Save solution to file in VTK format
        if (i%nplot==0):
            #u.vector.setValues(dofs1, np.array(u_cart.getArray()[4::N_dof_2]))
            #xdmf.write_function(u, t)
            HS_vec = CFx.wave.calculate_HS_actionbalance(u,V2,N_dof_1,N_dof_2,local_range2)
            HS.vector.setValues(dofs1,np.array(HS_vec))
            HS.vector.ghostUpdate()
            xdmf.write_function(HS,t)

    HS_vec = CFx.wave.calculate_HS_actionbalance(u,V2,N_dof_1,N_dof_2,local_range2)
    HS.vector.setValues(dofs1,np.array(HS_vec))
    HS.vector.ghostUpdate()
    xdmf.write_function(HS,t)
    xdmf.close()

    return u,xdmf

def Euler_Step(N,L,dt,sigmas,thetas,U_mag,theta_wind,c,k,S,rows,V2,local_size1,local_size2,local_range2):
    return  N+dt*L(S,sigmas,thetas,N,U_mag,theta_wind,c,k,rows,V2,local_size1,local_size2,local_range2)
    

def SSP_RK2(N,A,dt,sigmas,thetas,U_mag,theta_wind,c,S,rows,k,V2,local_size1,local_size2,local_range2):
    #takes state N, operator A and time step dt and advances the state using SSP_RK2
    N1 = Euler_Step(N,A,dt,sigmas,thetas,U_mag,theta_wind,c,k,S,rows,V2,local_size1,local_size2,local_range2)
    N2 = 0.5*N + 0.5*Euler_Step(N1,A,dt,sigmas,thetas,U_mag,theta_wind,c,k,S,rows,V2,local_size1,local_size2,local_range2)
    return N2


def strang_split(t,nt,dt,u,ksp2,RHS2,C,S,x,y,sigma,theta,c,cph,k,u_func,local_boundary_dofs,global_boundary_dofs,nplot,xdmf,HS,dofs1,V2,N_dof_1,N_dof_2,local_range2,U10,theta_wind,rows):
    #preforms time loop with strang splitting given 2 sets of operators ksp1,RHS1 and ksp2,RHS2


    #dt split
    dt_strang = dt/2

    #holds temporary values to contribute to RHS
    d = u.duplicate()
    #RHS of linear system of equations
    b = u.duplicate()
    #pointwise value of dirichlet boundary vector
    u_D = u.duplicate()
    #pointwise value for source term contribution
    S_D = u.duplicate()
    

    d.setFromOptions()
    b.setFromOptions()
    u_D.setFromOptions()
    S_D.setFromOptions()

    for i in range(nt):
        t+=dt


        #substep 1
        u = SSP_RK2(u,S,dt_strang,sigma,theta,U10,theta_wind,cph,S_D,rows,k,V2,N_dof_1,N_dof_2,local_range2)
        #correct boundary
        #u_d_vals = u_func(x,t,sigma,theta,c,t-dt_strang)[local_boundary_dofs]
        #u.setValues(global_boundary_dofs,u_d_vals)
        #u.assemble()

        #substep 2
        

        #B will hold RHS of system of equations
        RHS2.mult(u,b)

        #setting dirichlet BC
        u_d_vals = u_func(x,y,sigma,theta,c,t)[local_boundary_dofs]
        u_D.setValues(global_boundary_dofs,u_d_vals)
        C.mult(u_D,d)
        b = b - d
        b.setValues(global_boundary_dofs,u_d_vals)
        b.assemble()


        #solve for time t
        ksp2.solve(b, u)

        #is this necessary?
        b.zeroEntries()


        #substep3
        #u = SSP_RK2(u,S,dt_strang,sigma,theta,U10,theta_wind,cph,S_D,rows)
        u = SSP_RK2(u,S,dt_strang,sigma,theta,U10,theta_wind,cph,S_D,rows,k,V2,N_dof_1,N_dof_2,local_range2)
        #correct boundary
        #u.setValues(global_boundary_dofs,u_d_vals)
        #u.assemble()
        # Save solution to file in VTK format
        if (i%nplot==0):
            #t1 = time.time()
            #u.vector.setValues(dofs1, np.array(u_cart.getArray()[4::N_dof_2]))
            #xdmf.write_function(u, t)
            HS_vec = CFx.wave.calculate_HS_actionbalance(u,V2,N_dof_1,N_dof_2,local_range2)
            HS.vector.setValues(dofs1,np.array(HS_vec))
            HS.vector.ghostUpdate()
            xdmf.write_function(HS,t)
            #t2=time.time()
            #PETSc.Sys.Print("Time for computation and plotting at t=",t,"is ",t2-t1)

    if nplot !=1:
        HS_vec = CFx.wave.calculate_HS_actionbalance(u,V2,N_dof_1,N_dof_2,local_range2)
        HS.vector.setValues(dofs1,np.array(HS_vec))
        HS.vector.ghostUpdate()
        xdmf.write_function(HS,t)
    xdmf.close()

    return u,xdmf
