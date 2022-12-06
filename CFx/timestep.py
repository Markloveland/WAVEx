#any functions that deal with time stepping
import CFx.wave
import numpy as np

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
        u_d_vals = u_func(x,y,sigma,theta,c,t)[local_boundary_dofs]
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
