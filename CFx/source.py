import numpy as np
import CFx.wave


#computes source terms pointwise
#####################################################################
#ST6 Package. To be completed later
#this is from Rogers 2012 (ST6)
'''
def S_in(sigma,theta,E,U10,theta_wind,cg):
    #sigma,theta are numpy, E is PETSc vector
    W = (np.maximum(np.zeros(),U10/cg*np.cos(theta_wind-theta)-1))**2

    A = integral(E*dtheta)

    Bn = A*E*cg*k**3

    G = 2.8 - (1+np.tanh(10*np.sqrt(Bn)*W - 11))

    gamma = G*np.sqrt(Bn)*W

    B = gamma*sigma*rho_a/rho_w

    S = B*E

    return S
'''
#######################################################################


######################################################################
#Gen3 source terms as in default options from SWAN

def S_in(sigmas,thetas,N,U_mag,theta_wind,c,g=9.81):
    #exponential wave growth from Komen 1984
    #sigmas - vector np.array of sigma at each computational point (rn it is radian freq)
    #thetas - in radians
    #E - np.array vector with the given variance density spectrum (may want to switch to action density later)
    #U_mag - magnitude of wind 10 m above surface in m/s, just a float right now (need to add direction)
    #c - phase speed, should be same dimension as E
    #optional g - gravity
    
    #parameters that shouldn't change
    rho_a=1.225 #denisty of air at STP (could make this more sophisticated later on)
    rho_w=1000 #density of water

    #in an ideal world, do all these calculations in PETSc but I need to get this done quick so sorry:(
    E = N.getArray()

    C_d=1.2875*(10**(-3)) if U_mag < 7.5 else (0.8+0.065*U_mag)*(10**(-3))
    U_fric = np.sqrt(C_d*(U_mag**2))
    #sigmapm=0.13*g*2*np.pi/(28*U_fric)
    #H=np.exp(-(sigmas/sigmapm)**4)
    #Expression version for FENICS
    #A = Expression('1.5*pow(10,-3)/(2*pi*pow(g,2))*pow(U_fric,4)*exp(-pow(x[0]/(2*pi*0.13*g/(28*U_fric)),-4))'
    #               ,degree=p_degree, U_fric=U_fric, g=g)
    #B = Expression('max(0.0,0.25*rho_a/rho_w*(28*U_fric/c_ph-1))*x[0]',
    #               degree=p_degree,rho_a=rho_a,rho_w=rho_w,U_fric=U_fric,c_ph=c)
    #A = 1.5*10**(-3)/(2*np.pi*g**2)*(U_fric*np.maximum(0,np.cos(thetas-theta_wind)))**4*H
    B= np.maximum(np.zeros(c.shape),0.25*rho_a/rho_w*(28*U_fric/c*np.cos(thetas-theta_wind)-1))*sigmas
    
    #S.setValues(rows,B*E)
    #note that even though it is E, I left it as action balance N
    #S.assemble()
    return B*np.maximum(0.0,E)


def S_wc(sigmas,thetas,k,N,local_size2,Etot,sigma_factor,k_factor,opt=1):
    #########################################################
    #S_wc - whitecapping
    #follows WAM Cycle III formulation (see swantech manual)
    
    #Inputs:
    #sigmas - numpy array of the relative radian frequency of each d.o.f
    #k - wavenumber corresponding to each sigma
    #E - right now spectral density at each d.o.f but maybe change to action density (be careful!)
    #Output:
    #S_wc - numpy array same size as E with the spectral density per unit time

    #constant
    C_ds = 2.36e-5
    n_wc = 0.0
    p_wc=2
    mean_spm=np.sqrt(3.02e-3)

    
    #print('Max and min of integrated variables')
    #print(np.amax(Etot),np.amax(sigma_factor),np.amax(k_factor))
    #print(np.amin(Etot),np.amin(sigma_factor),np.amin(k_factor))
    if np.any(np.isnan(Etot)):
        print('Etot contains nans')
    if np.any(np.isnan(sigma_factor)):
        print('Sigma factor contains nans')
    if np.any(np.isnan(k_factor)):
        print('k_factor contains nans')
    
    #set a tolerance to prevent division by zero, if Etot is below tolerance then the S_wc =0 there
    tol = 1e-7

    #mask that lives in domain 1
    valid_idx1 = Etot>tol
    valid_idx2 = sigma_factor>tol
    valid_idx3 = k_factor>tol
    
    valid_idx = np.logical_and(np.logical_and(valid_idx1,valid_idx2),valid_idx3)

    gamma_factor = np.zeros(k.shape)
    #need check to see if any nonzeros, if valid_idx is all False then np.kron throws an error
    if np.any(valid_idx):
        #corresponding mask that lives in knronecker product space
        big_idx = np.kron(valid_idx,np.ones(local_size2))
        big_idx=np.array(big_idx, dtype=bool)

        sigma_tilde = np.zeros(Etot.shape)
        k_tilde = np.zeros(Etot.shape)
        s_tilde = np.zeros(Etot.shape)

        if opt==2:
            sigma_tilde[valid_idx] = sigma_factor[valid_idx]/Etot[valid_idx]
            k_tilde[valid_idx] = (k_factor[valid_idx]/Etot[valid_idx])**(2)
        else:
            sigma_tilde[valid_idx] = Etot[valid_idx]/sigma_factor[valid_idx]
            k_tilde[valid_idx] = (k_factor[valid_idx]/Etot[valid_idx])**(-2)
        #k_tilde[valid_idx] = Etot[valid_idx]**2/(k_factor[valid_idx]**2)
    

        s_tilde = k_tilde*np.sqrt(Etot)

        gamma = C_ds*((1-n_wc) + n_wc*k[big_idx]/(np.kron(k_tilde[valid_idx],np.ones(local_size2))) )*((np.kron(s_tilde[valid_idx],np.ones(local_size2))/mean_spm)**p_wc)

        gamma_factor[big_idx] = gamma*(np.kron(sigma_tilde[valid_idx]/k_tilde[valid_idx],np.ones(local_size2)))*k[big_idx] 


    if np.any(np.absolute(gamma_factor)>1):
        print('gamma may be blowing up',np.amax(gamma_factor),np.amin(gamma_factor))
        dum = np.absolute(gamma_factor)
        i1 = np.argmax(dum)
        i_small = int(np.floor(i1/local_size2))
        print('index and value of gamma blowing up',i1,i_small,gamma_factor[i1])
        print("Etot %.2E" % Etot[i_small])
        print('some values, Etot, k_tilde, s_tilde,sigma_tilde',Etot[i_small],k_tilde[i_small],s_tilde[i_small],sigma_tilde[i_small])
        print('integral params Etot, sigma_factor, k factor', Etot[i_small],sigma_factor[i_small],k_factor[i_small])
        
    S = -gamma_factor*np.maximum(0.0,N.getArray())
    return S

def calc_S_bfr(sigmas,k,E,depth,local_size2,g=9.81):
    ##########################################################
    #S_bfr (bottom friction)
    #seems only relevant in very shallow water (see swantech manual)
    #Inputs
    #sigmas - numpy array of rel. radian frequencies, should correspond to d.o.f in E
    #k - wavenumber, same configuration as sigmas
    #E - numpy array of spectral density, should correspond to d.o.f same as k and sigmas
    #depth- local water depth in meters
    #Outputs
    #S_br - numpy array same size as E with spectral density per unit time

    C_bfr=0.067
    #S_bfr=Expression('-C_bfr/g*pow(x[0]/sinh(k*d),2)*E',degree=p_degree, C_bfr=C_bfr,g=g,k=k,d=H,E=E) <- Fenics ver.
    #S_bfr=project(S_bfr,V)
    #print('Shape of sigmas,k,depth,N',k.shape,sigmas.shape,depth.shape,E.getArray().shape)
    #print('Shape of local size2 kronecker depth', local_size2,np.kron(depth,np.ones(local_size2)).shape )
    S_bfr=-C_bfr/(g**2)*(sigmas/np.sinh(k*np.kron(depth,np.ones(local_size2))))**2*np.maximum(0.0,E.getArray())
    return S_bfr


def S_brk(E,depth,local_size2,m0,sigma_factor):
    alpha_bj = 1
    Hrms = np.sqrt(8*m0)
    Hmax = 0.73*depth
    beta = Hrms/Hmax

    sigma_mean = np.zeros(m0.shape)
    Q0 = np.zeros(beta.shape)
    Qb = np.zeros(beta.shape)
    factor1 = np.zeros(m0.shape)

    tol = 1e-4
    #mask that lives in domain 1
    valid_idx1 = m0>tol


    if np.any(valid_idx1):
        sigma_mean[valid_idx1] = sigma_factor[valid_idx1]/m0[valid_idx1]
    
        mask1 = np.logical_and(beta>=0.5,beta<=1.0)
        Q0[mask1] = (2*beta[mask1]-1)**2
        
        
        
        mask2 = np.logical_and(beta>=0.2,beta<1)
        Qb[mask2] = Q0[mask2] - beta[mask2]**2*(Q0[mask2]-np.exp((Q0[mask2]-1)/beta[mask2]**2))/(beta[mask2]**2-np.exp((Q0[mask2]-1)/(beta[mask2]**2)));
        Qb[beta>=1] = 1

        factor1[valid_idx1] = -alpha_bj*Qb[valid_idx1]*sigma_mean[valid_idx1]/(beta[valid_idx1]**2*np.pi)
    
    if np.any(factor1>0):
        print("Breaking parameter is positive when it probably shouldnt be")
        print("Minimum and maximum Qb value", np.amax(Qb),np.amin(Qb))
    if np.any(factor1<-1):
        print('S_break getting big',np.amax(factor1))
    S_brk = np.kron(factor1,np.ones(local_size2))*np.maximum(0.0,E.getArray())
    return S_brk

def S_nl4_DIA(sigmas,thetas,N,V2,mesh2):
    ##DIA approximation
    #parameters for DIA
    #note I think we may still need to convert back to rel. rad. frequency
    #but idk
    #convert spectrum to from rad. freq. to freq by multiplying by Jacob
    Ef = N.getArray()*2*np.pi*sigmas

    #resonance conditions for quadruplets
    lambda1=1+0.25
    lambda2=1-.25
    theta1=11.48*np.pi/180#in degrees
    theta2=-33.56*np.pi/180#in degrees
    DAL1   = 1. / (lambda1)**4
    DAL2   = 1. / (lambda2)**4
    DAL3   = 2. * DAL1 * DAL2

    #first get the structured version of the dofs
    #need to convert to regular grid
    dof=np.zeros((len(sigmas),2))
    dof[:,0]=sigmas
    dof[:,1]=thetas
    Matrix_indeces=dof_2_meshgrid_indeces(dof)
    mini_sigma=np.unique(sigmas)
    mini_thets=np.unique(thetas)#indeces_to_meshgrid(thets,Matrix_indeces)[:,0]


    sig_low=min(mini_sigma)
    sig_high=max(mini_sigma)
    thet_low=min(mini_thets)
    thet_high=max(mini_thets)
    num_thet=len(mini_thets)
    num_sigma=len(mini_sigma)

    #establish extended spectrum to interpolate from
    sigmas_e=np.insert(mini_sigma,0,sig_low/lambda1)
    sigmas_e=np.append(sigmas_e,sig_high/lambda2)

    '''
    thetas_e=np.insert(mini_thets,0,thet_low-max(np.absolute([theta1,theta2])))
    thetas_e=np.append(thetas_e,thet_high+max(np.absolute([theta1,theta2])))


    SIGMAS_E,THETS_E=np.meshgrid(sigmas_e,thetas_e)

    #now interpolote given spectrum onto bigger grid, also need to figure out
    #to add exponential tail
    E2=indeces_to_meshgrid(E,Matrix_indeces)
    f=interp.interp2d(mini_sigma,mini_thets,E2,fill_value=0,bounds_error=False)
    E00=f(sigmas_e,thetas_e)
    
    #Need an even larger extenion to interpolate from
    #will keep this commented for now
    #sigmas_e2=[sig_low*(lambda2/lambda1);sigs';sig_high*lambda1/lambda2];
    #thetas_e2=[thetlow-2*max(abs([theta1,theta2]));thetas';thethigh+2*max(abs([theta1,theta2]))];
    #[SIGMAS_E2,THETAS_E2]=meshgrid(sigmas_e2,thetas_e2);
    #[N_ext2,~,~]=JONSWAP_2D_rad(sigmas_e2,thetas_e2,Hs,Pkper,MDIR,MSInput,LSHAPEin,DSHAPL);
    #N_ext2=N_ext2*2*pi;
    
    sigmas_e2=sigmas_e
    thetas_e2=thetas_e
    N_ext2=E00

    #now obtain the 4 shifted bins

    #here is spectra shifted by +1 in frequency,+1 in theta1
    sigmas_01=lambda1*sigmas_e
    thetas_01=theta1+thetas_e
    f=interp.interp2d(sigmas_e2,thetas_e2,E00,fill_value=0,bounds_error=False)
    EP1=f(sigmas_01,thetas_01)    
    

    #here is spectra shifted by -1 in frequency, -1 in theta2
    sigmas_02=lambda2*sigmas_e
    thetas_02=theta2+thetas_e
    EM1=f(sigmas_02,thetas_02)


    #now flipping directions we obtain +1 in freq, -1 in theta1
    sigmas_03=lambda1*sigmas_e
    thetas_03=-theta1+thetas_e
    EP2=f(sigmas_03,thetas_03)


    #and finally -1 in freq -1 in theta2
    sigmas_04=lambda2*sigmas_e
    thetas_04=-theta2+thetas_e
    EM2=f(sigmas_04,thetas_04)
    
    #calculate the nonlinear reaction as wavewatch does
    #CONS=3*10^7*9.81^(-4);
    SNLC1  = 1. / g**4
    SNLCS1 = 5.5
    SNLCS2 = 0.833
    SNLCS3 = -1.25
    X = 1#max( 0.75 * DEP2(KCGRD(1)) * KMESPC , 0.5 );
    X2 = 1#% MAX ( -1.E15, SNLCS3*X)
    CONS   = SNLC1 * ( 1. + SNLCS1/X * (1.-SNLCS2*X) * exp(X2))
    PQUAD2= 3.E7
    FACTOR = CONS * (SIGMAS_E/(2*np.pi))**11 * E00
    SA1A   = E00 * ( EP1*DAL1 + EM1*DAL2 )*PQUAD2
    SA1B   = SA1A - EP1*EM1*DAL3*PQUAD2
    SA2A   = E00 * ( EP2*DAL1 + EM2*DAL2 )*PQUAD2
    SA2B   = SA2A - EP2*EM2*DAL3*PQUAD2

    SA1 = FACTOR * SA1B
    SA2 = FACTOR * SA2B

    del_Snl1=SA1+SA2;
    
    del_Snl1=del_Snl1[1:num_thet+1,1:num_sigma+1]
    #print(del_Snl1.shape)
    #now interpolate one last time
    f_SA1=interp.interp2d(sigmas_01,thetas_01,SA1,fill_value=0,bounds_error=False)
    f_SA2=interp.interp2d(sigmas_03,thetas_03,SA2,fill_value=0,bounds_error=False)
    del_Snl2=f_SA1(mini_sigma,mini_thets)+f_SA2(mini_sigma,mini_thets)
    
    f_SA1=interp.interp2d(sigmas_02,thetas_02,SA1,fill_value=0,bounds_error=False)
    f_SA2=interp.interp2d(sigmas_04,thetas_04,SA2,fill_value=0,bounds_error=False)
    del_Snl3=f_SA1(mini_sigma,mini_thets)+f_SA2(mini_sigma,mini_thets)
    
    
    #now calculate source term
    S_nl4=-2*del_Snl1+del_Snl2+del_Snl3
    #and lastly reshape
    S_nl4=meshgrid_to_dof(S_nl4,Matrix_indeces)
    '''
    return S_nl4

def Gen3(S,sigmas,thetas,N,U_mag,theta_wind,c,k,depth,rows,V2,local_size1,local_size2,local_range2,g=9.81):
    #calculate any necessary integral parameters
    #int int E dsigma dtheta = int int N*sigma dsigma dtheta
    Etot = CFx.wave.calculate_Etot(N,V2,local_size1,local_size2,local_range2)
    #int int E/sigma dsigma dtheta = int int N dsgima dtheta
    sigma_factor = CFx.wave.calculate_sigma_tilde(N,V2,local_size1,local_size2,local_range2) 
    #alternative since something strange is happening with sigma factor
    sigma_factor2 = CFx.wave.calculate_sigma_tilde2(N,V2,local_size1,local_size2,local_range2)
    #int int E/sqrt(k) dsigma dtheta= int int N*sigma/sqrt(k) dsigma dtheta
    k_factor=CFx.wave.calculate_k_tilde(k,N,V2,local_size1,local_size2,local_range2)
    #int int Esqrt(k)
    k_factor2=CFx.wave.calculate_k_tilde2(k,N,V2,local_size1,local_size2,local_range2)

    Sin =   S_in(sigmas,thetas,N,U_mag,theta_wind,c,g=9.81) 
    Swc = S_wc(sigmas,thetas,k,N,local_size2,Etot,sigma_factor2,k_factor2,opt=2)
    Sbfr = calc_S_bfr(sigmas,k,N,depth,local_size2)
    Sbrk = S_brk(N,depth,local_size2,Etot,sigma_factor2)
    S.setValues(rows,Sin+Swc+Sbfr+Sbrk)
    #print("max/min of source terms",np.amax(Sin),np.amax(Swc),np.amin(Sin),np.amax(Swc))
    #print("max/min of incoming action balance",np.amax(N.getArray()),np.amin(N.getArray()))
    S.assemble()
    return S

'''
def calc_S_dsbr(sigmas,thetas,E,depth):
    #########################################################
    #S_dsbr
    #depth induced wave breaking

    #Inputs
    #sigmas
    #E
    #optional (need them but assuming named mesh1 and V)

    alpha_bj=1

    ###########################################
    #fenics version, doesn't work for 2d slice
    #B=Function(V)
    #B.vector()[:] = E
    #E_fenics=interpolate(B,V)
    #B=Function(V)
    #B.vector()[:]=sigmas
    #sigmas_fenics=interpolate(B,V)
    #m0=assemble(E_fenics*dx(mesh1))
    #print(m0)
    #mean_sigma = 1/m0*assemble(E_fenics*sigmas_fenics*dx(mesh1))
    #print(mean_sigma)

    ###########################################
    #manual integration, works on 2d slices
    dof=np.zeros((len(sigmas),2))
    dof[:,0]=sigmas
    dof[:,1]=thetas
    Matrix_indeces=dof_2_meshgrid_indeces(dof)
    mini_sigma=np.unique(sigmas)
    mini_thets=np.unique(thetas)#indeces_to_meshgrid(thets,Matrix_indeces)[:,0]
    m0=calc_double_int(E,mini_sigma,mini_thets,Matrix_indeces)
    mean_sigma=(1/m0*calc_double_int(sigmas*E,mini_sigma,mini_thets,Matrix_indeces))
    #print(m0)

    Hrms=np.sqrt(8*m0)
    H_max=0.73*depth
    beta=Hrms/H_max
    #print(beta)
    Q_0=0 if beta <= 0.5 else (2*beta-1)**2 if beta <=1 else 1
    num1=Q_0-(np.exp((Q_0-1)/(beta**2)))
    dem1=beta**2-(np.exp((Q_0-1)/(beta**2)))
    dum=Q_0-(beta**2)*(num1)/(dem1)
    Qb=0 if beta<0.2 else dum if beta <1 else 0.5
    print(Qb)


    S_dsbr=-alpha_bj*Qb*mean_sigma*E/(beta**2*np.pi)
    return S_dsbr
'''
