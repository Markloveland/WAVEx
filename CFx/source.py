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
    tol = 1e-4

    #mask that lives in domain 1
    valid_idx1 = Etot>tol
    valid_idx2 = sigma_factor>tol
    valid_idx3 = k_factor>tol
    
    valid_idx = np.logical_and(np.logical_and(valid_idx1,valid_idx2),valid_idx3)

    gamma_factor = np.zeros(k.shape)
    k_tilde = np.zeros(Etot.shape)
    #need check to see if any nonzeros, if valid_idx is all False then np.kron throws an error
    if np.any(valid_idx):
        #corresponding mask that lives in knronecker product space
        #big_idx = np.kron(valid_idx,np.ones(local_size2))
        #big_idx=np.array(big_idx, dtype=bool)


        big_idx = np.repeat(valid_idx,local_size2)
        big_idx=np.array(big_idx, dtype=bool)
        
        sigma_tilde = np.zeros(Etot.shape)
        
        s_tilde = np.zeros(Etot.shape)

        if opt==2:
            sigma_tilde[valid_idx] = sigma_factor[valid_idx]/Etot[valid_idx]
            k_tilde[valid_idx] = (k_factor[valid_idx]/Etot[valid_idx])**(2)
        else:
            sigma_tilde[valid_idx] = Etot[valid_idx]/sigma_factor[valid_idx]
            k_tilde[valid_idx] = (k_factor[valid_idx]/Etot[valid_idx])**(-2)
        #k_tilde[valid_idx] = Etot[valid_idx]**2/(k_factor[valid_idx]**2)
    

        s_tilde = k_tilde*np.sqrt(Etot)

        #gamma = C_ds*((1-n_wc) + n_wc*k[big_idx]/(np.kron(k_tilde[valid_idx],np.ones(local_size2))) )*((np.kron(s_tilde[valid_idx],np.ones(local_size2))/mean_spm)**p_wc)


        gamma = C_ds*((1-n_wc) + n_wc*k[big_idx]/(np.repeat(k_tilde[valid_idx],local_size2)) )*((np.repeat(s_tilde[valid_idx],local_size2)/mean_spm)**p_wc)

        #gamma_factor[big_idx] = gamma*(np.kron(sigma_tilde[valid_idx]/k_tilde[valid_idx],np.ones(local_size2)))*k[big_idx] 
        gamma_factor[big_idx] = gamma*(np.repeat(sigma_tilde[valid_idx]/k_tilde[valid_idx],local_size2))*k[big_idx] 


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
    return S,valid_idx,k_tilde

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
    #surpress over flow for deep water
    denom_min = 710
    S_bfr=-C_bfr/(g**2)*(sigmas/np.sinh( np.minimum( 710, k*np.repeat(depth,local_size2) ) ) )**2*np.maximum(0.0,E.getArray())
    
    return S_bfr


def S_brk(E,depth,local_size2,m0,sigma_factor):
    alpha_bj = 1
    Hrms = np.sqrt(8*m0)
    min_depth = 0.05 
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
        print('S_break getting big',np.amin(factor1))
    S_brk = np.repeat(factor1,local_size2)*np.maximum(0.0,E.getArray())
    return S_brk


def Snl_DIA(WWINT,WWAWG,WWSWG,NG,DIA_PARAMS,sigmas,thetas,N,all_sigmas,map_to_mat,map_to_dof,valid_idx,local_size2,k_mean,depth,g=9.81):
    MSC = DIA_PARAMS[0]
    MDC = DIA_PARAMS[1]
    sig_spacing = DIA_PARAMS[2]
    CONS = DIA_PARAMS[3]
    DAL1 = DIA_PARAMS[4]
    DAL2 = DIA_PARAMS[5]
    DAL3 = DIA_PARAMS[6]
    PQUAD2 = DIA_PARAMS[7]
    Extended_freq = DIA_PARAMS[8]
    #MSC = len(sigmas)
    #MDC = len(thetas)
    #half_nsig = int(np.floor(MSC/2))
    #half_nsig_minus = int(half_nsig - 1)
    #sig_spacing = sigmas[half_nsig]/sigmas[half_nsig_minus]
    
    MSC4MI = WWINT[14]
    MSCMAX = WWINT[18]
    MDCMAX = WWINT[19]
    ISM1 = WWINT[7]
    ISCHG = WWINT[11]
    MDC4MI = WWINT[16]
    MDC4MA = WWINT[15]
    IDCLOW = WWINT[12]
    IDCHGH = WWINT[13]
    ISP1 = WWINT[5]
    IDP1 = WWINT[1]
    IDP = WWINT[0]
    ISP = WWINT[4]
    ISM = WWINT[6]
    IDM1 = WWINT[3]
    IDM = WWINT[2]
    ISHGH = WWINT[9]    
    ISLOW = WWINT[8]
    ISCLW = WWINT[10]

    #############################3

    # need to create a matrix UE that will hold the extended spectrum
    UE = np.zeros((MSCMAX,MDCMAX,NG))
    #this will be like a meshgrid structure
    #For extended frequencies indeces 0:ISM1 are the appended low frequencies
    #ISM1:ISCHG should be original frequencies
    #ISCHG:MSCMAX should be higher than before
    #lets test
    #Extended_freq=np.zeros(MSCMAX)
    #Extended_freq[-MSC4MI+1:-MSC4MI+1+MSC]=sigmas

    #compute the parts that aren't in the range
    #Extended_freq[-MSC4MI+1+MSC:] = sig_spacing**(np.arange(1,ISHGH-MSC+1))*sigmas[-1]
    #Extended_freq[:-MSC4MI+1] = sig_spacing**(np.arange(ISLOW-1,0))*sigmas[0]

    #fill in extended freqs as needed...
    #compute extended spectrum, this is not set up for periodic full circle. directions outside of spectrum will be set to 0
    #need to get a map from the mesh to a structured matrix
    #print('ISM1,MSC4MI',ISM1,WWINT[14])
    #print('MSCMAX',MSCMAX)
    #print('ISP1,MSC4MA',WWINT[1],WWINT[15])

    Nvals = np.array(N.getArray())*all_sigmas*2*np.pi
    Narray = Nvals[map_to_mat].reshape(MSC,MDC,NG)
    #Narray = Narray
    #print('Narray shape',Narray.shape)
    UEvals = Narray

    temparr = N.array[map_to_mat].reshape(MSC,MDC,NG)
    #print('Narray max should be here',temparr[24,12,0])
    #print('actual max',np.amax(temparr),np.argmax(temparr))
    #this doesnt work
    #np.multiply(Narray.T,sigmas).T*2*np.pi
    I1 = -MSC4MI+1
    I2 = -MSC4MI+MSC+1
    J1 = -MDC4MI+1
    J2 = -MDC4MI+MDC+1

    #print('IDDUM, should be 72,45',J2,I2)
    UE[I1:I2,J1:J2,:] = UEvals
    
    
    #add spectral tail
    PWTAIL=4
    FACHFR = 1./(sig_spacing**PWTAIL)
    for a in range(MSC+1-MSC4MI,ISHGH-MSC4MI+1):
        UE[a,:,:] = UE[a-1,:,:]*FACHFR

    #print("where max should be")
    #print(UE.shape)
    #print(UE[44,48,0])
    #print('UE max and loc')
    #print(np.amax(UE))
    #print(np.argmax(UE))

    #looks like up to hear is fixed so far#######stopping point############3

    #bilinear interpolation
    I1 = ISCLW-MSC4MI
    I2 = ISCHG-MSC4MI+1
    J1 = IDCLOW -MDC4MI
    J2 = IDCHGH - MDC4MI+1


    #print('I range (should be 4 and 49)',I1,I2)
    #print('J1 (should be 25 and 72)',J1,J2)
    E00 = UE[I1:I2,J1:J2,:]
    EP1 = WWAWG[0]*UE[I1+ISP1:I2+ISP1,J1+IDP1:J2+IDP1,:] + \
        WWAWG[1]*UE[I1+ISP1:I2+ISP1,J1+IDP:J2+IDP,:] + \
        WWAWG[2]*UE[I1+ISP:I2+ISP,J1+IDP1:J2+IDP1,:] + \
        WWAWG[3]*UE[I1+ISP:I2+ISP,J1+IDP:J2+IDP,:]
    EM1 = WWAWG[4]*UE[I1+ISM1:I2+ISM1, J1-IDM1:J2-IDM1,:] + \
        WWAWG[5]*UE[I1+ISM1:I2+ISM1, J1-IDM:J2-IDM,:] + \
        WWAWG[6]*UE[I1+ISM:I2+ISM, J1-IDM1:J2-IDM1,:] + \
        WWAWG[7]*UE[I1+ISM:I2+ISM, J1-IDM:J2-IDM,:]

    EP2 = WWAWG[0]*UE[I1+ISP1:I2+ISP1,J1-IDP1:J2-IDP1,:] + \
        WWAWG[1]*UE[I1+ISP1:I2+ISP1,J1-IDP:J2-IDP,:] + \
        WWAWG[2]*UE[I1+ISP:I2+ISP,J1-IDP1:J2-IDP1,:] + \
        WWAWG[3]*UE[I1+ISP:I2+ISP,J1-IDP:J2-IDP,:]
    EM2 = WWAWG[4]*UE[I1+ISM1:I2+ISM1, J1+IDM1:J2+IDM1,:] + \
        WWAWG[5]*UE[I1+ISM1:I2+ISM1, J1+IDM:J2+IDM,:] + \
        WWAWG[6]*UE[I1+ISM:I2+ISM, J1+IDM1:J2+IDM1,:] + \
        WWAWG[7]*UE[I1+ISM:I2+ISM, J1+IDM:J2+IDM,:]
    AF11 = (Extended_freq/(2*np.pi))**11
    #print('AF11',AF11.shape,AF11[:10])
    #print('EP1 shape',EP1.shape)
    #print('max EP1',np.amax(EP1),np.argmax(EP1),EP1[21,20,0])    
    #print('max EM1',np.amax(EM1),np.argmax(EM1),EM1[27,33,0])    
    #print('max EP2',np.amax(EP2),np.argmax(EP2),EP2[21,26,0])    
    #print('max EM2',np.amax(EM2),np.argmax(EM2),EM2[27,13,0])    
    
    
    FACTOR = CONS*np.multiply(E00.T,AF11[I1:I2]).T
    #print('Factor shape,max',FACTOR.shape,np.max(FACTOR),FACTOR[44,23])
    
    SA1A = E00 *(EP1*DAL1 + EM1*DAL2)* PQUAD2
    SA1B = SA1A - EP1*EM1*DAL3 * PQUAD2
    SA2A = E00 * ( EP2*DAL1 + EM2*DAL2 )*PQUAD2
    SA2B   = SA2A - EP2*EM2*DAL3*PQUAD2
    

    #print('SA1A info',SA1A.shape,np.amax(SA1A),SA1A[23,21])
    #print('DAL1,DAl2,PQAD',DAL1,DAL2,PQUAD2)
    SA1 = np.zeros(UE.shape)
    SA2 = np.zeros(UE.shape)


    
    SA1[I1:I2,J1:J2,:] = FACTOR*SA1B
    SA2[I1:I2,J1:J2,:] = FACTOR*SA2B

    #print('max of sa1',np.amax(SA1),np.argmax(SA1))
    #print('random value of sa1')
    #compute DIA action
    I1 = -MSC4MI+1
    I2 = -MSC4MI + MSC +1
    
    J1 = - MDC4MI+1
    J2 = -MDC4MI + MDC+1

    #print('indeces should be 4,45,36,61',I1,I2,J1,J2)
    SFNL = - 2. * ( SA1[I1:I2,J1:J2,:] + SA2[I1:I2,J1:J2,:] ) \
            + WWAWG[0] * ( SA1[I1-ISP1:I2-ISP1,J1-IDP1:J2-IDP1,:] + SA2[I1-ISP1:I2-ISP1,J1+IDP1:J2+IDP1,:] ) \
            + WWAWG[1] * ( SA1[I1-ISP1:I2-ISP1,J1-IDP:J2-IDP,:] + SA2[I1-ISP1:I2-ISP1,J1+IDP:J2+IDP,:] ) \
            + WWAWG[2] * ( SA1[I1-ISP:I2-ISP,J1-IDP1:J2-IDP1,:] + SA2[I1-ISP:I2-ISP,J1+IDP1:J2+IDP1,:] ) \
            + WWAWG[3] * ( SA1[I1-ISP:I2-ISP ,J1-IDP:J2-IDP,:] + SA2[I1-ISP:I2-ISP ,J1+IDP:J2+IDP,:] ) \
            + WWAWG[4] * ( SA1[I1-ISM1:I2-ISM1,J1+IDM1:J2+IDM1,:] + SA2[I1-ISM1:I2-ISM1,J1-IDM1:J2-IDM1,:] ) \
            + WWAWG[5] * ( SA1[I1-ISM1:I2-ISM1,J1+IDM:J2+IDM,:] + SA2[I1-ISM1:I2-ISM1,J1-IDM:J2-IDM,:] ) \
            + WWAWG[6] * ( SA1[I1-ISM:I2-ISM ,J1+IDM1:J2+IDM1,:] + SA2[I1-ISM:I2-ISM ,J1-IDM1:J2-IDM1,:] ) \
            + WWAWG[7] * ( SA1[I1-ISM:I2-ISM ,J1+IDM:J2+IDM,:] + SA2[I1-ISM:I2-ISM ,J1-IDM:J2-IDM,:] )
    
    #convert back to action balance
    
    SFNL = np.multiply(SFNL.T,1/(2*np.pi*sigmas)).T

    #print('SFNL max min',np.amin(SFNL),np.amax(SFNL),SFNL[21,1])
    #now remap back to fem mesh
    S_nl_vals = SFNL.flatten()[map_to_dof] 
    
    S_nl = np.zeros(Nvals.shape)

    if np.any(valid_idx):
        #corresponding mask that lives in knronecker product space
        #big_idx = np.kron(valid_idx,np.ones(local_size2))
        big_idx = np.repeat(valid_idx,local_size2)
        big_idx = np.array(big_idx, dtype=bool)
        S_nl[big_idx] = S_nl_vals[big_idx]
    #if np.amax(S_nl)>1:
    #    print("warning,max Snl is blowing up",np.amax(S_nl))
    #if np.amin(S_nl)<-1:
    #    print("warning, min Snl is blowing up",np.amin(S_nl))

    #limiting
    tol = 1e-3

    limit_idx = np.where(S_nl>tol)[0]
    if np.any(limit_idx):
        temp = np.unique( np.floor(limit_idx/local_size2) )
        #locs = np.kron(temp,np.ones(local_size2))
        #dum = np.kron(np.ones(temp.shape),np.arange(local_size2))

        locs = np.repeat(temp,local_size2)
        dum = np.tile(np.arange(local_size2),temp.shape)

        idx2 = np.array(dum + locs*local_size2,dtype=np.int32)
        S_nl[idx2] = 0.0



    #compute shallow water correction term
    Csh1 = 5.5
    Csh2 = 5.0/6.0
    Csh3 = -5.0/4.0


    kp = 0.75*k_mean

    R = np.ones(kp.shape)

    R[valid_idx] = 1 + Csh1/(kp[valid_idx]*depth[valid_idx])*(1-Csh2*kp[valid_idx]*depth[valid_idx])*np.exp(Csh3*kp[valid_idx]*depth[valid_idx])

    #S_nl = np.kron(R,np.ones(local_size2))*S_nl
    S_nl = np.repeat(R,local_size2)*S_nl


    #print('SFNL',SFNL.shape)
    #print('Eoo shape',E00.shape)
    #print('Ep1 shape', EP1.shape)
    #print('EM1 shape',EM1.shape)
    #test by plotting the meshgrid
    #this can;t be correct
    #see if ignoring noise helps
    return S_nl




def Gen3(S,sigmas,thetas,N,U_mag,theta_wind,c,k,depth,rows,V2,local_size1,local_size2,local_range2,\
        WWINT,WWAWG,WWSWG,DIA_PARAMS,new_coords,thets_unique,inverse_map,flat_map,local_boundary_dofs,g=9.81):
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



    depth_min = 0.05
    depth = np.maximum(depth_min,depth)
    
    Sin =   S_in(sigmas,thetas,N,U_mag,theta_wind,c,g=9.81) 
    Swc,valid_idx,k_tilde = S_wc(sigmas,thetas,k,N,local_size2,Etot,sigma_factor2,k_factor2,opt=2)
    Sbfr = calc_S_bfr(sigmas,k,N,depth,local_size2)
    Sbrk = S_brk(N,depth,local_size2,Etot,sigma_factor2)
    Snl=Snl_DIA(WWINT,WWAWG,WWSWG,local_size1,DIA_PARAMS,new_coords,thets_unique,N,sigmas,inverse_map,flat_map,valid_idx,local_size2,k_tilde,depth)
    
    Snl[local_boundary_dofs] = 0.0
    Swc[local_boundary_dofs] = 0.0


    #S.setValues(rows,Sin+Swc+Sbfr+Sbrk+Snl)
    S.setValues(rows,Sin+Swc+Sbfr+Sbrk)
    
    #S.setValues(rows,Sin+Snl)
    #S.setValues(rows,Snl*0)
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
