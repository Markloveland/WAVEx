import numpy as np
from dolfinx import geometry,mesh
import ufl


def station_data(stations,domain,f):
    #takes in a numpy array of points in 2 dimensions
    #array should be Nx2
    #domain is a fenics mesh object
    #f is a fenics function defined on the given mesh
    if stations.shape[1] >= 4:
        print("Warning, input stations is not of correct dimension!!!")

    #now transpose this to proper format
    points = np.zeros((stations.shape[0],3))
    points[:,:stations.shape[1]] = stations
    
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    f_values = []
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    f_values = f.eval(points_on_proc, cells)

    return points_on_proc,f_values

def gather_station(comm,root,local_stats,local_vals):
    rank = comm.Get_rank()
    #PETSc.Sys.Print("Trying to mpi gather")
    gathered_coords = comm.gather(local_stats,root=root)
    gathered_vals = comm.gather(local_vals,root=root)
    #PETSc.Sys.Print("directly after mpi gather",gathered_coords,gathered_vals)
    #PETSc.Sys.Print("size of new list",gathered_coords,gathered_vals)
    coords=[]
    vals = []
    if rank == root:
        for a in gathered_coords:
            if a.shape[0] != 0:
                for row in a:
                    coords.append(row)
        coords = np.array(coords)
        coords,ind1 = np.unique(coords,axis=0,return_index=True)
        
        for n in gathered_vals:
            if n.shape[0] !=0:
                for row in n:
                    vals.append(np.array(row))
        vals = np.array(vals)
        vals = vals[ind1]
    return coords,vals
    #PETSc.Sys.Print('station coords',coords)
    #PETSc.Sys.Print('station vals',vals)

def fix_diag(A,local_start,rank):
    diag = A.getDiagonal()
    dry_dofs = np.where(diag.getArray()==0)[0]
    dry_dofs = np.array(dry_dofs,dtype=np.int32) + local_start
    #print('number of 0s found on diag on rank',rank)
    #print(dry_dofs.shape)
    #print(dry_dofs)
    
    #fill in and reset
    #diag.setValues(dry_dofs,np.ones(dry_dofs.shape))
    
    #fill in matrix
    #A.setDiagonal(diag)
    return dry_dofs

#read in an adcirc mesh and give a fenicsx mesh
def ADCIRC_mesh_gen(comm,file_path):
    #specify file path as a string, either absolute or relative to where script is run
    #only compatible for adcirc fort.14 format
    adcirc_mesh=open(file_path,'r')
    title=adcirc_mesh.readline()

    #NE number of elements, NP number of grid points
    NE,NP=adcirc_mesh.readline().split()
    NE=int(NE)
    NP=int(NP)

    #initiate data structures
    NODENUM=np.zeros(NP)
    LONS=np.zeros(NP)
    LATS=np.zeros(NP)
    DPS=np.zeros(NP)
    ELEMNUM=np.zeros(NE)
    NM = np.zeros((NE,3)) #stores connectivity at each element

    #read node information line by line
    for i in range(NP):
        NODENUM[i], LONS[i], LATS[i], DPS[i] = adcirc_mesh.readline().split()
    #read in connectivity
    for i in range(NE):
        ELEMNUM[i], DUM, NM[i,0],NM[i,1], NM[i,2]=adcirc_mesh.readline().split()

    #(we need to shift nodenum down by 1)
    ELEMNUM=ELEMNUM-1
    NM=NM-1
    NODENUM=NODENUM-1

    #close file
    adcirc_mesh.close()

    gdim, shape, degree = 2, "triangle", 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    element = ufl.VectorElement("Lagrange", cell, degree)
    #domain = ufl.Mesh(element)
    coords = np.array(list(zip(LONS,LATS)))
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree)) 
    #print(NM)
    #print(coords)
    #print(domain)
    
    domain1 = mesh.create_mesh(comm, NM, coords, domain)
    return domain1


#generate interpolation weights for DIA, same idea as SWAN
def DIA_weights(sigmas,thetas,geographic_nodes,g=9.81):
    #sigmas should be number of unique frequencies!! (NOT vector of dof in x)
    MSC = len(sigmas)
    print('MSC',MSC)
    MDC = len(thetas)
    NG = len(geographic_nodes)
    half_nsig = int(np.floor(MSC/2))
    half_nsig_minus = int(half_nsig - 1)
    sig_spacing = sigmas[half_nsig]/sigmas[half_nsig_minus]
    snl_c1 = 1/(g**4)
    lam1 = 0.25
    C = 3e-7
    snl_c2 = 5.5
    snl_c3 = 0.833
    snl_c4 = -1.25


    #compute offsets for resonance conditions
    #needed to get exxtents of extra grid
    LAMM2  = (1-lam1)**2
    LAMP2  = (1+lam1)**2
    DELTH3 = np.arccos( (LAMM2**2+4-LAMP2**2) / (4.*LAMM2) ) #angle 1 33.557 in rad
    AUX1 = np.sin(DELTH3)
    DELTH4 = np.arcsin(-AUX1*LAMM2/LAMP2) #angle 2 -11.4783 in rad

    #denominators for DIS
    DAL1 = 1/((1.+lam1)**4)
    DAL2   = 1. / ((1.-lam1)**4)
    DAL3   = 2. * DAL1 * DAL2

    #Compute directional indices in sigma and theta space ***
    DDIR = thetas[1]-thetas[0]
    CIDP   = abs(DELTH4/DDIR)
    IDP   = np.floor(CIDP)
    IDP1  = IDP + 1
    WIDP   = CIDP - (IDP)
    WIDP1  = 1.- WIDP
    CIDM   = abs(DELTH3/DDIR)
    IDM   = np.floor(CIDM)
    IDM1  = IDM + 1
    WIDM   = CIDM - (IDM)
    WIDM1  = 1.- WIDM
    XISLN  = np.log( sig_spacing )

    
    ISP = np.floor( np.log(1.+lam1) / XISLN )
    ISP1   = ISP + 1
    WISP   = (1.+lam1 - sig_spacing**ISP) / (sig_spacing**ISP1 - sig_spacing**ISP)
    WISP1  = 1. - WISP
    ISM    = np.floor( np.log(1.-lam1) / XISLN )
    ISM1   = int(ISM - 1)
    WISM   = (sig_spacing**ISM -(1.-lam1)) / (sig_spacing**ISM - sig_spacing**ISM1)
    WISM1  = 1. - WISM

    #calculate the max and min indeces for extended spectrum
    ISLOW =  int(1  + ISM1)
    ISHGH = int(MSC + ISP1 - ISM1)
    ISCLW =  1
    ISCHG = int(MSC - ISM1)
    IDLOW = int(1 - MDC - max(IDM1,IDP1)) #MARK CHECK, why is this
    IDHGH = int(MDC + MDC + max(IDM1,IDP1))
    MSC4MI = ISLOW
    MSC4MA = ISHGH
    MDC4MI = IDLOW
    MDC4MA = IDHGH
    MSCMAX = int(MSC4MA - MSC4MI + 1)
    MDCMAX = int(MDC4MA - MDC4MI + 1)

    #MSCMAX is the number of spectral nodes in extended frequency
    #MDCMAX is the number of directional nodes in extedned spectrum

    print('ISLOW',ISLOW)
    print('ISHGH',ISHGH)
    print('ISCHG',ISCHG)
    print('IDLOW',IDLOW)
    print('IDHGH',IDHGH)
    print('MSCMAX',MSCMAX)
    print('MDCMAX',MDCMAX)
    print('MDC4MI',MDC4MI)
    #*** Interpolation weights ***
    AWG1   = WIDP  * WISP;
    AWG2   = WIDP1 * WISP;
    AWG3   = WIDP  * WISP1;
    AWG4   = WIDP1 * WISP1;
    AWG5   = WIDM  * WISM;
    AWG6   = WIDM1 * WISM;
    AWG7   = WIDM  * WISM1;
    AWG8   = WIDM1 * WISM1;

    #     *** quadratic interpolation ***
    SWG1   = AWG1**2
    SWG2   = AWG2**2
    SWG3   = AWG3**2
    SWG4   = AWG4**2
    SWG5   = AWG5**2
    SWG6   = AWG6**2
    SWG7   = AWG7**2
    SWG8   = AWG8**2


    #     --- determine discrete counters for piecewise                       40.41
    #         constant interpolation                                          40.41

    if (AWG1<AWG2):
        if (AWG2<AWG3):
            if (AWG3<AWG4):
                ISPP=ISP
                IDPP=IDP
            else:
                ISPP=ISP;
                IDPP=IDP1;
    
        elif (AWG2<AWG4):
            ISPP=ISP
            IDPP=IDP
        else:
            ISPP=ISP1
            IDPP=IDP

    elif (AWG1<AWG3):
        if (AWG3<AWG4):
            ISPP=ISP
            IDPP=IDP
        else:
            ISPP=ISP
            IDPP=IDP1
    elif (AWG1<AWG4):
        ISPP=ISP
        IDPP=IDP
    else:
        ISPP=ISP1
        IDPP=IDP1
    if (AWG5<AWG6): 
        if(AWG6<AWG7):
            if (AWG7<AWG8): 
                ISMM=ISM
                IDMM=IDM
            else:
                ISMM=ISM
                IDMM=IDM1
        elif (AWG6<AWG8):
            ISMM=ISM
            IDMM=IDM
        else:
            ISMM=ISM1
            IDMM=IDM
    elif (AWG5<AWG7):
        if (AWG7<AWG8):
            ISMM=ISM
            IDMM=IDM
        else:
            ISMM=ISM
            IDMM=IDM1

    elif (AWG5<AWG8):
        ISMM=ISM
        IDMM=IDM
    else:
        ISMM=ISM1
        IDMM=IDM1


    #pack weights to pass to other function
    WWINT = np.array([IDP,IDP1,IDM,IDM1,ISP,ISP1,ISM,ISM1,ISLOW,ISHGH,ISCLW,ISCHG,IDLOW,IDHGH,MSC4MI,MSC4MA,MDC4MI,MDC4MA,MSCMAX,MDCMAX,IDPP,IDMM,ISPP,ISMM],dtype=np.int32)
    WWAWG = np.array([AWG1,AWG2,AWG3,AWG4,AWG5,AWG6,AWG7,AWG8])
    WWSWG = np.array([SWG1,SWG2,SWG3,SWG4,SWG5,SWG6,SWG7,SWG8])
    WWINT[12] = 1- max( WWINT[3], WWINT[1] )
    WWINT[13] = MDC + max( WWINT[3], WWINT[1] )
    print('WWINT',WWINT)
    return WWINT,WWAWG,WWSWG

def interpolate_for_DIA(WWINT,WWAWG,WWSWG,NG,sigmas,thetas,N,all_sigmas,map_to_mat,map_to_dof,g=9.81):
    
    MSC = len(sigmas)
    MDC = len(thetas)
    half_nsig = int(np.floor(MSC/2))
    half_nsig_minus = int(half_nsig - 1)
    sig_spacing = sigmas[half_nsig]/sigmas[half_nsig_minus]
    
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

    ########################
    #duplicate code, fix later
    PQUAD1 = 0.25
    PQUAD2 = 3e7

    snl_c1 = 1/(g**4)
    lam1 = 0.25
    C = 3e-7
    snl_c2 = 5.5
    snl_c3 = 0.833
    snl_c4 = -1.25


    #compute offsets for resonance conditions
    #needed to get exxtents of extra grid
    LAMM2  = (1-lam1)**2
    LAMP2  = (1+lam1)**2
    DELTH3 = np.arccos( (LAMM2**2+4-LAMP2**2) / (4.*LAMM2) ) #angle 1 33.557 in rad
    AUX1 = np.sin(DELTH3)
    DELTH4 = np.arcsin(-AUX1*LAMM2/LAMP2) #angle 2 -11.4783 in rad

    #denominators for DIS
    DAL1 = 1/((1.+lam1)**4)
    DAL2   = 1. / ((1.-lam1)**4)
    DAL3   = 2. * DAL1 * DAL2
    #############################3

    # need to create a matrix UE that will hold the extended spectrum
    UE = np.zeros((MSCMAX,MDCMAX,NG))
    #this will be like a meshgrid structure
    #For extended frequencies indeces 0:ISM1 are the appended low frequencies
    #ISM1:ISCHG should be original frequencies
    #ISCHG:MSCMAX should be higher than before
    #lets test
    Extended_freq=np.zeros(MSCMAX)
    Extended_freq[-MSC4MI+1:-MSC4MI+1+MSC]=sigmas

    #compute the parts that aren't in the range
    Extended_freq[-MSC4MI+1+MSC:] = sig_spacing**(np.arange(1,ISHGH-MSC+1))*sigmas[-1]
    Extended_freq[:-MSC4MI+1] = sig_spacing**(np.arange(ISLOW-1,0))*sigmas[0]

    #fill in extended freqs as needed...
    #compute extended spectrum, this is not set up for periodic full circle. directions outside of spectrum will be set to 0
    #need to get a map from the mesh to a structured matrix
    #print('ISM1,MSC4MI',ISM1,WWINT[14])
    #print('MSCMAX',MSCMAX)
    #print('ISP1,MSC4MA',WWINT[1],WWINT[15])

    Nvals = np.array(N.array)*all_sigmas*2*np.pi
    Narray = Nvals[map_to_mat].reshape(MSC,MDC)
    Narray = Narray
    print('Narray shape',Narray.shape)
    UEvals = Narray

    temparr = N.array[map_to_mat].reshape(MSC,MDC)
    print('Narray max should be here',temparr[24,12])
    print('actual max',np.amax(temparr),np.argmax(temparr))
    #this doesnt work
    #np.multiply(Narray.T,sigmas).T*2*np.pi
    I1 = -MSC4MI+1
    I2 = -MSC4MI+MSC+1
    J1 = -MDC4MI+1
    J2 = -MDC4MI+MDC+1

    print('IDDUM, should be 72,45',J2,I2)
    UE[I1:I2,J1:J2,0] = UEvals
    
    
    #add spectral tail
    PWTAIL=4
    FACHFR = 1./(sig_spacing**PWTAIL)
    for a in range(MSC+1-MSC4MI,ISHGH-MSC4MI+1):
        UE[a,:,0] = UE[a-1,:,0]*FACHFR

    print("where max should be")
    print(UE.shape)
    print(UE[44,48,0])
    print('UE max and loc')
    print(np.amax(UE))
    print(np.argmax(UE))

    #looks like up to hear is fixed so far#######stopping point############3

    #bilinear interpolation
    I1 = ISCLW-MSC4MI
    I2 = ISCHG-MSC4MI+1
    J1 = IDCLOW -MDC4MI
    J2 = IDCHGH - MDC4MI+1


    print('I range (should be 4 and 49)',I1,I2)
    print('J1 (should be 25 and 72)',J1,J2)
    E00 = UE[I1:I2,J1:J2,0]
    EP1 = WWAWG[0]*UE[I1+ISP1:I2+ISP1,J1+IDP1:J2+IDP1,0] + \
        WWAWG[1]*UE[I1+ISP1:I2+ISP1,J1+IDP:J2+IDP,0] + \
        WWAWG[2]*UE[I1+ISP:I2+ISP,J1+IDP1:J2+IDP1,0] + \
        WWAWG[3]*UE[I1+ISP:I2+ISP,J1+IDP:J2+IDP,0]
    EM1 = WWAWG[4]*UE[I1+ISM1:I2+ISM1, J1-IDM1:J2-IDM1,0] + \
        WWAWG[5]*UE[I1+ISM1:I2+ISM1, J1-IDM:J2-IDM,0] + \
        WWAWG[6]*UE[I1+ISM:I2+ISM, J1-IDM1:J2-IDM1,0] + \
        WWAWG[7]*UE[I1+ISM:I2+ISM, J1-IDM:J2-IDM,0]

    EP2 = WWAWG[0]*UE[I1+ISP1:I2+ISP1,J1-IDP1:J2-IDP1,0] + \
        WWAWG[1]*UE[I1+ISP1:I2+ISP1,J1-IDP:J2-IDP,0] + \
        WWAWG[2]*UE[I1+ISP:I2+ISP,J1-IDP1:J2-IDP1,0] + \
        WWAWG[3]*UE[I1+ISP:I2+ISP,J1-IDP:J2-IDP,0]
    EM2 = WWAWG[4]*UE[I1+ISM1:I2+ISM1, J1+IDM1:J2+IDM1,0] + \
        WWAWG[5]*UE[I1+ISM1:I2+ISM1, J1+IDM:J2+IDM,0] + \
        WWAWG[6]*UE[I1+ISM:I2+ISM, J1+IDM1:J2+IDM1,0] + \
        WWAWG[7]*UE[I1+ISM:I2+ISM, J1+IDM:J2+IDM,0]
    CONS = 3.7756e-04
    AF11 = (Extended_freq/(2*np.pi))**11
    print('AF11',AF11.shape,AF11[:10])
    print('EP1 shape',EP1.shape)
    print('max EP1',np.amax(EP1),np.argmax(EP1),EP1[21,20])    
    print('max EM1',np.amax(EM1),np.argmax(EM1),EM1[27,33])    
    print('max EP2',np.amax(EP2),np.argmax(EP2),EP2[21,26])    
    print('max EM2',np.amax(EM2),np.argmax(EM2),EM2[27,13])    
    
    
    FACTOR = CONS*np.multiply(E00.T,AF11[I1:I2]).T
    print('Factor shape,max',FACTOR.shape,np.max(FACTOR),FACTOR[44,23])
    
    SA1A = E00 *(EP1*DAL1 + EM1*DAL2)* PQUAD2
    SA1B = SA1A - EP1*EM1*DAL3 * PQUAD2
    SA2A = E00 * ( EP2*DAL1 + EM2*DAL2 )*PQUAD2
    SA2B   = SA2A - EP2*EM2*DAL3*PQUAD2
    

    print('SA1A info',SA1A.shape,np.amax(SA1A),SA1A[23,21])
    print('DAL1,DAl2,PQAD',DAL1,DAL2,PQUAD2)
    SA1 = np.zeros(UE.shape)
    SA2 = np.zeros(UE.shape)


    
    SA1[I1:I2,J1:J2,0] = FACTOR*SA1B
    SA2[I1:I2,J1:J2,0] = FACTOR*SA2B

    print('max of sa1',np.amax(SA1),np.argmax(SA1))
    print('random value of sa1')
    #compute DIA action
    I1 = -MSC4MI+1
    I2 = -MSC4MI + MSC +1
    
    J1 = - MDC4MI+1
    J2 = -MDC4MI + MDC+1

    print('indeces should be 4,45,36,61',I1,I2,J1,J2)
    SFNL = - 2. * ( SA1[I1:I2,J1:J2,0] + SA2[I1:I2,J1:J2,0] ) \
            + WWAWG[0] * ( SA1[I1-ISP1:I2-ISP1,J1-IDP1:J2-IDP1,0] + SA2[I1-ISP1:I2-ISP1,J1+IDP1:J2+IDP1,0] ) \
            + WWAWG[1] * ( SA1[I1-ISP1:I2-ISP1,J1-IDP:J2-IDP,0] + SA2[I1-ISP1:I2-ISP1,J1+IDP:J2+IDP,0] ) \
            + WWAWG[2] * ( SA1[I1-ISP:I2-ISP,J1-IDP1:J2-IDP1,0] + SA2[I1-ISP:I2-ISP,J1+IDP1:J2+IDP1,0] ) \
            + WWAWG[3] * ( SA1[I1-ISP:I2-ISP ,J1-IDP:J2-IDP,0] + SA2[I1-ISP:I2-ISP ,J1+IDP:J2+IDP,0] ) \
            + WWAWG[4] * ( SA1[I1-ISM1:I2-ISM1,J1+IDM1:J2+IDM1,0] + SA2[I1-ISM1:I2-ISM1,J1-IDM1:J2-IDM1,0] ) \
            + WWAWG[5] * ( SA1[I1-ISM1:I2-ISM1,J1+IDM:J2+IDM,0] + SA2[I1-ISM1:I2-ISM1,J1-IDM:J2-IDM,0] ) \
            + WWAWG[6] * ( SA1[I1-ISM:I2-ISM ,J1+IDM1:J2+IDM1,0] + SA2[I1-ISM:I2-ISM ,J1-IDM1:J2-IDM1,0] ) \
            + WWAWG[7] * ( SA1[I1-ISM:I2-ISM ,J1+IDM:J2+IDM,0] + SA2[I1-ISM:I2-ISM ,J1-IDM:J2-IDM,0] )
    
    #convert back to action balance
    print('SFNL max min',np.amin(SFNL),np.amax(SFNL))
    SFNL = np.multiply(SFNL.T,1/(2*np.pi*sigmas)).T

    #now remap back to fem mesh
    S_nl = SFNL.flatten()[map_to_dof] 

    print('SFNL',SFNL.shape)
    print('Eoo shape',E00.shape)
    print('Ep1 shape', EP1.shape)
    print('EM1 shape',EM1.shape)
    #test by plotting the meshgrid
    #this can;t be correct
    extended_thetas = np.zeros(MDCMAX)
    dtheta = thetas[1]-thetas[0]
    extended_thetas[-MDC4MI:(-MDC4MI+MDC)] = thetas
    extended_thetas[(-MDC4MI+MDC):] = thetas[-1]*np.arange(1,MDCMAX+MDC4MI-MDC+1)*dtheta

    #print('shape of shit',np.sum(UE==5))
    #print('should be ',MSC*MDC)
    #print('IDstuff',MDCMAX)
    print(Extended_freq)
    return S_nl

