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
def DIA_weights(sigmas,thetas,g=9.81):
    #sigmas should be number of unique frequencies!! (NOT vector of dof in x)
    MSC = len(sigmas)
    #print('MSC',MSC)
    MDC = len(thetas)
    #NG = len(geographic_nodes)
    half_nsig = int(np.floor(MSC/2))
    half_nsig_minus = int(half_nsig - 1)
    sig_spacing = sigmas[half_nsig]/sigmas[half_nsig_minus]
    PQUAD2=3e7
    snl_c1 = 1/(g**4)
    lam1 = 0.25
    C = 3e-7
    snl_c2 = 5.5
    snl_c3 = 0.833
    snl_c4 = -1.25
    
    X = 1
    X2 = 1
    CONS   = snl_c1* ( 1 + snl_c2/X * (1.-snl_c3*X) * np.exp(X2))
    

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

    #print('ISLOW',ISLOW)
    #print('ISHGH',ISHGH)
    #print('ISCHG',ISCHG)
    #print('IDLOW',IDLOW)
    #print('IDHGH',IDHGH)
    #print('MSCMAX',MSCMAX)
    #print('MDCMAX',MDCMAX)
    #print('MDC4MI',MDC4MI)
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
    
    Extended_freq=np.zeros(MSCMAX)
    Extended_freq[-MSC4MI+1:-MSC4MI+1+MSC]=sigmas
    #compute the parts that aren't in the range
    Extended_freq[-MSC4MI+1+MSC:] = sig_spacing**(np.arange(1,ISHGH-MSC+1))*sigmas[-1]
    Extended_freq[:-MSC4MI+1] = sig_spacing**(np.arange(ISLOW-1,0))*sigmas[0]
    
    DIA_PARAMS = [MSC,MDC,sig_spacing,CONS,DAL1,DAL2,DAL3,PQUAD2,Extended_freq]
    WWINT[12] = 1- max( WWINT[3], WWINT[1] )
    WWINT[13] = MDC + max( WWINT[3], WWINT[1] )
    #print('WWINT',WWINT)
    return WWINT,WWAWG,WWSWG,DIA_PARAMS

