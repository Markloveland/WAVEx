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

    # need to create a matrix UE that will hold the extended spectrum
    UE = np.zeros((MSCMAX,MDCMAX,NG))
    #this will be like a meshgrid structure
    #For extended frequencies indeces 0:ISM1 are the appended low frequencies
    #ISM1:ISCHG should be original frequencies
    #ISCHG:MSCMAX should be higher than before
    #lets test
    Extended_freq=np.zeros(MSCMAX)
    print(ISM1)
    print(ISCHG)
    Extended_freq[-ISM1:ISCHG]=sigmas
    print(Extended_freq)

    #fill in extended freqs as needed...

    return 0
