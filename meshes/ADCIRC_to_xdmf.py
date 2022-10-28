import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import sparse as sp
from petsc4py import PETSc
from dolfinx import fem,mesh,io,cpp,graph
import ufl
import time
import CFx.utils
import CFx.assemble
import CFx.transforms
import CFx.boundary
import CFx.wave
time_start = time.time()



#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

output_dir = 'Outputs/'

#read mesh directly from ADCIRC file
file_path = 'meshes/depth2.grd'

domain1 = CFx.utils.ADCIRC_mesh_gen(comm,file_path)
#now store as an xdmf file
out_name = 'meshes/shoaling_grid'
xdmf = io.XDMFFile(domain1.comm, out_name+".xdmf", "w")
xdmf.write_mesh(domain1)
xdmf.close()

