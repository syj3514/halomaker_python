import numpy as np
from multiprocessing import Pool, shared_memory
from collections import defaultdict
from scipy.io import FortranFile
from tqdm import tqdm
import time
import atexit, signal
import sys


#======================================================================
# Make Flags
#======================================================================
RENORM = False
BIG_RUN = False
METALS = True
TQDM = True
DEV = True
ANG_MOM_OF_R = False
Test_FOF = False

#======================================================================
# Memory allocation
#======================================================================
mem = {}
mem_address = {}
globvars = {}

pos_ref_0 = np.array([
    -1., -1., -1.,
     1., -1., -1.,
    -1.,  1., -1.,
     1.,  1., -1.,
    -1., -1.,  1.,
     1., -1.,  1.,
    -1.,  1.,  1.,
     1.,  1.,  1.], dtype=np.float64)
pos_ref_0 = pos_ref_0.reshape((3, 8), order='F')


def allocate(name, shape, dtype=np.float64):
    global mem, mem_address
    if(DEV): print(f"\t@Allocating memory `{name}` | {shape}")
    arr = np.empty(shape, dtype=dtype)
    temp = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=name)
    if(name in mem_address.keys()):
        if(mem_address[name] is not None):
            raise Exception(f"\t@Memory address `{name}` already exists")
    mem_address[name] = temp
    mem[name] = np.ndarray(arr.shape, dtype=dtype, buffer=mem_address[name].buf)
    arr = None

def allocated(name):
    global mem
    return name in mem.keys()

def deallocate(*names):
    global mem, mem_address
    for name in names:
        if(name in mem.keys()):
            if(DEV): print(f"\t@Deallocating memory `{name}`")
            if mem_address[name] is not None:
                mem[name] = None
                mem_address[name].close()
                mem_address[name].unlink()
                mem_address[name] = None
            del mem[name]
            del mem_address[name]
    if(list(mem.keys())==0):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)

def flush(msg=True, parent=''):
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    if(len(mem_address) > 0):
        if(msg or DEV): print(f"\t@{parent} Clearing memory")
        if(msg or DEV): print(f"\t   {[i.name for i in mem_address.values()]}")
        deallocate(*mem_address.keys())
    if(msg or DEV): print("\t@Memory Clear Done")
    sys.exit(0)
atexit.unregister(flush)

#======================================================================
# useful types :
#======================================================================
class vector:
    __slots__ = ["x", "y", "z"]
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None

class shape:
    __slots__ = ["a", "b", "c"]
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None

class baryon:
    __slots__ = ["rvir", "mvir", "tvir", "cvel"]
    def __init__(self):
        self.rvir = None
        self.mvir = None
        self.tvir = None
        self.cvel = None

class hprofile:
    __slots__ = ["rho_0", "r_c"]
    def __init__(self):
        self.rho_0 = None
        self.r_c = None

liste_halos_o0:list['halo'] = []
class halo:
    __slots__ = ["datas", "sh", "p", "v", "L", "halo_profile", "my_number", "my_timestep", "nbsub", "hosthalo", "hostsub", "level", "nextsub", "m", "r", "spin", "sigma", "ek", "ep", "et"]
    def __init__(self):
        self.datas = baryon()
        self.sh = shape()
        self.p = vector()
        self.v = vector()
        self.L = vector()
        self.halo_profile = hprofile()
        self.my_number = None
        self.my_timestep = None
        self.nbsub = None
        self.hosthalo = None
        self.hostsub = None
        self.level = None
        self.nextsub = None
        self.m = None
        self.r = None
        self.spin = None
        self.sigma = None
        self.ek = None
        self.ep = None
        self.et = None
    
    def clear_halo(self):
        self.my_number = 0
        self.my_timestep = 0
        self.nbsub = 0
        self.hosthalo = 0
        self.hostsub = 0
        self.level = 1
        self.nextsub = -1
        self.m = np.float64(0.0)
        self.p.x = np.float64(0.0)
        self.p.y = np.float64(0.0)    
        self.p.z = np.float64(0.0)  
        self.v.x = np.float64(0.0)
        self.v.y = np.float64(0.0)
        self.v.z = np.float64(0.0)
        self.L.x = np.float64(0.0)
        self.L.y = np.float64(0.0)
        self.L.z = np.float64(0.0)
        self.spin = np.float64(0.0)
        self.r = np.float64(0.0)
        self.sh.a = np.float64(0.0)
        self.sh.b = np.float64(0.0)
        self.sh.c = np.float64(0.0)
        self.ek = np.float64(0.0)
        self.ep = np.float64(0.0)
        self.et = np.float64(0.0)
        self.datas.rvir = np.float64(0.0)
        self.datas.mvir = np.float64(0.0)  
        self.datas.tvir = np.float64(0.0)
        self.datas.cvel = np.float64(0.0)
        self.halo_profile.rho_0 = np.float64(0.0)
        self.halo_profile.r_c = np.float64(0.0)
    
    def write_halo(self, fname):
        '''
        Masses (h%m,h%datas%mvir) are in units of 10^11 Msol, and 
        Lengths (h%p%x,h%p%y,h%p%z,h%r,h%datas%rvir) are in units of Mpc
        Velocities (h%v%x,h%v%y,h%v%z,h%datas%cvel) are in km/s
        Energies (h%ek,h%ep,h%et) are in
        Temperatures (h%datas%tvir) are in K
        Angular Momentum (h%L%x,h%L%y,h%L%z) are in
        Other quantities are dimensionless (h%my_number,h%my_timestep,h%spin)  
        '''
        with FortranFile(fname, 'w') as f:
            f.write_record(self.my_number)
            f.write_record(self.my_timestep)
            f.write_record(self.level,self.hosthalo,self.hostsub,self.nbsub,self.nextsub)
            f.write_record(self.m)
            f.write_record(self.p.x,self.p.y,self.p.z)
            f.write_record(self.v.x,self.v.y,self.v.z)
            f.write_record(self.L.x,self.L.y,self.L.z )
            f.write_record(self.r, self.sself.a, self.sself.b, self.sself.c)
            f.write_record(self.ek,self.ep,self.et)
            f.write_record(self.spin)
            f.write_record(self.sigma)
            f.write_record(self.datas.rvir,self.datas.mvir,self.datas.tvir,self.datas.cvel)
            f.write_record(self.halo_profile.rho_0,self.halo_profile.r_c)
#======================================================================


#======================================================================
# parameters relative to the simulation analysis
#======================================================================
gravsoft = 'cubsplin' # type of gravitational softening
nbPes = 1 # obsolete vars for reading treecode format
nsteps = 1 # number of timesteps
alpha = np.float64(1.0); tnow = np.float64(1.0)
nbodies = 1 # number of particles in the simulation
nMembers = 1 # minimal number of particles in a fof halo
b_init = np.float64(1.0) # linking length parameter of the fof at z=0
profile = 'TSIS' # type of halo profile (only isothermal_sphere yet)
ninterp = 1 # nb of bins for interp. of smoothed grav. field
FlagPeriod = 1 # flag for periodicity of boundary conditions
fPeriod = np.array([1.0, 1.0, 1.0], dtype=np.float64)
#----------- For gadget format: ----------------------------------------
# nhr = 1 # to read only the selected HR particles
# minlrmrat = 1.0 # to recognize contaminated halos
#======================================================================


#======================================================================
# Definitions specific to input/output
#======================================================================
data_dir = "."
file_num = "00001"
numstep = 1
# errunit = 0
write_resim_masses = True # for writing resim_masses.dat file
#======================================================================


#======================================================================
# Constants
#======================================================================
gravconst = np.float64(4.302e-6) # G in units of (km/s)^2 kpc/(10^11 Msol)
pi = np.pi
#======================================================================


#======================================================================
# Global variables 
#======================================================================
ndim = 3
#   real(kind=8),allocatable         :: pos(:,:),vel(:,:)
massp = np.float64(0.0)
#   real(kind=8),allocatable         :: epsvect(:),mass(:)
omega_t,omega_lambda_t,omega_f,omega_lambda_f,omega_c_f = np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)
rho_crit,aexp,Lboxp,mboxp,af,ai,Lf,H_f,H_i = np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)
age_univ,Lbox_pt,Lbox_pt2,Hub_pt,omega_0,hubble,omega_lambda_0 = np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0),np.float64(0.0)
vir_overdens,rho_mean = np.float64(0.0), np.float64(0.0)
#   integer(kind=4),allocatable      :: linked_list_oo(:), liste_parts(:)
#   integer(kind=4),allocatable      :: first_part_oo(:),nb_of_parts_o0(:)
#   type (halo),allocatable          :: liste_halos_o0(:)
phsmooth_oo: np.ndarray[np.float64, 2+ninterp] = np.zeros(2+ninterp)
nb_of_halos, nb_of_subhalos = 0, 0
numero_step = 1
simtype = 'Ra3'
#======================================================================


#======================================================================
# defs for Adaptahop
#======================================================================
nparmax=512*512*512
nparbuffer=128**3
ncellbuffermin=128**3
nlevelmax=30; npartpercell=20
lin=10; lsin=11; lin2=12
loudis=12; lounei=13; lounode=14; loupartnode=15; lounodedyn=16
bignum=1.e30
# Physical constants (units : m s kg) ->
gravitational_constant=np.float64(6.6726e-11)
critical_density= np.float64(1.8788e-26)
mega_parsec=np.float64(3.0857e22)
solar_mass=np.float64(1.989e30)
convert_in_mps2=np.float64(1.e6)
#    real(kind=8), allocatable    :: vxout(:),vyout(:),vzout(:),vdisout(:)
#    integer(kind=4), allocatable :: mass_cell(:)
#    real(kind=8), allocatable    :: tmass_cell(:)
#    real(kind=8), allocatable    :: vcell(:,:)
#    real(kind=8), allocatable    :: size_cell(:)
#    real(kind=8), allocatable    :: pos_cell(:,:)
#    integer(kind=4), allocatable :: sister(:)
#    integer(kind=4), allocatable :: firstchild(:)
#    integer(kind=4), allocatable :: idpart(:),idpart_tmp(:)
#    integer(kind=4), allocatable :: iparneigh(:,:)
#    real(kind=8), allocatable    :: distance(:)
#    real(kind=8), allocatable    :: density(:)
#    integer(kind=4), allocatable :: firstpart(:)
#    integer(kind=4), allocatable :: igrouppart(:)
#    integer(kind=4), allocatable :: idgroup(:),idgroup_tmp(:)
#    integer(kind=4), allocatable :: igroupid(:)
#    integer(kind=4), allocatable :: color(:)
#    integer(kind=4), allocatable :: partnode(:)
#    real(kind=8), allocatable    :: densityg(:)
sizeroot = np.float64(0.0)
xlong, ylong, zlong, boxsize, boxsize2 = np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)
xlongs2, ylongs2, zlongs2 = np.float64(0.0), np.float64(0.0), np.float64(0.0)
omega0,omegaL,mass_in_kg,GMphys = np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)
aexp_hubble_const,aexp_mega_parsec_sqrt3 = np.float64(0.0), np.float64(0.0)
aexp_mega_parsec,aexp_max = np.float64(0.0), np.float64(0.0)
npart = 0
nvoisins,nhop,ntype = 0, 0, 0
ncellmx = 0
ngroups,nmembthresh,nnodes,nnodesmax = 0, 0, 0, 0
ncpu,nmpi,niterations = 0, 0, 0
ncellbuffer = 0
rho_threshold = np.float64(0.0)
verbose,megaverbose,periodic = False, False, False
fgas = np.float64(0.0)
fudge,alphap,epsilon,fudgepsilon = np.float64(0.0), np.float64(0.0), np.float64(1e-2), np.float64(0.0)
pos_shift = np.zeros(3, dtype=np.float64)

class grp:
    __slots__ = ["nhnei", "njunk", "isad_gr", "rho_saddle_gr"]
    def __init__(self):
        self.nhnei = None
        self.njunk = None # To avoid missalignement in memory
        self.isad_gr = None # integer(kind=4), dimension(:),pointer
        self.rho_saddle_gr = None # real(kind=8), dimension(:),pointer

class supernode:
    __slots__ = ["level","mother","firstchild","nsisters","sister","rho_saddle","density","densmax","radius","mass","truemass","position"]
    def __init__(self):
        self.level = None
        self.mother = None
        self.firstchild = None
        self.nsisters = None
        self.sister = None
        self.rho_saddle = None
        self.density = None
        self.densmax = None
        self.radius = None
        self.mass = None
        self.truemass = None
        self.position = None

node_0:list['supernode'] = []
group:list['grp'] = []
#    type (grp), allocatable       :: group(:)
#    type (supernode), allocatable :: node_0(:)

#======================================================================
# Flags for halo finder selection
#======================================================================
method = 'MSM' # flag to notify which and how the halofinder is to be used
fsub = False # flag to notify whether subhaloes are included
cdm = False # flag to select particle closest to the cdm instead of the one with the highest density
DPMMC = False # flag to select the densest particle in the most massive cell of the halo (not with FOF)
SC = False # flag to select the com within concentric spheres (not with FOF)
dcell_min = np.float64(0.0)
eps_SC = np.float64(0.0)
dump_dms = False

#======================================================================
# array to build the structure tree
#======================================================================
# integer(kind=4), allocatable :: first_daughter(:), mother(:), first_sister(:), level(:)
nstruct = 0

# used for the merger history method
# integer(kind=4), allocatable :: npfather_0(:),ex_liste_parts(:),removesub(:)
# integer(kind=4), allocatable :: ex_level(:),ex_nb_of_parts(:)
ex_nb_of_structs = 0


if(ANG_MOM_OF_R):
    agor_unit = 19
    agor_file = ""
    nshells = 100
  

def frange(first, last, step=1):
    if(step>0):
        return range(first, last+1, step)
    else:
        return range(first, last-1, step)

