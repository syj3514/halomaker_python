from halo_defs import mem, halo, frange
import halo_defs as H
import numpy as np
import atexit, signal, sys
from scipy.io import FortranFile
from tqdm import tqdm
from multiprocessing import Pool

#///////////////////////////////////////////////////////////////////////
#***********************************************************************
def read_data():
    '''
     This routine read the output of N-body simulations (particles positions and speeds, 
     cosmological and technical parameters)
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
     WARNING: this routine just reads the data and converts positions       
              and velocities from CODE units to these units                 
              -- positions are between -0.5 and 0.5                         
              -- velocities are in km/s                                     
                 in units of Hubble velocity accross the box for SIMPLE (SN)
              -- total box mass is 1.0                                      
                 for simulation with hydro (only with -DRENORM) flag        
              -- initial (beg of simulation) expansion factor is ai=1.0      
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    '''

    print(f"\n> In read_data: timestep  ---> {H.numero_step}")

    if(H.numero_step == 1):
       print(f"> data_dir: `{H.data_dir}`")
       # contains the number of snapshots to analyze and their names, type and number (see below)
       f12 = open('inputfiles_HaloMaker.dat','r')

    # then read name of snapshot, its type (pm, p3m, SN, Nzo, Gd), num of procs used and number of snapshot
    name_of_file, H.simtype, H.nbPes, H.numstep = f12.readline().split()
    H.nbPes = int(H.nbPes); H.numstep = int(H.numstep)
    if(name_of_file[0]=="'")or(name_of_file[0]=='"'):
        name_of_file = name_of_file[1:-1]
    H.file_num = f"{int(H.numstep):05d}"
    if(name_of_file[0] != '/'):
        name_of_file = f'{H.data_dir}/{name_of_file}'
    print(f"> name_of_file: `{name_of_file}`")

    # Note 1: old treecode SNAP format has to be converted [using SNAP_to_SIMPLE (on T3E)] 
    #     into new treecode SIMPLE (SN) format.
    # Note 2: of the five format (pm, p3m, SN, Nzo, Gd) listed above, only  SN, Nzo and Gd 
    #     are fully tested so the code stops for pm and p3m

    if(H.numero_step == H.nsteps): f12.close()

    if(H.simtype=='SN'): raise NotImplementedError("`SN` format is not implemented yet")

    elif(H.simtype=='Ra'):
        read_ramses(name_of_file)
        # Computation of omega_t = omega_matter(t)
        #
        #                            omega_f*(1+z)^3
        # omega(z)   = ------------------------------------------------------------------
        #              omega_f*(1+z)^3+(1-omega_f-omega_lambda_f)*(1+z)^2+omega_lambda_f
        #
        #                              omega_lambda_0
        # omega_L(z) = ----------------------------------------------------------------
        #              omega_f*(1+z)^3+(1-omega_f-omega_lambda_f)*(1+z)^2+omega_lambda_f
        H.omega_t  = H.omega_f*(H.af/H.aexp)**3
        H.omega_t  = H.omega_t/(H.omega_t+(1.-H.omega_f-H.omega_lambda_f)*(H.af/H.aexp)**2+H.omega_lambda_f)
    elif(H.simtype[:2]=='Ra'):
        read_ramses_new(name_of_file, rver=H.simtype)
        H.omega_t  = H.omega_f*(H.af/H.aexp)**3
        H.omega_t  = H.omega_t/(H.omega_t+(1.-H.omega_f-H.omega_lambda_f)*(H.af/H.aexp)**2+H.omega_lambda_f)
    elif(H.simtype=='Nzo'): raise NotImplementedError("`Nzo` format is not implemented yet")
    elif(H.simtype=='Gd'): raise NotImplementedError("`Gd` format is not implemented yet")
    else: raise NotImplementedError(f"> Don''t know the snapshot format: `{H.simtype}`")

    print(f"> aexp = {H.aexp}")
    pos = mem['pos']
    print(f"> min max position (in box units)   : {np.min(pos)},{np.max(pos)}")
    vel = mem['vel']
    print(f"> min max velocities (in km/s)      : {np.min(vel)},{np.max(vel)}")
    print(f"> Reading done.")

def skip_records(f, skip_num=1):
    """
    Skips a record from current position, faster than read_ints.
    """
    for _ in range(skip_num):
        first_size = f._read_size()

        f._fp.seek(first_size, 1)

        second_size = f._read_size()
        if first_size != second_size:
            raise IOError(f'Sizes do not agree in the header({first_size}) and footer({second_size}) for '
                        'this record - check header dtype')

#***********************************************************************
def read_ramses(repository):
    ''' This routine reads DM particles dumped in the RAMSES format.
    implicit none

    character(len=*)            :: repository
    integer(kind=4)             :: ndim,npart,idim,icpu,ipos,H.ncpu,i,ipar
    integer(kind=4)             :: ncpu2,npart2,ndim2
    integer(kind=4)             :: nx,ny,nz,nlevelmax,ngridmax,nstep_coarse
    integer(kind=4),allocatable :: idp(:)
    real(kind=8)                :: boxlen,tco,aexp_ram,hexp
    real(kind=8)                :: omega_m,omega_l,omega_k,omega_b
    real(kind=8)                :: scale_l,scale_d,scale_t
    real(kind=8)                :: mtot,massres
    real(kind=8),allocatable    :: tmpp(:,:),tmpv(:,:),tmpm(:)
    character*200               :: nomfich
    character*5                 :: nchar,ncharcpu
    logical                     :: ok

    
    # NB: repository is directory containing output files
    # e.g. /horizon1/teyssier/ramses_simu/boxlen100_n256/output_00001/
    '''
    atexit.unregister(H.flush)
    signal.signal(signal.SIGINT, H.flush)
    signal.signal(signal.SIGPIPE, H.flush)
    signal.signal(signal.SIGTERM, H.flush)
    # read cosmological params in header of amr file
    ipos    = repository.find("output_")
    nchar   = repository[ipos+7:ipos+12]
    nomfich = f"{repository}/amr_{nchar}.out00001"
    with FortranFile(nomfich, 'r') as f:
        H.ncpu, = f.read_ints()
        H.ndim, = f.read_ints()   
        nx,ny,nz = f.read_ints()
        H.nlevelmax, = f.read_ints()
        ngridmax, = f.read_ints()
        nstep_coarse, = f.read_ints()
        boxlen, = f.read_reals()
        # temps conforme tau, expansion factor, da/dtau
        tco,aexp_ram,hexp = f.read_reals()
        omega_m,omega_l,omega_k,omega_b = f.read_reals()
        # to get units cgs multiply by these scale factors
        scale_l,scale_d,scale_t = f.read_reals()
    # use approximate comv from cm to Mpc to match Romain's conversion... 
    H.Lboxp          = boxlen*scale_l/3.08e24/aexp_ram # converts cgs to Mpc comoving
    #write(errunit,*) 'af,hf,lboxp,ai,aexp',af,h_f,lboxp,ai,aexp_ram
    H.aexp           = aexp_ram*H.af  
    H.omega_f        = omega_m+omega_b
    H.omega_lambda_f = omega_l
    H.omega_c_f      = omega_k
    print(f"> From AMR file: `{nomfich}`")
    print(f"\tncpu={H.ncpu}, ndim={H.ndim}, nstep_coarse={nstep_coarse}")
    print(f"\tnlevelmax={H.nlevelmax}, ngridmax={ngridmax}")
    print(f"\tt={tco:.3e}, aexp={aexp_ram:.3e}, hexp={hexp:.3e}")
    print(f"\tomega_m={omega_m:.3f}, omega_l={omega_l:.3f}, omega_k={omega_k:.3f}, omega_b={omega_b:.3f}")
    print(f"\tboxlen={H.Lboxp:.3e} h-1 Mpc")

    # now read the particle data files
    nomfich = f"{repository}/part_{nchar}.out00001"
    print(f"\n> From part file: `{nomfich}`")
    with FortranFile(nomfich, 'r') as f:
        H.ncpu, = f.read_ints()
        H.ndim, = f.read_ints()

    H.npart = 0
    for icpu1 in range(1,H.ncpu+1):
       nomfich = f"{repository}/part_{nchar}.out{icpu1:05d}"
       with FortranFile(nomfich, 'r') as f:
           ncpu2, = f.read_ints()
           ndim2, = f.read_ints()
           npart2, = f.read_ints()
       H.npart = H.npart+npart2
    
    H.nbodies = H.npart
    print(f"> Found {H.npart} particles")
    print(f"> Reading positions and masses...")
    
    H.allocate('pos', (H.npart, H.ndim), dtype=np.float64)
    H.allocate('vel', (H.npart, H.ndim), dtype=np.float64)
    H.allocate('mass', (H.npart,), dtype=np.float64)
  
    iterobj = range(1,H.ncpu+1)
    if(H.TQDM):
        iterobj = tqdm(range(1,H.ncpu+1), desc="Reading particles", unit="cpu")
    for icpu1 in iterobj:
        nomfich = f"{repository}/part_{nchar}.out{icpu1:05d}"
        with FortranFile(nomfich, 'r') as f:
            ncpu2, = f.read_ints()
            ndim2, = f.read_ints()
            npart2, = f.read_ints()
            H.allocate('tmpp', (npart2, ndim2), dtype=np.float64)
            H.allocate('tmpv', (npart2, ndim2), dtype=np.float64)
            H.allocate('tmpm', (npart2,), dtype=np.float64)
            H.allocate('idp', (npart2,), dtype=np.int32)
            
            # read all particle positions
            for idim0 in range(H.ndim):
                mem['tmpp'][:,idim0] = f.read_reals()
            # read all particle velocities
            for idim0 in range(H.ndim):
                mem['tmpv'][:,idim0] = f.read_reals()
            # read all particle masses
            mem['tmpm'][:] = f.read_reals()
            # read all particle ids
            mem['idp'][:] = f.read_ints()

        # now sort DM particles in ascending id order
        for idim0 in range(H.ndim):
            # put all positions between -0.5 and 0.5
            mem['pos'][mem['idp']-1,idim0] = mem['tmpp'][:,idim0] - 0.5
            # convert code units to km/s 
            mem['vel'][mem['idp']-1,idim0] = mem['tmpv'][:,idim0]*scale_l/scale_t*1e-5
            mem['mass'][mem['idp']-1] = mem['tmpm'][:]
        H.deallocate('tmpp','tmpv','tmpm','idp')
    
        mtot = np.sum(mem['mass'])
        # that is for the dark matter so let's add baryons now if there are any 
        # and renormalization flag is on ##
        massres = np.min(mem['mass'])*H.mboxp*1e11
        H.massp   = np.min(mem['mass'])
        print(f"> particle mass (in M_sun)               = {massres}")
        if(H.RENORM):
            massres /= mtot
            H.massp /= mtot
            print(f"> particle mass (in M_sun) after renorm  = {massres}")
        if(H.BIG_RUN):
            H.deallocate('mass')
#***********************************************************************
# def _read_ramses_new(icpu, cursors, nsize, kwargs):
def _read_ramses_new(icpu, kwargs):
    repository = kwargs['repository']
    rver = kwargs['rver']
    nchar = kwargs['nchar']
    H.ndim = kwargs['ndim']
    scale_l = kwargs['scale_l']
    scale_t = kwargs['scale_t']

    nomfich = f"{repository}/part_{nchar}.out{icpu:05d}"
    with FortranFile(nomfich, 'r') as f:
        ncpu2, = f.read_ints()
        ndim2, = f.read_ints()
        npart2, = f.read_ints()
        skip_records(f, 1)
        nstar, = f.read_ints()
        skip_records(f, 2)
        nsink, = f.read_ints()
        # assert nsize[icpu-1] == npart2

        tmpp = np.empty((npart2,3), dtype=np.float64)#mem['pos_tmp'][cursors[icpu-1]-nsize[icpu-1]:cursors[icpu-1], :].view()
        tmpv = np.empty((npart2,3), dtype=np.float64)#mem['vel_tmp'][cursors[icpu-1]-nsize[icpu-1]:cursors[icpu-1], :].view()
        tmpm = np.empty(npart2, dtype=np.float64)#mem['mass_tmp'][cursors[icpu-1]-nsize[icpu-1]:cursors[icpu-1]].view()           
        
        # read all particle positions
        # print(icpu, tmpp[:,0].shape, nsize[icpu-1], npart2, cursors[icpu-1], mem['pos_tmp'].shape)
        for idim0 in range(H.ndim):
            tmpp[:,idim0] = f.read_reals()
        # read all particle velocities
        for idim0 in range(H.ndim):
            tmpv[:,idim0] = f.read_reals()
        # read all particle masses
        tmpm = f.read_reals()
        # read all particle ids
        idp = f.read_ints()
        # read grid level of particles
        skip_records(f, 1)
        if(rver=='Ra4'):
            # read particle family
            fam = f.read_ints(dtype=np.int8)
            # read particle tag
            skip_records(f, 1)
        else:
            # read all particle creation times if necessary
            if((nstar>0)or(nsink>0)):
                tmpt = f.read_reals()
                if(H.METALS):
                    skip_records(f, 1)
    # now sort DM particles in ascending id order and get rid of stars
    if(rver=='Ra4'): mask = fam==1 # DM particles only
    else: mask = (idp>0)&(tmpt==0)
    npart_tmp = np.sum(mask)
    for idim0 in range(H.ndim):
        # put all positions between -0.5 and 0.5
        mem['pos_tmp'][idp[mask]-1,idim0] = tmpp[mask,idim0]-0.5
        # convert code units to km/s 
        mem['vel_tmp'][idp[mask]-1,idim0] = tmpv[mask,idim0]*scale_l/scale_t*1e-5
        mem['mass_tmp'][idp[mask]-1] = tmpm[mask]
    return npart_tmp
#***********************************************************************
def read_ramses_new(repository, rver='Ra3'):
    ''' This routine reads DM particles dumped in the RAMSES format.
    implicit none

    character(len=*)            :: repository
    integer(kind=4)             :: ndim,npart,idim,icpu,ipos,H.ncpu,i,ipar
    integer(kind=4)             :: ncpu2,npart2,ndim2
    integer(kind=4)             :: nx,ny,nz,nlevelmax,ngridmax,nstep_coarse
    integer(kind=4),allocatable :: idp(:)
    real(kind=8)                :: boxlen,tco,aexp_ram,hexp
    real(kind=8)                :: omega_m,omega_l,omega_k,omega_b
    real(kind=8)                :: scale_l,scale_d,scale_t
    real(kind=8)                :: mtot,massres
    real(kind=8),allocatable    :: tmpp(:,:),tmpv(:,:),tmpm(:)
    character*200               :: nomfich
    character*5                 :: nchar,ncharcpu
    logical                     :: ok

    
    # NB: repository is directory containing output files
    # e.g. /horizon1/teyssier/ramses_simu/boxlen100_n256/output_00001/
    '''
    atexit.unregister(H.flush)
    signal.signal(signal.SIGINT, H.flush)
    signal.signal(signal.SIGPIPE, H.flush)
    signal.signal(signal.SIGTERM, H.flush)
    print()
    print(f"\t#################################")
    print(f"\t# Reading RAMSES version {rver} #")
    print(f"\t#################################")
    # read cosmological params in header of amr file
    ipos    = repository.find("output_")
    nchar   = repository[ipos+7:ipos+12]
    nomfich = f"{repository}/amr_{nchar}.out00001"
    with FortranFile(nomfich, 'r') as f:
        H.ncpu, = f.read_ints()
        H.ndim, = f.read_ints()   
        nx,ny,nz = f.read_ints()
        H.nlevelmax, = f.read_ints()
        ngridmax, = f.read_ints()
        skip_records(f, 2) # nboundary, ngrid_current
        boxlen, = f.read_reals()
        nout,idum,idum = f.read_ints()
        skip_records(f, 2) # tout, aout
        tco, = f.read_reals()
        skip_records(f, 2) # dtold, dtnew
        idum, nstep_coarse = f.read_ints() # nstep, nstep_coarse
        skip_records(f, 1) # einit, mass_tot_0, rho_tot
        temp = f.read_reals()
        omega_m,omega_l,omega_k,omega_b,dummy = temp[:5]
        temp = f.read_reals()
        aexp_ram, hexp = temp[:2]
    print(f"\t> From AMR file: `{nomfich}`")
    print(f"\t\tncpu={H.ncpu}, ndim={H.ndim}, nstep_coarse={nstep_coarse}")
    print(f"\t\tnlevelmax={H.nlevelmax}, ngridmax={ngridmax}")
    print(f"\t\tt={tco:.3e}, aexp={aexp_ram:.3e}, hexp={hexp:.3e}")
    print(f"\t\tomega_m={omega_m:.3f}, omega_l={omega_l:.3f}, omega_k={omega_k:.3f}, omega_b={omega_b:.3f}")

    nomfich = f"{repository}/info_{nchar}.txt"
    with open(nomfich, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading and trailing whitespace
            if not line or line.startswith('#'):
                continue  # Skip empty lines and lines starting with '#'

            # Split the line at the '=' character
            name, value = map(str.strip, line.split('='))

            # Check for a comment at the end of the value
            if '!' in value:
                value = value.split('!')[0].strip()

            # Process the name and value
            if name in ('unit_l', 'scale_l'):
                scale_l = float(value)
            elif name in ('unit_d', 'scale_d'):
                scale_d = float(value)
            elif name in ('unit_t', 'scale_t'):
                scale_t = float(value)
                break

    H.Lboxp          = boxlen*scale_l/3.08e24/aexp_ram # converts cgs to Mpc comoving
    H.aexp           = aexp_ram*H.af  
    H.omega_f        = omega_m
    H.omega_lambda_f = omega_l
    H.omega_c_f      = omega_k
    print(f"\t\tboxlen={boxlen*scale_l/3.08e24:.3e} h-1 Mpc")
    # print(f"\t> From AMR file: `{nomfich}`")
    # print(f"\t\tncpu={H.ncpu}, ndim={H.ndim}, nstep_coarse={nstep_coarse}")
    # print(f"\t\tnx={nx}, ny={ny}, nz={nz}")
    # print(f"\t\tnlevelmax={H.nlevelmax}, ngridmax={ngridmax}")
    # print(f"\t\tt={tco:.3e}, aexp={H.aexp:.3e}, hexp={hexp:.3e}")
    # print(f"\t\tomega_m={omega_m:.3f}, omega_l={omega_l:.3f}, omega_k={omega_k:.3f}, omega_b={omega_b:.3f}")
    # print(f"\t\tboxlen={boxlen*scale_l:.3e} h-1 Mpc")

    # now read the particle data files
    nomfich = f"{repository}/part_{nchar}.out00001"
    print(f"\t> From part file: `{nomfich}`")
    with FortranFile(nomfich, 'r') as f:
        H.ncpu, = f.read_ints()
        H.ndim, = f.read_ints()

    H.npart = 0
    # nsize = np.zeros(H.ncpu, dtype=np.int32)
    for icpu1 in range(1,H.ncpu+1):
        nomfich = f"{repository}/part_{nchar}.out{icpu1:05d}"
        with FortranFile(nomfich, 'r') as f:
            ncpu2, = f.read_ints()
            ndim2, = f.read_ints()
            npart2, = f.read_ints()
            idum = f.read_ints()
            nstar, = f.read_ints()
            idum = f.read_ints()
            idum = f.read_ints()
            nsink, = f.read_ints()
        H.npart += npart2
    #     nsize[icpu-1] = npart2
    # cursors = np.cumsum(nsize)
    # print(nsize)

    print(f"\t> Found {H.npart} Total particles")
    H.npart -= nstar
    H.nbodies = H.npart
    print(f"\t        {H.npart} non-stellar particles")
    print(f"\t        {nstar} star particles")
    print(f"\t> Reading positions and masses...")
    
    H.allocate('pos_tmp', (H.npart, H.ndim), dtype=np.float64)
    H.allocate('vel_tmp', (H.npart, H.ndim), dtype=np.float64)
    H.allocate('mass_tmp', (H.npart,), dtype=np.float64)
  
    ##### MultiProcessing Start #####
    # H.ncpu=4
    # H.nbPes = 4
    kwargs = {'repository':repository, 'rver':rver, 'nchar':nchar, 'ndim':H.ndim, 'scale_l':scale_l, 'scale_t':scale_t}
    iterobj = range(1,H.ncpu+1)
    if(H.nbPes==1): # Sequential reading
        if(H.TQDM):
            iterobj = tqdm(range(1,H.ncpu+1), desc=f"Reading parts(nbPes={H.nbPes})", unit="cpu")
        npart_tmp = 0
        for icpu1 in iterobj:
            # npart_tmp += _read_ramses_new(icpu1, cursors, nsize, kwargs)
            npart_tmp += _read_ramses_new(icpu1, kwargs)
    else: # Multiprocessing
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        with Pool(processes=H.nbPes) as pool:
            # async_results = [pool.apply_async(_read_ramses_new, (icpu1, cursors, nsize, kwargs)) for icpu1 in iterobj]
            async_results = [pool.apply_async(_read_ramses_new, (icpu1, kwargs)) for icpu1 in iterobj]
            npart_tmp = 0
            iterobj = async_results
            if(H.TQDM):
                iterobj = tqdm(async_results, desc=f"Reading parts(nbPes={H.nbPes})", unit="cpu", total=H.ncpu)
            for async_result in iterobj:
                npart_tmp += async_result.get()
        signal.signal(signal.SIGTERM, H.flush)

    H.npart = npart_tmp
    H.nbodies = H.npart
    print(f"\t> Found {H.npart} DM particles after masking")
    H.allocate('pos', (H.npart, H.ndim), dtype=np.float64)
    H.allocate('vel', (H.npart, H.ndim), dtype=np.float64)
    H.allocate('mass', (H.npart,), dtype=np.float64)
    mem['pos'][:H.npart, :] = mem['pos_tmp'][:H.npart, :]
    mem['vel'][:H.npart, :] = mem['vel_tmp'][:H.npart, :]
    mem['mass'][:H.npart] = mem['mass_tmp'][:H.npart]
    H.deallocate('pos_tmp','vel_tmp','mass_tmp')

    mtot = np.sum(mem['mass'])
    # that is for the dark matter so let's add baryons now if there are any 
    # and renormalization flag is on ##
    massres = np.min(mem['mass'])*H.mboxp*1e11
    H.massp   = np.min(mem['mass'])
    print(f"\t> particle mass (in M_sun)               = {massres}")
    if(H.RENORM):
        massres /= mtot
        H.massp /= mtot
        print(f"\t> particle mass (in M_sun) after renorm  = {massres}")
    if(H.BIG_RUN):
        H.deallocate('mass')
    print(f"\t#################################\n")

#***********************************************************************
def write_tree_brick():
    '''
    This subroutine writes the information relevant to building a halo 
    merging tree (using the build_tree program) i.e. for each halo:
      1/ the list of all the particles it contains (this enables us --- as  
         particle numbers are time independent --- to follow the halo history) 
      2/ its properties which are independent of merging history (mass ...)
    '''
#     integer(kind=4)                                         :: i,unitfile,start,j,idim,ndim=3
#     character(LEN=5)                                        :: nchar
#     character(LEN=7)                                        :: ncharg
#     character(LEN=300)                                      :: nomfich
# #ifndef H.BIG_RUN
#     character(len=len_trim(H.data_dir)+16)                    :: file
# #endif
#     character(len=len_trim(H.data_dir)+len_trim(H.file_num)+11) :: filename
#     integer(kind=4) ,allocatable                            :: members(:)
#     real(kind=8) ,allocatable                            :: mass_memb(:),mdump(:)
#     real(kind=8) ,allocatable                            :: pos_memb(:,:),vel_memb(:,:)
#     logical                                                 :: done
    import os
    nchar   = f'{int(H.file_num):05d}'
    if(H.dump_dms):
        #    call system('mkdir HAL_'//TRIM(nchar))
        os.mkdir(f'HAL_{nchar}')    

    done = False
    if(H.BIG_RUN):
        if(H.write_resim_masses):
            f44 = FortranFile(f'{H.data_dir}/resim_masses.dat', 'w')
            f44.write_record(H.nbodies)
            f44.write_record(mem['mass'])
            f44.close()
            H.write_resim_masses = False

    if(not H.fsub):
        filename = f"{H.data_dir}/tree_brick_{nchar}"
    else:
        filename = f"{H.data_dir}/tree_bricks_{nchar}"
    f44 = FortranFile(filename, 'w')
    print()
    print('> Output data to build halo merger tree to: ',filename)
    f44.write_record(H.nbodies)
    f44.write_record(H.massp)
    f44.write_record(H.aexp)
    f44.write_record(H.omega_t)
    f44.write_record(H.age_univ)
    f44.write_record(H.nb_of_halos, H.nb_of_subhalos)
    for i0 in range(H.nb_of_halos + H.nb_of_subhalos):
        # write list of particles in each halo
        H.allocate('members',mem['nb_of_parts_o0'][i0+1], dtype=np.int32)
        if(H.dump_dms):
            H.allocate('mass_memb', mem['nb_of_parts_o0'][i0+1], dtype=np.float64)
            H.allocate('pos_memb', (mem['nb_of_parts_o0'][i0+1],3), dtype=np.float64)
            H.allocate('vel_memb', (mem['nb_of_parts_o0'][i0+1],3), dtype=np.float64)
            H.allocate('mdump', mem['nb_of_parts_o0'][i0+1], dtype=np.float64)
        start = mem['first_part_oo'][i0+1]
        for j0 in range(mem['nb_of_parts_o0'][i0+1]):
            mem['members'][j0] = start
            if(H.dump_dms):
                mem['mass_memb'][j0] = mem['mass'][start-1]
                mem['pos_memb'][j0,0]=mem['pos'][start-1,0]
                mem['pos_memb'][j0,1]=mem['pos'][start-1,1]
                mem['pos_memb'][j0,2]=mem['pos'][start-1,2]
                mem['vel_memb'][j0,0]=mem['vel'][start-1,0]
                mem['vel_memb'][j0,1]=mem['vel'][start-1,1]
                mem['vel_memb'][j0,2]=mem['vel'][start-1,2]
            start = mem['linked_list_oo'][start]
        f44.write_record(mem['nb_of_parts_o0'][i0+1])
        f44.write_record(mem['members'])

        if(H.dump_dms):
            ncharg = f"{H.liste_halos_o0[i0+1].my_number:07d}"
            nomfich = f"HAL_{nchar}/halo_dms_{ncharg}"
            f9 = FortranFile(nomfich, 'w')
            f9.write_record(H.liste_halos_o0[i0+1].my_number)
            f9.write_record(H.liste_halos_o0[i0+1].level)
            f9.write_record(H.liste_halos_o0[i0+1].m)
            f9.write_record(H.liste_halos_o0[i0+1].p.x,H.liste_halos_o0[i0+1].p.y,H.liste_halos_o0[i0+1].p.z)
            f9.write_record(H.liste_halos_o0[i0+1].v.x,H.liste_halos_o0[i0+1].v.y,H.liste_halos_o0[i0+1].v.z)
            f9.write_record(H.liste_halos_o0[i0+1].L.x,H.liste_halos_o0[i0+1].L.y,H.liste_halos_o0[i0+1].L.z)
            f9.write_record(mem['nb_of_parts_o0'][i0+1])
            for idim0 in range(H.ndim):
                mem['mdump'][mem['nb_of_parts_o0'][i0+1]]=mem['pos_memb'][mem['nb_of_parts_o0'][i0+1],idim0]
                f9.write_record( mem['mdump'] )
            for idim0 in range(H.ndim):
                mem['mdump'][mem['nb_of_parts_o0'][i0+1]]=mem['vel_memb'][mem['nb_of_parts_o0'][i0+1],idim0]
                f9.write_record( mem['mdump'] )
            f9.write_record( mem['mass_memb'] )
            f9.write_record( mem['members'] )
            H.deallocate('mass_memb','pos_memb','vel_memb','mdump')

        H.deallocate('members')
        # write each halo properties
        write_halo(H.liste_halos_o0[i0+1],f44)
    f44.close()

#***********************************************************************
def write_halo(h:halo,unitfile:FortranFile):
    # integer(kind=4) :: unitfile
    # type (halo)     :: h

    # Masses (h.m,h.datas.mvir) are in units of 10^11 Msol, and 
    # Lengths (h.p.x,h.p.y,h.p.z,h.r,h.datas.rvir) are in units of Mpc
    # Velocities (h.v.x,h.v.y,h.v.z,h.datas.cvel) are in km/s
    # Energies (h.ek,h.ep,h.et) are in
    # Temperatures (h.datas.tvir) are in K
    # Angular Momentum (h.L.x,h.L.y,h.L.z) are in
    # Other quantities are dimensionless (h.my_number,h.my_timestep,h.spin)  

    unitfile.write_record( h.my_number )
    unitfile.write_record( h.my_timestep  )
    unitfile.write_record( h.level,h.hosthalo,h.hostsub,h.nbsub,h.nextsub )
    unitfile.write_record( h.m )
    unitfile.write_record( h.p.x,h.p.y,h.p.z )
    unitfile.write_record( h.v.x,h.v.y,h.v.z )
    unitfile.write_record( h.L.x,h.L.y,h.L.z  )
    unitfile.write_record( h.r, h.sh.a, h.sh.b, h.sh.c )
    unitfile.write_record( h.ek,h.ep,h.et )
    unitfile.write_record( h.spin )
    unitfile.write_record( h.sigma )
    unitfile.write_record( h.datas.rvir,h.datas.mvir,h.datas.tvir,h.datas.cvel )
    unitfile.write_record( h.halo_profile.rho_0,h.halo_profile.r_c )