from input_output import *
import halo_defs as H
from halo_defs import mem,halo,vector,frange
import time
import numpy as np
from scipy.io import FortranFile

#//////////////////////////////////////////////////////////////////////////
#**************************************************************************
def init_0():
    H.write_resim_masses = True

    # initialize gravitationalsoftening 
    initgsoft_00()

    # initialize cosmological and technical parameters of the simulation
    init_cosmo_01()

    return

#*************************************************************************
def initgsoft_00():
    '''
    subroutine to initialize the required arrays for the gravitational 
    field smoothing interpolation in routine SOFTGRAV. This is only
    required for a cubic spline kernel; interpolation is performed in 
    distance.
    '''
    # integer(kind=4) :: i
    # real(kind=8)    :: deldrg,xw,xw2,xw3,xw4,tiny,one,two

    tiny = np.float64(1.e-19)
    one  = np.float64(1.0)
    two  = np.float64(2.0)

    if(H.gravsoft=='harmonic'): return None

    deldrg      = 2./H.ninterp
    H.phsmooth_oo[0] = 7.*np.sqrt(tiny)/5.

    for i0 in range(H.ninterp):
       xw  = (i0+1)*deldrg
       xw2 = xw*xw
       xw3 = xw2*xw
       xw4 = xw2*xw2
       if(xw <= one):
          H.phsmooth_oo[i0+1] = -2.*xw3*(one/3.-3.*xw2/20.+xw3/20.)+7.*xw/5.
       else:
          H.phsmooth_oo[i0+1] = -one/15.+8.*xw/5.-xw3*(4./3.-xw+3.*xw2/10.-xw3/30.)
       if (xw >= two):
          H.phsmooth_oo[i0+1] = one
    return None

#*************************************************************************
def init_cosmo_01():
    '''
    This routine reads in the `input_HaloMaker.dat` file which contains the cosmological 
    and technical parameters of the N-Body simulation to analyze.
    '''
    import compute_neiKDtree_mod as neiKDtree
    # use fof
    # integer(kind=4)      :: i
    # # cannot do otherwise than setting all the strings to a value larger than that of a line 
    # # in the input file and trim them whenever it is needed
    # character(len=200)   :: line,name,value

    # Initial (beginning of the simulation) expansion factor
    H.ai             = np.float64(1.0)
    # The file is based on a keyword list. Keywords are optional, so we set defaults here.  
    H.omega_f        = np.float64(0.3333)      #   mass density at final timestep
    H.omega_lambda_f = np.float64(0.6667)      #   lambda at final timestep 
    H.af             = np.float64(36.587)      #   expansion factor of the final timestep  
    H.Lf             = np.float64(150.0)       #   final length of box in physical Mpc 
    H.H_f            = np.float64(66.667)      #   Hubble constant in km/s/Mpc, at final timestep
    H.b_init         = np.float64(0.2)         #   linking length friend-of-friend parameter @ z=0
    Nmembers       = 20          #   minimum number of particles for a halo
    H.nsteps         = 1           #   number of files to analyse (listed in 'files.dat')
    H.method         = "FOF"
    if(H.ANG_MOM_OF_R): H.agor_file      = "ang_mom_of_r"

    print('\n> Values of input parameters:  ')
    print('> ---------------------------- ')
    print(f'> Looking for `input_HaloMaker.dat` in directory: `{H.data_dir}`')
    f20 = open(f'input_HaloMaker.dat', 'r')
    for line in f20:
        i = line.find('=')
        if (i == -1 or line[0] == '#'): continue
        name  = line[:i].strip()
        value = line[i+1:].strip()
        # check for a comment at end of line
        i     = value.find('!')
        if (i != -1): value = value[:i].strip()
        print(f'> {name:>15} : {str(value):>10}')
        if (name == 'omega_0' or name == 'Omega_0' or name == 'omega_f'):
            H.omega_f = np.float64(value)
        elif (name == 'omega_l' or name == 'lambda_0' or name == 'lambda_f'):
            H.omega_lambda_f = np.float64(value)
        elif (name == 'af' or name == 'afinal' or name == 'a_f'):
            H.af = np.float64(value)
        elif (name == 'Lf' or name == 'lf' or name == 'lbox'):
            H.Lf = np.float64(value)
        elif (name == 'H_f' or name == 'H_0' or name == 'H'):
            H.H_f = np.float64(value)
        elif (name == 'FlagPeriod'):
            H.FlagPeriod = np.int32(value)
        elif (name == 'n' or name == 'N' or name == 'npart'):
            H.nMembers = np.int32(value)
        elif (name == 'cdm'):
            H.cdm = value=='.true.'
        elif (name == 'method'):
            H.method = value
        elif (name == 'b'):
            H.b_init = np.float64(value)
        elif (name == 'nvoisins'):
            H.nvoisins = np.int32(value)
        elif (name == 'nhop'):
            H.nhop = np.int32(value)
        elif (name == 'rhot'):
            H.rho_threshold = np.float64(value)
        elif (name == 'fudge'):
            H.fudge = np.float64(value)
        elif (name == 'fudgepsilon'):
            H.fudgepsilon = np.float64(value)
        elif (name == 'alphap'):
            H.alphap = np.float64(value)
        elif (name == 'verbose'):
            H.verbose = value=='.true.'
        elif (name == 'megaverbose'):
            H.megaverbose = value=='.true.'
        elif (name == 'DPMMC'):
            H.DPMMC = value=='.true.'
        elif (name == 'SC'):
            H.SC = value=='.true.'
        elif (name == 'dcell_min'):
            H.dcell_min = np.float64(value)
        elif (name == 'eps_SC'):
            H.eps_SC = np.float64(value)
        elif (name == 'nsteps' or name == 'nsteps_do'):
            H.nsteps = np.int32(value)
        elif (name == 'dump_DMs'):
            H.dump_dms = value=='.true.'
        elif (name == 'H.agor_file'):
            if(H.ANG_MOM_OF_R): H.agor_file = f"{H.data_dir}/{value}"
        else:
            print(f'>   dont recognise parameter: {name}')
    f20.close()
    
    # initial size of the box in physical Mpc (NB: f index stands for final quantities)
    H.Lboxp = H.Lf*(H.ai/H.af)

    if( ((H.omega_f+H.omega_lambda_f) != 1.0)and(H.omega_lambda_f != 0.0) ):
        raise NotImplementedError('> lambda + non flat Universe not implemented yet')

    # In the most general of cases:
    #     af/aexp         = 1+z
    #     omega(z)        = (H_f/H(z))**2 * (1+z)**3  * omega_f
    #     omega_lambda(z) = omega_lambda_f * (H_f/H(z))**2
    #     omega_c(z)      = omega_c_f * (1+z)**2 * (H_f/H(z))**2
    #     H(z)**2         = H_f**2*( omega_f*(1+z)**3 + omega_c_f*(1+z)**2 + omega_lambda_f)

    H.omega_c_f = 1. - H.omega_f - H.omega_lambda_f
    H.H_i       = H.H_f * np.sqrt( H.omega_f*(H.af/H.ai)**3 + H.omega_c_f*(H.af/H.ai)**2 + H.omega_lambda_f)
    # rho_crit = 2.78782 h^2  (critical density in units of 10**11 M_sol/Mpc^3)
    H.mboxp     = 2.78782*(H.Lf**3)*(H.H_f/100.)**2*H.omega_f 

    print()
    print( f'> Initial/Final values of parameters:  ')
    print( f'> -----------------------------------  ')
    print( f'> redshift                         : ',H.af/H.ai-1.)
    print( f'> box size (Mpc)                   : ',H.Lboxp)
    print( f'> Hubble parameter  (km/s/Mpc)     : ',H.H_i)
    print( f'> box mass (10^11 Msol)            : ',H.mboxp)
    print()

#*************************************************************************
def new_step_1():
    '''
    This is the main subroutine: it builds halos from the particle simulation and 
    computes their properties ...
    '''
#     integer(kind=4)                      :: indexp,ierr,i
#     integer(kind=4)                      :: found,n_halo_contam,n_subs_contam
#     real(kind=8)                         :: read_time_ini,read_time_end
#     real(kind=8)                         :: t0,t1
#     logical                              :: printdatacheckhalo !put to true if bug after make_linked_list 
# #ifdef H.ANG_MOM_OF_R
#     character(200)                       :: filename
# #endif
    print(f'\n\n> Timestep  ---> {H.numero_step}')
    print('> -------------------')

    read_time_ini = time.time()

    # read N-body info for this new step
    print("\n $ Read data...")
    read_data_10()
    if(H.nbodies<H.nMembers):
        print()
        print('ERROR in this snapshot:')
        print('> nbodies=',H.nbodies)
        print('> nMembers=',H.nMembers)
        print('> nbodies < nMembers_threshold !')
        print('> Skip this step')
        if(H.allocated('pos_10')): H.deallocate('pos_10')
        if(H.allocated('vel_10')): H.deallocate('vel_10')
        if(H.allocated('mass_10')): H.deallocate('mass_10')
        if(H.allocated('liste_parts')): H.deallocate('liste_parts')
        if(len(H.liste_halos_o0)>0): H.liste_halos_o0 = []
        return

    # determine the age of the universe and current values of cosmological parameters
    print("\n $ Determine the Age...")
    det_age_11()
    
    # first compute the virial overdensity (with respect to the average density of the universe) 
    # predicted by the top-hat collapse model at this redshift 
    print("\n $ Compute the virial overdensity...")
    virial_12()

    if(H.method!="FOF"):
       H.allocate('liste_parts',H.nbodies,dtype=np.int32)
    print("\n $ Make halos...")
    make_halos_13()  

    if(H.Test_FOF):
        filelisteparts = f"liste_parts_{H.numstep}"
        with FortranFile(filelisteparts, 'w') as f:
            f.write_record(H.nbodies,H.nb_of_halos)
            f.write_record(mem['liste_parts'][:H.nbodies])
        # reset nb of halos not to construct halos
        H.nb_of_halos = 0
    # if there are no halos go to the next timestep
    if (H.nb_of_halos == 0):
        print('no halos deallocating')
        H.deallocate('pos_10','vel_10')
        if(H.allocated('density_1312')): H.deallocate('density_1312')
        if(H.allocated('mass_10')): H.deallocate('mass_10')
        H.deallocate('liste_parts')
        return

    H.allocate('first_part_oo_1', H.nb_of_halos+H.nb_of_subhalos+1, dtype=np.int32)
    H.allocate('nb_of_parts_o0_1', H.nb_of_halos+H.nb_of_subhalos+1, dtype=np.int32)
    # make a linked list of the particles so that each member of a halo points to the next 
    # until there are no more members (last particles points to -1)
    H.allocate('linked_list_oo_1', 1+H.nbodies+1, dtype=np.int32)

    print("\n $ Make linked list...")
    make_linked_list_14()

    # H.deallocate liste_parts bc it has been replaced by linked_list_oo
    H.deallocate('liste_parts')

    # in case of resimulation (or individual particle masses) count how many halos are contaminated 
    # (i.e. contain "low res" particles) 
    # NB: for numerical reasons, it can happen that mass is sligthly larger 
    #     than massp. Thus, for detecting LR particles, the criterion 
    #     if (mass(indexp) > massp) ...  may not work (it can get ALL parts, also HR parts!).
    #     Thus, to be conservative, use: if (mass(indexp) > massp * (1+1e-5))) ...
    if(H.allocated('mass_10')):
        n_halo_contam = 0 
        n_subs_contam = 0  
        for i0 in range(H.nb_of_halos+H.nb_of_subhalos):
            indexp = mem['first_part_oo_1'][i0+1]
            found  = 0
            while( (indexp != -1)and(found == 0) ):
                if(mem['mass_10'][indexp] > H.massp* 1.00001 ):
                    if(H.fsub):
                        if(mem['level_1319'][i0] == 1): 
                            n_halo_contam += 1
                        else:
                            n_subs_contam += 1
                    else:
                        n_halo_contam += 1
                    found = 1
                indexp = mem['linked_list_oo_1'][indexp]
        print('> # of halos, # of CONTAMINATED halos :',H.nb_of_halos,n_halo_contam,H.nb_of_subhalos,n_subs_contam)
        f222 = open('ncontam_halos.dat', 'r')
        f222.write(f'{H.numero_step:6d} {H.nb_of_halos:6d} {n_halo_contam:6d} {H.nb_of_subhalos:6d} {n_subs_contam:6d}\n')
        f222.close()

    # until now we were using code units for positions and velocities
    # this routine changes that to physical (non comoving) coordinates for positions 
    # and peculiar (no Hubble flow) velocities in km/s
    # The masses are changed from code units into 10^11 M_sun as well.
    change_units_15()

    # allocation and initialization of the halo list
    H.liste_halos_o0 = [halo() for _ in range(H.nb_of_halos+H.nb_of_subhalos+1)]
    # (liste_halos_o0 is list of `halo` class, and defined in `halo_defs`)
    # (It may should be changed to shared memory if you want to implement the multiprocessing)

    init_halos_16()

    fagor=None
    if(H.ANG_MOM_OF_R):
        filename = f"{H.agor_file}.{H.numstep:03d}"
        fagor = FortranFile(filename, 'w')
        fagor.write_record(H.nb_of_halos,H.nb_of_subhalos)
        fagor.write_record(H.nshells)

    printdatacheckhalo = False
    for i1 in frange(1,H.nb_of_halos + H.nb_of_subhalos):
        if(printdatacheckhalo):
            print('> halo:', i1,'nb_of_parts_o0_1',mem['nb_of_parts_o0_1'][i1])
            t0 = time.time()
        # determine mass of halo       
        det_mass_17(H.liste_halos_o0[i1])
        if(printdatacheckhalo): print('> mass:', H.liste_halos_o0[i1].m)
        # compute center of halo there as position of "most dense" particle
        # and give it the velocity of the true center of mass of halo
        det_center_18(H.liste_halos_o0[i1])
        if(printdatacheckhalo): print('> center:',H.liste_halos_o0[i1].p)
        # compute angular momentum of halos
        compute_ang_mom_19(H.liste_halos_o0[i1])
        if(printdatacheckhalo): print('> angular momentum:',H.liste_halos_o0[i1].L)
        # compute r = max(distance of halo parts to center of halo)
        r_halos_1a(H.liste_halos_o0[i1])
        if(printdatacheckhalo): print('> radius:',H.liste_halos_o0[i1].r)
        # compute energies and virial properties of the halos depending on density profile
        # (so this profile is also computed in the routine)
        det_vir_1b(H.liste_halos_o0[i1], fagor=fagor)
        if(printdatacheckhalo): print('> mvir,rvir:',H.liste_halos_o0[i1].datas.mvir,H.liste_halos_o0[i1].datas.mvir)
        # compute dimensionless spin parameter of halos
        compute_spin_parameter_1c(H.liste_halos_o0[i1])
        if(printdatacheckhalo): print('> spin:',H.liste_halos_o0[i1].spin)
        
        if(printdatacheckhalo):
            t1 = time.time()
            print('> halo computation took:',int(t1- t0) ,'s')
            print()

    if(H.ANG_MOM_OF_R): fagor.close()

    write_tree_brick_1d()

    H.liste_halos_o0 = []
    H.deallocate('nb_of_parts_o0_1','first_part_oo_1','linked_list_oo_1')
    H.deallocate('pos_10','vel_10')
    if(H.allocated('mass_10')): H.deallocate('mass_10')
    if(not H.cdm): H.deallocate('density_1312')

    read_time_end = time.time()

    print('> time_step computations took : ',round(read_time_end - read_time_ini),' seconds')
    print()

#***********************************************************************
def make_halos_13():
    '''
    subroutine which builds the halos from particle data using fof or adaptahop
    '''
    # use fof
    import compute_neiKDtree_mod as neiKDtree
    #real(kind=8)    :: read_time_ini,read_time_end

    print('> In routine make_halos ')
    print('> ----------------------')
    
    print()
    H.fPeriod[:]    = H.FlagPeriod
    if(H.FlagPeriod==1):
        print('> WARNING: Assuming PERIODIC boundary conditions --> make sure this is correct')
        periodic = True
    else:
        print('> WARNING: Assuming NON PERIODIC boundary conditions --> make sure this is correct')
        periodic = False
    
    if(H.numero_step==1):
        if(H.cdm):
            print('> Center of haloes and subhaloes are defined as the particle the closest to the cdm')
        else:
            print('> Center of haloes and subhaloes are defined as the particle with the highest density' )

        if(H.method == "FOF"):
            raise NotImplementedError('FOF is not implemented yet')
            print('> HaloMaker is using Friend Of Friend algorithm')
            # fof_init()
            H.fsub = False
        elif("HOP"):
            print('> HaloMaker is using Adaptahop in order to' )
            print('> Detect halos, subhaloes will not be selected'    )
            neiKDtree.init_adaptahop_130()
            H.fsub = False
        elif("DPM"):
            print('> HaloMaker is using Adaptahop in order to' )
            print('> Detect halos, and subhaloes with the Density Profile Method'    )
            neiKDtree.init_adaptahop_130()
            H.fsub = True
        elif("MSM"):
            print('> HaloMaker is using Adaptahop in order to' )
            print('> Detect halos, and subhaloes with the Most massive Subhalo Method')
            neiKDtree.init_adaptahop_130()
            H.fsub = True
        elif("BHM"):
            print('> HaloMaker is using Adaptahop in order to' )
            print('> Detect halos, and subhaloes with the Branch History Method'    )
            neiKDtree.init_adaptahop_130()
            H.fsub = True
        else:
            print('> Selection method: ',H.method,' is not included')
            print('> Please check input file:, input_HaloMaker.dat')
    
    read_time_ini = time.time()
    if(H.method =='FOF'):
        # compute density when most dense particle is choosen as center
        raise NotImplementedError('FOF is not implemented yet')
        fof_main()
        if( (not H.cdm)and(H.nb_of_halos>0) ): compute_density_for_fof()
        H.nb_of_subhalos = 0       
    else:
        neiKDtree.compute_adaptahop_131()
        # no nead to keep information about density when choosing particle closest to the cdm as center
        if(H.cdm): H.deallocate('density_1312')

    read_time_end = time.time()

    print(' > Number of halos with more than     ', H.nMembers,' particles:',H.nb_of_halos)
    print(' > Number of sub-halos with more than ', H.nMembers,' particles:',H.nb_of_subhalos)
    print('> time_step computations took : ',round(read_time_end - read_time_ini),' seconds')
  
    return

#***********************************************************************
def make_linked_list_14():
    '''
    Subroutine builds a linked list of parts for each halo which 
    contains all its particles.
    '''
    # integer(kind=4)             :: i,index1,index2,ierr
    # integer(kind=4),allocatable :: current_ptr_o0(:)  

    current_ptr_o0 = np.zeros(1+H.nb_of_halos+H.nb_of_subhalos, dtype=np.int32)-1
    # initialization of linked list
    mem['first_part_oo_1'][:]  = -1
    mem['nb_of_parts_o0_1'][:] =  0
    mem['linked_list_oo_1'][:] = -1
 
    # make linked list: a few (necessary) explanations ....
    #   1/ index1 (liste_parts(i)) is the number of the halo to which belongs particle i (i is in [1..nbodies]) 
    #   2/ first_part_oo(j) is the number of the first particle of halo j  (j is in [1..nhalo])
    #   3/ current_ptr_o0(index1) is the number of the latest particle found in halo number index1 
    # to sum up, first_part_oo(i) contains the number of the first particle of halo i,
    # linked_list_oo(first_part_oo(i)) the number of the second particle of halo i, etc ... until 
    # the last particle which points to number -1

    for i0 in range(H.nbodies):
        index1 = mem['liste_parts'][i0]
        if(index1>(H.nb_of_halos+H.nb_of_subhalos)): raise IndexError('error in liste_parts')
        if (mem['first_part_oo_1'][index1] == -1):
            mem['first_part_oo_1'][index1]  = i0
            mem['nb_of_parts_o0_1'][index1] = 1
            current_ptr_o0[index1] = i0
        else:
            index2              = current_ptr_o0[index1]
            mem['linked_list_oo_1'][index2] = i0
            current_ptr_o0[index1] = i0
            mem['nb_of_parts_o0_1'][index1] += 1

    # close linked list
    for i0 in range(0,H.nb_of_halos+H.nb_of_subhalos+1):
        if(current_ptr_o0[i0] == -1): continue
        index2              = current_ptr_o0[i0]
        mem['linked_list_oo_1'][index2] = -1


#*************************************************************************
def init_halos_16():
    # integer(kind=4) :: ihalo,ihtmp,imother
    
    for ih1 in frange(1, H.nb_of_halos + H.nb_of_subhalos):
        H.liste_halos_o0[ih1].clear_halo()
        H.liste_halos_o0[ih1].my_number   = ih1
        H.liste_halos_o0[ih1].my_timestep = H.numstep
        if(H.fsub):
            if(mem['level_1319'][ih1-1]==1):
                if(mem['first_daughter_1319'][ih1-1]>0):
                    H.liste_halos_o0[ih1].nextsub = mem['first_daughter_1319'][ih1-1]
                H.liste_halos_o0[ih1].hosthalo   = ih1
            else:
                H.liste_halos_o0[ih1].level    = mem['level_1319'][ih1]
                H.liste_halos_o0[ih1].hostsub = mem['mother_1319'][ih1]
                imother1 = H.liste_halos_o0[ih1].hostsub
                H.liste_halos_o0[imother1].nbsub += 1
                if(mem['first_daughter_1319'][ih1-1]>0):
                    H.liste_halos_o0[ih1].nextsub = mem['first_daughter_1319'][ih1-1]
                elif(mem['first_sister_1319'][ih1-1]>0):
                    H.liste_halos_o0[ih1].nextsub = mem['first_sister_1319'][ih1-1]
                else:
                    ihtmp1 = ih1
                    while((mem['first_sister_1319'][ihtmp1-1]<=0)and(mem['level_1319'][ihtmp1-1]>1)):
                        ihtmp1 = mem['mother_1319'][ihtmp1-1]
                    if(mem['level_1319'][ihtmp1-1]>1):
                        ihtmp1 = mem['first_sister_1319'][ihtmp1-1] = ihtmp1
                        H.liste_halos_o0[ih1].nextsub = ihtmp1
                imother1 = H.liste_halos_o0[ih1].hostsub
                imother1 = H.liste_halos_o0[imother1].hosthalo
                if(H.liste_halos_o0[imother1].level!=1): raise ValueError('wrong id for halo host')
                H.liste_halos_o0[ih1].hosthalo = imother1
    
    if(H.fsub): H.deallocate('mother_1319','first_daughter_1319','first_sister_1319','level_1319')

#*************************************************************************
def det_mass_17(h:halo):
    '''
    adds up masses of particles to get total mass of the halo. 
    '''
    # integer(kind=4)    :: indexp,npch
    # real(kind=8)       :: masshalo
    # type(halo)         :: h
    
    masshalo      = 0e0
    npch          = 0
    indexp        = mem['first_part_oo_1'][h.my_number]
    while(indexp != -1):
        if(H.allocated('mass_10')):
            masshalo += mem['mass_10'][indexp]
        else:
            masshalo += H.massp
        npch += 1
        indexp = mem['linked_list_oo_1'][indexp]       
    h.m = masshalo  # in 10^11 M_sun

    if(npch != mem['nb_of_parts_o0_1'][h.my_number]):
       print('nb_of_parts_o0, npch:',h.my_number,npch)
       raise ValueError('> Fatal error in det_mass for', h.my_number)

#***********************************************************************
def compute_ang_mom_19(h:halo):
    '''
    compute angular momentum of all halos
    we compute r * m * v, where r & v are pos and vel of halo particles relative to center of halo
    (particle closest to center of mass or most dense particle)
    '''
    # integer(kind=4) :: indexp
    # real(kind=8)    :: lx,ly,lz
    # type (halo)     :: h
    # type (vector)   :: dr,p

    dr=vector(); p=vector()
    indexp = mem['first_part_oo_1'][h.my_number]
    lx =0 ; ly = 0 ; lz = 0
    while(indexp != -1):
        dr.x   = mem['pos_10'][indexp,1] - h.p.x
        dr.y   = mem['pos_10'][indexp,2] - h.p.y
        dr.z   = mem['pos_10'][indexp,3] - h.p.z
        
        correct_for_periodicity(dr)
        
        if (H.allocated('mass_10')):
            p.x = mem['mass_10'][indexp]*(mem['vel_10'][indexp,1]-h.v.x)
            p.y = mem['mass_10'][indexp]*(mem['vel_10'][indexp,2]-h.v.y)
            p.z = mem['mass_10'][indexp]*(mem['vel_10'][indexp,3]-h.v.z)
        else:
            p.x = H.massp*(mem['vel_10'][indexp,1]-h.v.x)
            p.y = H.massp*(mem['vel_10'][indexp,2]-h.v.y)
            p.z = H.massp*(mem['vel_10'][indexp,3]-h.v.z)

        lx  += dr.y*p.z - dr.z*p.y   # in 10**11 Msun * km/s * Mpc
        ly  += dr.z*p.x - dr.x*p.z
        lz  += dr.x*p.y - dr.y*p.x        
        
        indexp = mem['linked_list_oo_1'][indexp]   

    h.L.x = lx
    h.L.y = ly
    h.L.z = lz

#***********************************************************************
def r_halos_1a(h:halo):
    '''
    compute distance of the most remote particle (with respect to center of halo, which
    is either center of mass or most bound particle)
    '''
    # integer(kind=4) :: indexp
    # real(kind=8)    :: dr2max,dr2
    # type (vector)   :: dr
    # type (halo)     :: h
    dr = vector()
    dr2max  = 0.0
    indexp = mem['first_part_oo_1'][h.my_number]

    while(indexp != -1):
        dr.x = mem['pos_10'][indexp,1] - h.p.x
        dr.y = mem['pos_10'][indexp,2] - h.p.y
        dr.z = mem['pos_10'][indexp,3] - h.p.z         
        
        correct_for_periodicity(dr)
        
        dr2    = (dr.x*dr.x + dr.y*dr.y + dr.z*dr.z)
        
        if (dr2 > dr2max):
            dr2max         = dr2
        indexp=mem['linked_list_oo_1'][indexp]
        
    h.r = np.sqrt(dr2max)

#***********************************************************************
def correct_for_periodicity(dr:vector, copy=False):
    '''
    subroutine corrects for the fact that if you have periodic boundary conditions,
    then groups of particles that sit on the edge of the box can have part of their
    particles on one side and part on the other. So we have to take out a box-length
    when measuring the distances between group members if needed.
    '''

    if (H.FlagPeriod == 0): return  #--> NO PERIODIC BCs 
    
    if(copy):
        temp = vector()
        temp.x = dr.x*1
        temp.y = dr.y*1
        temp.z = dr.z*1
        if (temp.x > + H.Lbox_pt2): temp.x = temp.x - H.Lbox_pt
        if (temp.x < - H.Lbox_pt2): temp.x = temp.x + H.Lbox_pt 

        if (temp.y > + H.Lbox_pt2): temp.y = temp.y - H.Lbox_pt
        if (temp.y < - H.Lbox_pt2): temp.y = temp.y + H.Lbox_pt 

        if (temp.z > + H.Lbox_pt2): temp.z = temp.z - H.Lbox_pt
        if (temp.z < - H.Lbox_pt2): temp.z = temp.z + H.Lbox_pt 
        return temp
    else:
        if (dr.x > + H.Lbox_pt2): dr.x = dr.x - H.Lbox_pt
        if (dr.x < - H.Lbox_pt2): dr.x = dr.x + H.Lbox_pt 

        if (dr.y > + H.Lbox_pt2): dr.y = dr.y - H.Lbox_pt
        if (dr.y < - H.Lbox_pt2): dr.y = dr.y + H.Lbox_pt 

        if (dr.z > + H.Lbox_pt2): dr.z = dr.z - H.Lbox_pt
        if (dr.z < - H.Lbox_pt2): dr.z = dr.z + H.Lbox_pt 

#***********************************************************************                
def det_center_18(h:halo):
    '''
    compute position of center of mass of halo, and its velocity.
    '''
    # type (halo)        :: h
    # integer(kind=4)    :: indexp, icenter,ifirst 
    # real(kind=8)       :: maxdens, distmin
    # real(kind=8)       :: pcx,pcy,pcz,vcx,vcy,vcz,vmean,v2mean,sigma2,vnorm
    # type(vector)       :: dr,pc

    # real(kind=8)       ::aaa,bbb,ccc,ddd,eee,half_radius,mhalf
    # integer(kind=4)    ::i,j,nmax
    # real(kind=8),dimension(:),allocatable::drr,mm,vxx,vyy,vzz

    icenter = -1

    if(H.cdm):
        # compute cdm
        dr = vector(); pc = vector()
        pcx   = 0 ; pcy   = 0 ; pcz = 0
        ifirst = mem['first_part_oo_1'][h.my_number]
        indexp = ifirst
        while(indexp != -1):
            dr.x = mem['pos_10'][indexp,1] - mem['pos_10'][ifirst,1]
            dr.y = mem['pos_10'][indexp,2] - mem['pos_10'][ifirst,2]
            dr.z = mem['pos_10'][indexp,3] - mem['pos_10'][ifirst,3]
            correct_for_periodicity(dr)
            if(H.allocated('mass_10')):
                pcx += mem['mass_10'][indexp]*dr.x,
                pcy += mem['mass_10'][indexp]*dr.y,
                pcz += mem['mass_10'][indexp]*dr.z,
            else:
                pcx += H.massp*dr.x,
                pcy += H.massp*dr.y,
                pcz += H.massp*dr.z,
            indexp = mem['linked_list_oo_1'][indexp]
        # It's simply the center of mass.
        pcx  = pcx / h.m + mem['pos_10'][ifirst,1]
        pcy  = pcy / h.m + mem['pos_10'][ifirst,2]
        pcz  = pcz / h.m + mem['pos_10'][ifirst,3]
        pc.x = pcx
        pc.y = pcy
        pc.z = pcz
        correct_for_periodicity(pc)
        # search particule closest to the cdm
        indexp  = ifirst
        distmin = H.Lbox_pt
        while (indexp != -1):
            dr.x = mem['pos_10'][indexp,1] - pc.x
            dr.y = mem['pos_10'][indexp,2] - pc.y
            dr.z = mem['pos_10'][indexp,3] - pc.z
            correct_for_periodicity(dr)
            if (np.sqrt(dr.x**2+dr.y**2+dr.z**2)<distmin):
                icenter = indexp
                distmin = np.sqrt(dr.x**2+dr.y**2+dr.z**2)
            indexp = mem['linked_list_oo_1'][indexp]
    else:
        maxdens = 0.0
        indexp  = mem['first_part_oo_1'][h.my_number]
        while (indexp != -1):
            if(mem['density_1312'][indexp]>maxdens):
                maxdens = mem['density_1312'][indexp]
                icenter = indexp
            indexp = mem['linked_list_oo_1'][indexp]

    if (icenter<0):
        print('> Could not find a center for halo: ',h.my_number,icenter)
        print('  h.m,massp,h.m/massp             : ',h.m,H.massp,h.m/H.massp)
        print('  Lbox_pt,distmin                 : ',H.Lbox_pt,distmin)
        print('  pcx,pcy,pcz                  : ',pcx,pcy,pcz)
        print('  periodicity flag                : ',H.FlagPeriod)
        raise ValueError('> Check routine det_center')

    h.p.x  = mem['pos_10'][icenter,1]
    h.p.y  = mem['pos_10'][icenter,2]
    h.p.z  = mem['pos_10'][icenter,3]

    # velocity of center is set equal velocity of center of mass:
    indexp = mem['first_part_oo_1'][h.my_number]
    vcx = 0 ; vcy = 0 ; vcz =0
    v2mean= 0
    i=0
    while (indexp != -1):
        i=i+1
        if (H.allocated('mass_10')):
            vcx += mem['mass_10'][indexp]*mem['vel_10'][indexp,1]
            vcy += mem['mass_10'][indexp]*mem['vel_10'][indexp,2]
            vcz += mem['mass_10'][indexp]*mem['vel_10'][indexp,3]
        else:
            vcx += H.massp*mem['vel_10'][indexp,1]
            vcy += H.massp*mem['vel_10'][indexp,2]
            vcz += H.massp*mem['vel_10'][indexp,3]
        indexp   = mem['linked_list_oo_1'][indexp]
    nmax=i
    
    h.v.x = vcx/h.m
    h.v.y = vcy/h.m
    h.v.z = vcz/h.m

    vmean=np.sqrt(vcx**2+vcy**2+vcz**2)/h.m

    indexp = mem['first_part_oo_1'][h.my_number]
    sigma2= 0
    i=0
    while (indexp != -1):
        i += 1
        vnorm = np.sqrt(mem['vel_10'][indexp,1]**2+mem['vel_10'][indexp,2]**2+mem['vel_10'][indexp,3]**2)
        if (H.allocated('mass_10')):
            sigma2 += mem['mass_10'][indexp]*(vnorm-vmean)**2
        else:
            sigma2 += H.massp       *(vnorm-vmean)**2
        indexp   = mem['linked_list_oo_1'][indexp]
    h.sigma = np.sqrt(sigma2/h.m)

#***********************************************************************
def interact_1b30(i,j):
    # integer(kind=4) :: i,j,ifirst
    # real(kind=8)    :: dist2ij
    # real(kind=8)    :: interact_1b30,rinveff,r3inveff
    # real(kind=8)    :: epstmp,massp2,lbox2
    # type (vector)   :: dr

    # Check if the attributes exist, and initialize them if they don't
    if not hasattr(interact_1b30, 'ifirst'):
        interact_1b30.ifirst = 1
    if not hasattr(interact_1b30, 'epstmp'):
        interact_1b30.epstmp = 0.0
    if not hasattr(interact_1b30, 'massp2'):
        interact_1b30.massp2 = 0.0

    # Constants
    ifirst = interact_1b30.ifirst
    epstmp = interact_1b30.epstmp
    massp2 = interact_1b30.massp2

    if (ifirst == 1):
        # epstmp is mean interparticular distance / 20.0 (i.e. smoothing length)
        # true for tree code but RAMSES and ENZO ?
        epstmp = (H.massp/H.mboxp)**(1./3.) / 20.0
        massp2 = H.massp*H.massp
        ifirst = 0
        interact_1b30.ifirst = ifirst
        interact_1b30.epstmp = epstmp
        interact_1b30.massp2 = massp2

    dr = vector()
    Lbox2   = H.Lbox_pt**2
    dr.x    = mem['pos_10'][j,1] - mem['pos_10'][i,1]  
    dr.y    = mem['pos_10'][j,2] - mem['pos_10'][i,2]
    dr.z    = mem['pos_10'][j,3] - mem['pos_10'][i,3]
    correct_for_periodicity(dr)
    dist2ij = (dr.x**2) + (dr.y**2) + (dr.z**2)
    dist2ij = dist2ij / Lbox2

    if (H.allocated('mass_10')):
        if (H.allocated('epsvect')):
            rinveff,r3inveff = softgrav_1b300(mem['epsvect'][i],mem['epsvect'][j],dist2ij)
            ans = -mem['mass_10'][i] * mem['mass_10'][j] * rinveff
        else:
            # do not correct for softening --> have to change that
            ans =-mem['mass_10'][i]*mem['mass_10'][j]/np.sqrt(dist2ij)
    else:
        rinveff,r3inveff = softgrav_1b300(epstmp,epstmp,dist2ij,rinveff,r3inveff)
        ans = -massp2 * rinveff
    return ans

#***********************************************************************
def softgrav_1b300(epsp,epsi,drdotdr):
    '''
    subroutine to compute the effective distance between particles of
    smoothing lengths, epsp and epsi, given their real distance**2, 
    drdotdr, in order to get the smoothed values of the potential and 
    acceleration phi and acc (in GRAVSUM). Calculations are for an 
    harmonic smoothing or a cubic spline smoothing kernel. 
    For the spline smoothing, phsmooth_oo and acsmooth must have 
    been initialized by initgsoft.
    '''
    # integer(kind=4) :: smindex
    # real(kind=8)    :: epsp,epsi,drdotdr,sdrdotdr,rinveff,r3inveff,drdeldrg
    # real(kind=8)    :: drsm,phsm,epseff,epsinv,dr,tiny,one

    tiny = 1.e-19
    one  = 1.0

    if (H.gravsoft == 'harmonic'):

        drdotdr  = drdotdr+tiny
        rinveff  = 1.0/np.sqrt(drdotdr)
        r3inveff = rinveff*rinveff*rinveff
        epseff   = 0.5*(epsp+epsi)
        if (drdotdr < epseff*epseff):
            epsinv   = 1.0/epseff
            r3inveff = epsinv*epsinv*epsinv
            rinveff  = 1.5*epsinv-0.5*drdotdr*r3inveff

    elif(H.gravsoft == 'cubsplin'):
        dr       = epsp+epsi
        drdotdr  = drdotdr+tiny*0.25*dr**2
        sdrdotdr = np.sqrt(drdotdr)
        drdeldrg = sdrdotdr*H.ninterp/dr
        smindex  = int(drdeldrg)
        if (smindex > H.ninterp): 
            phsm = H.phsmooth_oo[1+H.ninterp]
        else:
            if (one < drdeldrg-smindex):
                phsm = H.phsmooth_oo[1+smindex]
            else:
                drsm = drdeldrg-smindex
                phsm = (1.-drsm)*H.phsmooth_oo[smindex]+drsm*H.phsmooth_oo[1+smindex]
        rinveff = phsm/sdrdotdr
        # NB: r3inveff is used to compute the acceleration and necessitates
        #     the definition of accsmooth which is not available here.  
        #     For the treecode, the relation should be:
        #     r3inveff=accsmooth*r3inveff (to be checked)
    return rinveff,r3inveff

#***********************************************************************
def tab_props_inside_1b40(h:halo,nr,tabm2_o0,tabk2_o0,tabp2_o0,v,amax,bmax,cmax):
    '''
    returns the cumulative mass contained in concentric ellipsoids centered on the center of the
    halo (cdm or mbp)
    '''
    # integer(kind=4)        :: nr,num_h,indexp,i,i_ell,louped_parts  
    # real(kind=8)           :: amax,bmax,cmax,v(3,3)
    # real(kind=8)           :: tabm2_o0(0:nr-1),tabk2_o0(0:nr-1),tabp2_o0(0:nr-1) ## double
    # real(kind=8)           :: srm,srk                                   ## double
    # real(kind=8)           :: rmax,dra,drb,drc
    # real(kind=8)           :: r_ell,v2,rf
    # real(kind=8),parameter :: epsilon = 1.d-2
    # type (vector)          :: posp,vt
    # type (halo)            :: h
    from num_rec import rf
    posp = vector(); vt = vector()
    # rescale to get ellipsoid  concentric to principal ellipsoid
    # which contains all the particles of the halo
    rmax = 0.0
    indexp = mem['first_part_oo_1'][h.my_number]
    while(indexp > 0):
       posp.x = mem['pos_10'][indexp,0] - h.p.x
       posp.y = mem['pos_10'][indexp,1] - h.p.y
       posp.z = mem['pos_10'][indexp,2] - h.p.z
       correct_for_periodicity(posp)
        # project position vector along the principal ellipsoid axis
       dra    = posp.x*v[0,0]+posp.y*v[1,0]+posp.z*v[2,0]
       drb    = posp.x*v[0,1]+posp.y*v[1,1]+posp.z*v[2,1]
       drc    = posp.x*v[0,2]+posp.y*v[1,2]+posp.z*v[2,2]
       r_ell  = np.sqrt((dra / h.sh.a)**2 + (drb / h.sh.b)**2 + (drc / h.sh.c)**2)
       rmax   = max(rmax,r_ell)
       indexp = mem['linked_list_oo_1'][indexp]
    
    amax = rmax * h.sh.a * (1.0 + H.epsilon)
    bmax = rmax * h.sh.b * (1.0 + H.epsilon)
    cmax = rmax * h.sh.c * (1.0 + H.epsilon)

    # initialize loop quantities
    tabm2_o0[:] = 0
    tabk2_o0[:] = 0
    louped_parts = 0
    indexp       = mem['first_part_oo_1'][h.my_number]

    while (indexp != -1):
        posp.x = mem['pos_10'][indexp,0] - h.p.x
        posp.y = mem['pos_10'][indexp,1] - h.p.y
        posp.z = mem['pos_10'][indexp,2] - h.p.z
        correct_for_periodicity(posp)
        # compute velocities in the halo frame adding in the Hubble flow
        vt.x   = mem['vel_10'][indexp,0] - h.v.x + posp.x * H.Hub_pt
        vt.y   = mem['vel_10'][indexp,1] - h.v.y + posp.y * H.Hub_pt
        vt.z   = mem['vel_10'][indexp,2] - h.v.z + posp.z * H.Hub_pt
        v2     = vt.x**2 + vt.y**2 + vt.z**2
        # project position vector along the principal ellipsoid axis
        dra    = posp.x*v[0,0]+posp.y*v[1,0]+posp.z*v[2,0]
        drb    = posp.x*v[0,1]+posp.y*v[1,1]+posp.z*v[2,1]
        drc    = posp.x*v[0,2]+posp.y*v[1,2]+posp.z*v[2,2]
        # biggest ellipsoid is divided in nr concentric ellipsoid shells: we
        # calculate below the ellipsoid bin in which each particle falls and fill up the 
        # mass and energy tables accordingly
        # NB: if by chance the most distant particle from the center is ON the shortest
        #     axis, then r_ell is equal to 1-epsilon (one of the terms below is one and the others 0)
        #     otherwise r_ell is between 0 and 1-epsilon and so we just multiply it by nr to 
        #     find the ellipsoid shell containing the particle.
        r_ell  = np.sqrt((dra / amax)**2 + (drb / bmax)**2 + (drc / cmax)**2)
        i_ell  = int(r_ell*nr)
        if (i_ell < nr):
            if (H.allocated('mass_10')):
                tabm2_o0[i_ell] += mem['mass_10'][indexp]
                tabk2_o0[i_ell] += 0.5*mem['mass_10'][indexp]*v2
            else:
                tabm2_o0[i_ell] += +H.massp
                tabk2_o0[i_ell] += +0.5*H.massp*v2
        else:
            louped_parts += 1

        indexp = mem['linked_list_oo_1'][indexp]

    if (louped_parts  >  0):
       raise ValueError('> Problem in tab_props_inside : missed ',louped_parts,' particles\n')

    srm = tabm2_o0[0]
    srk = tabk2_o0[0]
    for i1 in frange(1,nr-1):
        srm      += tabm2_o0[i1]
        srk      += tabk2_o0[i1]
        tabm2_o0[i1] = srm
        tabk2_o0[i1] = srk
        # approximation based on appendix B of paper GALICS 1:
        # better than 10-15 . accuracy on average
        tabp2_o0[i1] = -0.3 * H.gravconst * tabm2_o0[i1]**2 * rf(h.sh.a**2,h.sh.b**2,h.sh.c**2)
    # correct potential energy estimate for small halos which are calculated by direct summation 
    if (h.ep != tabp2_o0[nr-1]): tabp2_o0 = tabp2_o0/tabp2_o0[nr-1]*h.ep
    # return tabm2_o0,tabk2_o0,tabp2_o0,amax,bmax,cmax

#***********************************************************************

def det_vir_props_1b4(h:halo,v,amax=0,bmax=0,cmax=0,ttab=1000):
    '''
    computes the virial properties (radius, mass) of a halo
    '''
    # integer(kind=4)           :: i,ii
    # # ttab = 1000 bins for virial radius precision better than 1. of halo size 
    # integer(kind=4),parameter :: ttab = 1000 
    # real(kind=8)              :: rvir,mvir,kvir,pvir,v(3,3)
    # real(kind=8)              :: amax,bmax,cmax,avir,bvir,cvir
    # real(kind=8)              :: tab_mass_o0(0:ttab-1),tab_ekin_o0(0:ttab-1),tab_epot_o0(0:ttab-1)  ## double
    tab_mass_o0 = np.zeros(ttab); tab_ekin_o0 = np.zeros(ttab); tab_epot_o0 = np.zeros(ttab)
    # real(kind=8)              :: virth,virth_old,volmin
    # type (halo)               :: h

    # compute properties inside ttab concentric principal ellipsoids centered on center of halo
    tab_props_inside_1b40(h,ttab,tab_mass_o0,tab_ekin_o0,tab_epot_o0,v,amax,bmax,cmax)

    # find the outermost ellipsoid bin where virial theorem is either satisfied better than 20 .
    # or satisfied best ... 
    mvir      = tab_mass_o0[ttab-1]
    kvir      = tab_ekin_o0[ttab-1]
    pvir      = tab_epot_o0[ttab-1]
    # initialize rvir to be the geometric average of the axis radii of the outermost ellipsoid shell
    # which contains at least one particle
    for i1 in frange(ttab-1,1,-1):
        if (tab_mass_o0[i1-1] < tab_mass_o0[i1]): break

    avir      = i1/(ttab-1)*amax
    bvir      = i1/(ttab-1)*bmax
    cvir      = i1/(ttab-1)*cmax
    rvir      = (avir*bvir*cvir)**(1./3.)
    # assume initial departure from virialization is 100 .
    virth_old = 1.0
    virths = np.abs((2.0*tab_ekin_o0+tab_epot_o0)/(tab_ekin_o0+tab_epot_o0))
    i0 = np.where(virths <= 0.2)[0][-1]
    virth = virths[i0]
    mvir      = tab_mass_o0[i0] 
    # take the min here bc initialization throws away all the empty outer shells
    avir      = min(avir,i0/(ttab-1)*amax)
    bvir      = min(bvir,i0/(ttab-1)*bmax)
    cvir      = min(cvir,i0/(ttab-1)*cmax)
    rvir      = (avir*bvir*cvir)**(1./3.)
    kvir      = tab_ekin_o0[i0] 
    pvir      = tab_epot_o0[i0]
    virth_old = virth
    print("#-----------#")
    print(virth, mvir)
    for i0 in frange(ttab-1,0,-1):
        # if region is unbound, it cannot be virialized in the same time 
        if( (tab_ekin_o0[i0]+tab_epot_o0[i0]) >= 0.0): continue
        # the region is bound so compute relative virialization |2*K+P|/|K+P|
        virth = abs((2.0*tab_ekin_o0[i0]+tab_epot_o0[i0])/(tab_ekin_o0[i0]+tab_epot_o0[i0]))
        # if region is better virialized then update virial quantities
        if (virth < virth_old):
            mvir      = tab_mass_o0[i0] 
            # take the min here bc initialization throws away all the empty outer shells
            avir      = min(avir,i0/(ttab-1)*amax)
            bvir      = min(bvir,i0/(ttab-1)*bmax)
            cvir      = min(cvir,i0/(ttab-1)*cmax)
            rvir      = (avir*bvir*cvir)**(1./3.)
            kvir      = tab_ekin_o0[i0] 
            pvir      = tab_epot_o0[i0]
            virth_old = virth
            # if virial theorem holds with better than 20 . accuracy, exit do loop
            if (virth <= 0.20): break
    print(virth, mvir)
    print("#-----------#")
 
    # for small halos it may happen that virial theorem is not enforced to within 15 .
    # bc the halo is not fully virialized yet or that it is valid by fluke (right combination
    # of potential and kinetic energy) ... so .... 
    # 1/ in the latter case, we further check that the halo density is high enough 
    # (order of what predicted by the spherical top-hat model, vir_overdens) 
    # for us to believe in the measurement of the virial theorem. 
    # 2/ in the former case, as an inner region measurement of the virialization would be too noisy for 
    # lack of mass resolution (not enough particles) we only use the overdensity criterion.
    # NB: this criterion is similar to NFW but with vir_overdens * average density and NOT 200. * critical density
    #     which does not makes sense when Omega_matter(z) != 1 (not critical) and we take ellipsoids
    #     NOT spheres bc they yield more accurate volumes and average halo densities
    # average density of the universe at current timestep (in 10^11 M_sun / Mpc^3)
 
    # volume of the smallest concentric ellipsoid
    volmin   = 4./3.*H.pi*(amax/(ttab-1))*(bmax/(ttab-1))*(cmax/(ttab-1))
    if( (virth > 0.20)or(mvir < H.vir_overdens*H.rho_mean*4./3.*H.pi*avir*bvir*cvir)):
        for ii1 in frange(ttab-1,1,-1):
            # assume that the virial mass and radii are obtained when the density inside the ellipsoid 
            # is greater than vir_overdens * rho_mean AND there is at least one particle inside the outermost
            # ellipsoid shell
            mvir = H.vir_overdens * H.rho_mean * volmin * ii1**3
            #mvir = 200d0 * 3d0*Hub_pt**2/8d0/acos(-1d0)/gravconst * volmin * ii1,4)**3
            if( (tab_mass_o0[ii1] >= mvir)and(tab_mass_o0[ii1-1] < tab_mass_o0[ttab-1]) ): break
        mvir   = tab_mass_o0[ii1]
        kvir   = tab_ekin_o0[ii1]
        pvir   = tab_epot_o0[ii1]
        avir   = ii1/(ttab-1)*amax
        bvir   = ii1/(ttab-1)*bmax
        cvir   = ii1/(ttab-1)*cmax
        rvir   = (avir*bvir*cvir)**(1./3.)

    # check if virialization conditions were met --> if not set relevant quantities to zero: this is a 
    # non-virialized halo ....
    if( (mvir > 0.0)and(rvir > 0.0) ):
       # it may happen (although it is very rare bc vector linking center of halo to most distant 
       # particle has to be roughly parallel to minor axis of the halo in such a case) that the virial 
       # radius estimated from the geometric mean of the 3 principal axis is slightly larger than the 
       # distance of the most distant particle to the center of the halo (what we call r)
       # when this happens we set the virial radius to be r
       h.datas.rvir         = min(rvir,h.r)    # in Mpc
       h.datas.mvir         = mvir             # in 10^11 M_sun
       # circular velocity at r_vir in km/s
       h.datas.cvel         = np.sqrt(H.gravconst*h.datas.mvir/h.datas.rvir) 
       # temperature at r_vir in K
       h.datas.tvir         = 35.9*H.gravconst*h.datas.mvir/h.datas.rvir
       # compute halo density profile within the virialized region
       compute_halo_profile_1b41(h)
    else:
       print('halo bugged (ID, Mvir, Rvir)',h.my_number,mvir,rvir)
       raise ValueError('at ',h.p.x,h.p.y,h.p.z)
    return amax,bmax,cmax

#***********************************************************************
def compute_halo_profile_1b41(h:halo):
    # type (halo)     :: h
    if (H.profile == 'TSIS'):
       # for the singular isothermal sphere the profile is defined @ rvir for it is singular at r=0.
       h.halo_profile.rho_0 = h.datas.mvir / (4.0 * H.pi * h.datas.rvir**3)
       h.halo_profile.r_c   = h.datas.rvir
    else:
       raise NotImplementedError('Other profiles than TSIS not yet fully implemented')

#***********************************************************************
def det_vir_1b(h:halo, fagor:FortranFile=None):
    '''
    determine virial properties of the halos, energies and profiles
    '''
    # type (halo)     :: h
    # real(kind=8)    :: v(3,3)
    # real(kind=8)    :: x0,y0,z0,r0
    # #ifdef H.ANG_MOM_OF_R
    # real(kind=8)    :: amax,bmax,cmax
    # #endif
    # integer         :: nx
    v = np.zeros((3,3), dtype=np.float64)

    if(H.method != "FOF" and H.DPMMC):
        x0=h.p.x;y0=h.p.y;z0=h.p.z;r0=h.r
        nx=9 # The initial mesh have a size of 2 times the maximum radius (~virial radius)
        det_halo_center_multiscale_1b0(h,x0,y0,z0,r0,nx)
    if(H.method != "FOF" and H.SC):
        x0=h.p.x;y0=h.p.y;z0=h.p.z;r0=h.r
        det_halo_center_sphere_1b1(h,x0,y0,z0,r0)

    # compute principal axis of halo
    det_main_axis_1b2(h,v)
 
    # compute halo energies if necessary i.e. in the case where the center of the halo is the 
    # center of mass bc if it is the most bound particle then energies are already available  
    det_halo_energies_1b3(h)

    # compute virial properties based on conditions necessary for the virial theorem to apply
    amax,bmax,cmax = det_vir_props_1b4(h,v)
    if(H.ANG_MOM_OF_R):
        det_ang_momentum_per_shell_1b5(h,amax,bmax,cmax,v,fagor=fagor)
        

#***********************************************************************
def det_age_11():
    '''
    subroutine which determines the current age of the Universe (Gyr) and
    values of current cosmological parameters 
    '''
    from num_rec import qromo, age_temps
    from functools import partial
    import scipy.integrate as integrate
    # real(kind=8) :: age0,age1,somme0,somme1,omm,oml
    # save age0

    omm  = H.omega_f
    oml  = H.omega_lambda_f
    # age0 is the age of the universe @ the beginning of the simulation (in Gyr)
    # somme0 = qromo(age_temps,(H.af/H.ai),10001,omm=omm,oml=oml)
    func = partial(age_temps, oml=oml, omm=omm)
    somme0 = integrate.quad(func, (H.af/H.ai), 10001)[0]
    age0 = 977.78*somme0/H.H_f            

    # age1 is the time between the beginning of the simulation and the current timestep (in Gyr)
    if(H.aexp != H.ai):
    #    somme1 = qromo(age_temps,(H.af/H.aexp),(H.af/H.ai),omm=omm,oml=oml)
       somme1 = integrate.quad(func, (H.af/H.aexp), (H.af/H.ai))[0]
    else:
       somme1 = 0.0

    age1 = 977.78*somme1/H.H_f 

    # finally the age of the universe is the sum 
    H.age_univ = age0+age1  

    print()
    print( '> Current values of parameters: ')
    print( '> ----------------------------- ')
    print( '> aexp                        : ',H.aexp)
    print( '> Redshift                    : ',H.af/H.aexp-1.)
    print( '> Age of the Universe (Gyr)   : ',H.age_univ)

    H.Hub_pt   = H.H_f * np.sqrt(H.omega_f*(H.af/H.aexp)**3 +  H.omega_c_f*(H.af/H.aexp)**2 + H.omega_lambda_f)
    H.Lbox_pt  = H.Lboxp*(H.aexp/H.ai)
    H.Lbox_pt2 = H.Lbox_pt / 2.0

    print( '> Hubble Parameter  (km/s/Mpc): ',H.Hub_pt)
    print( '> Box Length (Mpc)            : ',H.Lbox_pt)
    print()

    return

#***********************************************************************
def virial_12():
    '''
    compute the overdensity factor for virialization in a tophat collapse model at a given redshift 
    for a given cosmological model
    '''
    # real(kind=8) :: a,b,eta,omega_maxexp,age,reduce,cubic
    from num_rec import cubic
    
    # convert age of universe from Gyr back into inverse Hubble parameter units
    age       = H.age_univ/977.78*H.H_f
    # compute the overdensity needed to reach maximum expansion by half the age of the universe
    omega_maxexp = H.omega_f
    collapse_120(age/2.0,omega_maxexp,1.e-6)
    # calculate how far an object collapses to virial equilibrium
    eta = 2.0*H.omega_lambda_f/omega_maxexp*(H.af/H.aexp)**3
    if(eta == 0.0):
       reduce = 0.5
    else:
       a      = 2.0*eta
       b      = -(2.0+eta)
       reduce = cubic(a,0.0,b,1.0)
    
    H.vir_overdens = omega_maxexp/H.omega_f/reduce**3*(H.aexp/H.af)**3
    H.rho_mean     = H.mboxp/H.Lbox_pt**3


#***********************************************************************
def collapse_120(age0,omm,acc):
    '''
    this subroutine performs a brute-force search using bracketing to find the value of the cosmic curvature 
    that gives turnaround at the specified expansion parameter. The expansion parameter is defined to
    be 1 at turnaround.

    note that for a constant cosmological constant, the age at a given
    expansion factor decreases monotonically with omegam (before turnaround;
    after, there is more than one age at a given expansion factor).

    third argument is the desired fractional accurracy.
    '''    
    # real(kind=8)            :: age0,omm,acc,oml
    # real(kind=8)            :: age,omax,omin,age_f
    # real(kind=8), parameter :: omax0=1.d7,omin0=1.d0 
    # IMPORTANT NOTE: omax0 corresponds to a perturbation turning around at z=140 in a LCDM standard cosmology
    #                 and needs to be increased if you analyze outputs before this redshift ...
    from num_rec import age_temps_turn_around, qromo
    from functools import partial
    from scipy import integrate
    # import time

    omax0=1.e7 ;omin0=1.e0; age_f=0
    age  = -1.0  # impossible value
    omax = omax0
    omin = omin0
    oml  = H.omega_lambda_f
    while( (abs(age-age0) > acc*age0)and((omax-omin) > acc*omm) ):
        omm = 0.5*(omax+omin)
        # age_f = qromo(age_temps_turn_around,101,10001,omm=omm,oml=oml)
        func = partial(age_temps_turn_around, oml=oml, omm=omm)
        age_f = integrate.quad(func, 101, 10001)[0]
        # age = qromo(age_temps_turn_around,1,101,omm=omm,oml=oml)
        age = integrate.quad(func, 1, 101)[0]
        if(age+age_f > age0):
            omin = omm
        else:
            omax = omm

    if( (omax==omax0)or(omin==omin0) ):
        print('WARNING: presumed bounds for omega are inadequate in collapse.')
        print('WARNING: omax,omax0,omin,omin0=',omax,omax0,omin,omin0)

#***********************************************************************
def change_units_15():
    '''
    subroutine which goes from positions in code units to physical (non comoving)
    Mpc and from velocities in code units to peculiar (no Hubble flow) velocities 
    in km/s. masses are also changed from code units to 10^11 M_sun
    '''
    mem['pos_10']   = mem['pos_10'] * H.Lbox_pt
    if(type == 'SN'): mem['vel_10'] = mem['vel_10']*H.Hub_pt*H.Lbox_pt
    massp = massp * H.mboxp
    if (H.allocated('mass_10')): mem['mass_10']  = mem['mass_10'] * H.mboxp

#***********************************************************************
def det_halo_energies_1b3(h:halo, full_PE=1000):
    # integer(kind=4)           :: indexp,indexpp,np
    # integer(kind=4),parameter :: full_PE = 1000 # below this number of parts, we calculate full potential energy 
    # real(kind=8)              :: v2,rf          # rf is elliptic integral function from numrec
    # real(kind=8)              :: ped,ked        # need hi precision for potential and ke energy sum.   
    # logical(kind=4)           :: count_pairs
    # type (halo)               :: h
    # type (vector)             :: vt,dr
    from num_rec import rf

    vt = vector(); dr = vector()
    inp          = mem['nb_of_parts_o0_1'][h.my_number]
    count_pairs = (inp < full_PE)
    # get potential energy 
    if (not count_pairs):
       # formula B1 of appendix of paper GalICS 1 :
       # EP = -3/5 G M^2 Rf, 
       # with Rf = 0.5 * Int_0^infty dt/sqrt((t+x)(t+y)(t+z)) = 0.5 * rf_numrec
       # NB : rf_numrec returns RF in inverse Mpc (because x is in Mpc^2)
       h.ep = (-0.3) * H.gravconst * h.m**2 * rf(h.sh.a**2,h.sh.b**2,h.sh.c**2)
    else:
       indexp = mem['first_part_oo_1'][h.my_number]
       ped    = 0
       while (indexp != -1):
          indexpp = mem['linked_list_oo_1'][indexp] # only count pairs once
          while (indexpp != -1):
             ped     += interact_1b30(indexp,indexpp)
             indexpp = mem['linked_list_oo_1'][indexpp]
          indexp = mem['linked_list_oo_1'][indexp]
       h.ep = ped * H.gravconst / H.Lbox_pt

    # get kinetic energy (in center-of-halo frame)
    indexp = mem['first_part_oo_1'][h.my_number]
    ked    = 0
    while (indexp != -1):
       vt.x = mem['vel_10'][indexp,0] - h.v.x
       vt.y = mem['vel_10'][indexp,1] - h.v.y
       vt.z = mem['vel_10'][indexp,2] - h.v.z
       dr.x = mem['pos_10'][indexp,0] - h.p.x
       dr.y = mem['pos_10'][indexp,1] - h.p.y
       dr.z = mem['pos_10'][indexp,2] - h.p.z
       correct_for_periodicity(dr)
       # add Hubble flow 
       vt.x = vt.x + dr.x * H.Hub_pt
       vt.y = vt.y + dr.y * H.Hub_pt
       vt.z = vt.z + dr.z * H.Hub_pt
       v2   = vt.x**2 + vt.y**2 + vt.z**2
       if (H.allocated('mass_10')):
          ked += mem['mass_10'][indexp]*v2
       else:
          ked += H.massp*v2
       indexp  = mem['linked_list_oo_1'][indexp]
    h.ek = 0.5*ked
    # get total energy 
    h.et = h.ek + h.ep
    
    return

#***********************************************************************
def det_halo_center_multiscale_1b0(h:halo,x0,y0,z0,r0,nx):
    '''
    The initial guesss of the C.O.M. was calculated by det_center().
    '''
    # integer(kind=4)           :: indexp
    # type (halo)               :: h
    # integer                   :: ii,jj,kk,imax,jmax,kmax,itarget,nx,nxnew
    # real(kind=8)              :: xmin,ymin,zmin,deltax,mass_max
    # real(kind=8)              :: x0,y0,z0,r0,xc,yc,zc
    # real(kind=8),dimension(:,:,:),allocatable:: mass_grid

    mass_grid = np.zeros((nx,nx,nx), dtype=np.float64)

    xmin=x0-r0
    ymin=y0-r0
    zmin=z0-r0
    deltax=2*r0/nx

    # Assign mass to a uniform mesh with NGP
    indexp = mem['first_part_oo_1'][h.my_number]
    while (indexp != -1):
        ii= int( (mem['pos_10'][indexp,1] - xmin)/deltax )+1
        jj= int( (mem['pos_10'][indexp,2] - ymin)/deltax )+1
        kk= int( (mem['pos_10'][indexp,3] - zmin)/deltax )+1
        if( (ii>0)and(ii<=nx)and(jj>0)and(jj<=nx)and(kk>0)and(kk<=nx) ):
            if (H.allocated('mass_10')):
                mass_grid[ii,jj,kk] += mem['mass_10'][indexp]
            else:
                mass_grid[ii,jj,kk] += H.massp
        indexp  = mem['linked_list_oo_1'][indexp]

    # Search for the cell containing the maximum mass
    mass_max = np.max(mass_grid)
    imax,jmax,kmax = np.where(mass_grid == mass_max)
    imax = imax[0]; jmax = jmax[0]; kmax = kmax[0]
    del mass_grid


    #if(deltax > 0.02*h.r)then
    #if(deltax > 2.0*dcell_min)then

    nxnew=3
    if(deltax > nxnew*H.dcell_min):
        xc=xmin+(imax-0.5)*deltax
        yc=ymin+(jmax-0.5)*deltax
        zc=zmin+(kmax-0.5)*deltax
        det_halo_center_multiscale_1b0(h,xc,yc,zc,deltax,nxnew)
    else:
        # Find the particle with the maximum density within 
        # the cell with the maximum mass
        mass_max=0.0
        indexp = mem['first_part_oo_1'][h.my_number]
        while (indexp != -1):
            ii= int( (mem['pos_10'][indexp,1] - xmin)/deltax )+1
            if(ii==imax):
                jj= int( (mem['pos_10'][indexp,2] - ymin)/deltax )+1
                if(jj==jmax):
                    kk= int( (mem['pos_10'][indexp,3] - zmin)/deltax )+1
                    if(kk==kmax):         
                        if(mem['density_1312'][indexp]>mass_max):
                            mass_max=mem['density_1312'][indexp]
                            itarget=indexp
            indexp  = mem['linked_list_oo_1'][indexp]
        # Assign the new halo center
        h.p.x=mem['pos_10'][itarget,1]
        h.p.y=mem['pos_10'][itarget,2]
        h.p.z=mem['pos_10'][itarget,3]

#***********************************************************************
def det_halo_center_sphere_1b1(h:halo,x0,y0,z0,r0):
    # integer(kind=4)    :: indexp,ifirst,itarget
    # type (halo)        :: h
    # integer            :: nxnew
    # real(kind=8)       :: x0,y0,z0,r0,r02,xc,yc,zc,mtot
    # real(kind=4)       :: distmin
    # real(kind=8)       :: pcx,pcy,pcz,dr2,deltax
    # type(vector)       :: dr,pc

    r02=r0*r0
    dr = vector(); pc = vector()

    # compute cdm
    pcx   = 0 ; pcy   = 0 ; pcz = 0 ; mtot=0
    ifirst = mem['first_part_oo_1'][h.my_number]
    indexp = ifirst
    while (indexp != -1):
        dr.x = mem['pos_10'][indexp,1] - x0
        dr.y = mem['pos_10'][indexp,2] - y0
        dr.z = mem['pos_10'][indexp,3] - z0
        correct_for_periodicity(dr)
        dr2=    dr.x*dr.x
        if(dr2 <= r02):
            dr2=dr2+dr.y*dr.y
            if(dr2 <= r02):
                dr2=dr2+dr.z*dr.z
                if(dr2 <= r02):
                    if(H.allocated('mass_10')):
                        pcx += mem['mass_10'][indexp]*dr.x
                        pcy += mem['mass_10'][indexp]*dr.y
                        pcz += mem['mass_10'][indexp]*dr.z
                        mtot+= mem['mass_10'][indexp]
                    else:
                        pcx += H.massp*dr.x
                        pcy += H.massp*dr.y
                        pcz += H.massp*dr.z
                        mtot+= H.massp
        indexp = mem['linked_list_oo_1'][indexp]
    if(mtot > 0):
       xc  = pcx / mtot + x0
       yc  = pcy / mtot + y0
       zc  = pcz / mtot + z0
    else:
       xc = x0
       yc = y0
       zc = z0   

    if(r0 > H.dcell_min and mtot>0):
       det_halo_center_sphere_1b1(h,xc,yc,zc,(1-H.eps_SC)*r0)
    else:
       pc.x = xc
       pc.y = yc
       pc.z = zc
       correct_for_periodicity(pc)
       # search particule closest to the cdm
       distmin = r02
       itarget = -1
       while (itarget == -1):
          indexp  = ifirst
          while (indexp != -1):
             dr.x = mem['pos_10'][indexp,1] - pc.x
             dr.y = mem['pos_10'][indexp,2] - pc.y
             dr.z = mem['pos_10'][indexp,3] - pc.z
             correct_for_periodicity(dr)
             dr2=dr.x**2+dr.y**2+dr.z**2
             if (dr2 < distmin):
                itarget = indexp
                distmin = dr2
             indexp = mem['linked_list_oo_1'][indexp]
          distmin=distmin*2**2
          if(distmin > 1.0): raise ValueError(f'distmin(={distmin}) > 1')
       # Assign the new halo center
       h.p.x=mem['pos_10'][itarget,1]
       h.p.y=mem['pos_10'][itarget,2]
       h.p.z=mem['pos_10'][itarget,3]

#***********************************************************************
def compute_spin_parameter_1c(h:halo):
    # real(kind=8)                :: hl,spin
    # type (halo)                 :: h

    hl                  = h.L.x**2 + h.L.y**2 + h.L.z**2
    hl                  = np.sqrt(hl)        
    spin                = hl * np.sqrt(abs(h.et)) / h.m**2.5
    spin                = spin / H.gravconst
    h.spin              = spin

#***********************************************************************
def det_inertial_tensor_1b20(h:halo,mat:np.ndarray):
    '''
    Compute inertial tensor with respect to center of halo (either cdm or mbp)
    '''
    # integer(kind=4) :: num_h,indexp
    # real(kind=8)    :: mat(1:3,1:3)
    # real(kind=8)    :: md(1:3,1:3)
    # type (vector)   :: dr
    # type (halo)     :: h

    indexp = mem['first_part_oo_1'][h.my_number]
    dr = vector()

    while (indexp != -1):
        dr.x=mem['pos_10'][indexp,1]-h.p.x
        dr.y=mem['pos_10'][indexp,2]-h.p.y
        dr.z=mem['pos_10'][indexp,3]-h.p.z

        correct_for_periodicity(dr)

        if (H.allocated('mass_10')):
            mat[0,0] += mem['mass_10'][indexp]*dr.x*dr.x
            mat[0,1] += mem['mass_10'][indexp]*dr.x*dr.y
            mat[0,2] += mem['mass_10'][indexp]*dr.x*dr.z
            mat[1,0] += mem['mass_10'][indexp]*dr.x*dr.y
            mat[1,1] += mem['mass_10'][indexp]*dr.y*dr.y
            mat[1,2] += mem['mass_10'][indexp]*dr.y*dr.z
            mat[2,0] += mem['mass_10'][indexp]*dr.x*dr.z
            mat[2,1] += mem['mass_10'][indexp]*dr.y*dr.z
            mat[2,2] += mem['mass_10'][indexp]*dr.z*dr.z
        else:
            mat[0,0] += H.massp*dr.x*dr.x
            mat[0,1] += H.massp*dr.x*dr.y
            mat[0,2] += H.massp*dr.x*dr.z
            mat[1,0] += H.massp*dr.x*dr.y
            mat[1,1] += H.massp*dr.y*dr.y
            mat[1,2] += H.massp*dr.y*dr.z
            mat[2,0] += H.massp*dr.x*dr.z
            mat[2,1] += H.massp*dr.y*dr.z
            mat[2,2] += H.massp*dr.z*dr.z

        indexp = mem['linked_list_oo_1'][indexp]

#***********************************************************************
def det_main_axis_1b2(h:halo,v):
    '''
    determine the principal axis of the halo (h.sh.a,b,c)
    '''
    # integer(kind=4) :: nrot
    # real(kind=8)    :: mat(1:3,1:3)
    # real(kind=8)    :: d(3),v(3,3)
    # type (halo)     :: h
    from num_rec import jacobi

    mat = np.zeros((3,3), dtype=np.float64)
    det_inertial_tensor_1b20(h,mat)

    d,v = jacobi(mat)

    d      = np.sqrt(d/h.m)
    h.sh.a = d[0]
    h.sh.b = d[1]
    h.sh.c = d[2]

#***********************************************************************
def det_ang_momentum_per_shell_1b5(h,amax,bmax,cmax,v,fagor:FortranFile=None):    
    # integer(kind=4) :: indexp
    # type(halo)      :: h
    # type(vector)    :: dr, dv
    dr=vector();dv=vector()
    # real(kind=8)    :: amax,bmax,cmax,v(3,3) # computed in tab_props_inside
    # real(kind=8)    :: dra, drb, drc, r_ell
    # integer(kind=4) :: i_ell
    # type(vector)    :: L(nshells)  # ang mom of a shell (in 10**11 Msun * km/s * Mpc)
    L = [vector() for _ in range(H.nshells)]
    # real(kind=8)    :: m(nshells)  # mass of a shell (in 10**11 Msun)
    m = np.zeros(H.nshells, dtype=np.float64)
    # integer(kind=4) :: i

    for i0 in range(H.nshells):
       L[i0].x = 0.0
       L[i0].y = 0.0
       L[i0].z = 0.0
    m[:]   = 0.0
    indexp = mem['first_part_oo_1'][h.my_number]
    while(indexp != -1):
        # particle positions relative to center of halo
        dr.x   = mem['pos_10'][indexp,0] - h.p.x
        dr.y   = mem['pos_10'][indexp,1] - h.p.y
        dr.z   = mem['pos_10'][indexp,2] - h.p.z
        correct_for_periodicity(dr)
        # convert dr into ellipsoid coords
        dra    = dr.x*v[0,0]+dr.y*v[1,0]+dr.z*v[2,0]
        drb    = dr.x*v[0,1]+dr.y*v[1,1]+dr.z*v[2,1]
        drc    = dr.x*v[0,2]+dr.y*v[1,2]+dr.z*v[2,2]
        # index of shell containing particle
        r_ell  = np.sqrt((dra / amax)**2 + (drb / bmax)**2 + (drc / cmax)**2)
        i_ell  = int(r_ell*H.nshells)
        if (i_ell > H.nshells):
            print('> Problem in get_ang_momentum_per_shell : i_ell > nshells ')
            raise ValueError(f"{i_ell},{H.nshells}")
        # velocity relative to halo velocity (cdm)
        dv.x   = mem['vel_10'][indexp,0]-h.v.x
        dv.y   = mem['vel_10'][indexp,1]-h.v.y
        dv.z   = mem['vel_10'][indexp,2]-h.v.z
        # update mass and angular momentum of shell
        if(H.BIG_RUN):
            m[i_ell]   += H.massp
            L[i_ell].x += H.massp*(dr.y*dv.z - dr.z*dv.y) # in 10**11 Msun * km/s * Mpc
            L[i_ell].y += H.massp*(dr.z*dv.x - dr.x*dv.z)
            L[i_ell].z += H.massp*(dr.x*dv.y - dr.y*dv.x)
        else:
            m[i_ell]   += mem['mass_10'][indexp]
            L[i_ell].x += mem['mass_10'][indexp]*(dr.y*dv.z - dr.z*dv.y)
            L[i_ell].y += mem['mass_10'][indexp]*(dr.z*dv.x - dr.x*dv.z)
            L[i_ell].z += mem['mass_10'][indexp]*(dr.x*dv.y - dr.y*dv.x)
        indexp = mem['linked_list_oo_1'][indexp]   
    
    fagor.write_record(
        h.my_number, h.p.x, h.p.y, h.p.z, h.v.x, h.v.y, h.v.z,
        h.datas.rvir, h.datas.mvir, h.m, h.r, h.spin,
        amax, bmax, cmax, v[0,:3], v[1,:3], v[2,:3],
        m[0:H.nshells], L[0:H.nshells].x, L[0:H.nshells].y, L[0:H.nshells].z
        )
    
    return