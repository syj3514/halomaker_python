#=======================================================================
#                           COMPUTE_NEIKDTREE
#=======================================================================
# Author : S. Colombi
#          Institut d'Astrophysique de Paris
#          98 bis bd Arago, F-75014, Paris, France 
#          colombi@iap.fr
#
# This program has multiple usage and can do three things
# (1) Read an input particle distribution file and compute the mean
#     square distance between each particle and its nearest neighbourgs
# (2) Read an input particle distribution file and compute the SPH
#     density associated to each particle + the list of its nearest
#     neighbourgs
# (3) Read an input particle distribution and a neighbourgs file which
#     is an output of step 2), and output the tree of the structures in
#     structures.
#
# If steps (1) and (2) are probably correct, step (3) is still under
# test and construction. Step (1) was rather extensively tested, while
# step (2) needs further tests although I am quite confident it should
# be correct.
#
#=======================================================================
#                           COMPUTE_NEIKDTREE_MOD
#=======================================================================
# Modication of compute_neiKDtree to fit inside HaloMaker
# Modification : D. Tweed
#
# PARAMETERS IN THE CONFIG FILE 
# +++++++++++++++++++++++++++++
# 
# A example of config file is given in compute_neiKDtree.config
#
#
# H.verbose     : normal H.verbose mode (True or False)
# H.megaverbose : full H.verbose mode (True or False)
# filein      : name of the input particles/velocities file ('myfilename')
# Ntype       : format of the file (integer number)
#               0 : PM simple precision unformatted
#               1 : GADGET simple precision unformatted, cosmological simulation
#                   of dark matter only with periodic boundaries
#               2 : RAMSES unformatted (particles) simulation
#                   MPI multiple output
#               3 : CONE unformatted 
#               4 : Ninin treecode unformatted
# remove_degenerate : if this set to True, check for particles at the same 
#               position and add a random displacement at approximately 
#               floating representation accuracy in order to avoid infinite
#               number of KD tree cells creation. This parameter is global since
#               if a random displacement has been applied once for computing
#               SPH density, the same one must be applied again for subsequent 
#               calculations in order to be self-consistent.
#               Setting remove_degenerate=False will of course speep-up
#               the program at the cost of a risk of crash due to infinite
#               KD tree cell creation. This is however safe in most cases since
#               KD tree creation can now deal with particles at the same position.
#               It is therefore recommended to set remove_degenerate to False
#               first before trying True
# action      : 'distances' for computing mean square distance between each
#                           particle and its nearest neighbourgs
#               'neighbors' for computing the list of the nearest neighbourgs of
#                           each particle and its SPH density
#               'adaptahop' for computing the tree of structures and substructures
#                           in the simulation 
#               'posttreat' for computing the physical properties of dynamically
#                           selected haloes and subhaloes
#
# if (Ntype=1) : GADGET format
# --------------
#
# nmpigadget  : number of CPU used to perform the simulation (in order to read the 
#               appropriate number of files). If there is no multiple file 
#               extension (only one file), set nmpigadget=-1.
#
# if (Ntype=2) : RAMSES unformatted (dark matter particles)
# --------------
#
# mtot        : total mass in the full simulation box, of the population considered,
#               in internal RAMSES units.
#               mtot=-1 works if full simulation box is analysed.
# Ltot        : size of the full simulation box, in internal RAMSES units.
#               Ltot=-1 gives default value, valid for cosmological simulations
#               only.
# 
# if (Ntype=3) : CONE unformatted (dark matter particles) 
# ------------------
#
# boxsize     : comoving size of the simulation box in Mpc
#
# if (Ntype=4) : Ninin treecode unformatted 
# --------------
#
# boxsize2    : comoving size of the simulation box, in Mpc
# hubble      : value of H0/100 in km/s/Mpc
# omega0      : value of cosmological density parameter
# omegaL      : value of cosmological constant
# aexp_max    : value of expansion factor at present time
#
# if (action='distances') :
# -------------------------
#
# nvoisdis    : number of neighbors considered for each particle
# filedis     : name of the output file for the mean square distances 
#               ('mydistancename')
#               (this is a binary unformatted or formatted file which will be 
#               described more in detail in the near future) 
# formatted   : True if the output file is in ASCII, False if the output
#               file is in binary unformatted double float.
# ncpu        : number of virtual threads for a parallel calculation with openMP
#               This integer number should be a multiple of the real number of 
#               threads used during the run. The larger it will be, the better
#               the load balancing will be. 
# velocities  : True or False : if set to True, include velocities 
#               treatment in the calculation and the output of the results
#
# if (action='neighbors') :
# -------------------------
#
# nvoisnei    : number of neighbors considered for computing the SPH density
#               (integer number). Typically this number should vary between
#               20 and 100 (template value 64).
# H.nhop        : number of stored nearest neighbourgs in the output file
#              (integer number).
#              H.nhop must smaller smaller or equal to nvoisnei. Typically H.nhop
#              should be of order 10-30 (template value 16).
# fileneinei  : name of the output file for the SPH density and the list of
#              H.nhop nearest neighbors for each particle ('myfileneighbors')
#              (this is a binary unformatted file which will be described more 
#              in detail in the near future)
#
# if (action='adaptahop') :
# -------------------------
#
# H.rho_threshold : density threshold. Particles with SPH density below this
#              density threshold are not selected. This thresholding will
#              define a number of connected regions, each of which corresponding
#              to a structure. For each of these structures, we aim to 
#              build a tree of substructures. Typically, one is interested
#              in finding the substructures of highly nonlinear objects. A value
#              often taken for H.rho_threshold is 80, which corresponds roughly
#              to friend-of-friend algorithm parameter b=2.
# H.nmembthresh: threshold on the number of particles that a structure or a
#              a substructure (above some density threshold rhot) must contain 
#              for being considered as significant.
#              The choice of H.nmembthresh is related to effects of N-body relaxation
#              (treecode) or over-softening of the forces (AMR or PM codes) or
#              SPH smoothing (SPH smoothing over a number N of particles tends
#              to ``erase'' structures with less than N particles).
#              Basically, due to either of these effects, structures or 
#              substructures with a number of particles smaller than H.nmembthresh
#              are not considered. A typical choice of H.nmembthresh is 64.
# fudgepsilon: This parameter can be seen as an Eulerian version of the thresholding
#              controlled by H.nmembthresh. It defines the size of the smallest structures
#              that can exist physically and is related to force softening.
#              Basically, if epsilon is the softening parameter, substructures with
#              typical radius smaller than epsilon are just coincidences and should
#              not be considered. The typical radius of a structure is given by the
#              mean square distance of particles belonging to it with respect to its
#              center of gravity. 
#              The criterion for selecting a structure is thus
#              
#              radius_{substructure} > epsilon,
#             
#              where epsilon=fudgepsilon*L/Npart^{1/3}. Fudgepsilon is thus expressed
#              in units of mean interparticle separation.
#              - For a treecode, fudgepsilon is typically of the order of 1/20;
#              - For a PM code, fudgepsilon is typically of the order of N_g/Npart^{1/3}
#              where N_g is the resolution of the grid used for the force calculations,
#              i.e. of the order of 1 or 0.5.
#              - For a quasi-Lagrangian code such as RAMSES, putting constrains from fudgespilon 
#              is not really useful. Basically it corresponds to N_g/Npart^{1/3} where N_g
#              would be the equivalent of the PM grid probed by the smallest AMR cells.              
# alpha      : a criterion can be applied to decide weither a substructure
#              is selected or not. Here the choice of alpha dictates by how
#              much the maximum local density in this substructure should be larger
#              than the local average density in this substructure:
#
#              rho_max_{substructure} >= alpha*< rho >_{substructure}.
#
#              Basically, the choice of alpha dictates the ``peakyness'' of a 
#              substructure. For instance, a substructure might contain itself
#              5 local maxima. For this substructure to be dynamically significant
#              we want the largest of the local maxima to be at least alpha
#              times the mean local density, i.e. the average density of all the
#              particles within the substructure. Dynamically bounded substructures
#              are expected to be very peaky. A typical choice of alpha would be
#              alpha=0 or a few unites, e.g. alpha=4.
# H.fudge      : a criterion can be applied to decide wither a substructure is 
#              statistically significant in terms of Poisson noise. Indeed,
#              even if Poisson noise is considerably reduced by SPH smoothing
#              it is still present to some extent, and some substructures might
#              be simply due to local Poisson fluctuations. 
#              Here the choice of H.fudge dictates by how many ``sigma's'' 
#              a structure must be compared to a random Poisson fluctuation.
#              The criterion applied is
#
#              < rho >_{substructure} > rhot*[1+H.fudge/np.sqrt(N)]
#
#              where N is the number of particles in the substructure and rhot
#              the density threshold corresponding to this substructure (in 
#              other worlds, rhot is the minimum density of particles whithin
#              the substructure). 
#              This criterion can be understood as follows : if a given substructure
#              contains N particles, the corresponding random poisson fluctuations are
#              of the order of np.sqrt(N). So typically the uncertainty on the density
#              estimate of a structure containing N particles is of the order of
#              np.sqrt(N) (this of course neglects the fact that SPH softening reduces
#              considerably the effects of poisson fluctuations). For the
#              fluctuation associated to the substructure to be significant, we
#              impose that (< rho >_{substructure} - rhot)/rhot > H.fudge*sigma with
#              sigma=1/np.sqrt(N)=np.sqrt(< (M-<M>)^2 >)/<M> where M is a Poisson process
#              of average N.
#              A typical value is H.fudge=4.
#
#              IMPORTANT NOTE : we should always have H.fudge > 0. Small values of
#              H.fudge will slow down the program, while large values of H.fudge will
#              make the hierarchical decomposition in terms of substructures less
#              accurate. 
#              The reason for that is that despite the fact we know all the saddle points
#              connecting all the substructures, it was impossible to figure out a simple
#              way of sorting these saddle points in order to construct automatically
#              the tree of structures and substructures. As a result, we operate 
#              iteratively by increasing the local threshold density as follows:
#              [rhot NEXT] = [rhot OLD]*[1+H.fudge/np.sqrt(N)] where N is the number of particles
#              in the substructure. Then we see weither saddle points within this
#              substructure are below the new value of rhot: if this happens, it
#              means that at the density level [rhot NEXT], the substructure is composed
#              of disconnected subsubstructures and the threshold value of connection
#              between these subsubstructures is approximated by [rhot NEXT] (the real value
#              should be between [rhot NEXT] and [rhot OLD]).
# filenode   : output file for the tree of structures and substructures. See below
#              for explanation of its format
# simu_unitsnei : True or .false : to specifie if the nodes positions and radii in the
#              file filenode are specified in the same units as in the simulation input
#              file or in Mpc
# filepartnodenei : ouput file for particule H.node_0 number. To each particle, a
#              integer is associated, which is the H.node_0 number the deepest possible in the tree
#              given in filenode. This way, at any level in the tree, it will be possible to
#              find recursively all the particles belonging to the tree. If the H.node_0 number 
#              is zero, it means the SPH density of particle is below the threshold H.rho_threshold 
#              and is therefore associated to no structure. 
# formattedpartnodenei : True for having filepartnodenei in ascii format, False for
#              having filepartnodenei in binary format. The format is in both
#              case, the number of particles H.npart in the first line, followed by id(:), 
#              where id(:) is an array of H.npart integers.
# fileneihop : input file with SPH density and list of nearest neighbors of each
#              particle, obtained by running comptu_neiKDtree with action='neighbors'.
#
# if (action='posttreat') :
# -------------------------
# This option is still in development and test phase
#
#=======================================================================
#
# HISTORY
# +++++++
#
# 02/05/02/SC/IAP : first operational clean version with namelist and
#                   three options for the calculations.
#                   + action='distances' tested by comparison with the
#                   results of a different algorithm based on link list
#                   approach instead of KDtree (compute_neighbourgs2.f)
#                   The agreement between the 2 softwares is good within
#                   float calculation accuracy (5.10^{-6} relative 
#                   maximum differences). The differences can be 
#                   explained by a slightly different approach used
#                   in the 2 programs to do floating calculations.
#                   + action='neighbors' is not fully tested but should be
#                   okay. 
#                   + action='adaptahop' has been seen to work fine on
#                   a few data sets, but is not by any mean extensively
#                   tested. (No comments on the parameters of the namelist
#                   to avoid the users using this option)
#                   --> Version 0.0
# 04/05/02/SC/IAP : Full comments on the namelist parameters corresponding
#                   to action='adaptahop'. A few comments are added to
#                   the program. A new input particle file format is 
#                   added (RAMSES). 
#                   --> Version 0.1
# 04/17/02/SC&RT/IAP : Clean up the RAMSES format. Speed up the SPH 
#                   smoothing procedure
#                   --> Version 0.2
# 12/02/02/SC/IAP : Parallelize action='distances' and action='neighbors'
#                   Add RAMSES dark matter format with periodic boundaries
#                   (Ntype=-2), add CONE format (Ntype=3)
#                   --> Version 0.3
# 27/05/03/SC&RT/IAP&STR : Add RAMSES MPI dark matter + stars format with
#                   periodic boundaries (still under work, Ntype=-20) and
#                   a subroutine to remove degeneracy in particle positions
#                   Add GADGET multiple file MPI format.
# 04/06/03/SC/IAP : Dynamical selection of haloes and subhaloes :
#                   + new action='posttreat'
# 26/10/03/SC/IAP : add Ninin treecode format Ntype=4
# 16/10/06/SC/IAP : Change the treecode algorithm to be able to deal with
#                   2 or more particles at the same position. It is not
#                   yet completely safe in terms of the calculation of
#                   the SPH density (if all the particles are at the
#                   same position one gets 0/0).
#                   Add a few comments on file formats.
#                   --> Version 0.8
# 18/10/06/SC&RT/IAP : remove obsolete formats and improve RAMSES MPI
#                   format (now swapped from -20 to 2).
# 28/09/06/DT/CRAL: module compute_neiKDtree_mod.f90 compatibible with HaloMaker2.0
# 24/09/07/DT/CRAL: choice of 3 selection method using flag method
#                   HOP: Adaptahop is only used to detect haloes
#                   DPM: Subhaloes are selected through the density profile method, 
#                   accurate if studiing one step only
#                   MHM: Subhaloes are selected through the merger history method, 
#                   use this method if you need to obtain an accurate merger tree 
#                   containing subhaloes
#=======================================================================
import halo_defs as H
from halo_defs import mem,frange
import numpy as np

#   use halo_defs
#   integer::ncall         #YDdebug
ncall = 0
ntest = 0
inccell = 0
#   integer::icolor_select #YDdebug
icolor_select = 0
#   contains

#=======================================================================
def compute_adaptahop_131():
#=======================================================================
    change_pos_1310()
    # action neighbors
    create_tree_structure_1311()
    compute_mean_density_and_np_1312()
    # action adaptahop
    find_local_maxima_1313()
    create_group_tree_1314()
    change_pos_back_1315()
    # check that we have halos
    count_halos_1316()
    if(H.nb_of_halos>0):
        # reinit halo and subhalo count
        H.nb_of_halos    = 0
        H.nb_of_subhalos = 0
        # H.node_0 structure tree -> halo structure tree: 3 methods avalable
        if(H.method=="HOP"):
            select_halos_1317()
        elif(H.method=="DPM"):
            select_with_DP_method_1318()
        elif(H.method=="MSM"):
            select_with_MS_method_1319()
        elif(H.method=="BHM"):
            select_with_BH_method_131a()
        else:
            print('> could not recognise selection method:',H.method)
    else:
        H.node_0 = []

#=======================================================================
def list_parameters_1300():
#=======================================================================


    print('============================================================')
    print('Compute_neiKDtree was called with the following attributes :')
    print(    'verbose         :',H.verbose)
    print(    'megaverbose     :',H.megaverbose)
    print(    'boxsize2        :',H.boxsize2)
    print(    'hubble          :',H.hubble)
    print(    'omega0          :',H.omega0)
    print(    'omegaL          :',H.omegaL)
    print(    'aexp_max        :',H.aexp_max)
    print(    'action          : neighbors')
    print(    'nvoisins        :',H.nvoisins)
    print(    'nhop            :',H.nhop)
    print(    'action          : adaptahop')
    print(    'rho_threshold   :',H.rho_threshold)
    print(    'nmembthresh     :',H.nmembthresh)
    print(    'alphap          :',H.alphap)
    print(    'fudge           :',H.fudge)
    print(    'fudgepsilon     :',H.fudgepsilon)
    print(    'Selection method:',' ',H.method)
    print('============================================================')


#=======================================================================
def init_adaptahop_130():
#=======================================================================
    H.omegaL   = H.omega_lambda_f
    H.omega0   = H.omega_f
    H.aexp_max = H.af
    H.hubble   = H.H_f*1e-2
    H.boxsize2 = H.Lf
    H.xlong    = H.boxsize2
    H.ylong    = H.xlong
    H.zlong    = H.xlong
    H.xlongs2  = H.xlong*0.5
    H.ylongs2  = H.ylong*0.5
    H.zlongs2  = H.zlong*0.5

    pos_renorm  =H.xlong
    H.pos_shift[0]=0.0
    H.pos_shift[1]=0.0
    H.pos_shift[2]=0.0
    H.mass_in_kg=(H.xlong*H.ylong*H.zlong/H.npart)*H.mega_parsec**3 \
        *H.omega0*H.critical_density*H.hubble**2 
    H.Hub_pt = 100.*H.hubble * np.sqrt(H.omega0*(H.aexp_max/H.aexp)**3 + (1-H.omegaL-H.omega0)*(H.aexp_max/H.aexp)**2 + H.omegaL) 
    
    H.nmembthresh = H.nMembers  
    
    list_parameters_1300()

#=======================================================================
def change_pos_1310():
#=======================================================================
    H.npart    = H.nbodies
    H.epsilon  = H.fudgepsilon*H.xlong/H.npart**(1/3)
    mem['pos_10']      = mem['pos_10'] * H.boxsize2

#=======================================================================
def change_pos_back_1315():
#=======================================================================
    mem['pos_10'] = mem['pos_10'] / H.boxsize2

#=======================================================================
def count_halos_1316():
#=======================================================================
    # integer(kind=4) :: inode
    H.nb_of_halos = 0
    for inode1 in frange(1,H.nnodes):
        if(H.node_0[inode1].level==1): H.nb_of_halos += 1

#=======================================================================
def select_halos_1317():
#=======================================================================
    #   integer(kind=4) :: inode, ihalo, ipar
    #   integer(kind=4) :: node_to_halo(H.nnodes)
    node_to_halo = np.zeros(H.nnodes, dtype=np.int32)-1

    # counting number of halos
    H.nb_of_halos = 0
    for inode1 in frange(1,H.nnodes):
        if(H.node_0[inode1].mother<=0): H.nb_of_halos += 1
        ih1 = inode1
        while(H.node_0[ih1].mother !=0):
            ih1 = H.node_0[ih1].mother if(node_to_halo[ih1-1]<=0) else node_to_halo[ih1-1]
        node_to_halo[inode1-1] = ih1 
    print('number of nodes :',H.nnodes)
    print('number of haloes:', H.nb_of_halos)

    for ipar0 in range(H.npart):
        inode1 = mem['liste_parts'][ipar0]
        if(inode1>0):
            ih1 = node_to_halo[inode1-1]
            if((ih1<=H.nb_of_halos)and(ih1>0)):
                mem['liste_parts'][ipar0] = ih1
            else:
                raise ValueError('ihalo > nb_of_halos or nil')

    H.node_0 = []

#=======================================================================
def select_with_DP_method_1318():
#=======================================================================
    raise NotImplementedError('`select_with_DP_method` not implemented')

#=======================================================================
def select_with_MS_method_1319(mass_acc = 1.0e-2):
#=======================================================================
    # integer(kind=4) :: ihalo,inode,isub,ip
    # integer(kind=4) :: imother,istruct 
    # integer(kind=4) :: nb_halos,nb_sub
    # integer(kind=4) :: mostmasssub(H.nnodes),node_to_struct(H.nnodes)
    # real(kind=8)    :: maxmasssub,mass_acc
    # integer(kind=4), allocatable :: npartcheck_0(:)

    
    mostmasssub = np.zeros(H.nnodes, dtype=np.int32)
    node_to_struct = np.zeros(H.nnodes, dtype=np.int32)
    if(H.verbose): print('Using MS method')

    # First recompute H.node_0 mass accordingly to their particle list
    for inode1 in frange(1,H.nnodes):
        isub1 = H.node_0[inode1].firstchild
        while(isub1>0):
            H.node_0[inode1].mass -= H.node_0[isub1].mass
            H.node_0[inode1].truemass -= H.node_0[isub1].truemass
            isub1 = H.node_0[isub1].sister
        if(H.node_0[inode1].mass<=0 or H.node_0[inode1].truemass<=0.0):
            print(' Error in computing H.node_0',inode1,'mass_10')
            raise ValueError(' mass, truemass:', H.node_0[inode1].mass, H.node_0[inode1].truemass)
    if(H.verbose):
        # check that the new masses are correct
        npartcheck_0 = np.zeros(H.nnodes+1, dtype=np.int32); npartcheck_0[:] = 0
        for ip1 in frange(1,H.nbodies):
            if(mem['liste_parts'][ip1-1]<0): raise ValueError('liste_parts is smaller than 0')
            npartcheck_0[mem['liste_parts'][ip1-1]] += 1
        if(sum(npartcheck_0)!=H.nbodies): raise ValueError('Error in particles count')
        for inode1 in frange(1,H.nnodes):
            if(H.node_0[inode1].mass!=npartcheck_0[inode1]):
                print('Error in H.node_0 particle count, for H.node_0',inode1)
                print('it first subnode is:',H.node_0[inode1].firstchild)
                if(H.node_0[inode1].firstchild>0): print('it has:',H.node_0[H.node_0[inode1].firstchild].nsisters,'subnodes' )
                raise ValueError("")
        del npartcheck_0
    
    mostmasssub[:]    = -1
    for inode1 in frange(H.nnode1, 1, -1):
        # Search inode1 subnodes for the most massive one
        isub1 = H.node_0[inode1].firstchild
        if(isub1>0):
            # init all on the first subnode
            maxmasssub         = H.node_0[isub1].truemass
            mostmasssub[inode1-1] = isub1
            isub1               = H.node_0[isub1].sister
        while(isub1>0):
            if(H.node_0[isub1].truemass>maxmasssub):
                maxmasssub         = H.node_0[isub1].truemass
                mostmasssub[inode1-1] = isub1
            isub1               = H.node_0[isub1].sister
        # add mostmasssub mass to inode1 mass and inode1's densmax = mostmasssub's densmax
        if (mostmasssub[inode1-1] > 0):
            H.node_0[inode1].mass     += H.node_0[mostmasssub[inode1-1]].mass
            H.node_0[inode1].truemass += H.node_0[mostmasssub[inode1-1]].truemass
            H.node_0[inode1].densmax  = H.node_0[mostmasssub[inode1-1]].densmax

    # Second run to write node_to_struct array and count the number of substructures
    nb_sub         = 0
    nb_halos       = 0
    istruct        = 0
    node_to_struct[:] = -1
    
    for inode1 in frange(1,H.nnodes):
        if(node_to_struct[inode1-1]>0): raise ValueError('node_to_struct is greater than 0')
        if(H.node_0[inode1].mother<=0):
            nb_halos              += 1
            istruct               += 1
            node_to_struct[inode1-1] = istruct
        else:
            imother1  = H.node_0[inode1].mother
            if(inode1==mostmasssub[imother1-1]):
                if(node_to_struct[imother1-1]<=0): raise ValueError('node_to_struct not defined for imother1')
                node_to_struct[inode1-1] = node_to_struct[imother1-1]
            else:
                nb_sub                += 1
                istruct               += 1
                node_to_struct[inode1-1] = istruct
    
    # check halos and subhalos number 
    H.nstruct = nb_halos + nb_sub
    if(H.nstruct!=istruct):
        raise ValueError('Error in subroutines count,',H.nstruct, istruct)

    if(H.verbose):
        # Check that the structure tree is ordered as it should be 
        for inode1 in range(1,H.nnodes):
            if(inode1<=nb_halos):
                if( not( (H.node_0[inode1].level==1)and(node_to_struct[inode1-1]==inode1) )): raise ValueError('error in H.node_0 order' )
            else:
                if(H.node_0[inode1].level==1): raise ValueError('error shouldn''t be any halo here' )
        print("")
        print('> number of nodes               :', H.nnodes)
        print('> number of halos               :', nb_halos)
        print('> number of substructures       :', nb_sub)
        print('> number of H.node_0 removed        :', H.nnodes - H.nstruct)
        print("")
        print('> Cleaning liste_parts')
        npartcheck_0 = np.zeros(H.nstruct, dtype=np.int32); npartcheck_0[:] = 0

    # Cleaning liste_parts
    for ip0 in range(H.nbodies):
        inode1  = mem['liste_parts'][ip0]
        if(inode1>0):
            if(node_to_struct[inode1-1]<=0):
                print(ip0, inode1,  node_to_struct[inode1])
                raise ValueError('error in node_to_struct')
            mem['liste_parts'][ip0] = node_to_struct[inode1-1]
            if(H.verbose): npartcheck_0[node_to_struct[inode1-1]] += 1
    if(H.verbose):
        # Check H.node_0[inode].mass it should now correspond to npartcheck_0 count
        for inode0 in range(H.nnodes):
            if(node_to_struct[inode0]<=0): raise ValueError('node_to_struct is nil')
            imother1 = H.node_0[inode0+1].mother
            if((imother1<=0) or (imother1>0 and (node_to_struct[imother1-1]!=node_to_struct[inode0]))):
                if(H.node_0[inode0+1].mass!=npartcheck_0[node_to_struct[inode0]]):
                    print('Wrong nb of part in struct: ', node_to_struct[inode0])
                    print('inode0,H.node_0[inode0+1].mass,istruct,npartcheck_0(istruct)',inode0, \
                        H.node_0[inode0+1].mass,node_to_struct[inode0],npartcheck_0[node_to_struct[inode0]])
                    raise ValueError("")
        del npartcheck_0

    if(H.verbose): print('> Creating new structure tree')
    # creating new structure tree
    H.allocate('mother_1319', H.nstruct, dtype=np.int32)
    H.allocate('first_sister_1319', H.nstruct, dtype=np.int32)
    H.allocate('first_daughter_1319', H.nstruct, dtype=np.int32)
    H.allocate('level_1319', H.nstruct, dtype=np.int32)
    mem['mother_1319'][:]         = -1
    mem['first_sister_1319'][:]   = -1
    mem['first_daughter_1319'][:] = -1
    mem['level_1319'][:]          =  0
    ihalo1          = -1 

    for inode0 in range(H.nnodes):
        istruct1 = node_to_struct[inode0]
        if(istruct1<=0): raise ValueError('index nil for istrut')
        if(mem['mother_1319'][istruct1-1]<0):
            if(H.node_0[inode0+1].mother<=0):
                mem['mother_1319'][istruct1-1] = 0
                if(ihalo1>0): mem['first_sister_1319'][ihalo1-1] = istruct1
                ihalo1 = istruct1
                mem['level_1319'][istruct1-1] = 1
            else:
                imother1          = node_to_struct[H.node_0[inode0+1].mother]
                mem['mother_1319'][istruct1-1]  = imother1
                mem['level_1319'][istruct1-1]   = mem['level_1319'][imother1-1] + 1
                if(mem['first_daughter_1319'][imother1]<=0):
                    mem['first_daughter_1319'][imother1-1] = istruct1
                else:
                    isub1 = mem['first_daughter_1319'][imother1-1]
                    while(mem['first_sister_1319'][isub1-1]>0):
                        isub1 = mem['first_sister_1319'][isub1-1]
                    mem['first_sister_1319'][isub1-1] = istruct1

    if(H.megaverbose):
        # For test we shall output the structure tree in file struct_tree.dat'
        f120 = open("struct_tree.dat", mode='w+')
        for istruct0 in range(H.nstruct):
            if(mem['mother_1319'][istruct0]<=0):
                f120.write('---------\n')
                f120.write(f"halo:{istruct0:6d} first_child:{mem['first_daughter_1319'][istruct0]:6d} sister:{mem['first_sister_1319'][istruct0]:6d}\n")
                isub1 = mem['first_daughter_1319'][istruct0]
                while(isub1>0):
                    f120.write(f"sub:{isub1:6d} mother:{mem['mother_1319'][isub1-1]:6d} first_child:{mem['first_daughter_1319'][isub1-1]:6d} sister:{mem['first_sister_1319'][isub1-1]:6d}\n")
                    isub1 = mem['first_sister_1319'][isub1-1]
            else:
                isub1 = mem['first_daughter_1319'][istruct0]
                if(isub1!=0):
                    f120.write(f"sub:{istruct0:6d} mother:{mem['mother_1319'][istruct0]:6d} first_child:{mem['first_daughter_1319'][istruct0]:6d} sister:{mem['first_sister_1319'][istruct0]:6d}\n")
                    while(isub1>0):
                        # write(120,'(a5,1x,i6,2x,a7,1x,i6,2x,a12,1x,i6,2x,a7,1x,i6)')   &
                        #     ' sub:',isub1, 'mother:', mother(isub1), 'first_child:',   &
                        #     first_daughter(isub1), 'sister:', mem['first_sister_1319'][isub1-1]
                        f120.write(f"sub:{isub1:6d} mother:{mem['mother_1319'][isub1-1]:6d} first_child:{mem['first_daughter_1319'][isub1-1]:6d} sister:{mem['first_sister_1319'][isub1-1]:6d}\n")
                        isub1 = mem['first_sister_1319'][isub1-1]
        f120.close()

    H.nb_of_halos    = nb_halos
    H.nb_of_subhalos = nb_sub
    H.node_0 = []

#=======================================================================
def select_with_BH_method_131a():
#=======================================================================
    raise NotImplementedError('`select_with_BH_method` not implemented')

#=======================================================================
def main_prog(inode,mainprog):
#=======================================================================
    raise NotImplementedError("`main_prog` is used in BHM which is not implemented")

#=======================================================================
def main_prog_node_sub(inode,isub,mainprog):
#=======================================================================
    raise NotImplementedError("`main_prog_node_sub` is used in BHM which is not implemented")

#=======================================================================
def make_linked_node_list():
#=======================================================================
    raise NotImplementedError("`make_linked_node_list` is used in BHM which is not implemented")

#=======================================================================
def compute_mean_density_and_np_1312():
#=======================================================================
    # integer(kind=4)                     :: ipar
    # real(kind=8), dimension(0:H.nvoisins) :: dist2_0
    # integer, dimension(H.nvoisins)        :: iparnei
    # real(kind=8)                        :: densav
    dist2_0 = np.empty(H.nvoisins+1, dtype=np.float64)
    iparnei = np.empty(H.nvoisins, dtype=np.int32)

    if (H.verbose): print('Compute mean density for each particle...')

    H.allocate('iparneigh_1312',(H.nhop,H.npart), dtype=np.int32)
    H.allocate('density_1312',H.npart, dtype=np.float64)

    if (H.verbose): print('First find nearest particles')

    # !$OMP PARALLEL DO &
    # !$OMP DEFAULT(SHARED) &
    # !$OMP PRIVATE(ipar,dist2_0,iparnei)
    for ipar1 in frange(1,H.npart):
        dist2_0, iparnei = find_nearest_parts_13120(ipar1,dist2_0,iparnei)
        compute_density_13121(ipar1,dist2_0,iparnei)
        mem['iparneigh_1312'][:H.nhop,ipar1-1]=iparnei[:H.nhop]
    # !$OMP END PARALLEL DO

    # Check for average density
    if (H.verbose):
        densav=0
        for ipar1 in frange(1,H.npart):
            densav=(densav*(ipar1-1)+mem['density_1312'][ipar1-1])/ipar1
        print('Average density :',densav)
    
    H.deallocate('mass_cell_1311')
    H.deallocate('size_cell_1311')
    H.deallocate('pos_cell_1311')
    H.deallocate('sister_1311')
    H.deallocate('firstchild_1311')

#=======================================================================
def find_local_maxima_1313():
#=======================================================================
#   integer(kind=4)             :: ipar,idist,iparid,iparsel,igroup,nmembmax,nmembtot
#   integer(kind=4),allocatable :: nmemb(:)
#   real(kind=8)                :: denstest

    if (H.verbose): print('Now Find local maxima...')

    H.allocate('igrouppart_1313', H.npart)

    mem['idpart_1311'][:]=0
    H.ngroups=0
    for ipar1 in frange(1,H.npart):
        denstest=mem['density_1312'][ipar1-1]
        if (denstest>H.rho_threshold):
            iparsel1=ipar1
        for idist0 in range(H.nhop):
            iparid1=mem['iparneigh_1312'][idist0,ipar1-1]
            if (mem['density_1312'][iparid1-1]>denstest):
                iparsel1=iparid1
                denstest=mem['density_1312'][iparid1-1]
            elif (mem['density_1312'][iparid1-1]==denstest):
                iparsel1=min(iparsel1,iparid1)
                if (H.verbose): print('WARNING : equal densities in find_local_maxima.')
        if (iparsel1==ipar1):
            H.ngroups += 1
            mem['idpart_1311'][ipar1-1]=-H.ngroups
        else:
            mem['idpart_1311'][ipar1-1]=iparsel1
    
    if (H.verbose): print('Number of local maxima found :',H.ngroups)

    # Now Link the particles associated to the same maximum
    if (H.verbose): print('Create link list...')

    H.allocate('densityg_1313',H.ngroups, dtype=np.float64)
    nmemb = np.zeros(H.ngroups, dtype=np.int32)
    H.allocate('firstpart_1313',H.ngroups, dtype=np.int32)

    for ipar0 in range(H.npart):
        if (mem['density_1312'][ipar0]>H.rho_threshold):
            iparid1=mem['idpart_1311'][ipar0]
            if (iparid1<0): mem['densityg_1313'][-iparid1-1]=mem['density_1312'][ipar0]
            while (iparid1>0):
                iparid1=mem['idpart_1311'][iparid1-1]
            mem['igrouppart_1313'][ipar0]=-iparid1
        else:
            mem['igrouppart_1313'][ipar0]=0

    nmemb[:H.ngroups]=0
    mem['firstpart_1313'][:H.ngroups]=0
    for ipar0 in range(H.npart):
        igroup1=mem['igrouppart_1313'][ipar0]
        if (igroup1>0):
            mem['idpart_1311'][ipar0]=mem['firstpart_1313'][igroup1-1]
            mem['firstpart_1313'][igroup1-1]=ipar0+1
            nmemb[igroup1-1] += 1

    nmembmax=0
    nmembtot=0
    for igroup0 in range(H.ngroups):
        nmembmax=max(nmembmax,nmemb[igroup0])
        nmembtot += nmemb[igroup0]

    if (H.verbose):
        print('Number of particles of the largest group :',nmembmax)
        print('Total number of particles in groups ',nmembtot)
    del nmemb

#=======================================================================
def create_group_tree_1314():
#=======================================================================
    #   integer(kind=4) :: inode,mass_loc,masstmp,igroup,igrA,igrB,igroupref
    #   real(kind=8)    :: rhot,posg[2],posref(3),rsquare,densmoy,truemass,truemasstmp
    posg = np.empty(3, dtype=np.float64)
    posref = np.empty(3, dtype=np.float64)

    if (H.verbose): print('Create the tree of structures of structures')

    # End of the branches of the tree
    compute_saddle_list_13140()

    if (H.verbose): print('Build the hierarchical tree')

    H.nnodesmax=2*H.ngroups
    # Allocations
    H.node_0 = [H.supernode() for _ in range(H.nnodesmax)]
    H.allocate('idgroup_1314',H.ngroups, dtype=np.int32)
    H.allocate('color_1314',H.ngroups, dtype=np.int32)
    H.allocate('igroupid_1314',H.ngroups, dtype=np.int32)
    H.allocate('idgroup_tmp_1314',H.ngroups, dtype=np.int32)

    # Initializations
    mem['liste_parts'][:] = 0

    # Iterative loop to build the rest of the tree
    inode1               = 0
    H.nnodes              = 0
    rhot                = H.rho_threshold
    H.node_0[inode1].mother  = 0
    mass_loc            = 0
    truemass            = 0
    igroupref           = 0
    masstmp             = 0
    rsquare             = 0
    densmoy             = 0
    truemasstmp         = 0
    for igroup1 in frange(1,H.ngroups):
        treat_particles_13141(igroup1,rhot,posg,masstmp,igroupref,posref, rsquare,densmoy,truemasstmp)
        mass_loc += masstmp
        truemass += truemasstmp

    H.node_0[inode1].mass          = mass_loc
    H.node_0[inode1].truemass      = truemass
    H.node_0[inode1].radius        = 0
    H.node_0[inode1].density       = 0
    H.node_0[inode1].position[:3] = 0
    H.node_0[inode1].densmax       = np.max(mem['densityg_1313'])
    H.node_0[inode1].rho_saddle    = 0.
    H.node_0[inode1].level         = 0
    H.node_0[inode1].nsisters      = 0
    H.node_0[inode1].sister        = 0
    igrA = 1
    igrB = H.ngroups
    for igroup0 in range(H.ngroups):
        mem['idgroup_1314'][igroup0]=igroup0+1
        mem['igroupid_1314'][igroup0]=igroup0+1
    create_nodes_13143(rhot,inode1,igrA,igrB)

    H.deallocate('idgroup_1314')
    H.deallocate('color_1314')
    H.deallocate('igroupid_1314')
    H.deallocate('idgroup_tmp_1314')
    H.deallocate('idpart_1311')
    H.deallocate('group')
    H.deallocate('densityg_1313')
    H.deallocate('firstpart_1313')
 
#=======================================================================
def create_nodes_13143(rhot,inode1,igrA,igrB):
#=======================================================================
    global ncall, icolor_select
    # integer(kind=4)              :: inode1,igrA,igrB
    # real(kind=8)                 :: rhot
    # integer(kind=4)              :: igroup,icolor,igr,igr_eff
    # integer(kind=4)              :: inc_color_tot,inode1,mass_loc,masstmp,igroupref
    # integer(kind=4)              :: inodeout,igrAout,igrBout,isisters,nsisters
    # integer(kind=4)              :: mass_comp,icolor_ref
    # real(kind=8)                 :: posg[2],posgtmp[2],posref(3),rhotout,rsquaretmp,rsquareg
    posg = np.empty(3, dtype=np.float64)
    posgtmp = np.empty(3, dtype=np.float64)
    posref = np.empty(3, dtype=np.float64)
    # real(kind=8)                 :: densmoyg,densmoytmp
    # real(kind=8)                 :: densmoy_comp_max,truemass,truemasstmp
    # real(kind=8)                 :: posfin[2]
    posfin = np.empty(3, dtype=np.float64)
    # real(kind=8)                 :: densmaxgroup
    # integer(kind=4), allocatable :: igrpos_0(:),igrinc(:)
    # integer(kind=4), allocatable :: igrposnew_0(:)
    # integer(kind=4), allocatable :: massg(:)
    # real(kind=8),  allocatable   :: truemassg(:)
    # real(kind=8),  allocatable   :: densmaxg(:)
    # real(kind=8),  allocatable   :: densmoy_comp_maxg(:)
    # integer(kind=4), allocatable :: mass_compg(:)
    # real(kind=8),  allocatable   :: posgg(:,:)
    # real(kind=8),  allocatable   :: rsquare(:)
    # real(kind=8),  allocatable   :: densmoy(:)
    # logical, allocatable         :: ifok(:)  

    mem['color_1314'][igrA-1:igrB] = 0
    # Percolate the groups
    icolor_select=0
    for igr1 in frange(igrA, igrB):
        igroup1=mem['idgroup_1314'][igr1-1]
        if (mem['color_1314'][igr1-1]==0):
            icolor_select += 1
            ncall=0                                               #YDdebug
            do_colorize_131430(igroup1,igr1,rhot) #YDdebug
            #write(errunit,'(A,3I8,e10.2,I8)')'End of do_colorize',icolor_select,igroup,igr,rhot,ncall

    # We select only groups where we are sure of having at least one
    # particle above the threshold density rhot
    # Then sort them to gather them on the list
    igrpos_0 = np.zeros(icolor_select+1, dtype=np.int32)
    igrinc = np.zeros(icolor_select, dtype=np.int32)
    igrpos_0[0]=igrA-1
    igrpos_0[1:icolor_select+1]=0
    for igr1 in frange(igrA, igrB):
        icolor=mem['color_1314'][igr1-1]
        igroup=mem['idgroup_1314'][igr1-1]
        if(mem['densityg_1313'][igroup]>rhot): igrpos_0[icolor] += 1
    for icolor1 in frange(1,icolor_select):
        igrpos_0[icolor1] += igrpos_0[icolor1-1]
    if (igrpos_0[icolor_select]-igrA+1 == 0):
        print('ERROR in create_nodes :')
        raise ValueError('All subgroups are below the threshold.')

    igrinc[:icolor_select]=0
    for igr1 in frange(igrA, igrB):
        icolor1=mem['color_1314'][igr1-1]
        igroup1=mem['idgroup_1314'][igr1-1]
        if (mem['densityg_1313'][igroup1-1]>rhot):
            igrinc[icolor1-1] += 1
            igr_eff1=igrinc[icolor1-1]+igrpos_0[icolor1-1]
            mem['idgroup_tmp_1314'][igr_eff1-1]=igroup1
            mem['igroupid_1314'][igroup1-1]=igr_eff1
    igrB=igrpos_0[icolor_select]
    mem['idgroup_1314'][igrA-1:igrB]=mem['idgroup_tmp_1314'][igrA-1:igrB]

    inc_color_tot = np.sum(igrinc[:icolor_select]>0)

    H.allocate('igrposnew_0_13143',1+inc_color_tot, dtype=np.int32)
    mem['igrposnew_0_13143'][0] = igrpos_0[0]
    inc_color_tot=0
    for icolor1 in frange(1,icolor_select):
        if (igrinc[icolor1-1]>0):
            inc_color_tot += 1
            mem['igrposnew_0_13143'][inc_color_tot]=igrpos_0[icolor1]

    del igrpos_0
    del igrinc

    isisters=0
    H.allocate('posgg_13143', (3,inc_color_tot), dtype=np.float64)
    H.allocate('massg_13143', inc_color_tot, dtype=np.int32)
    H.allocate('truemassg_13143', inc_color_tot, dtype=np.float64)
    H.allocate('densmaxg_13143', inc_color_tot, dtype=np.float64)
    H.allocate('densmoy_comp_maxg_13143', inc_color_tot, dtype=np.float64)
    H.allocate('mass_compg_13143', inc_color_tot, dtype=np.int32)
    H.allocate('rsquare_13143', inc_color_tot, dtype=np.float64)
    H.allocate('densmoy_13143', inc_color_tot, dtype=np.float64)
    H.allocate('ifok_13143', inc_color_tot, dtype=np.bool_)
    mem['ifok_13143'][:inc_color_tot]=False

    for icolor0 in range(inc_color_tot):
        posg[:3]=0
        mass_loc=0
        truemass=0
        rsquareg=0
        densmoyg=0
        igrA=mem['igrposnew_0_13143'][icolor0]+1
        igrB=mem['igrposnew_0_13143'][icolor0+1]
        densmaxgroup=-1.
        mass_comp=0
        densmoy_comp_max=-1.
        igroupref=0
        masstmp = 0
        rsquaretmp = 0
        truemasstmp = 0
        for igr1 in frange(igrA, igrB):
            igroup1=mem['idgroup_1314'][igr1-1]
            densmaxgroup=max(densmaxgroup,mem['densityg_1313'][igroup1-1])
            igroupref = treat_particles_13141(
                igroup1,rhot,igroupref,
                posref,masstmp,
                truemasstmp,rsquaretmp,densmoytmp,posgtmp)
            posg[0] += posgtmp[0]
            posg[1] += posgtmp[1]
            posg[2] += posgtmp[2]
            rsquareg += rsquaretmp
            mass_loc += masstmp
            truemass += truemasstmp
            densmoyg += densmoytmp
            densmoytmp /= masstmp
            mass_comp=max(mass_comp,masstmp)
            if (masstmp > 0): densmoy_comp_max=max(densmoy_comp_max, densmoytmp/(1+H.fudge/np.sqrt(masstmp)))

        mem['massg_13143'][icolor0]=mass_loc
        mem['truemassg_13143'][icolor0]=truemass
        mem['posgg_13143'][:3,icolor0]=posg[:3]
        mem['densmaxg_13143'][icolor0]=densmaxgroup
        mem['densmoy_comp_maxg_13143'][icolor0]=densmoy_comp_max
        mem['mass_compg_13143'][icolor0]=mass_comp
        mem['rsquare_13143'][icolor0]=np.sqrt(abs((truemass*rsquareg-(posg[0]**2+posg[1]**2+posg[2]**2) )/ truemass**2 ))
        mem['densmoy_13143'][icolor0]=densmoyg/mass_loc

        mem['ifok_13143'][icolor0]=(mass_loc>=H.nmembthresh) and \
            (mem['densmoy_13143'][icolor0] > rhot*(1+H.fudge/np.sqrt(mass_loc)) or  mem['densmoy_comp_maxg_13143'][icolor0]>rhot) and \
            (mem['densmaxg_13143'][icolor0] >= H.alphap*mem['densmoy_13143'][icolor0]) and \
            (mem['rsquare_13143'][icolor0] >= H.epsilon)

        if (mem['ifok_13143'][icolor0]):
            isisters += 1
            icolor_ref=icolor0

    nsisters=isisters
    if (nsisters>1):
        isisters=0
        inodetmp=H.nnodes+1
        for icolor0 in range(inc_color_tot):
            if (mem['ifok_13143'][icolor0]):
                isisters += 1
                H.nnodes=H.nnodes+1
                if (H.nnodes>H.nnodesmax):
                    print('ERROR in create_nodes :')
                    raise ValueError(f'H.nnodes({H.nnodes}) > H.nnodes max({H.nnodesmax})')
                if((H.nnodes%max(H.nnodesmax/10000,1))==0 and H.megaverbose): print('H.nnodes=',H.nnodes)
                H.node_0[H.nnodes].mother=inode1
                H.node_0[H.nnodes].densmax=mem['densmaxg_13143'][icolor0]
                if (isisters>1): H.node_0[H.nnodes].sister=H.nnodes-1
                else: H.node_0[H.nnodes].sister=0

                H.node_0[H.nnodes].nsisters=nsisters
                H.node_0[H.nnodes].mass=mem['massg_13143'][icolor0]
                H.node_0[H.nnodes].truemass=mem['truemassg_13143'][icolor0]
                if (mass_loc==0):
                    print('ERROR in create_nodes :')
                    raise ValueError('NULL mass for H.nnodes=',H.nnodes)
                posfin[:3] = mem['posgg_13143'][:3,icolor0]/mem['truemassg_13143'][icolor0]
                H.node_0[H.nnodes].radius=mem['rsquare_13143'][icolor0]
                H.node_0[H.nnodes].density=mem['densmoy_13143'][icolor0]
                posfin[0]=posfin[0]-H.xlong if(posfin[0]>=H.xlongs2) else posfin[0]+H.xlong
                posfin[1]=posfin[1]-H.ylong if(posfin[1]>=H.ylongs2) else posfin[1]+H.ylong
                posfin[2]=posfin[2]-H.zlong if(posfin[2]>=H.zlongs2) else posfin[2]+H.zlong
                H.node_0[H.nnodes].position[:3]=posfin[:3]
                H.node_0[H.nnodes].rho_saddle=rhot
                H.node_0[H.nnodes].level=H.node_0[inode1].level+1
                if (H.megaverbose and (H.node_0[H.nnodes].mass>=H.nmembthresh)):
                    print('*****************************************')
                    print('new H.node_0 :',H.nnodes)
                    print('level    :',H.node_0[H.nnodes].level)
                    print('nsisters :',H.node_0[H.nnodes].nsisters)
                    print('mass     :',H.node_0[H.nnodes].mass)
                    print('true mass:',H.node_0[H.nnodes].truemass)
                    print('radius   :',H.node_0[H.nnodes].radius)
                    print('position :',H.node_0[H.nnodes].position)
                    print('rho_saddl:',H.node_0[H.nnodes].rho_saddle)
                    print('rhomax   :',H.node_0[H.nnodes].densmax)
                    print('*****************************************')

        H.node_0[inode1].firstchild=H.nnodes
        inodeout1=inodetmp
        for icolor0 in range(inc_color_tot):
            if (mem['ifok_13143'][icolor0]):
                igrAout=mem['igrposnew_0_13143'][icolor0]+1
                igrBout=mem['igrposnew_0_13143'][icolor0+1]
                for igr1 in frange(igrAout, igrBout):
                    paint_particles_131431(mem['idgroup_1314'][igr1-1],inodeout1,rhot)
                rhotout=rhot*(1+H.fudge/np.sqrt(mem['mass_compg_13143'][icolor0]))
                if (igrBout!=igrAout):
                    create_nodes_13143(rhotout,inodeout1,igrAout,igrBout)
                else:
                    H.node_0[inodeout1].firstchild=0
                inodeout1 += 1
    elif (nsisters==1):
        inodeout1=inode1
        rhotout=rhot*(1+H.fudge/np.sqrt(mem['mass_compg_13143'][icolor_ref]))
        igrAout=mem['igrposnew_0_13143'][0]+1
        igrBout=mem['igrposnew_0_13143'][inc_color_tot]
        if (igrBout!=igrAout): create_nodes_13143(rhotout,inodeout1,igrAout,igrBout)
        else: H.node_0[inode1].firstchild=0
    else:
        H.node_0[inode1].firstchild=0

    H.deallocate('igrposnew_0_13143')
    H.deallocate('posgg_13143')
    H.deallocate('massg_13143')
    H.deallocate('truemassg_13143')
    H.deallocate('densmaxg_13143')
    H.deallocate('densmoy_comp_maxg_13143')
    H.deallocate('densmoy_13143')
    H.deallocate('mass_compg_13143')
    H.deallocate('rsquare_13143')
    H.deallocate('ifok_13143')

#=======================================================================
def paint_particles_131431(igroup1,inode1,rhot):
#=======================================================================
    # integer(kind=4) :: igroup1,inode1
    # real(kind=8)    :: rhot
    # integer(kind=4) :: ipar

    ipar1=mem['firstpart_1313'][igroup1-1]
    while(ipar1>0):
        if(mem['density_1312'][ipar1-1]>rhot):
            mem['liste_parts'][ipar1-1]=inode1
        ipar1=mem['idpart_1311'][ipar1-1]

#=======================================================================
def treat_particles_13141(igroup1,rhot,igroupref,posref,imass,truemass,rsquare,densmoy,posg):
#=======================================================================
    # real(kind=8)    :: rhot
    # real(kind=8)    :: posg[2],posref(3)
    # real(kind=8)    :: posdiffx,posdiffy,posdiffz,rsquare,densmoy,truemass,xmasspart
    # real(kind=8)    :: densmax,densmin
    # integer(kind=4) :: imass,ipar,iparold,igroup1,igroupref
    # logical         :: first_good

    imass=0
    truemass=0
    rsquare=0
    densmoy=0
    posg = np.zeros(3, dtype=np.float64)
    ipar1=mem['firstpart_1313'][igroup1-1]
    first_good=False
    while (ipar1>0):
        if (mem['density_1312'][ipar1-1] > rhot):
            if ( not first_good):
                if (igroupref==0):
                    posref[:3]=mem['pos_10'][ipar1-1,:3]
                    igroupref=igroup1
                first_good=True
                mem['firstpart_1313'][igroup1-1]=ipar1
                densmin=mem['density_1312'][ipar1-1]
                densmax=densmin
            else:
                mem['idpart_1311'][iparold]=ipar1

            iparold=ipar1
            imass += 1
            xmasspart = mem['mass_10'][ipar1-1] if(H.allocated('mass_10')) else H.massp
            truemass += xmasspart
            posdiffx=mem['pos_10'][ipar1-1,0]-posref[0]
            posdiffy=mem['pos_10'][ipar1-1,1]-posref[1]
            posdiffz=mem['pos_10'][ipar1-1,2]-posref[2]

            posdiffx=posdiffx-H.xlong if(posdiffx>=H.xlongs2) else posdiffx+H.xlong
            posdiffy=posdiffy-H.ylong if(posdiffy>=H.ylongs2) else posdiffy+H.ylong
            posdiffz=posdiffz-H.zlong if(posdiffz>=H.zlongs2) else posdiffz+H.zlong
            posdiffx += posref[0]
            posdiffy += posref[1]
            posdiffz += posref[2]
            posg[0] += posdiffx*xmasspart
            posg[1] += posdiffy*xmasspart
            posg[2] += posdiffz*xmasspart
            rsquare += xmasspart*(posdiffx**2+posdiffy**2+posdiffz**2)
            densmoy += mem['density_1312'][ipar1-1]
            densmax = max(densmax,mem['density_1312'][ipar1-1])
            densmin = min(densmin,mem['density_1312'][ipar1-1])
        ipar1=mem['idpart_1311'][ipar1-1]
    if ( not first_good): mem['firstpart_1313'][igroup1-1]=0

    if ( (densmin<=rhot)or(densmax!=mem['densityg_1313'][igroup1-1]) ):
        print('ERROR in treat_particles')
        print('igroup1, densmax, rhot=',igroup1,mem['densityg_1313'][igroup1-1],rhot)
        raise ValueError('denslow, denshigh    =',densmin,densmax)
    return igroupref

#=======================================================================
# !!$recursive subroutine do_colorize(icolor_select,igroup,igr,rhot)
def do_colorize_131430(igroup1,igr1,rhot): #YDdebug
#=======================================================================
    # integer(kind=4) :: igroup1,igr1
    # integer(kind=4) :: ineig,igroup2,igrB,neig
    # real(kind=8)  :: rhot  
    global ncall, icolor_select

    ncall += 1
    # !!$  write(errunit,'(A,3I8,e10.2,I8)')'do_colorize',icolor_select,igroup1,igr1,rhot,ncall
    mem['color_1314'][igr1-1]=icolor_select
    neig=H.group[igroup1-1].nhnei
    for ineig0 in range(H.group[igroup1-1].nhnei):
        if (H.group[igroup1-1].rho_saddle_gr[ineig0] > rhot):
    # We connect this group to its neighbourg
            igroup2=H.group[igroup1-1].isad_gr[ineig0]
            igrB=mem['igroupid_1314'][igroup2-1]
            if (mem['color_1314'][igrB-1]==0):
    # !!$           call do_colorize(icolor_select,igroup2,igrB,rhot)
                do_colorize_131430(igroup2,igrB,rhot) #YDdebug
            elif (mem['color_1314'][igrB-1]!=icolor_select):
                print(f"ERROR in do_colorize : color(igrB)({mem['color_1314'][igrB-1]}) <> icolor_select({icolor_select})")
                raise ValueError('The connections are not symmetric.')
            else:
                pass
        else:
    # We do not need this saddle anymore (and use the fact that saddles 
    # are ranked in decreasing order)
            neig -= 1
    H.group[igroup1-1].nhnei=neig

#=======================================================================
def compute_saddle_list_13140():
#=======================================================================
# Compute the lowest density threshold below which each group is 
# connected to an other one
#=======================================================================
    # integer(kind=4) :: ipar1,ipar2,igroup2,ineig,idist,igroup1,ineig2
    # integer(kind=4) :: neig,ineigal,in1,in2,idestroy,icon_count
    # integer(kind=4) :: i
    # real(kind=8)  :: density1,density2,rho_sad12
    # logical :: exist
    # logical, allocatable :: touch(:)
    # integer, allocatable :: listg(:)
    # real(kind=8),  allocatable :: rho_sad(:)  
    # integer, allocatable :: isad(:)
    # integer, allocatable :: indx(:)

    if (H.verbose): print('Fill the end of the branches of the group tree')

    # Allocate the array of nodes
    H.group = [H.grp() for _ in range(H.ngroups)]

    if (H.verbose): print('First count the number of neighbourgs of each elementary group...')

    touch = np.zeros(H.ngroups, dtype=np.bool_); touch[:H.ngroups] = False
    listg = np.zeros(H.ngroups, dtype=np.int32); listg[:H.ngroups] = 0


    # First count the number of neighbourgs for each group to H.allocate
    # arrays isad_in,isad_out,rho_saddle_gr
    for igroupA0 in range(H.ngroups):
        ineig=0
        ipar1=mem['firstpart_1313'][igroupA0]
    # Loop on all the members of the group
        while (ipar1>0):
            for idist0 in range(H.nhop):
                ipar2=mem['iparneigh_1312'][idist0,ipar1-1]
                igroupB1=mem['igrouppart_1313'][ipar2-1]
        # we test that we are in a group (i.e. that mem['density_1312'][ipar] >= rho_hold
        # and that this group is different from the one we are sitting on
                if( (igroupB1>0)and(igroupB1 != igroupA0+1) ):
                    if ( not touch[igroupB1-1]):
                        ineig += 1
                        touch[igroupB1-1]=True
                        listg[ineig-1]=igroupB1
    # Next member
            ipar1 = mem['idpart_1311'][ipar1-1]
    # Reinitialize touch
        for inA0 in range(ineig):
            igroupB1=listg[inA0]
            touch[igroupB1-1]=False
    # Allocate the nodes 
        H.group[igroupA0].nhnei=ineig
        ineigal=max(ineig,1)
        H.group[igroupA0].isad_gr = np.zeros(ineigal, dtype=np.int32)
        H.group[igroupA0].rho_saddle_gr = np.zeros(ineigal, dtype=np.float64)

    if (H.verbose): print('Compute lists of neighbourgs and saddle points...')


    # arrays isad_in,isad_out,rho_saddle_gr
    for igroupA0 in range(H.ngroups):
    # No calculation necessary if no neighbourg
        neig=H.group[igroupA0].nhnei
        if(neig>0):
            ineig=0
            ipar1=mem['firstpart_1313'][igroupA0]
            rho_sad = np.zeros(neig, dtype=np.float64)
    # Loop on all the members of the group
            while (ipar1>0):
                density1=mem['density_1312'][ipar1-1]
                for idist0 in range(H.nhop):
                    ipar2=mem['iparneigh_1312'][idist0,ipar1-1]
                    igroupB1=mem['igrouppart_1313'][ipar2-1]
        # we test that we are in a group (i.e. that mem['density_1312'][ipar] >= rho_hold
        # and that this group is different from the one we are sitting on
                    if( (igroupB1>0)and(igroupB1 != igroupA0+1) ):
                        density2=mem['density_1312'][ipar2-1]
                        if ( not touch[igroupB1-1] ):
                            ineig += 1
                            touch[igroupB1-1]=True
                            listg[igroupB1-1]=ineig
                            rho_sad12=min(density1,density2)
                            rho_sad[ineig-1]=rho_sad12
                            H.group[igroupA0].isad_gr[ineig-1]=igroupB1
                        else:
                            ineig2=listg[igroupB1-1]
                            rho_sad12=min(density1,density2)
                            rho_sad[ineig2-1]=max(rho_sad[ineig2-1],rho_sad12)
        # Next member
                ipar1=mem['idpart_1311'][ipar1-1]
            if (ineig!=neig):
    # Consistency checking
                print('ERROR in compute_saddle_list :')
                print('The number of neighbourgs does not match.')
                raise ValueError('ineig, neig =',ineig,neig)

            H.group[igroupA0].rho_saddle_gr[:ineig]=rho_sad[:ineig]
            del rho_sad
    # Reinitialize touch
            for inA0 in range(ineig):
                igroupB1=H.group[igroupA0].isad_gr[inA0]
                touch[igroupB1-1]=False
    # No neighbourg

    del touch
    del listg

    if (H.verbose): print('Establish symmetry in connections...')

    # Total number of connections count
    icon_count=0

    # Destroy the connections between 2 groups which are not symmetric
    # This might be rather slow and might be discarded later
    idestroy=0
    for igroupA0 in range(H.ngroups):
        if (H.group[igroupA0].nhnei>0):
            for inA0 in range(H.group[igroupA0].nhnei):
                exist=False
                igroupB1=H.group[igroupA0].isad_gr[inA0]
                if (igroupB1>0):
                    for inB0 in range(H.group[igroupB1-1].nhnei):
                        if (H.group[igroupB1-1].isad_gr[inB0] == igroupA0+1):
                            exist=True
                            rho_sad12=min(H.group[igroupB1-1].rho_saddle_gr[inB0], H.group[igroupA0].rho_saddle_gr[inA0])
                            H.group[igroupB1-1].rho_saddle_gr[inB0]=rho_sad12
                            H.group[igroupA0].rho_saddle_gr[inA0]=rho_sad12
                if ( not exist):
                    H.group[igroupA0].isad_gr[inA0]=0
                    idestroy += 1
                else:
                    icon_count += 1

    if (H.verbose): print('Number of connections removed :',idestroy)
    if (H.verbose): print('Total number of connections remaining :',icon_count)

    if (H.verbose): print('Rebuild groups with undesired connections removed...')


    # Rebuild the group list correspondingly with the connections removed
    # And sort the list of saddle points
    for igroupA0 in range(H.ngroups):
        neig=H.group[igroupA0].nhnei
        if (neig>0):
            rho_sad = np.zeros(neig, dtype=np.float64)
            isad = np.zeros(neig, dtype=np.int32)
            ineig=0
            for inA0 in range(neig):
                igroupB1=H.group[igroupA0].isad_gr[inA0]
                if (igroupB1>0):
                    ineig += 1
                    rho_sad[ineig]=H.group[igroupA0].rho_saddle_gr[inA0]
                    isad[ineig]=igroupB1

            H.group[igroupA0].isad_gr = None
            H.group[igroupA0].rho_saddle_gr = None
            ineigal=max(ineig,1)
            H.group[igroupA0].isad_gr = np.zeros(ineigal, dtype=np.int32)
            H.group[igroupA0].rho_saddle_gr = np.zeros(ineigal, dtype=np.float64)
            H.group[igroupA0].nhnei=ineig
            if (ineig>0):
    # sort the saddle points by decreasing order
                indx = np.argsort(rho_sad[:ineig])
                H.group[igroupA0].isad_gr = isad[indx]
                H.group[igroupA0].rho_saddle_gr = rho_sad[indx]
            del rho_sad
            del isad

    H.deallocate('iparneigh_1312')
    H.deallocate('igrouppart_1313')

#=======================================================================
def compute_density_13121(ipar1,dist2_0,iparnei):
#=======================================================================
    # real(kind=8)          :: dist2_0(0:H.nvoisins)
    # integer(kind=4)       :: iparnei(H.nvoisins)
    # real(kind=8)          :: r,unsr,contrib
    # real(kind=8),external :: spline
    # integer(kind=4)       :: idist,ipar1

    from num_rec import spline

    r=np.sqrt(dist2_0[H.nvoisins])*0.5
    unsr=1./r
    contrib=0.
    for idist0 in range(H.nvoisins-1):
        if(H.allocated('mass_10')):
            contrib += mem['mass_10'][iparnei[idist0]-1]*spline(np.sqrt(dist2_0[idist0+1])*unsr)
        else:
            contrib += H.massp*spline(np.sqrt(dist2_0[idist0+1])*unsr)
    # Add the contribution of the particle itself and normalize properly
    # to get a density with average unity (if computed on an uniform grid)
    # note that this assumes that the total mass in the box is normalized to 1.
    if(H.allocated('mass_10')):
        mem['density_1312'][ipar1-1]=(H.xlong*H.ylong*H.zlong)*(contrib+mem['mass_10'][ipar1-1]) /(H.pi*r**3)
    else:
        mem['density_1312'][ipar1-1]=(H.xlong*H.ylong*H.zlong)*(contrib + H.massp) /(H.pi*r**3)

#=======================================================================
def find_nearest_parts_13120(ipar1,dist2_0,iparnei):
#=======================================================================
    # integer(kind=4) :: ipar1,idist,icell_identity,inccellpart
    # real(kind=8)    :: dist2_0(0:H.nvoisins)
    # integer(kind=4) :: iparnei(H.nvoisins)
    # real(kind=8)    :: poshere(1:3)
    poshere = np.zeros(3, dtype=np.float64)

    poshere[:3]=mem['pos_10'][ipar1-1,:3]
    dist2_0[0]=0.
    dist2_0[1:] = H.bignum
    # for idist0 in range(H.nvoisins):
    #     dist2_0[idist0+1]=H.bignum
    icell_identity1 =1
    # inccellpart    =0
    # walk_tree(icell_identity1,poshere,dist2_0,ipar1,inccellpart,iparnei)
    dist2_0, iparnei = walk_tree_131200(icell_identity1,poshere,dist2_0,ipar1,iparnei)
    return dist2_0, iparnei


#=======================================================================
# def walk_tree(icellidin1,poshere,dist2_0, iparid,inccellpart,iparnei):
def walk_tree_131200(icellidin1,poshere,dist2_0, iparid1,iparnei):
#=======================================================================
    '''
    The subroutine 
    calculates the distances between the particle and cells (or the particles within cells),
    stores the closest cells/particles in `discell2_0` and `icid` arrays, and
    updates `dist2_0` and `iparnei` arrays with the closest particles.
    The subroutine recursively calls itself with the children cells of the current cell until it reaches the leaf cells that contain particles.
    The algorithm is an implementation of a tree structure, where cells are nodes and the particles are leaves.
    
    - by ChatGPT Jan 9
    '''
    # integer(kind=4) :: icellidin1,icell_identity,iparid1,inccellpart,ic,iparcell
    # real(kind=8)    :: poshere[2],dist2_0(0:H.nvoisins)
    # real(kind=8)    :: dx,dy,dz,distance2,sc
    # integer(kind=4) :: idist,inc
    # integer(kind=4) :: icellid_out
    # real(kind=8)    :: discell2_0(0:8)
    discell2_0 = np.zeros(9, dtype=np.float64)
    # integer(kind=4) :: iparnei(H.nvoisins)
    # integer(kind=4) :: icid(8)
    icid = np.zeros(8, dtype=np.int32)

    # integer(kind=4) :: i,first_pos_this_node
    # real(kind=8)    :: distance2p

    icell_identity1 = mem['firstchild_1311'][icellidin1-1]
    inc=1
    discell2_0[0]=0
    discell2_0[1:]=1e30
    # Until icell_identity1==0: (Final leaf of tree)
    # Calc distance (poshere <-> cells)
    while (icell_identity1 != 0):
        sc=mem['size_cell_1311'][icell_identity1-1]
        dx=abs(mem['pos_cell_1311'][0,icell_identity1-1]-poshere[0])
        dx=max(0.,min(dx,H.xlong-dx)-sc)
        dy=abs(mem['pos_cell_1311'][1,icell_identity1-1]-poshere[1])
        dy=max(0.,min(dy,H.ylong-dy)-sc)
        dz=abs(mem['pos_cell_1311'][2,icell_identity1-1]-poshere[2])
        dz=max(0.,min(dz,H.zlong-dz)-sc)
        distance2=dx**2+dy**2+dz**2
        if (distance2 < dist2_0[H.nvoisins]):
            idist=inc-1
            while (discell2_0[idist]>distance2):
                discell2_0[idist+1]=discell2_0[idist]
                icid[idist]=icid[idist-1]
                idist -= 1
            discell2_0[idist+1]=distance2
            icid[idist]=icell_identity1
            inc += 1
        icell_identity1=mem['sister_1311'][icell_identity1-1]
    # inccellpart += inc-1
    # Loop for counted cells,
    # Update the closest particle ID(in iparnei) and the distance to that part(in dist2_0)
    for ic0 in range(inc-1):
        icellid_out1=icid[ic0]
        if (mem['firstchild_1311'][icellid_out1-1] < 0):
            if (discell2_0[ic0+1]<dist2_0[H.nvoisins]):
                first_pos_this_node=-mem['firstchild_1311'][icellid_out1-1]-1
                for i1 in frange(first_pos_this_node+1, first_pos_this_node+mem['mass_cell_1311'][icellid_out1]):
                    iparcell1=mem['idpart_1311'][i1-1]
                    dx=abs(mem['pos_10'][iparcell1-1, 0]-poshere[0])
                    dx=max(0.,min(dx,H.xlong-dx))
                    dy=abs(mem['pos_10'][iparcell1-1, 1]-poshere[1])
                    dy=max(0.,min(dy,H.ylong-dy))
                    dz=abs(mem['pos_10'][iparcell1-1, 2]-poshere[2])
                    dz=max(0.,min(dz,H.zlong-dz))
                    distance2p=dx**2+dy**2+dz**2
                    if (distance2p < dist2_0[H.nvoisins]): 
                        if (iparcell1 != iparid1):
                            idist1=H.nvoisins-1
                            while (dist2_0[idist1]>distance2p):
                                dist2_0[idist1+1]=dist2_0[idist1]
                                iparnei[idist1]=iparnei[idist1-1]
                                idist1 -= 1
                            dist2_0[idist1+1]=distance2p
                            iparnei[idist1]=iparcell1
        elif (discell2_0[ic0+1] < dist2_0[H.nvoisins]):
            # walk_tree(icellid_out1,poshere,dist2_0,iparid1,inccellpart,iparnei)
            dist2_0, iparnei = walk_tree_131200(icellid_out1,poshere,dist2_0,iparid1,iparnei)
        else: pass
    return dist2_0, iparnei

#=======================================================================
def create_tree_structure_1311():
#=======================================================================
    # integer(kind=4) :: nlevel,inccell,idmother,ipar
    # integer(kind=4) :: npart_this_node,first_pos_this_node
    # integer(kind=4) :: ncell
    # real(kind=8)    :: pos_this_node(3)
    global inccell
    pos_this_node = np.empty(3, dtype=np.float64)

    if (H.verbose): print('Create tree structure...')

    # we modified to put 2*H.npart-1 instead of 2*H.npart so that AdaptaHOP can work on a 1024^3, 2*(1024^3)-1 is still an integer(kind=4), 2*(1024^3) is not 
    H.ncellmx=2*H.npart -1
    H.ncellbuffer=max(round(0.1*H.npart),H.ncellbuffermin)
    H.allocate('idpart_1311',H.npart, dtype=np.int32)
    # H.allocate('idpart_tmp_1311',H.npart, dtype=np.int32)
    H.allocate('mass_cell_1311',H.ncellmx, dtype=np.int32)
    H.allocate('size_cell_1311',H.ncellmx, dtype=np.float64)
    H.allocate('pos_cell_1311',(3,H.ncellmx), dtype=np.float64)
    H.allocate('sister_1311',H.ncellmx, dtype=np.int32)
    H.allocate('firstchild_1311',H.ncellmx, dtype=np.int32)
    
    mem['idpart_1311'][:] = np.arange(H.npart, dtype=np.int32)+1

    nlevel=0
    inccell=0
    idmother=0
    pos_this_node[:]=0.
    npart_this_node=H.npart
    first_pos_this_node=0
    # mem['idpart_tmp_1311'][:]=0
    mem['pos_cell_1311'][:]=0
    mem['size_cell_1311'][:]=0
    mem['mass_cell_1311'][:]=0
    mem['sister_1311'][:]=0
    mem['firstchild_1311'][:]=0
    H.sizeroot = np.double( np.max([H.xlong,H.ylong,H.zlong]) )

    create_KDtree_13110(nlevel,pos_this_node, npart_this_node,first_pos_this_node, idmother)
    ncell=inccell

    if (H.verbose): print('total number of cells =',ncell)

    # H.deallocate('idpart_tmp_1311')
    raise ValueError("stop")


from multiprocessing import Pool
#=======================================================================
def create_KDtree_13110(nlevel:np.int32,pos_this_node:np.ndarray[np.float64],npart_this_node, first_pos_this_node,idmother,pos_ref_0=None):
#=======================================================================
#  nlevel : level of the H.node_0 in the octree. Level zero corresponds to 
#           the full box
#  pos_this_node : position of the center of this H.node_0
#  H.npart  : total number of particles 
#  idpart : array of dimension H.npart containing the id of each
#           particle. It is sorted such that neighboring particles in
#           this array belong to the same cell H.node_0.
#  idpart_tmp : temporary array of same size used as a buffer to sort
#           idpart.
#  npart_this_node : number of particles in the considered H.node_0
#  first_pos_this_node : first position in idpart of the particles 
#           belonging to this H.node_0
#  pos :    array of dimension 3.H.npart giving the positions of 
#           each particle belonging to the halo
#  inccell : cell id number for the newly created structured grid site
#  pos_cell : array of dimension 3.H.npart (at most) corresponding
#           to the positions of each H.node_0 of the structured grid
#  mass_cell : array of dimension H.npart (at most) giving the 
#           number of particles in each H.node_0 of the structured grid
#  size_cell : array of dimension H.npart (at most) giving half the
#           size of the cube forming each H.node_0 of the structured grid
#  H.sizeroot : size of the root cell (nlevel=0)
#  idmother : id of the mother cell
#  sister   : sister of a cell (at the same level, with the same mother)
#  firstchild : first child of a cell (then the next ones are found 
#             with the array sister). If it is a cell containing only
#           one particle, it gives the id of the particle.
#  ncellmx : maximum number of cells
#  H.megaverbose : detailed H.verbose mode
#=======================================================================
    global inccell
    print(f"[create_KDtree] nlevel={nlevel}, pos_this_node={pos_this_node}, npart_this_node={npart_this_node}, inccell={inccell}")
    # integer(kind=4)           :: nlevel,first_pos_this_node,npart_this_node
    # real(kind=8)              :: pos_ref_0(3,0:7)
    if(pos_ref_0 is None):
        pos_ref_0 = np.array([-1., -1., -1.,
                             1., -1., -1.,
                            -1.,  1., -1.,
                             1.,  1., -1.,
                            -1., -1.,  1.,
                             1., -1.,  1.,
                            -1.,  1.,  1.,
                             1.,  1.,  1.], dtype=np.float64)
        pos_ref_0 = pos_ref_0.reshape((3, 8), order='F')  # 'F' for Fortran-style ordering
    # real(kind=8)              :: pos_this_node(3)   
    # integer(kind=4)           :: ipar,icid,j,inccell,nlevel_out
    # integer(kind=4), external :: icellid
    from num_rec import icellid, icellids
    import time
    # integer(kind=4)           :: first_pos_this_node_out,npart_this_node_out
    # integer(kind=4)           :: incsubcell_0(0:7),nsubcell_0(0:7)
    timereport = []; icount=1
    ref = time.time()
    incsubcell_0 = np.zeros(8, dtype=np.int32)
    nsubcell_0 = np.zeros(8, dtype=np.int32)
    # real(kind=8)              :: xtest(3),pos_this_node_out(3)
    # xtest = np.empty(3, dtype=np.float64)
    # pos_this_node_out = np.empty(3, dtype=np.float64)
    timereport.append((f'{icount} init', time.time()-ref)); ref = time.time(); icount+=1
    # integer(kind=4)           :: idmother,idmother_out

    # integer(kind=8)           :: ncellmx_old
    # integer, allocatable      :: mass_cell_tmp(:),sister_tmp(:),firstchild_tmp(:)
    # real(kind=8), allocatable :: size_cell_tmp(:),pos_cell_tmp(:,:)

    #  pos_ref_0 : an array used to find positions of the 8 subcells in this
    #           H.node_0.

    if (npart_this_node>0):
        inccell += 1
        if ( ((inccell%1000000)==0)and(H.megaverbose) ): print('inccell=',inccell)
        if (inccell>H.ncellmx):
            # If we have reached the maximum number of cells, we increase
            # the size of the arrays and reallocate them
            ncellmx_old=H.ncellmx
            H.ncellmx += H.ncellbuffer
            if(H.megaverbose): print(f'ncellmx{ncellmx_old} is too small. Increase(+{H.ncellbuffer}) it and reallocate arrays accordingly')
            tmp = np.zeros(ncellmx_old, dtype=np.int32)
            tmp[:ncellmx_old]=mem['mass_cell_1311'][:ncellmx_old]
            H.deallocate('mass_cell_1311')
            H.allocate('mass_cell_1311',H.ncellmx, dtype=np.int32)
            mem['mass_cell_1311'][:ncellmx_old]=tmp[:ncellmx_old]
            tmp[:ncellmx_old]=mem['sister_1311'][:ncellmx_old]
            H.deallocate('sister_1311')
            H.allocate('sister_1311',H.ncellmx, dtype=np.int32)
            mem['sister_1311'][:ncellmx_old]=tmp[:ncellmx_old]
            tmp[:ncellmx_old]=mem['firstchild_1311'][:ncellmx_old]
            H.deallocate('firstchild_1311')
            H.allocate('firstchild_1311',H.ncellmx, dtype=np.int32)
            mem['firstchild_1311'][:ncellmx_old]=tmp[:ncellmx_old]
            mem['firstchild_1311'][ncellmx_old:H.ncellmx]=0
            del tmp
            tmp = np.zeros(ncellmx_old, dtype=np.float64)
            tmp[:ncellmx_old]=mem['size_cell_1311'][:ncellmx_old]
            H.deallocate('size_cell_1311')
            H.allocate('size_cell_1311',H.ncellmx,dtype=np.float64)
            mem['size_cell_1311'][:ncellmx_old]=tmp[:ncellmx_old]
            del tmp
            pos_cell_tmp = np.zeros((3,ncellmx_old), dtype=np.float64)
            pos_cell_tmp[:3,:ncellmx_old]=mem['pos_cell_1311'][:3,:ncellmx_old]
            H.deallocate('pos_cell_1311')
            H.allocate('pos_cell_1311',(3,H.ncellmx))
            mem['pos_cell_1311'][:3,:ncellmx_old]=pos_cell_tmp[:3,:ncellmx_old]
            del pos_cell_tmp
        mem['pos_cell_1311'][:3,inccell-1]=pos_this_node[:3]
        mem['mass_cell_1311'][inccell-1]=npart_this_node
        mem['size_cell_1311'][inccell-1]=2.**(-nlevel)*H.sizeroot*0.5
        if (idmother>0):
            # If this is not the root cell, we link it to its mother
            mem['sister_1311'][inccell-1]=mem['firstchild_1311'][idmother-1]
            mem['firstchild_1311'][idmother-1]=inccell
        if ((npart_this_node <= H.npartpercell) or (nlevel==H.nlevelmax)):
            # If there is only `H.npartpercell` particles in the `H.node_0` or we have reach
            # maximum level of refinement, we are done
            mem['firstchild_1311'][inccell-1]=-(first_pos_this_node+1)
            return
    else:
        # Stop refinement due to no particle in leaf cell
        return
    timereport.append((f'{icount} check', time.time()-ref)); ref = time.time(); icount+=1

    #  Count the number of particles in each subcell of this H.node_0
    incsubcell_0[:]=0
    idpart1s = mem['idpart_1311'][first_pos_this_node : first_pos_this_node+npart_this_node]
    # xtests = mem['pos_10'][idpart1s-1,:3] - pos_this_node[:3]
    # timereport.append((f'{icount} xtests', time.time()-ref)); ref = time.time(); icount+=1
    icids = icellids(mem['pos_10'][idpart1s-1,:3] - pos_this_node[:3])
    timereport.append((f'{icount} icellids', time.time()-ref)); ref = time.time(); icount+=1
    uni,count = np.unique(icids, return_counts=True)
    timereport.append((f'{icount} unique', time.time()-ref)); ref = time.time(); icount+=1
    incsubcell_0[uni] = count
    timereport.append((f'{icount} uni count', time.time()-ref)); ref = time.time(); icount+=1

    

    #  Create the array of positions of the first particle of the lists
    #  of particles belonging to each subnode
    nsubcell_0[0]=0
    nsubcell_0[1:] = np.cumsum(incsubcell_0[:-1])
    timereport.append((f'{icount} cumsum', time.time()-ref)); ref = time.time(); icount+=1

    #  Sort the array of ids (idpart) to gather the particles belonging
    #  to the same subnode. Put the result in `idpart_tmp`.
    argsort = np.argsort(icids, kind='mergesort')
    timereport.append((f'{icount} argsort', time.time()-ref)); ref = time.time(); icount+=1
    mem['idpart_1311'][first_pos_this_node : first_pos_this_node+npart_this_node] = idpart1s[argsort] # <- This is main part?
    timereport.append((f'{icount} idpart', time.time()-ref)); ref = time.time(); icount+=1
    
    #  Put back the sorted ids in idpart
    # for ipar1 in frange(first_pos_this_node+1, first_pos_this_node+npart_this_node):
    #     mem['idpart_1311'][ipar1-1]=mem['idpart_tmp_1311'][ipar1-1]
    for tmp in timereport:
        print(tmp)

    #  Call again the routine for the 8 subnodes:
    #  Compute positions of subnodes, new level of refinement, 
    #  positions in the array idpart corresponding to the subnodes,
    #  and call for the treatment recursively.
    nlevel_out=nlevel+1
    idmother_out=inccell
    for j0 in range(7+1):
        pos_this_node_out=pos_this_node[:3] + H.sizeroot*pos_ref_0[:3,j0]*2**(-nlevel-2)
        first_pos_this_node_out=first_pos_this_node+nsubcell_0[j0]
        npart_this_node_out=incsubcell_0[j0]
        timereport.append((f'{icount} before recursive', time.time()-ref)); ref = time.time(); icount+=1
        create_KDtree_13110(nlevel_out,pos_this_node_out,npart_this_node_out,first_pos_this_node_out,idmother_out,pos_ref_0=pos_ref_0)



# def create_KDtree_mod():
#     # cpos = np.zeros((pos.shape[0],3), dtype=np.float128)
#     # octarr["done"] = False
#     octarr = np.zeros(
#         mem['pos_10'].shape[0], 
#         dtype=[
#             ("oct1", np.int64),("oct2", np.int64), ("lvl", np.int8), ("done", bool), 
#             ("inccell", np.int64), ("idmother", np.int64)
#             ])
#     octarr["done"] = False
#     cpos = np.zeros((mem['pos_10'].shape[0],3), dtype=np.float128)
#     octarr = refine(mem['pos_10'], octarr, cpos)
#     lexsort = np.lexsort((octarr["oct2"], octarr["oct1"]))
#     mem['idpart_1311'] = mem['idpart_1311'][lexsort]


# def refine(pos, octarr, cpos):
#     from num_rec import icellids
#     yet = ~octarr["done"]
#     ncell_new = 1
#     inccell = 1
#     while(True in yet):
#         nlevel = int( np.max(octarr["lvl"]) )
#         if(nlevel >= H.nlevelmax): break
#         octid = octarr['oct1'] if(nlevel < 15) else octarr['oct1']+1j*octarr['oct2']
#         ncell_old = ncell_new
#         icids = icellids(pos[yet]-cpos[yet])+1
#         if(nlevel > 14):
#             octarr['oct2'] *= 10
#             octarr['oct2'][yet] += icids
#         else:
#             octarr['oct1'] *= 10
#             octarr['oct1'][yet] += icids
#         octarr['lvl'][yet] += 1

#         octid = octarr['oct1'] if(nlevel < 15) else octarr['oct1']+1j*octarr['oct2']
#         uni, count = np.unique(octid[yet], return_counts=True)
#         if(1 in count):
#             leafind = np.where(count==1)[0]
#             leaf = uni[leafind]
#             isin = np.isin(octid, leaf)
#             octarr["done"][isin] = True
#         a,b = np.unique(octarr['lvl'], return_counts=True)
#         ncell_new = np.sum(b[:-1]) + len( np.unique(octid[octarr['lvl'] == a[-1]]) )
#         print(f"[lvl={nlevel}] {ncell_old} -> {ncell_new} [{octarr[0]}, {octarr[yet][0]}]")
#         cpos[yet] += H.sizeroot*H.pos_ref_0[:3,icids-1].T * 2**(-nlevel-2)
#         yet = ~octarr["done"]
#     print(f"[lvl={nlevel}] KDtree done")
#     return octarr




#=======================================================================
def remove_degenerate_particles():
#=======================================================================
    raise NotImplementedError('remove_degenerate_particles is not implemented yet')

#   real(kind=8), parameter :: accurac=1.d-6

#   integer, allocatable, dimension(:) :: idgene
#   real(kind=8),allocatable, dimension(:) :: tmp
#   real(kind=8),allocatable, dimension(:,:) :: possav
#   logical,allocatable,dimension(:) :: move
#   real(kind=8) :: xref,tolerance,tolerance2,phi,costeta,teta,sinteta
#   real(kind=8) :: ran2
#   logical :: doneiter,done
#   integer(kind=4) :: i,niter,j,ipar,jpar,idd,jdd,incdege
#   integer(kind=4) :: idum

#   if (H.megaverbose) print('Move randomly particles at exactly the same position')

#   idum=-111
#   tolerance=max(H.xlongs2,H.ylongs2,H.zlongs2)*accurac
#   H.allocate(idgene(H.npart),tmp(H.npart),move(H.npart))
#   tolerance2=2.0*tolerance
#   H.allocate(possav(3,H.npart))
#   do i=1,H.npart
#      possav(1:3,i)=pos(i,1:3)
#   enddo

#   niter=1
#   doneiter=False
#   while ( not doneiter) 
#      if (H.megaverbose) print('Iteration :',niter)
#      niter=niter+1
#      doneiter=False
#      do i=1,H.npart
#         tmp(i)=pos(i,1)
#         move(i)=False
#      enddo
#      call indexx(H.npart,tmp,idgene)

#      done=False
#      i=1
#      while ( not done)
#         xref=tmp(idgene(i))
#         j=i+1
#         if (j > H.npart) then
#            j=1
#            done=True
#         endif
#         while (abs(xref-tmp(idgene(j))) < tolerance)
#            j=j+1
#            if (j > H.npart) then
#               j=1
#               done=True
#            endif
#         enddo
#         do ipar=i,j-1
#            idd=idgene(ipar)
#            do jpar=ipar+1,j-1
#               jdd=idgene(jpar)
#               if (abs(pos(jdd,2)-pos(idd,2)) < tolerance and &
#  &                abs(pos(jdd,3)-pos(idd,3)) < tolerance) then
#                  move(idgene(jpar))=True
#               endif
#            enddo
#         enddo
#         i=j
#      enddo
#      incdege=0
#      do i=1,H.npart
#         if (move(i)) then
#            incdege=incdege+1
#         endif
#      enddo
#      if (H.megaverbose) print(&)
#  &                 'Found the following number of degeneracies :',incdege
#      do i=1,H.npart
#         if (move(i)) then
#            phi=2.*H.pi*ran2(idum)
#            costeta=1.0-2.0*ran2(idum)
#            teta=acos(costeta)           
#            sinteta=sin(teta)
#            pos(i,1)=possav(1,i)+tolerance2*cos(phi)*sinteta
#            pos(i,2)=possav(2,i)+tolerance2*sin(phi)*sinteta
#            pos(i,3)=possav(3,i)+tolerance2*costeta
#         endif
#      enddo
     
#      if (incdege==0) doneiter=True
#   enddo

#   deallocate(possav)
#   deallocate(tmp,idgene,move)
  
# end subroutine remove_degenerate_particles

#=======================================================================
def convtoasc(number,sstring):
#=======================================================================
# To convert an integer(kind=4) smaller than 999999 to a 6 characters string
#=======================================================================
    #   integer(kind=4) :: number, istring, num, nums10, i
    #   character*6 :: sstring
    #   character*10,parameter :: nstring='0123456789'
    sstring = f"{number:06d}"
    return sstring

#=======================================================================




# #=======================================================================
# function HsurH0(z,omega0,omegaL,omegaR)
# #=======================================================================
#   real(kind=8) :: z,omega0,omegaL,omegaR,HsurH0

#   HsurH0=np.sqrt(Omega0*(1+z)**3+OmegaR*(1+z)**2+OmegaL)
# end function HsurH0


