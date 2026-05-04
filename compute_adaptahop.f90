!=====================================================================
!=====================================================================
!
!
!
!  halo_defs.f90
!
!
!
!=====================================================================
!=====================================================================
module fhalo_defs

  public

  !======================================================================
  ! parameters relative to the simulation analysis
  !======================================================================
!   character(len=8),parameter :: gravsoft  = 'cubsplin'         ! type of gravitational softening 
  integer(kind=4)            :: nbPes                          ! obsolete vars for reading treecode format
!   integer(kind=4)            :: nsteps                         ! number of timesteps
!   real(kind=8)               :: alpha,tnow                     ! 
  integer(kind=4)            :: nbodies                        ! number of particles in the simulation 
  integer(kind=4)            :: nMembers                       ! minimal number of particles in a fof halo
!   real(kind=8)               :: b_init                         ! linking length parameter of the fof at z=0
!   character(len=4),parameter :: profile   = 'TSIS'             ! type of halo profile (only isothermal_sphere yet)
!   integer(kind=4),parameter  :: ninterp   = 30000              ! nb of bins for interp. of smoothed grav. field
!   integer(kind=4)            :: FlagPeriod                     ! flag for periodicity of boundary conditions 
!   real(kind=8)               :: fPeriod(3)
  !----------- For gadget format: ----------------------------------------
!   integer (kind=4)           :: nhr                            ! to read only the selected HR particles
!   real (kind=4)              :: minlrmrat                      ! to recognize contaminated halos
  !======================================================================


  !======================================================================
  ! Definitions specific to input/output
  !======================================================================
!   character(len=80)         :: data_dir
!   character(len=5)          :: file_num
!   integer(kind=4)           :: numstep
  integer(kind=4),parameter :: errunit = 0
!   logical(kind=4) :: write_resim_masses                        ! for writing resim_masses.dat file
!   integer(kind=4)           :: dchmod = int(o'755')
!   character(len=5)          :: dchmod_str = '755'
!   integer(kind=4)           :: fchmod = int(o'644')
!   character(len=5)          :: fchmod_str = '644'
!   integer(kind=4)           :: uid = -1
!   integer(kind=4)           :: gid = -1
  !======================================================================


  !======================================================================
  ! Constants
  !======================================================================
!   real(kind=8),parameter    :: gravconst = 430.1               ! G in units of (km/s)^2 Mpc/(10^11 Msol)
  real(kind=8),parameter    :: pi        = 3.141592654
  !======================================================================


  !======================================================================
  ! Global variables 
  !======================================================================
  real(kind=8),allocatable         :: pos(:,:)!,vel(:,:)
  real(kind=8)                     :: massp
!   real(kind=8),allocatable         :: epsvect(:)
  real(kind=8),allocatable         :: mass(:)
  real(kind=8)                     :: omega_t,omega_lambda_t,omega_f,omega_lambda_f,omega_c_f
  real(kind=8)                     :: rho_crit,aexp,Lboxp,mboxp,af,ai,Lf,H_f,H_i
  real(kind=8)                     :: age_univ,Lbox_pt,Lbox_pt2,Hub_pt,omega_0,hubble,omega_lambda_0
!   real(kind=8)                     :: vir_overdens,rho_mean
!   integer(kind=4),allocatable      :: linked_list(:)!, liste_parts(:)
!   integer(kind=4),allocatable      :: first_part(:),nb_of_parts(:)
!   type (halo),allocatable          :: liste_halos(:)
!   real(kind=8)                     :: phsmooth(0:1+ninterp)
!   integer(kind=4)                  :: nb_of_halos, nb_of_subhalos
!   integer(kind=4)                  :: numero_step  
  character(len=3)                 :: type
  !======================================================================


  !======================================================================
  ! defs for Adaptahop
  !======================================================================
   ! integer(kind=4), parameter :: nparmax=512*512*512
   ! integer(kind=4),parameter  :: nparbuffer=128**3
   integer(kind=4),parameter  :: ncellbuffermin=128**3
   ! integer(kind=4),parameter  :: nlevelmax=30,npartpercell=100
   integer(kind=4),parameter  :: npartpercell=100
   ! integer(kind=4)            :: lin=10,lsin=11,lin2=12
   ! integer(kind=4)            :: loudis=12,lounei=13,lounode=14,loupartnode=15,lounodedyn=16
   real(kind=8)            :: bignum=1.d30
   ! Physical constants (units : m s kg) ->
   real(kind=8), parameter :: gravitational_constant=6.6726d-11
   real(kind=8), parameter :: critical_density= 1.8788d-26
   real(kind=8), parameter :: mega_parsec=3.0857d22
   real(kind=8), parameter :: solar_mass=1.989d30
   real(kind=8), parameter :: convert_in_mps2=1.d6
   ! real(kind=8), allocatable    :: vxout(:),vyout(:),vzout(:),vdisout(:)
   integer(kind=4), allocatable :: mass_cell(:)
   ! real(kind=8), allocatable    :: tmass_cell(:)
   ! real(kind=8), allocatable    :: vcell(:,:)
   real(kind=8), allocatable    :: size_cell(:)
   real(kind=8), allocatable    :: pos_cell(:,:)
   integer(kind=4), allocatable :: sister(:)
   integer(kind=4), allocatable :: firstchild(:)
   integer(kind=4), allocatable :: idpart(:),idpart_tmp(:)
   integer(kind=4), allocatable :: iparneigh(:,:)
   real(kind=8), allocatable    :: distance(:)
   ! real(kind=8), allocatable    :: density(:)
   integer(kind=4), allocatable :: firstpart(:)
   integer(kind=4), allocatable :: igrouppart(:)
   integer(kind=4), allocatable :: idgroup(:),idgroup_tmp(:)
   integer(kind=4), allocatable :: igroupid(:)
   integer(kind=4), allocatable :: color(:)
   ! integer(kind=4), allocatable :: partnode(:)
   real(kind=8), allocatable    :: densityg(:)
   real(kind=8)    :: sizeroot
   real(kind=8)    :: xlong, ylong, zlong, boxsize, boxsize2
   real(kind=8)    :: xlongs2, ylongs2, zlongs2
   real(kind=8)    :: omega0,omegaL,mass_in_kg,GMphys
   real(kind=8)    :: aexp_hubble_const,aexp_mega_parsec_sqrt3
   real(kind=8)    :: aexp_mega_parsec,aexp_max
   integer(kind=4) :: npart
   integer(kind=4) :: nvoisins,nhop,ntype,nlevelmax
   integer(kind=4) :: ncellmx
   integer(kind=4) :: ngroups,nmembthresh,nnodes,nnodesmax
   ! integer(kind=4) :: ncpu,nmpi,niterations
   integer(kind=4) :: ncellbuffer
   real(kind=8)    :: rho_threshold
   logical         :: verbose,megaverbose,periodic
   ! real(kind=8)    :: fgas
   real(kind=8)    :: fudge,alphap,epsilon,fudgepsilon
   real(kind=8)    :: pos_shift(3),pos_renorm,velrenorm

   type grp
      sequence
      integer(kind=4) :: nhnei
      integer(kind=4) :: njunk ! To avoid missalignement in memory
      integer(kind=4), dimension(:),pointer :: isad_gr(:)
      real(kind=8), dimension(:),pointer    :: rho_saddle_gr(:)
   end type grp

   type supernode
      sequence
      integer(kind=4) :: mass
      integer(kind=4) :: level
      integer(kind=4) :: mother
      integer(kind=4) :: firstchild
      integer(kind=4) :: nsisters
      integer(kind=4) :: sister
      real(kind=8)    :: rho_saddle
      real(kind=8)    :: density
      real(kind=8)    :: densmax
      real(kind=8)    :: radius
      real(kind=8)    :: truemass
      real(kind=8)    :: position(3)
   end type supernode
   type (supernode), allocatable :: node(:)

   type (grp), allocatable       :: group(:)
   

!======================================================================
! Flags for halo finder selection
!======================================================================
   character(len=3)    :: method       ! flag to notify which and how the halofinder is to be used
   logical             :: fsub         ! flag to notify whether subhaloes are included               
   logical             :: cdm          ! flag to select particle closest to the cdm instead of the one with the highest density
   logical             :: DPMMC=.false.! flag to select the densest particle in the most massive cell of the halo (not with FOF)
   logical             :: SC=.false.   ! flag to select the com within concentric spheres (not with FOF)
   real(kind=8)        :: dcell_min
   real(kind=8)        :: eps_SC
   logical             :: dump_dms=.false.

!======================================================================
! array to build the structure tree
!======================================================================
integer(kind=4), allocatable :: first_daughter(:), mother(:), first_sister(:), level(:)
integer(kind=4) :: nstruct

! used for the merger history method
! integer(kind=4), allocatable :: npfather(:),ex_liste_parts(:),removesub(:)
! integer(kind=4), allocatable :: ex_level(:),ex_nb_of_parts(:),ex_linked_list(:)
! integer(kind=4) ::  ex_nb_of_structs


! #ifdef ANG_MOM_OF_R
! integer(kind=4),parameter :: agor_unit = 19
! character(200)            :: agor_file
! integer(kind=4),parameter :: nshells = 100
! #endif

! logical :: isprinting=.false.
! integer :: tqdm_icalced=0

contains


!=====================================================================
! num_rec.f90
!=====================================================================
subroutine indexx(n,arr,indx)
   implicit none

   integer(kind=4)           :: n,indx(n)
   real(kind=8)              :: arr(n)
   integer(kind=4),parameter :: m=7,nstack=50
   integer(kind=4)           :: i,indxt,ir,itemp,j,jstack,k,l,istack(nstack)
   real(kind=8)              :: a

   do j = 1,n
      indx(j) = j
   enddo

   jstack = 0
   l      = 1
   ir     = n
   1 if (ir-l .lt. m) then
      do j = l+1,ir
         indxt = indx(j)
         a     = arr(indxt)
         do i = j-1,1,-1
            if (arr(indx(i)) .le. a) goto 2
            indx(i+1) = indx(i)
         enddo
         i         = 0
   2       indx(i+1) = indxt
      enddo
      if (jstack .eq. 0) return
      ir     = istack(jstack)
      l      = istack(jstack-1)
      jstack = jstack-2
   else
      k         = (l+ir)/2
      itemp     = indx(k)
      indx(k)   = indx(l+1)
      indx(l+1) = itemp
      if (arr(indx(l+1)) .gt. arr(indx(ir))) then
         itemp     = indx(l+1)
         indx(l+1) = indx(ir)
         indx(ir)  = itemp
      endif
      if (arr(indx(l)) .gt. arr(indx(ir))) then
         itemp    = indx(l)
         indx(l)  = indx(ir)
         indx(ir) = itemp
      endif
      if (arr(indx(l+1)) .gt. arr(indx(l))) then
         itemp     = indx(l+1)
         indx(l+1) = indx(l)
         indx(l)   = itemp
      endif
      i     = l+1
      j     = ir
      indxt = indx(l)
      a     = arr(indxt)
   3    continue
      i     = i+1
      if (arr(indx(i)) .lt. a) goto 3
   4    continue
      j     = j-1
      if (arr(indx(j)) .gt. a) goto 4
      if (j .lt. i) goto 5
      itemp   = indx(i)
      indx(i) = indx(j)
      indx(j) = itemp
      goto 3
   5    continue
      indx(l) = indx(j)
      indx(j) = indxt
      jstack  = jstack+2
      if (jstack .gt. nstack) stop 'nstack too small in indexx'
      if (ir-i+1 .ge. j-l) then
         istack(jstack)   = ir
         istack(jstack-1) = i
         ir               = j-1
      else
         istack(jstack)   = j-1
         istack(jstack-1) = l
         l                = i
      endif
   endif
   goto 1

end subroutine indexx

end module fhalo_defs























































module neiKDtree
   use fhalo_defs
   private

   public :: compute_adaptahop, sync_others, sync_from_change_pos, sync_from_init_adaptahop, &
             close, liste_parts, density, nnodes, real_table, integer_table

   integer(kind=4)::ncall         !YDdebug
   integer(kind=4)::icolor_select !YDdebug

   ! Output to python
   integer(kind=4),allocatable      :: liste_parts(:)
   real(kind=8),   allocatable      :: density(:)

   ! Convert node(:) to python-understandable array
   real(kind=8),    dimension(:,:),   allocatable :: real_table
   integer(kind=4), dimension(:,:),   allocatable :: integer_table

   contains

!=======================================================================
subroutine compute_adaptahop(pos_in, mass_in)
!=======================================================================
   implicit none
   real(kind=8), intent(in)         :: pos_in(:,:)
   real(kind=8), intent(in)         :: mass_in(:)
   real(kind=8)    :: dtdtdt

   allocate(liste_parts(1:nbodies))
   allocate(pos(1:npart,1:3))
   pos(:,:) = pos_in(:,:)
   allocate(mass(npart))
   mass(:) = mass_in(:)
   
   call create_tree_structure
   call compute_mean_density_and_np
   call find_local_maxima
   call create_group_tree

   deallocate(pos)
   return
end subroutine compute_adaptahop



!=====================================================================
! For f2py
!=====================================================================
subroutine node2table
   integer(kind=4) :: inodee
   allocate(real_table(8,nnodes)) !  rho_saddle, density, densmax, radius, truemass, position(3)
   allocate(integer_table(6,nnodes)) ! mass, level, mother, firstchild, nsisters, sister
   do inodee=1,nnodes
      real_table(1,inodee) = node(inodee)%rho_saddle
      real_table(2,inodee) = node(inodee)%density
      real_table(3,inodee) = node(inodee)%densmax
      real_table(4,inodee) = node(inodee)%radius
      real_table(5,inodee) = node(inodee)%truemass
      real_table(6:8,inodee) = node(inodee)%position(:)
      integer_table(1,inodee) = node(inodee)%mass
      integer_table(2,inodee) = node(inodee)%level
      integer_table(3,inodee) = node(inodee)%mother
      integer_table(4,inodee) = node(inodee)%firstchild
      integer_table(5,inodee) = node(inodee)%nsisters
      integer_table(6,inodee) = node(inodee)%sister
   end do
   deallocate(node)
end subroutine node2table


subroutine sync_from_change_pos(npart_in, nbodies_in, epsilon_in, fudgepsilon_in, xlong_in, boxsize2_in)
   implicit none

   integer(kind=4), intent(in) :: npart_in, nbodies_in
   real(kind=8), intent(in) :: epsilon_in, fudgepsilon_in, xlong_in, boxsize2_in

   npart = npart_in
   nbodies = nbodies_in
   epsilon = epsilon_in
   fudgepsilon = fudgepsilon_in
   xlong = xlong_in
   boxsize2 = boxsize2_in
end subroutine sync_from_change_pos

subroutine sync_from_init_adaptahop( &
            npart_in,nmembthresh_in,nMembers_in, &
            omegaL_in, omega_lambda_f_in,  &
            omega0_in, omega_f_in,  &
            aexp_max_in, af_in,  &
            hubble_in, H_f_in,  &
            boxsize2_in, Lf_in,  &
            xlong_in, ylong_in, zlong_in,  &
            xlongs2_in, ylongs2_in, zlongs2_in, &
            Hub_pt_in,aexp_in, &
            pos_shift_in)
   implicit none

   integer(kind=4), intent(in) :: npart_in,nmembthresh_in,nMembers_in
   real(kind=8), intent(in) :: omegaL_in, omega_lambda_f_in,  &
            omega0_in, omega_f_in,  &
            aexp_max_in, af_in,  &
            hubble_in, H_f_in,  &
            boxsize2_in, Lf_in,  &
            xlong_in, ylong_in, zlong_in,  &
            xlongs2_in, ylongs2_in, zlongs2_in, &
            Hub_pt_in,aexp_in
   real(kind=8), intent(in) :: pos_shift_in(3)

   npart = npart_in
   nmembthresh = nmembthresh_in
   nMembers = nMembers_in
   omegaL = omegaL_in
   omega_lambda_f = omega_lambda_f_in
   omega0 = omega0_in
   omega_f = omega_f_in
   aexp_max = aexp_max_in
   af = af_in
   hubble = hubble_in
   H_f = H_f_in
   boxsize2 = boxsize2_in
   Lf = Lf_in
   xlong = xlong_in
   ylong = ylong_in
   zlong = zlong_in
   xlongs2 = xlongs2_in
   ylongs2 = ylongs2_in
   zlongs2 = zlongs2_in
   Hub_pt = Hub_pt_in
   aexp = aexp_in
   pos_shift(1:3) = pos_shift_in(1:3)
   fsub=.true.

end subroutine sync_from_init_adaptahop

subroutine sync_others( &
   verbose_in, npart_in, nbPes_in,&
   rho_threshold_in, massp_in, boxsize_in, &
   nhop_in, nvoisins_in, &
   fudge_in, alphap_in, &
   method_in, nlevelmax_in)

   implicit none

   logical, intent(in) :: verbose_in
   integer(kind=4), intent(in) :: npart_in, nbPes_in
   integer(kind=4), intent(in) :: nhop_in, nvoisins_in
   real(kind=8), intent(in) :: rho_threshold_in, massp_in, boxsize_in
   real(kind=8), intent(in) :: fudge_in, alphap_in
   character(len=*), intent(in) :: method_in
   integer(kind=4), intent(in) :: nlevelmax_in

   verbose         = verbose_in
   npart           = npart_in
   nbPes           = nbPes_in
   rho_threshold   = rho_threshold_in
   massp           = massp_in
   boxsize         = boxsize_in

   nhop            = nhop_in
   nvoisins        = nvoisins_in
   fudge           = fudge_in
   alphap          = alphap_in
   
   method          = method_in(1:min(len(method), len(method_in)))
   nlevelmax       = nlevelmax_in

end subroutine sync_others


!=======================================================================
subroutine compute_mean_density_and_np
!=======================================================================
  use omp_lib
  implicit none

  integer(kind=4)                     :: ipar
  real(kind=8), dimension(0:nvoisins) :: dist2
  integer, dimension(nvoisins)        :: iparnei
  real(kind=8)                        :: densav
  integer(kind=4) :: tttt0, tttt1, ttttrate
  real(kind=8)    :: dtdtdtdt

  call system_clock(count=tttt0, count_rate=ttttrate)
  if (verbose) write(errunit,*) '    Compute mean density for each particle...'

  allocate(iparneigh(nhop,npart))
  allocate(density(npart))


  call omp_set_num_threads(nbPes)
  if(verbose) write(errunit,*) "    [OMP] compute density with ncore=",nbPes
  !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ipar,dist2,iparnei)
  !$OMP DO
  do ipar=1,npart
     call find_nearest_parts(ipar,dist2,iparnei)
     call compute_density(ipar,dist2,iparnei)
     iparneigh(1:nhop,ipar)=iparnei(1:nhop)
  enddo
  !$OMP END DO
  !$OMP END PARALLEL

! Check for average density
  if (verbose) then
     write(errunit,*) '        Calc average density...'
     densav=0.d0
     do ipar=1,npart
        densav=(densav*dble(ipar-1)+dble(density(ipar)))/dble(ipar)
     enddo
     write(errunit,*) '    --> Average density :',densav
  endif
  
  deallocate(mass_cell)
  deallocate(size_cell)
  deallocate(pos_cell)
  deallocate(sister)
  deallocate(firstchild)
  call system_clock(count=tttt1, count_rate=ttttrate)
  dtdtdtdt=real(tttt1-tttt0,8)/real(ttttrate,8)
  if (verbose) write(errunit,'(A,F10.2,A)') "     --> ",dtdtdtdt," seconds to compute mean density"

end subroutine compute_mean_density_and_np

!=======================================================================
subroutine find_local_maxima
!=======================================================================

  implicit none

  integer(kind=4)             :: ipar,idist,iparid,iparsel,igroup,nmembmax,nmembtot
  integer(kind=4),allocatable :: nmemb(:)
  real(kind=8)                :: denstest

  if (verbose) write(errunit,*) '    Now Find local maxima...'

  allocate(igrouppart(npart))

  idpart=0
  ngroups=0
  do ipar=1,npart
     denstest=density(ipar)
     if (denstest.gt.rho_threshold) then
        iparsel=ipar
        do idist=1,nhop
           iparid=iparneigh(idist,ipar)
           if (density(iparid).gt.denstest) then
              iparsel=iparid
              denstest=density(iparid)
           elseif (density(iparid).eq.denstest) then
              iparsel=min(iparsel,iparid)
              if (verbose) &
 &               write(errunit,*) '    WARNING : equal densities in find_local_maxima.'
           endif
        enddo
        if (iparsel.eq.ipar) then
           ngroups=ngroups+1
           idpart(ipar)=-ngroups
        else
           idpart(ipar)=iparsel
        endif
     endif
  enddo
 
  if (verbose) write(errunit,*) '    --> Number of local maxima found :',ngroups

! Now Link the particles associated to the same maximum
  if (verbose) write(errunit,*) '    Create link list...'

  allocate(densityg(ngroups))
  allocate(nmemb(ngroups))
  allocate(firstpart(ngroups))

  do ipar=1,npart
     if (density(ipar).gt.rho_threshold) then
        iparid=idpart(ipar)
        if (iparid.lt.0) densityg(-iparid)=density(ipar)
        do while (iparid.gt.0)
           iparid=idpart(iparid)
        enddo
        igrouppart(ipar)=-iparid
     else
        igrouppart(ipar)=0
     endif
  enddo

  nmemb(1:ngroups)=0
  firstpart(1:ngroups)=0
  do ipar=1,npart
     igroup=igrouppart(ipar)
     if (igroup.gt.0) then
        idpart(ipar)=firstpart(igroup)   
        firstpart(igroup)=ipar
        nmemb(igroup)=nmemb(igroup)+1
     endif
  enddo

  nmembmax=0
  nmembtot=0
  do igroup=1,ngroups
     nmembmax=max(nmembmax,nmemb(igroup))
     nmembtot=nmembtot+nmemb(igroup)
  enddo

  if (verbose) then
     write(errunit,*) '    --> Number of particles of the largest group :',nmembmax
     write(errunit,*) '    --> Total number of particles in groups ',nmembtot
  endif
  deallocate(nmemb)
  
end subroutine find_local_maxima

!=======================================================================
subroutine create_group_tree
!=======================================================================
  implicit none

  integer(kind=4) :: mass_loc,masstmp,igroupref
  integer(kind=4) :: igroup,igr1,igr2,inode
  integer(kind=4) :: ttt0, ttt1, tttrate
  real(kind=8)    :: dtdtdt
  real(kind=8)    :: rhot,posg(3),posref(3),rsquare,densmoy,truemass,truemasstmp

  if (verbose) write(errunit,*) '    Create the tree of structures of structures'

! End of the branches of the tree
  call system_clock(count=ttt0, count_rate=tttrate)
  call compute_saddle_list
  call system_clock(count=ttt1, count_rate=tttrate)
  dtdtdt=real(ttt1-ttt0,8)/real(tttrate,8)
  if (verbose) write(errunit,'(A,F10.2,A)') "     --> ",dtdtdt," seconds to saddle_list"

  if (verbose) write(errunit,*) '    Build the hierarchical tree'

  nnodesmax=2*ngroups
! Allocations
  allocate(node(0:nnodesmax))
  allocate(idgroup(ngroups))
  allocate(color(ngroups))
  allocate(igroupid(ngroups))
  allocate(idgroup_tmp(ngroups))

! Initializations
  liste_parts = 0

! Iterative loop to build the rest of the tree
  inode              = 0
  nnodes             = 0
  rhot               = rho_threshold
  node(inode)%mother = 0
  mass_loc           = 0
  truemass           = 0.d0
  igroupref          = 0
  do igroup=1,ngroups
     call treat_particles(igroup,rhot,posg,masstmp,igroupref,posref, &
&                          rsquare,densmoy,truemasstmp)
     mass_loc = mass_loc+masstmp
     truemass = truemass+truemasstmp
  enddo

  node(inode)%mass          = mass_loc
  node(inode)%truemass      = truemass
  node(inode)%radius        = 0.d0
  node(inode)%density       = 0.d0
  node(inode)%position(1:3) = 0.d0
  node(inode)%densmax       = maxval(densityg(1:ngroups))
  node(inode)%rho_saddle    = 0.
  node(inode)%level         = 0
  node(inode)%nsisters      = 0
  node(inode)%sister        = 0
  igr1 = 1
  igr2 = ngroups
  do igroup=1,ngroups
     idgroup(igroup)=igroup
     igroupid(igroup)=igroup
  enddo
  if (verbose) write(errunit,*) '    Create nodes ...'
  call system_clock(count=ttt0, count_rate=tttrate)
  call create_nodes(rhot,inode,igr1,igr2)
  call system_clock(count=ttt1, count_rate=tttrate)
  dtdtdt=real(ttt1-ttt0,8)/real(tttrate,8)
  if (verbose) write(errunit,'(A,F10.2,A)') "     --> ",dtdtdt," seconds to create nodes"

  deallocate(idgroup)
  deallocate(color)
  deallocate(igroupid)
  deallocate(idgroup_tmp)
  deallocate(idpart)
  deallocate(group)
  deallocate(densityg)
  deallocate(firstpart)

  call node2table
 
end subroutine create_group_tree

!=======================================================================
recursive subroutine create_nodes(rhot,inode,igr1,igr2)
!=======================================================================

  implicit none
  integer(kind=4)              :: inode
  real(kind=8)                 :: rhot
  integer(kind=4)              :: icolor,igr_eff
  integer(kind=4)              :: igroup, igr,igr1,igr2
  integer(kind=4)              :: inc_color_tot,inode1,mass_loc,masstmp,igroupref
  integer(kind=4)              :: inodeout,igr1out,igr2out,isisters,nsisters
  integer(kind=4)              :: mass_comp,icolor_ref
  real(kind=8)                 :: posg(3),posgtmp(3),posref(3),rhotout,rsquaretmp,rsquareg
  real(kind=8)                 :: densmoyg,densmoytmp
  real(kind=8)                 :: densmoy_comp_max,truemass,truemasstmp
  real(kind=8)                 :: posfin(3)
  real(kind=8)                 :: densmaxgroup
  integer(kind=4), allocatable :: igrpos(:),igrinc(:)
  integer(kind=4), allocatable :: igrposnew(:)
  integer(kind=4), allocatable :: massg(:)
  real(kind=8),  allocatable   :: truemassg(:)
  real(kind=8),  allocatable   :: densmaxg(:)
  real(kind=8),  allocatable   :: densmoy_comp_maxg(:)
  integer(kind=4), allocatable :: mass_compg(:)
  real(kind=8),  allocatable   :: posgg(:,:)
  real(kind=8),  allocatable   :: rsquare(:)
  real(kind=8),  allocatable   :: densmoy(:)
  logical, allocatable         :: ifok(:)  

  color(igr1:igr2)=0
! Percolate the groups
  icolor_select=0
  do igr=igr1,igr2
     igroup=idgroup(igr)
     if (color(igr).eq.0) then
        icolor_select=icolor_select+1
!!$        call do_colorize(icolor_select,igroup,igr,rhot)
        ncall=0                                               !YDdebug
        call do_colorize(igroup,igr,rhot) !YDdebug
        !write(errunit,'(A,3I8,e10.2,I8)')'End of  ',icolor_select,igroup,igr,rhot,ncall
     endif
  enddo

! We select only groups where we are sure of having at least one
! particle above the threshold density rhot
! Then sort them to gather them on the list
  allocate(igrpos(0:icolor_select))
  allocate(igrinc(1:icolor_select))
  igrpos(0)=igr1-1
  igrpos(1:icolor_select)=0
  do igr=igr1,igr2
     icolor=color(igr)
     igroup=idgroup(igr)
     if (densityg(igroup).gt.rhot) &
 &           igrpos(icolor)=igrpos(icolor)+1
  enddo
  do icolor=1,icolor_select
     igrpos(icolor)=igrpos(icolor-1)+igrpos(icolor)
  enddo
  if (igrpos(icolor_select)-igr1+1.eq.0) then
     write(errunit,*) 'ERROR in create_nodes :'
     write(errunit,*) 'All subgroups are below the threshold.'
!     STOP
  endif      

  igrinc(1:icolor_select)=0
  do igr=igr1,igr2
     icolor=color(igr)
     igroup=idgroup(igr)
     if (densityg(igroup).gt.rhot) then
        igrinc(icolor)=igrinc(icolor)+1
        igr_eff=igrinc(icolor)+igrpos(icolor-1)
        idgroup_tmp(igr_eff)=igroup
        igroupid(igroup)=igr_eff
     endif
  enddo
  igr2=igrpos(icolor_select)
  idgroup(igr1:igr2)=idgroup_tmp(igr1:igr2)

  inc_color_tot=0
  do icolor=1,icolor_select
     if (igrinc(icolor).gt.0) inc_color_tot=inc_color_tot+1
  enddo 
  allocate(igrposnew(0:inc_color_tot))
  igrposnew(0)=igrpos(0)
  inc_color_tot=0
  do icolor=1,icolor_select
     if (igrinc(icolor).gt.0) then
        inc_color_tot=inc_color_tot+1
        igrposnew(inc_color_tot)=igrpos(icolor)
     endif
  enddo
  deallocate(igrpos)
  deallocate(igrinc)

  isisters=0
  allocate(posgg(1:3,1:inc_color_tot))
  allocate(massg(1:inc_color_tot))
  allocate(truemassg(1:inc_color_tot))
  allocate(densmaxg(1:inc_color_tot))
  allocate(densmoy_comp_maxg(1:inc_color_tot))
  allocate(mass_compg(1:inc_color_tot))
  allocate(rsquare(1:inc_color_tot))
  allocate(densmoy(1:inc_color_tot))
  allocate(ifok(1:inc_color_tot))
  ifok(1:inc_color_tot)=.false.

  do icolor=1,inc_color_tot
     posg(1:3)=0.d0
     mass_loc=0
     truemass=0.d0
     rsquareg=0.d0
     densmoyg=0.d0
     igr1=igrposnew(icolor-1)+1
     igr2=igrposnew(icolor)
     densmaxgroup=-1.d0
     mass_comp=0
     densmoy_comp_max=-1.d0
     igroupref=0
     do igr=igr1,igr2
        igroup=idgroup(igr)
        densmaxgroup=max(densmaxgroup,densityg(igroup))
        call treat_particles(igroup,rhot,posgtmp,masstmp, &
 &                           igroupref,posref,rsquaretmp, &
 &                           densmoytmp,truemasstmp)
        posg(1)=posg(1)+posgtmp(1)
        posg(2)=posg(2)+posgtmp(2)
        posg(3)=posg(3)+posgtmp(3)
        rsquareg=rsquareg+rsquaretmp
        mass_loc=mass_loc+masstmp
        truemass=truemass+truemasstmp
        densmoyg=densmoyg+densmoytmp
        densmoytmp=densmoytmp/dble(masstmp)
        mass_comp=max(mass_comp,masstmp)
        if (masstmp > 0) then
           densmoy_comp_max=max(densmoy_comp_max, &
 &             densmoytmp/(1.d0+fudge/sqrt(dble(masstmp))))
        endif
     enddo
     massg(icolor)=mass_loc
     truemassg(icolor)=truemass
     posgg(1:3,icolor)=posg(1:3)
     densmaxg(icolor)=densmaxgroup
     densmoy_comp_maxg(icolor)=densmoy_comp_max
     mass_compg(icolor)=mass_comp
     rsquare(icolor)=sqrt(abs( &
 &             (truemass*rsquareg- &
 &             (posg(1)**2+posg(2)**2+posg(3)**2) )/ &
 &              truemass**2 ))
     densmoy(icolor)=densmoyg/dble(mass_loc)

     ifok(icolor)=mass_loc.ge.nmembthresh.and. &
 &      (densmoy(icolor).gt.rhot*(1.d0+fudge/sqrt(dble(mass_loc))).or. &
 &       densmoy_comp_maxg(icolor).gt.rhot).and. &
 &      densmaxg(icolor).ge.alphap*densmoy(icolor).and. &
 &      rsquare(icolor).ge.epsilon

     if (ifok(icolor)) then
        isisters=isisters+1
        icolor_ref=icolor
     endif
  enddo
  nsisters=isisters
  if (nsisters.gt.1) then
     isisters=0
     inode1=nnodes+1
     do icolor=1,inc_color_tot
        if (ifok(icolor)) then
           isisters=isisters+1
           nnodes=nnodes+1
           if (nnodes.gt.nnodesmax) then
              write(errunit,*) 'ERROR in create_nodes :'
              write(errunit,*) 'nnodes > nnodes max'
              STOP
           endif
         !   if (mod(nnodes,max(nnodesmax/10000,1)).eq.0.and.megaverbose) then
         !      write(errunit,*) 'nnodes=',nnodes
         !   endif
           node(nnodes)%mother=inode
           node(nnodes)%densmax=densmaxg(icolor)
           if (isisters.gt.1) then
              node(nnodes)%sister=nnodes-1
           else
              node(nnodes)%sister=0
           endif
           node(nnodes)%nsisters=nsisters
           node(nnodes)%mass=massg(icolor)
           node(nnodes)%truemass=truemassg(icolor)
           ! I think this is wrong!!! not mass_loc but massg(icolor) that should be checked against nmembthresh 
           !if (mass_loc.eq.0) then
           if (massg(icolor).eq.0) then
              write(errunit,*) 'ERROR in create_nodes :'
              write(errunit,*) 'NULL mass for nnodes=',nnodes
              STOP
           endif
           posfin(1:3)=real(posgg(1:3,icolor)/truemassg(icolor),8)
           node(nnodes)%radius=real(rsquare(icolor),8)    
           node(nnodes)%density=real(densmoy(icolor),8)
           if (posfin(1).ge.xlongs2) then
              posfin(1)=posfin(1)-xlong
           elseif (posfin(1).lt.-xlongs2) then
              posfin(1)=posfin(1)+xlong
           endif
           if (posfin(2).ge.ylongs2) then
              posfin(2)=posfin(2)-ylong
           elseif (posfin(2).lt.-ylongs2) then
              posfin(2)=posfin(2)+ylong
           endif
           if (posfin(3).ge.zlongs2) then
              posfin(3)=posfin(3)-zlong
           elseif (posfin(3).lt.-zlongs2) then
              posfin(3)=posfin(3)+zlong
           endif
           node(nnodes)%position(1:3)=posfin(1:3)
           node(nnodes)%rho_saddle=rhot
           node(nnodes)%level=node(inode)%level+1
         !   if (megaverbose.and.node(nnodes)%mass.ge.nmembthresh) then
         !      write(errunit,*) '*****************************************'
         !      write(errunit,*) 'new node :',nnodes
         !      write(errunit,*) 'level    :',node(nnodes)%level
         !      write(errunit,*) 'nsisters :',node(nnodes)%nsisters
         !      write(errunit,*) 'mass     :',node(nnodes)%mass
         !      write(errunit,*) 'true mass:',node(nnodes)%truemass
         !      write(errunit,*) 'radius   :',node(nnodes)%radius
         !      write(errunit,*) 'position :',node(nnodes)%position
         !      write(errunit,*) 'rho_saddl:',node(nnodes)%rho_saddle
         !      write(errunit,*) 'rhomax   :',node(nnodes)%densmax
         !      write(errunit,*) '*****************************************'
         !   endif
        endif
     enddo
     node(inode)%firstchild=nnodes
     inodeout=inode1
     do icolor=1,inc_color_tot
        if (ifok(icolor)) then
           igr1out=igrposnew(icolor-1)+1
           igr2out=igrposnew(icolor)
           do igr=igr1out,igr2out
              call paint_particles(idgroup(igr),inodeout,rhot)
           enddo
           rhotout=rhot*(1.d0+fudge/sqrt(dble(mass_compg(icolor))))
           if (igr2out.ne.igr1out) then
              call create_nodes(rhotout,inodeout,igr1out,igr2out)
           else
              node(inodeout)%firstchild=0
           endif
           inodeout=inodeout+1
        endif
     enddo
  elseif (nsisters.eq.1) then
     inodeout=inode
     rhotout=rhot*(1.d0+fudge/sqrt(dble(mass_compg(icolor_ref))))
     igr1out=igrposnew(0)+1
     igr2out=igrposnew(inc_color_tot)
     if (igr2out.ne.igr1out) then
        call create_nodes(rhotout,inodeout,igr1out,igr2out)
     else
        node(inode)%firstchild=0
     endif
  else
     node(inode)%firstchild=0
  endif
  deallocate(igrposnew)
  deallocate(posgg)
  deallocate(massg)
  deallocate(truemassg)
  deallocate(densmaxg)
  deallocate(densmoy_comp_maxg)
  deallocate(densmoy)
  deallocate(mass_compg)
  deallocate(rsquare)
  deallocate(ifok)

end subroutine create_nodes

!=======================================================================
subroutine paint_particles(igroup,inode,rhot)
!=======================================================================

  implicit none

  integer(kind=4) :: igroup,inode
  real(kind=8)    :: rhot
  integer(kind=4) :: ipar

  ipar=firstpart(igroup)
  do while (ipar.gt.0)
     if (density(ipar).gt.rhot) then
        liste_parts(ipar)=inode
     endif
     ipar=idpart(ipar)
  enddo
end subroutine paint_particles

!=======================================================================
subroutine treat_particles(igroup,rhot,posg,imass,igroupref,posref, &
&                          rsquare,densmoy,truemass)
!=======================================================================

  implicit none

  real(kind=8)    :: rhot
  real(kind=8)    :: posg(3),posref(3)
  real(kind=8)    :: posdiffx,posdiffy,posdiffz,rsquare,densmoy,truemass,xmasspart
  real(kind=8)    :: densmax,densmin
  integer(kind=4) :: imass,ipar,iparold,igroupref
  integer(kind=4) :: igroup
  logical         :: first_good

  imass=0
  truemass=0.d0
  rsquare=0.d0
  densmoy=0.d0
  posg(1:3)=0.d0
  ipar=firstpart(igroup)
  first_good=.false.
  do while (ipar.gt.0)
     if (density(ipar).gt.rhot) then
        if (.not.first_good) then
           if (igroupref.eq.0) then
              posref(1:3)=dble(pos(ipar,1:3))
              igroupref=igroup
           endif
           first_good=.true.
           firstpart(igroup)=ipar
           densmin=density(ipar)
           densmax=densmin
        else
           idpart(iparold)=ipar
        endif
        iparold=ipar
        imass=imass+1
        if(allocated(mass)) then
           xmasspart=mass(ipar)
        else
           xmasspart=massp
        end if
        truemass=truemass+xmasspart
        posdiffx=dble(pos(ipar,1))-posref(1)
        posdiffy=dble(pos(ipar,2))-posref(2)
        posdiffz=dble(pos(ipar,3))-posref(3)
        if (posdiffx.ge.xlongs2) then
           posdiffx=posdiffx-xlong
        elseif (posdiffx.lt.-xlongs2) then
           posdiffx=posdiffx+xlong
        endif
        if (posdiffy.ge.ylongs2) then
           posdiffy=posdiffy-ylong
        elseif (posdiffy.lt.-ylongs2) then
           posdiffy=posdiffy+ylong
        endif
        if (posdiffz.ge.zlongs2) then
           posdiffz=posdiffz-zlong
        elseif (posdiffz.lt.-zlongs2) then
           posdiffz=posdiffz+zlong
        endif
        posdiffx=posdiffx+posref(1)
        posdiffy=posdiffy+posref(2)
        posdiffz=posdiffz+posref(3)
        posg(1)=posg(1)+posdiffx*xmasspart
        posg(2)=posg(2)+posdiffy*xmasspart
        posg(3)=posg(3)+posdiffz*xmasspart
        rsquare=rsquare+xmasspart*(posdiffx**2+posdiffy**2+posdiffz**2)
        densmoy=densmoy+dble(density(ipar))
        densmax=max(densmax,density(ipar))
        densmin=min(densmin,density(ipar))
     endif
     ipar=idpart(ipar)
  enddo
  if (.not.first_good) firstpart(igroup)=0

  if (densmin.le.rhot.or.densmax.ne.densityg(igroup)) then
     write(errunit,*) 'ERROR in treat_particles'
     write(errunit,*) 'igroup, densmax, rhot=',igroup,densityg(igroup),rhot
     write(errunit,*) 'denslow, denshigh    =',densmin,densmax
     STOP
  endif

end subroutine treat_particles


!=======================================================================
subroutine do_colorize(igroup, igr, rhot)
   !=======================================================================
   implicit none
   integer(kind=4) :: igroup, igr
   integer(kind=4) :: ineig, igroup2, igr2, neig
   real(kind=8) :: rhot
   integer(kind=4), dimension(:), allocatable :: queue
   integer(kind=4) :: front, back
   
   ! Orginal do_colorize sometimes stuck maybe due to recursion depth
   ! do_colorize_norec stuck due to too large VIRT memory required
   ! Here code is suggested by ChatGPT, and I confirmed it returns consistent results.
 
   allocate(queue(ngroups))
   queue(1) = igroup
   front = 1
   back = 2
   color(igr) = icolor_select
 
   do while (front < back)
     igroup = queue(front)
     front = front + 1
     neig = group(igroup)%nhnei
     do ineig = 1, group(igroup)%nhnei
       if (group(igroup)%rho_saddle_gr(ineig) > rhot) then
         ! We connect this group to its neighbour
         igroup2 = group(igroup)%isad_gr(ineig)
         igr2 = igroupid(igroup2)
         if (color(igr2) == 0) then
           color(igr2) = icolor_select
           queue(back) = igroup2
           back = back + 1
         elseif (color(igr2) /= icolor_select) then
           write(*,*) 'ERROR in do_colorize : color(igr2) <> icolor_select'
           write(*,*) 'The connections are not symmetric.'
           STOP
         endif
       else
         ! We do not need this saddle anymore (and use the fact that saddles
         ! are ranked in decreasing order)
         neig = neig - 1
       endif
     enddo
     group(igroup)%nhnei = neig
   enddo
 
   deallocate(queue)
 end subroutine do_colorize
 
!=======================================================================
subroutine compute_saddle_list
!=======================================================================
! Compute the lowest density threshold below which each group is 
! connected to an other one
!=======================================================================

  implicit none
  integer(kind=4) :: ipar1,ipar2,igroup2,idist,igroup1,ineig2
  integer(kind=4) :: ineig
  integer(kind=4) :: neig,ineigal,in1,in2,idestroy,icon_count
  integer(kind=4) :: i
  real(kind=8)  :: density1,density2,rho_sad12
  logical :: exist
  logical, allocatable :: touch(:)
  integer, allocatable :: listg(:)
  real(kind=8),  allocatable :: rho_sad(:)  
  integer, allocatable :: isad(:)
  integer(kind=4), allocatable :: indx(:)

  if (verbose) write(errunit,*) '    Fill the end of the branches of the group tree'

! Allocate the array of nodes
  allocate(group(ngroups))

  if (verbose) write(errunit,*) '    First count the number of neighbourgs'// &
&                         ' of each elementary group...'

  allocate(touch(ngroups))
  allocate(listg(ngroups))

  touch(1:ngroups)=.false.


! First count the number of neighbourgs for each group to allocate
! arrays isad_in,isad_out,rho_saddle_gr
  do igroup1=1,ngroups
     ineig=0
     ipar1=firstpart(igroup1)
! Loop on all the members of the group
     do while (ipar1.gt.0)
        do idist=1,nhop
           ipar2=iparneigh(idist,ipar1)
           igroup2=igrouppart(ipar2)
! we test that we are in a group (i.e. that density(ipar) >= rho_threshold)
! and that this group is different from the one we are sitting on
           if (igroup2.gt.0.and.igroup2.ne.igroup1) then
              if (.not.touch(igroup2)) then
                 ineig=ineig+1
                 touch(igroup2)=.true.
                 listg(ineig)=igroup2
              endif
           endif
        enddo
! Next member
        ipar1=idpart(ipar1)
     enddo
! Reinitialize touch
     do in1=1,ineig
        igroup2=listg(in1)
        touch(igroup2)=.false.
     enddo
! Allocate the nodes 
     group(igroup1)%nhnei=ineig
     ineigal=max(ineig,1)
     allocate(group(igroup1)%isad_gr(1:ineigal))
     allocate(group(igroup1)%rho_saddle_gr(1:ineigal))
  enddo 

  if (verbose) write(errunit,*) '    Compute lists of neighbourgs and saddle points...'


! arrays isad_in,isad_out,rho_saddle_gr ! BOTTLENECK
  do igroup1=1,ngroups
! No calculation necessary if no neighbourg
     neig=group(igroup1)%nhnei
     if (neig.gt.0) then
        ineig=0
        ipar1=firstpart(igroup1)
        allocate(rho_sad(1:neig))
! Loop on all the members of the group
        do while (ipar1.gt.0)
           density1=density(ipar1)
           do idist=1,nhop
              ipar2=iparneigh(idist,ipar1)
              igroup2=igrouppart(ipar2)
! we test that we are in a group (i.e. that density(ipar) >= rho_threshold)
! and that this group is different from the one we are sitting on
              if (igroup2.gt.0.and.igroup2.ne.igroup1) then
                 density2=density(ipar2)
                 if (.not.touch(igroup2)) then
                    ineig=ineig+1
                    touch(igroup2)=.true.
                    listg(igroup2)=ineig
                    rho_sad12=min(density1,density2)
                    rho_sad(ineig)=rho_sad12
                    group(igroup1)%isad_gr(ineig)=igroup2
                 else
                    ineig2=listg(igroup2)
                    rho_sad12=min(density1,density2)
                    rho_sad(ineig2)=max(rho_sad(ineig2),rho_sad12)
                 endif
              endif
           enddo
! Next member
           ipar1=idpart(ipar1)
        enddo
        if (ineig.ne.neig) then
! Consistency checking
           write(errunit,*) 'ERROR in compute_saddle_list :'
           write(errunit,*) 'The number of neighbourgs does not match.'
           write(errunit,*) 'ineig, neig =',ineig,neig
           STOP
        endif
        group(igroup1)%rho_saddle_gr(1:ineig)=rho_sad(1:ineig)
        deallocate(rho_sad)
! Reinitialize touch
        do in1=1,ineig
           igroup2=group(igroup1)%isad_gr(in1)
           touch(igroup2)=.false.
        enddo
! No neighbourg
     endif
  enddo 

  deallocate(touch)
  deallocate(listg)

  if (verbose) write(errunit,*) '    Establish symmetry in connections...'

! Total number of connections count
  icon_count=0

! Destroy the connections between 2 groups which are not symmetric
! This might be rather slow and might be discarded later
  idestroy=0
  do igroup1=1,ngroups
     if (group(igroup1)%nhnei.gt.0) then
        do in1=1,group(igroup1)%nhnei
           exist=.false.
           igroup2=group(igroup1)%isad_gr(in1)
           if (igroup2.gt.0) then
              do in2=1,group(igroup2)%nhnei
                 if (group(igroup2)%isad_gr(in2).eq.igroup1) then
                    exist=.true.
                    rho_sad12=min(group(igroup2)%rho_saddle_gr(in2), &
 &                                group(igroup1)%rho_saddle_gr(in1))
                    group(igroup2)%rho_saddle_gr(in2)=rho_sad12
                    group(igroup1)%rho_saddle_gr(in1)=rho_sad12
                 endif
              enddo
           endif
           if (.not.exist) then
              group(igroup1)%isad_gr(in1)=0
              idestroy=idestroy+1
           else
              icon_count=icon_count+1
           endif
        enddo
     endif
  enddo

  if (verbose) write(errunit,*) '    --> Number of connections removed :',idestroy
  if (verbose) write(errunit,*) '    --> Total number of connections remaining :',icon_count

  if (verbose) then
     write(errunit,*) '    Rebuild groups with undesired connections removed...'
  endif


! Rebuild the group list correspondingly with the connections removed
! And sort the list of saddle points
  do igroup1=1,ngroups
     neig=group(igroup1)%nhnei
     if (neig.gt.0) then
        allocate(rho_sad(neig))
        allocate(isad(neig))
        ineig=0
        do in1=1,neig
           igroup2=group(igroup1)%isad_gr(in1)
           if (igroup2.gt.0) then
              ineig=ineig+1
              rho_sad(ineig)=group(igroup1)%rho_saddle_gr(in1)
              isad(ineig)=igroup2
           endif
        enddo
        deallocate(group(igroup1)%isad_gr)
        deallocate(group(igroup1)%rho_saddle_gr)
        ineigal=max(ineig,1)
        allocate(group(igroup1)%isad_gr(ineigal))
        allocate(group(igroup1)%rho_saddle_gr(ineigal))
        group(igroup1)%nhnei=ineig
        if (ineig.gt.0) then
! sort the saddle points by decreasing order
           allocate(indx(ineig))
           call indexx(ineig,rho_sad,indx)
           do i=1,ineig
              ineig2=indx(i)
              group(igroup1)%isad_gr(ineig-i+1)=isad(ineig2)
              group(igroup1)%rho_saddle_gr(ineig-i+1)=rho_sad(ineig2)
           enddo
           deallocate(indx)
        endif
        deallocate(rho_sad)
        deallocate(isad)
     endif
  enddo

  deallocate(iparneigh)
  deallocate(igrouppart)

end subroutine compute_saddle_list

!=======================================================================
subroutine compute_density(ipar,dist2,iparnei)
!=======================================================================

  implicit none
  real(kind=8)          :: dist2(0:nvoisins)
  integer(kind=4)       :: iparnei(nvoisins)
  real(kind=8)          :: r,unsr,contrib
  real(kind=8),external :: spline
  integer(kind=4)       :: idist,ipar

  r=sqrt(dist2(nvoisins))*0.5d0
  unsr=1.d0/r
  contrib=0.d0
  do idist=1,nvoisins-1
     if(allocated(mass)) then
        contrib=contrib+mass(iparnei(idist))*spline(sqrt(dist2(idist))*unsr)
     else
        contrib=contrib+massp*spline(sqrt(dist2(idist))*unsr)
     end if
  enddo
! Add the contribution of the particle itself and normalize properly
! to get a density with average unity (if computed on an uniform grid)
! note that this assumes that the total mass in the box is normalized to 1.
  if(allocated(mass)) then
     density(ipar)=(xlong*ylong*zlong)*(contrib+mass(ipar)) &
          &              /(pi*r**3)
  else
     density(ipar)=(xlong*ylong*zlong)*(contrib+massp) &
          &              /(pi*r**3)
  end if

end subroutine compute_density

!=======================================================================
subroutine find_nearest_parts(ipar,dist2,iparnei)
!=======================================================================
! For ID(ipar) particle, find `nvoisins` nearest particles
! and store their distances in dist2 and their IDs in iparnei
  implicit none

  integer(kind=4) :: ipar,idist,icell_identity,inccellpart
  real(kind=8)    :: dist2(0:nvoisins)
  integer(kind=4) :: iparnei(nvoisins)
  real(kind=8)    :: poshere(1:3)

  poshere(1:3)=pos(ipar,1:3)
  dist2(0)=0.
  do idist=1,nvoisins
     dist2(idist)=bignum
  enddo
  icell_identity =1
  inccellpart    =0
  call walk_tree(icell_identity,poshere,dist2,ipar,inccellpart,iparnei)

end subroutine find_nearest_parts

!=======================================================================
recursive subroutine walk_tree(icellidin,poshere,dist2,    &
 &                   iparid,inccellpart,iparnei)
!=======================================================================
!The subroutine 
!calculates the distances between the particle and cells (or the particles within cells),
!stores the closest cells/particles in `discell2` and `icid` arrays, and
!updates `dist2` and `iparnei` arrays with the closest particles.
!The subroutine recursively calls itself with the children cells of the current cell until it reaches the leaf cells that contain particles.
!The algorithm is an implementation of a tree structure, where cells are nodes and the particles are leaves.
!
! - by ChatGPT Jan 9
  implicit none


  integer(kind=4) :: icellidin,icell_identity,iparid,inccellpart,ic,iparcell
  real(kind=8)    :: poshere(3),dist2(0:nvoisins)
  real(kind=8)    :: dx,dy,dz,distance2,sc
  integer(kind=4) :: idist,inc
  integer(kind=4) :: icellid_out
  real(kind=8)    :: discell2(0:8)
  integer(kind=4) :: iparnei(nvoisins)
  integer(kind=4) :: icid(8)

  integer(kind=4) :: i,npart_pos_this_node
  real(kind=8)    :: distance2p

  icell_identity=firstchild(icellidin)
  inc=1
  discell2(0)=0
  discell2(1:8)=1.d30
  ! Until icell_identity==0: (Final leaf of tree)
  ! Calc distance (poshere <-> cells)
  do while (icell_identity.ne.0)
     sc=size_cell(icell_identity)
     dx=abs(pos_cell(1,icell_identity)-poshere(1))
     dy=abs(pos_cell(2,icell_identity)-poshere(2))
     dz=abs(pos_cell(3,icell_identity)-poshere(3))
     dx=max(0.,min(dx,real(xlong,8)-dx)-sc)
     dy=max(0.,min(dy,real(ylong,8)-dy)-sc)
     dz=max(0.,min(dz,real(zlong,8)-dz)-sc)
     distance2=dx**2+dy**2+dz**2
     if (distance2.lt.dist2(nvoisins)) then
        idist=inc-1
        do while (discell2(idist).gt.distance2)
           discell2(idist+1)=discell2(idist)
           icid(idist+1)=icid(idist)
           idist=idist-1
        enddo
        discell2(idist+1)=distance2
        icid(idist+1)=icell_identity
        inc=inc+1
     endif
     icell_identity=sister(icell_identity)
  enddo
  inccellpart=inccellpart+inc-1
  ! Loop for counted cells,
  ! Update the closest particle ID(in iparnei) and the distance to that part(in dist2)
  do ic=1,inc-1
     icellid_out=icid(ic)
     if (firstchild(icellid_out) < 0) then ! i.e., if leaf cell
        if (discell2(ic).lt.dist2(nvoisins)) then ! Check if cell is closer than the farthest known particle
           npart_pos_this_node=-firstchild(icellid_out)-1
           do i=npart_pos_this_node+1,npart_pos_this_node+mass_cell(icellid_out)
              iparcell=idpart(i)
              dx=abs(pos(iparcell,1)-poshere(1))
              dx=max(0.,min(dx,real(xlong,8)-dx))
              dy=abs(pos(iparcell,2)-poshere(2))
              dy=max(0.,min(dy,real(ylong,8)-dy))
              dz=abs(pos(iparcell,3)-poshere(3))
              dz=max(0.,min(dz,real(zlong,8)-dz))
              distance2p=dx**2+dy**2+dz**2
              if (distance2p .lt. dist2(nvoisins)) then 
                 if (iparcell.ne.iparid) then
                    idist=nvoisins-1
                    do while (dist2(idist).gt.distance2p)
                       dist2(idist+1)=dist2(idist)
                       iparnei(idist+1)=iparnei(idist)
                       idist=idist-1
                    enddo
                    dist2(idist+1)=distance2p
                    iparnei(idist+1)=iparcell
                 endif
              endif
           enddo
        endif              
     elseif (discell2(ic).lt.dist2(nvoisins)) then
        call walk_tree(icellid_out,poshere,dist2,iparid,inccellpart,iparnei)
     endif
  enddo
end subroutine walk_tree

!=======================================================================
subroutine create_tree_structure
!=======================================================================
   implicit none
   integer(kind=4) :: nlevel,inccell,idmother,ipar
   integer(kind=4) :: npart_this_node,npart_pos_this_node
   integer(kind=4) :: ncell
   real(kind=8)    :: pos_this_node(3)
   integer(kind=4) :: ttt0, ttt1, tttrate
   real(kind=8)    :: dtdtdt

   if (verbose) write(errunit,*) '    Create tree structure...'

   ! we modified to put 2*npart-1 instead of 2*npart so that AdaptaHOP can work on a 1024^3, 2*(1024^3)-1 is still an integer(kind=4), 2*(1024^3) is not 
   ncellmx=2*npart -1
   ncellbuffer=max(nint(0.1*npart),ncellbuffermin)
   allocate(idpart(npart))
   allocate(idpart_tmp(npart))
   allocate(mass_cell(ncellmx))
   allocate(size_cell(ncellmx))
   allocate(pos_cell(3,ncellmx))
   allocate(sister(1:ncellmx))
   allocate(firstchild(1:ncellmx))
   do ipar=1,npart
      idpart(ipar)=ipar
   enddo
   nlevel=0
   inccell=0
   idmother=0
   pos_this_node(1:3)=0.
   npart_this_node=npart
   npart_pos_this_node=0
   idpart_tmp(1:npart)=0
   pos_cell(1:3,1:ncellmx)=0.
   size_cell(1:ncellmx)=0.
   mass_cell(1:ncellmx)=0
   sister(1:ncellmx)=0
   firstchild(1:ncellmx)=0
   sizeroot=real(max(xlong,ylong,zlong),8)
   call system_clock(count=ttt0, count_rate=tttrate)
   
   call create_KDtree(nlevel,pos_this_node, &
   &                   npart_this_node,npart_pos_this_node,inccell, &
   &                   idmother)
   ncell=inccell

   call system_clock(count=ttt1, count_rate=tttrate)
   dtdtdt=real(ttt1-ttt0,8)/real(tttrate,8)
   
   if (verbose) write(errunit,*) '    --> total number of cells =',ncell
   if (verbose) write(errunit,'(A,F10.2,A)') "     --> ",dtdtdt," seconds to create the tree structure"

   deallocate(idpart_tmp)

end subroutine create_tree_structure


!=======================================================================
recursive subroutine create_KDtree(nlevel,pos_this_node,npart_this_node, &
 &                npart_pos_this_node,inccell,idmother)
!=======================================================================
!  nlevel : level of the node in the octree. Level zero corresponds to 
!           the full box
!  pos_this_node : position of the center of this node
!  npart  : total number of particles 
!  idpart : array of dimension npart containing the id of each
!           particle. It is sorted such that neighboring particles in
!           this array belong to the same cell node.
!  idpart_tmp : temporary array of same size used as a buffer to sort
!           idpart.
!  npart_this_node : number of particles in the considered node
!  npart_pos_this_node : first position in idpart of the particles 
!           belonging to this node
!  pos :    array of dimension 3.npart giving the positions of 
!           each particle belonging to the halo
!  inccell : cell id number for the newly created structured grid site
!  pos_cell : array of dimension 3.npart (at most) corresponding
!           to the positions of each node of the structured grid
!  mass_cell : array of dimension npart (at most) giving the 
!           number of particles in each node of the structured grid
!  size_cell : array of dimension npart (at most) giving half the
!           size of the cube forming each node of the structured grid
!  sizeroot : size of the root cell (nlevel=0)
!  idmother : id of the mother cell
!  sister   : sister of a cell (at the same level, with the same mother)
!  firstchild : first child of a cell (then the next ones are found 
!             with the array sister). If it is a cell containing only
!           one particle, it gives the id of the particle.
!  ncellmx : maximum number of cells
!  megaverbose : detailed verbose mode
!=======================================================================

   implicit none

   integer(kind=4)           :: nlevel,npart_pos_this_node,npart_this_node
   real(kind=8)              :: pos_ref(3,0:7)
   real(kind=8)              :: pos_this_node(3)   
   integer(kind=4)           :: ipar,icid,j,inccell,nlevel_out
   integer(kind=4), external :: icellid
   integer(kind=4)           :: npart_pos_this_node_out,npart_this_node_out
   integer(kind=4)           :: incsubcell(0:7),nsubcell(0:7)
   real(kind=8)              :: xtest(3),pos_this_node_out(3)
   integer(kind=4)           :: idmother,idmother_out

   integer(kind=4)           :: ncellmx_old
   integer, allocatable      :: mass_cell_tmp(:),sister_tmp(:),firstchild_tmp(:)
   real(kind=8), allocatable :: size_cell_tmp(:),pos_cell_tmp(:,:)

!  pos_ref : an array used to find positions of the 8 subcells in this
!           node.
   data  pos_ref /-1.,-1.,-1., &
 &                 1.,-1.,-1., &
 &                -1., 1.,-1., &
 &                 1., 1.,-1., &
 &                -1.,-1., 1., &
 &                 1.,-1., 1., &
 &                -1., 1., 1., &
 &                 1., 1., 1.  / 


   if (npart_this_node.gt.0) then
      inccell=inccell+1
      if (mod(inccell,1000000).eq.0.and.megaverbose) write(errunit,*) '    inccell=',inccell,nlevelmax
      if (inccell.gt.ncellmx) then
         ncellmx_old=ncellmx
         ncellmx=ncellmx+ncellbuffer
         if (megaverbose) write(errunit,*) &
 &          'ncellmx is too small. Increase it and reallocate arrays accordingly'
         allocate(mass_cell_tmp(ncellmx_old))
         mass_cell_tmp(1:ncellmx_old)=mass_cell(1:ncellmx_old)
         deallocate(mass_cell)
         allocate(mass_cell(ncellmx))
         mass_cell(1:ncellmx_old)=mass_cell_tmp(1:ncellmx_old)
         deallocate(mass_cell_tmp)
         allocate(sister_tmp(ncellmx_old))
         sister_tmp(1:ncellmx_old)=sister(1:ncellmx_old)
         deallocate(sister)
         allocate(sister(ncellmx))
         sister(1:ncellmx_old)=sister_tmp(1:ncellmx_old)
         deallocate(sister_tmp)
         allocate(firstchild_tmp(ncellmx_old))
         firstchild_tmp(1:ncellmx_old)=firstchild(1:ncellmx_old)
         deallocate(firstchild)
         allocate(firstchild(ncellmx))
         firstchild(1:ncellmx_old)=firstchild_tmp(1:ncellmx_old)
         firstchild(ncellmx_old:ncellmx)=0
         deallocate(firstchild_tmp)
         allocate(size_cell_tmp(ncellmx_old))
         size_cell_tmp(1:ncellmx_old)=size_cell(1:ncellmx_old)
         deallocate(size_cell)
         allocate(size_cell(ncellmx))
         size_cell(1:ncellmx_old)=size_cell_tmp(1:ncellmx_old)
         deallocate(size_cell_tmp)
         allocate(pos_cell_tmp(3,ncellmx_old))
         pos_cell_tmp(1:3,1:ncellmx_old)=pos_cell(1:3,1:ncellmx_old)
         deallocate(pos_cell)
         allocate(pos_cell(3,ncellmx))
         pos_cell(1:3,1:ncellmx_old)=pos_cell_tmp(1:3,1:ncellmx_old)
         deallocate(pos_cell_tmp)
      endif
      pos_cell(1:3,inccell)=pos_this_node(1:3)
      mass_cell(inccell)=npart_this_node
      size_cell(inccell)=2.**(-nlevel)*sizeroot*0.5d0
      if (idmother.gt.0) then
         sister(inccell)=firstchild(idmother)
         firstchild(idmother)=inccell
      endif
!     If there is only one particle in the node or we have reach
!     maximum level of refinement, we are done
      if ((npart_this_node <= npartpercell).or.(nlevel.eq.nlevelmax)) then
         firstchild(inccell)=-(npart_pos_this_node+1)
         return
      endif
   else
      return
   endif


!  Count the number of particles in each subcell of this node
   incsubcell(0:7)=0
   do ipar=npart_pos_this_node+1,npart_pos_this_node+npart_this_node
      xtest(1:3)=pos(idpart(ipar),1:3)-pos_this_node(1:3)
      icid=icellid(xtest)
      incsubcell(icid)=incsubcell(icid)+1
   enddo

!  Create the array of positions of the first particle of the lists
!  of particles belonging to each subnode
   nsubcell(0)=0
   do j=1,7
      nsubcell(j)=nsubcell(j-1)+incsubcell(j-1)
   enddo

!  Sort the array of ids (idpart) to gather the particles belonging
!  to the same subnode. Put the result in idpart_tmp.
   incsubcell(0:7)=0
   do ipar=npart_pos_this_node+1,npart_pos_this_node+npart_this_node
      xtest(1:3)=pos(idpart(ipar),1:3)-pos_this_node(1:3)
      icid=icellid(xtest)
      incsubcell(icid)=incsubcell(icid)+1
      idpart_tmp(incsubcell(icid)+nsubcell(icid)+npart_pos_this_node)=idpart(ipar)
   enddo
   
!  Put back the sorted ids in idpart
   do ipar=npart_pos_this_node+1,npart_pos_this_node+npart_this_node
      idpart(ipar)=idpart_tmp(ipar)
   enddo

!  Call again the routine for the 8 subnodes:
!  Compute positions of subnodes, new level of refinement, 
!  positions in the array idpart corresponding to the subnodes,
!  and call for the treatment recursively.
   nlevel_out=nlevel+1
   idmother_out=inccell
   do j=0,7
      pos_this_node_out(1:3)=pos_this_node(1:3)+ &
 &                           sizeroot*pos_ref(1:3,j)*2.**(-nlevel-2)
      npart_pos_this_node_out=npart_pos_this_node+nsubcell(j)
      npart_this_node_out=incsubcell(j)
      call create_KDtree(nlevel_out,pos_this_node_out,npart_this_node_out, &
 &                npart_pos_this_node_out,inccell,idmother_out)
   enddo
end subroutine create_KDtree


!=======================================================================

!#####################################################################
subroutine close()
!#####################################################################
   ! Clean old data table is used more then once
   implicit none
   if(allocated(real_table)) deallocate(real_table)
   if(allocated(integer_table)) deallocate(integer_table)
   if(allocated(density)) deallocate(density)
   if(allocated(liste_parts)) deallocate(liste_parts)
end subroutine close

end module neiKDtree


!=======================================================================
function spline(x)
!=======================================================================
  implicit none
  real(kind=8) :: spline,x

  if (x.le.1.) then
     spline=1.d0 - 1.5d0*x**2 + 0.75d0*x**3
  elseif (x.le.2.) then
     spline=0.25d0*(2.d0 - x)**3
  else
     spline=0.
  endif

end function spline


!=======================================================================
function icellid(xtest)
!=======================================================================
!  Compute cell id corresponding to the signs of coordinates of xtest
!  as follows :
!  (-,-,-) : 0
!  (+,-,-) : 1
!  (-,+,-) : 2
!  (+,+,-) : 3
!  (-,-,+) : 4
!  (+,-,+) : 5
!  (-,+,+) : 6
!  (+,+,+) : 7
!  For self-consistency, the array pos_ref should be defined exactly 
!  with the same conventions
!=======================================================================
   implicit none
   integer(kind=4) :: icellid,j,icellid3d(3)
   real(kind=8) :: xtest(3)
   do j=1,3
      if (xtest(j).ge.0) then
         icellid3d(j)=1
      else
         icellid3d(j)=0
      endif
   enddo
   icellid=icellid3d(1)+2*icellid3d(2)+4*icellid3d(3)
end function icellid














