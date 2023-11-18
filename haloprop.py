import numpy as np


gravconst = 430.1               # G in units of (km/s)^2 Mpc/(10^11 Msol)



def Execute(halo, member):
    snap = member.snap
    h = initialize(halo, member)
    print(h.keys())
    h, v = det_main_axis(h)
    print(h.keys())
    h = det_vir_props(h, v=v)
    print(h.keys())

def colllapse(age, omega_maxexp, tol):
    pass

def cubic(a,b,c,d):
    pass

def virial(snap):
    age       = snap.age_univ/977.78*snap.H0
    # compute the overdensity needed to reach maximum expansion by half the age of the universe
    omega_maxexp = snap.omega_m
    collapse(age/2.0,omega_maxexp,1e-6)
    # calculate how far an object collapses to virial equilibrium
    eta = 2.0*snap.omega_l/omega_maxexp*(1/snap.aexp)**3
    if (eta == 0.0) then
       redc = 0.5
    else
       a      = 2.0*eta
       b      = -(2.0+eta)
       redc = cubic(a,0.0,b,1.0)
    end if
    
    vir_overdens = omega_maxexp/omega_f/redc**3*(aexp/af)**3
    Lboxp          = snap.boxlen*snap.unit_l/3.08e24/snap.aexp
    Lbox_pt  = Lboxp*snap.aexp
    rho_mean     = mboxp/Lbox_pt**3




def initialize(halo, member):
    snap = member.snap
    Lboxp          = snap.boxlen*snap.unit_l/3.08e24/snap.aexp
    Lbox_pt  = Lboxp*snap.aexp
    Lbox_pt2 = Lbox_pt / 2.0
    h = {}
    h['id'] = halo['id']
    h['x'] = halo['x'] - 0.5
    h['y'] = halo['y'] - 0.5
    h['z'] = halo['z'] - 0.5
    h['vx'] = halo['vx']
    h['vy'] = halo['vy']
    h['vz'] = halo['vz']
    member['x'] -= 0.5
    member['y'] -= 0.5
    member['z'] -= 0.5
    member['vx'] *= snap.unit_l / snap.unit_t * 1e-5
    member['vy'] *= snap.unit_l / snap.unit_t * 1e-5
    member['vz'] *= snap.unit_l / snap.unit_t * 1e-5
    member['m'] = member['m', 'Msol'] / 1e11
    h['m'] = np.sum(member['m'])
    dist = np.sqrt((h['x'] - member['x'])**2 + (h['y'] - member['y'])**2 + (h['z'] - member['z'])**2)
    h['r'] = np.max(dist)
    h['member'] = member
    h['snap'] = snap
    h['box_size'] = Lbox_pt2
    return h

def det_main_axis(h):
    mat = det_inertial_tensor(h, h['member'])
    d, v = jacobi(mat)
    d = np.sqrt(d / h['m'])
    a,b,c = d
    h['a'] = a
    h['b'] = b
    h['c'] = c
    return h, v

def correct_for_periodicity(pos, box_size=None):
    ind = np.where(pos > 0.5 * box_size)
    if(True in ind): pos[ind] -= box_size
    ind = np.where(pos < -0.5 * box_size)
    if(True in ind): pos[ind] += box_size
    return pos


def det_inertial_tensor(h, member):
    xs = correct_for_periodicity(member['x'] - h['x'], box_size=h['box_size'])
    ys = correct_for_periodicity(member['y'] - h['y'], box_size=h['box_size'])
    zs = correct_for_periodicity(member['z'] - h['z'], box_size=h['box_size'])
    dr = [xs, ys, zs]
    mat = np.zeros((3,3))
    for i in range(3):
        ri = dr[i]
        for j in range(3):
            rj = dr[j]
            mat[i,j] = np.sum(member['m'] * ri * rj) 
    return mat

def jacobi(a, nmax=500, itermax=50):
    n = a.shape[0]
    assert n == a.shape[1]
    if(n>nmax): raise BufferError(f"n(={n}) is too small (nmax={nmax})")
    b = np.diag(a).copy()
    z=np.zeros(n)

    d = np.diag(a).copy()
    v = np.eye(n)
    nbinot = 0
    for i in range(itermax):
        sm = np.sum(np.abs(np.triu(a, k=1)))
        if(sm == 0): return d, v
        tresh = 0.2*sm/(n**2) if(i<4) else 0.0

        for ip in range(n-1):
            for iq in range(ip+1,n):
                g = 100*abs(a[ip,iq])
                if (i > 4) and (abs(d[ip])+g == abs(d[ip])) and (abs(d[iq])+g == abs(d[iq])):
                    a[ip,iq] = 0.0
                elif (abs(a[ip,iq]) > tresh):
                    h = d[iq] - d[ip]
                    if (abs(h)+g == abs(h)):
                        t = a[ip,iq]/h
                    else:
                        theta = 0.5*h/a[ip,iq]
                        t = 1.0/(abs(theta)+np.sqrt(1.0+theta**2))
                        if(theta < 0): t = -t
                    c = 1.0/np.sqrt(1+t**2)
                    s = t*c
                    tau = s/(1.0+c)
                    h = t*a[ip,iq]
                    z[ip] = z[ip] - h
                    z[iq] = z[iq] + h
                    d[ip] = d[ip] - h
                    d[iq] = d[iq] + h
                    a[ip,iq] = 0.0
                    for j in range(ip):
                        g = a[j,ip]
                        h = a[j,iq]
                        a[j,ip] = g - s*(h+g*tau)
                        a[j,iq] = h + s*(g-h*tau)
                    for j in range(ip+1,iq):
                        g = a[ip,j]
                        h = a[j,iq]
                        a[ip,j] = g - s*(h+g*tau)
                        a[j,iq] = h + s*(g-h*tau)
                    for j in range(iq+1,n):
                        g = a[ip,j]
                        h = a[iq,j]
                        a[ip,j] = g - s*(h+g*tau)
                        a[iq,j] = h + s*(g-h*tau)
                    for j in range(n):
                        g = v[j,ip]
                        h = v[j,iq]
                        v[j,ip] = g - s*(h+g*tau)
                        v[j,iq] = h + s*(g-h*tau)
                    nbinot = nbinot + 1
        for ip in range(n):
            b[ip] = b[ip] + z[ip]
            d[ip] = b[ip]
            z[ip] = 0.0
    raise RuntimeError(f"Too many iterations ({itermax}) in routine jacobi")
    


def rf(x, y, z):
    errtol = 0.08
    tiny = 1.5e-38
    big = 3.e37
    third = 1. / 3.
    c1 = 1. / 24.
    c2 = 0.1
    c3 = 3. / 44.
    c4 = 1. / 14.

    if min(x, y, z) < 0.0 or min(x + y, x + z, y + z) < tiny or max(x, y, z) > big:
        raise ValueError("Invalid arguments in rf")

    xt = x
    yt = y
    zt = z

    while True:
        sqrtx = np.sqrt(xt)
        sqrty = np.sqrt(yt)
        sqrtz = np.sqrt(zt)
        alamb = sqrtx * (sqrty + sqrtz) + sqrty * sqrtz
        xt = 0.25 * (xt + alamb)
        yt = 0.25 * (yt + alamb)
        zt = 0.25 * (zt + alamb)
        ave = third * (xt + yt + zt)
        delx = (ave - xt) / ave
        dely = (ave - yt) / ave
        delz = (ave - zt) / ave
        if max(abs(delx), abs(dely), abs(delz)) > errtol:
            continue
        e2 = delx * dely - delz ** 2
        e3 = delx * dely * delz
        return (1. + (c1 * e2 - c2 - c3 * e3) * e2 + c4 * e3) / np.sqrt(ave)




def tab_props_inside(h, nbin, v):
    # Returns the cumulative mass contained in concentric ellipsoids centered on the center of the halo
    # Rescale to get ellipsoid concentric to the principal ellipsoid containing all the particles of the halo
    rmax = 0.0

    pxs = correct_for_periodicity(h['member']['x'] - h['x'], box_size=h['box_size'])
    pys = correct_for_periodicity(h['member']['y'] - h['y'], box_size=h['box_size'])
    pzs = correct_for_periodicity(h['member']['z'] - h['z'], box_size=h['box_size'])
    posp = np.vstack((pxs, pys, pzs)).T
    dras = np.dot(posp, v[0])
    drbs = np.dot(posp, v[1])
    drcs = np.dot(posp, v[2])
    r_ells = np.sqrt((dras / h['a'])**2 + (drbs / h['b'])**2 + (drcs / h['c'])**2)
    rmax = np.max(r_ells)
    
    amax = rmax * h['a'] * (1.0 + 1e-2)
    bmax = rmax * h['b'] * (1.0 + 1e-2)
    cmax = rmax * h['c'] * (1.0 + 1e-2)

    # Initialize loop quantities
    louped_parts = 0
    num_h = h['id']
    snap = h['snap']
    Hub_pt = snap.H0 * np.sqrt(snap.omega_m*(1/snap.aexp)**3 +  snap.omega_k*(1/snap.aexp)**2 + snap.omega_l)
    pvxs = h['member']['vx'] - h['vx'] + pxs * Hub_pt
    pvys = h['member']['vy'] - h['vy'] + pys * Hub_pt
    pvzs = h['member']['vz'] - h['vz'] + pzs * Hub_pt
    vt = np.vstack((pvxs, pvys, pvzs)).T
    v2s = np.sum(vt**2, axis=1)
    dras = np.dot(posp, v[0])
    drbs = np.dot(posp, v[1])
    drcs = np.dot(posp, v[2])
    r_ells = np.sqrt((dras / amax)**2 + (drbs / bmax)**2 + (drcs / cmax)**2)

    tabm2,_ = np.histogram(r_ells, bins=nbin, weights=h['member']['m'])
    tabk2,_ = np.histogram(r_ells, bins=nbin, weights=0.5 * h['member']['m'] * v2s)



    srm = np.cumsum(tabm2)
    srk = np.cumsum(tabk2)
    tabm2 = srm
    tabk2 = srk
    tabp2 = np.zeros(nbin)
    tabp2[:-1] = -0.3 * gravconst * tabm2[:-1]**2 * rf(h['a']**2, h['b']**2, h['c']**2)

    # if h['ep'] != tabp2[nbin - 1]:
    #     tabp2 = tabp2 / tabp2[nbin - 1] * h['ep']

    return amax,bmax,cmax, tabm2, tabk2, tabp2


def det_vir_props(h, v=None):
    # Computes the virial properties (radius, mass) of a halo

    nbin = 1000
    rvir, mvir, kvir, pvir = 0.0, 0.0, 0.0, 0.0

    # Compute properties inside nbin concentric principal ellipsoids centered on center of halo
    amax,bmax,cmax, tab_mass, tab_ekin, tab_epot = tab_props_inside(h, nbin, v)

    # Find the outermost ellipsoid bin where the virial theorem is satisfied best
    i_arr = np.arange(nbin - 1, 0, -1)
    exit_mask = (tab_mass[i_arr - 1] < tab_mass[i_arr])
    i = i_arr[exit_mask][0]

    avir = i / (nbin - 1) * amax
    bvir = i / (nbin - 1) * bmax
    cvir = i / (nbin - 1) * cmax
    rvir = (avir * bvir * cvir) ** (1./3.)

    virth = abs((2.0 * tab_ekin + tab_epot) / (tab_ekin + tab_epot))
    idx_min = np.argmin(virth)

    mvir = tab_mass[idx_min]
    avir = np.minimum(avir, i_arr[idx_min] / (nbin - 1) * amax)
    bvir = np.minimum(bvir, i_arr[idx_min] / (nbin - 1) * bmax)
    cvir = np.minimum(cvir, i_arr[idx_min] / (nbin - 1) * cmax)
    rvir = (avir * bvir * cvir) ** (1./3.)
    kvir = tab_ekin[idx_min]
    pvir = tab_epot[idx_min]

    virth_min = virth[idx_min]

    if virth_min > 0.20 or mvir < vir_overdens * rho_mean * (4./3. * np.pi) * avir * bvir * cvir:
        ii_arr = np.arange(nbin - 1, 0, -1)
        exit_mask = (tab_mass[ii_arr] >= vir_overdens * rho_mean * (4./3. * np.pi) * (ii_arr**3)) & (tab_mass[ii_arr - 1] < tab_mass[nbin - 1])
        ii = ii_arr[exit_mask][0]

        mvir = vir_overdens * rho_mean * (4./3. * np.pi) * (ii**3)
        mvir = np.maximum(mvir, 0)  # Ensure that mvir is non-negative

        avir = ii / (nbin - 1) * amax
        bvir = ii / (nbin - 1) * bmax
        cvir = ii / (nbin - 1) * cmax
        rvir = (avir * bvir * cvir) ** (1./3.)

        kvir = tab_ekin[ii]
        pvir = tab_epot[ii]

    if mvir > 0.0 and rvir > 0.0:
        h['rvir'] = min(rvir, h['r'])  # in Mpc
        h['mvir'] = mvir  # in 10^11 M_sun
        h['cvel'] = np.sqrt(gravconst * h['mvir'] / h['rvir'])  # circular velocity at r_vir in km/s
        h['tvir'] = 35.9 * gravconst * h['mvir'] / h['rvir']  # temperature at r_vir in K
        # compute halo density profile within the virialized region
        compute_halo_profile(h)
    else:
        raise ValueError(f"Halo bugged (ID, Mvir, Rvir): {h['id']}, {mvir}, {rvir}")

    return h
