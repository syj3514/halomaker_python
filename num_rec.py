import numpy as np

# For calculation of the age of the universe
#***********************************************************************
def polint(xa, ya, x):
    n = len(xa)
    if n > 10:
        raise ValueError("Maximum allowed size for xa and ya is 10")
    
    c = ya.copy()
    d = ya.copy()

    # Find the index of the closest value in xa to x
    ns = np.abs(xa - x).argmin()
    y = ya[ns]
    
    for m in range(1, n):
        den = xa[:n-m] - x - xa[m:] + x
        w = c[1:n-m+1] - d[:n-m]
        with np.errstate(divide='raise'):
            try:
                den = w / den
            except FloatingPointError:
                raise ValueError("Denominator is zero!")

        d[:n-m] = (xa[m:] - x) * den
        c[:n-m] = (xa[:n-m] - x) * den

        if 2 * ns < n - m:
            dy = c[ns]
        else:
            dy = d[ns-1]
            ns -= 1
        
        y += dy
        
    return y, dy

#***********************************************************************
def qromo(func, a, b, **kwargs):
    def midpnt(func, a, b, s, n, **kwargs):
        # If `it` was a static or saved variable in Fortran subroutine,
        # then in Python, it can be made an attribute of the function to mimic this behavior.
        if not hasattr(midpnt, "it"):
            midpnt.it = 0

        if n == 1:
            s = (b - a) * func(0.5 * (a + b), **kwargs)
            midpnt.it = 1
        else:
            tnm = midpnt.it
            del_ = (b - a) / (3.0 * tnm)  # using del_ instead of del because del is a keyword in Python
            ddel = del_ + del_
            x = a + 0.5 * del_
            sum_ = 0.0  # using sum_ instead of sum because sum is a built-in function in Python

            for j in range(midpnt.it):
                sum_ += func(x, **kwargs)
                x += ddel
                sum_ += func(x, **kwargs)
                x += del_
            
            s = (s + (b - a) * sum_ / tnm) / 3.0
            midpnt.it *= 3

        return s

    jmax = 14; k = 5; km = k - 1; jmaxp = jmax + 1; dss=0
    eps = 1e-6
    s = np.zeros(jmaxp)
    h = np.zeros(jmaxp)
    h[0] = 1.0
    for j in range(jmax):
        s[j] = midpnt(func, a, b, s[j], j+1, **kwargs)
        if j >= k:
            ss, dss = polint(h[j-km : j-km+k], s[j-km : j-km+k], 0.0)
            if abs(dss) < eps * abs(ss):
                return ss
        s[j+1] = s[j]
        h[j+1] = h[j] / 9.0
    print('> too many steps in qromo')
    return ss

#***********************************************************************
def age_temps(x,omm=0,oml=0):
    # Maybe this should be moved to `compute_halo_props`
    return 1/np.sqrt(omm*x**5 + (1.0-omm-oml)*x**4 + oml*x**2)

#***********************************************************************
def age_temps_turn_around(x,omm=0,oml=0):
    # Maybe this should be moved to `compute_halo_props`
    temps = omm*x**5 + (-omm-oml)*x**4 + oml*x**2
    if(temps > 0.0):
       return 1/np.sqrt(temps)
    else:
       return 0.0

#***********************************************************************
# subroutine `locate` is not used in the code

#***********************************************************************
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

#***********************************************************************
# subroutine `indexx` can be replaced by `np.argsort`

#***********************************************************************
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

#***********************************************************************
def cubic(a1, a2, a3, a4):
    """
    Function for evaluating a cubic equation. In the case where
    there is one real root, this routine returns it. If there are three
    roots, it returns the smallest positive one.

    The cubic equation is a1*x**3 + a2*x**2 + a3*x + a4 = 0.

    Ref: Tallarida, R. J., "Pocket Book of Integrals and Mathematical
    Formulas," 2nd ed., Boca Raton: CRC Press 1992, pp. 8-9.
    """

    if a1 == 0.0:
        raise ValueError('Quadratic/linear equation passed to cubic function')

    p = a3/a1 - (a2/a1)**2/3.0
    q = a4/a1 - a2*a3/a1**2/3.0 + 2.0*(a2/a1)**3/27.0
    d = p**3/27.0 + q**2/4.0

    if d > 0.0:
        a = -0.5*q + np.sqrt(d)
        b = -0.5*q - np.sqrt(d)
        a = a**(1/3) if a > 0 else -(-a)**(1/3)
        b = b**(1/3) if b > 0 else -(-b)**(1/3)
        y = a + b
    else:
        ar = -0.5 * q
        ai = np.sqrt(-d)
        r = (ar**2 + ai**2)**(1/6)
        theta = np.arctan2(ai, ar)
        y1 = 2.0 * r * np.cos(theta/3.0) - a2/a1/3.0
        y = y1
        y2 = 2.0 * r * np.cos(theta/3.0 + 2.0*np.pi/3.0) - a2/a1/3.0
        if y < 0 or (y2 > 0 and y2 < y): y = y2
        y3 = 2.0 * r * np.cos(theta/3.0 - 2.0*np.pi/3.0) - a2/a1/3.0
        if y < 0 or (y3 > 0 and y3 < y): y = y3

    return y

#***********************************************************************
# subroutine `int_indexx` can be replaced by `np.argsort`

#***********************************************************************
# function `ran2` can be replaced by `np.random.rand``


#=======================================================================
def spline(x):
#=======================================================================
#   Moved from `compute_neiKDtree_mod`
#   real(kind=8) :: spline,x

    if (x<=1.):
        ans=1.-1.5*x**2+0.75*x**3
    elif (x<=2.):
        ans=0.25*(2.-x)**3
    else:
        ans=0.
    return ans

#=======================================================================
def icellid(xtest:np.ndarray):
#=======================================================================
#  Compute cell id corresponding to the signs of coordinates of xtest
#  as follows :
#  (-,-,-) : 0
#  (+,-,-) : 1
#  (-,+,-) : 2
#  (+,+,-) : 3
#  (-,-,+) : 4
#  (+,-,+) : 5
#  (-,+,+) : 6
#  (+,+,+) : 7
#  For self-consistency, the array pos_ref_0 should be defined exactly 
#  with the same conventions
#=======================================================================
    # Moved from `compute_neiKDtree_mod`
    # integer(kind=4) :: icellid,j,icellid3d(3)
    # real(kind=8) :: xtest(3)
    assert xtest.shape == (3,)
    icellid3d = (xtest>=0).astype(np.int32)
    return icellid3d[0]+2*icellid3d[1]+4*icellid3d[2]

#=======================================================================
def icellids(xtests:np.ndarray):
    '''
    (N,3) -> (N,)
    '''
#=======================================================================
#  Compute cell id corresponding to the signs of coordinates of xtest
#  as follows :
#  (-,-,-) : 0
#  (+,-,-) : 1
#  (-,+,-) : 2
#  (+,+,-) : 3
#  (-,-,+) : 4
#  (+,-,+) : 5
#  (-,+,+) : 6
#  (+,+,+) : 7
#  For self-consistency, the array pos_ref_0 should be defined exactly 
#  with the same conventions
#=======================================================================
    # Moved from `compute_neiKDtree_mod`
    # integer(kind=4) :: icellid,j,icellid3d(3)
    # real(kind=8) :: xtest(3)
    # assert xtests.shape == (xtests.shape[0],3)
    # icellid3d = (xtests>=0).astype(np.int32)
    is_positive = xtests >= 0
    icellid = np.sum(is_positive * 2 ** np.arange(3), axis=1)
    return icellid
    # return icellid3d[:,0]+2*icellid3d[:,1]+4*icellid3d[:,2] #shape = (xtests.shape[0],)



