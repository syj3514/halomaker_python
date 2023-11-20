# this program reads in the particle data from N-Body simulation snapshots and 
# puts them into halos (using a friend-of-friend algorithm). It then goes on to
# compute various properties for these halos (mass, spin, energy ...)

import compute_halo_props
import sys
# from halo_defs import *
import halo_defs as H

print()
print( '_______________________________________________________________________')
print()
print( '              HaloMaker                                                ')
print( '              ---------                                                ' )
print()
print()
print( ' first version    :        S. Ninin                  (1999)            ')
print( ' modif.           :        J. Devriendt              (1999-2002)       ')
print( ' modif.           :        B. Lanzoni & S. Hatton    (2000-2001)       ')
print( ' galics v1.0      :        J. Blaizot & J. Devriendt (2002)            ')
print( ' horizon v1.0     :        J. Devriendt & D. Tweed   (2006)            ')
print( ' horizon v2.0     :        J. Devriendt & D. Tweed   (2007)            ')
print( ' horizon v2.0.2   :        J. Devriendt, D. Tweed & J.Blaizot (2008)   ')
print( ' horizon python   :        S. Jeon                   (2023)            ')
print( )
print( '_______________________________________________________________________'  )
print( )


# get directory where to input/output the data
print("Usage: python HaloMaker.py [data_dir]")
if(len(sys.argv)<2): H.data_dir = '.'
else: H.data_dir = sys.argv[1]
# initialize cosmological and technical parameters of the N_Body simulation 
print("\n[          init_cosmo          ]")
compute_halo_props.init_0()

# loop over snapshots
print("\n[      loop over snapshots     ]")
for istep1 in range(1,H.nsteps,1):
    H.numero_step = istep1
    compute_halo_props.new_step_1()
    print(H.numero_step,' of ',H.nsteps)
    print()
    print()

print()
print( '> Bricks now available to build halo merger tree with TreeMaker')
print()
print( '_______________________________________________________________________'  )
print()
print( '      End of HaloMaker                                        ')
print()
print( '_______________________________________________________________________'  )