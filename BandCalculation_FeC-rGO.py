from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton

from gpaw import GPAW, PW, FermiDirac
from gpaw.xc import XC

import math
import numpy as np

"""Experimental Parameters"""
dist_C_C            = 1.4 # C-C distance
dist_C_H            = 1.104
dist_Cp_Fe          = 1.66 # Cp-Fe distance

theta               = 2*math.pi/5
radius_Cp           = (dist_C_C/2)/math.sin(theta)

dist_C_C_graphene     = 1.42
len_graphene_unitcell = dist_C_C_graphene*math.sqrt(3)
dist_Plane_to_Cp      = 3.3

"""
Graphene unit cell is like a diamond which contains 2 Carbon atoms
On the oblique coordinate of an unitcell, 
C1 is located at (1/3, 1/3, 0), C2 is at (2/3, 2/3, 0)
They're translated to (dist_C_C_G, 0, 0), (2*dist...) in Rectangle coordinate
"""

relative_loc_C1 = np.array([dist_C_C_graphene, 0, 0])
relative_loc_C2 = np.array([2*dist_C_C_graphene, 0, 0])
unitcell_vec_1  = np.array([0.5*math.sqrt(3)*len_graphene_unitcell, -0.5*len_graphene_unitcell, 0])
unitcell_vec_2  = np.array([0.5*math.sqrt(3)*len_graphene_unitcell, 0.5*len_graphene_unitcell,  0])
upper_plane_shift = np.array([0, 0,  (dist_Plane_to_Cp + dist_Cp_Fe)])
lower_plane_shift = np.array([0, 0, -(dist_Plane_to_Cp + dist_Cp_Fe)])
supercellsize_1 = 4
supercellsize_2 = 4
origin_shift_1  = -2
origin_shift_2  = -2

# Setup the atoms
# Fe atom of Ferrocene is located at center (0, 0, 0)
system = Atoms('C74H10Fe', 
               positions=[
                  # Ferrocene's Cp ring
                   np.array([radius_Cp*math.cos(0*theta), radius_Cp*math.sin(0*theta), -dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(1*theta), radius_Cp*math.sin(1*theta), -dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(2*theta), radius_Cp*math.sin(2*theta), -dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(3*theta), radius_Cp*math.sin(3*theta), -dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(4*theta), radius_Cp*math.sin(4*theta), -dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(0*theta), radius_Cp*math.sin(0*theta), dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(1*theta), radius_Cp*math.sin(1*theta), dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(2*theta), radius_Cp*math.sin(2*theta), dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(3*theta), radius_Cp*math.sin(3*theta), dist_Cp_Fe]),
                   np.array([radius_Cp*math.cos(4*theta), radius_Cp*math.sin(4*theta), dist_Cp_Fe]),

                  # Graphene
                  # Each atom's location is described as the sum of
                  # 1. atom's relative location in unitcell
                  # 2. unitcell's relative location in supercell
                  # 3. supercell's relative location in the system
                   relative_loc_C1 + (0 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (0 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (0 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (0 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 0 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 1 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 2 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 3 * unitcell_vec_2) + (upper_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                  
                   relative_loc_C1 + (0 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (0 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (0 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (0 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (0 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (1 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (1 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (2 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (2 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 0 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 1 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 2 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C1 + (3 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),
                   relative_loc_C2 + (3 * unitcell_vec_1 + 3 * unitcell_vec_2) + (lower_plane_shift + origin_shift_1 * unitcell_vec_1 + origin_shift_2 * unitcell_vec_2),

                  # Ferrocene's Cp ring
                   np.array([(dist_C_H+radius_Cp)*math.cos(0*theta), (dist_C_H+radius_Cp)*math.sin(0*theta), -dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(1*theta), (dist_C_H+radius_Cp)*math.sin(1*theta), -dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(2*theta), (dist_C_H+radius_Cp)*math.sin(2*theta), -dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(3*theta), (dist_C_H+radius_Cp)*math.sin(3*theta), -dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(4*theta), (dist_C_H+radius_Cp)*math.sin(4*theta), -dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(0*theta), (dist_C_H+radius_Cp)*math.sin(0*theta), dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(1*theta), (dist_C_H+radius_Cp)*math.sin(1*theta), dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(2*theta), (dist_C_H+radius_Cp)*math.sin(2*theta), dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(3*theta), (dist_C_H+radius_Cp)*math.sin(3*theta), dist_Cp_Fe]),
                   np.array([(dist_C_H+radius_Cp)*math.cos(4*theta), (dist_C_H+radius_Cp)*math.sin(4*theta), dist_Cp_Fe]),
                   np.array([0.0, 0.0, 0.0]),
               ])

system.set_cell((20.0, 20.0, 20.0))
system.center()

"""
    Recognized names are: 
    LDA, PW91, PBE, revPBE, RPBE, BLYP, HCTH407, TPSS, 
    M06-L, revTPSS, vdW-DF, vdW-DF2, EXX, PBE0, B3LYP, BEE, GLLBSC.
"""

calc = GPAW(mode = "lcao",
            xc = XC('PBE'),
            # kpts=(8, 8, 8),
            random=True,
            occupations=FermiDirac(0.01),
            txt='FeC-rGO_GroundState.txt')
system.calc = calc

from ase.visualize import view
from ase.io import write
view(system)

system.get_potential_energy()
fermienergy = calc.get_fermi_level()
calc.write('FeC-rGO_GroundState.gpw')

# P2
# Restart from ground state and fix potential:
# system is recognized as Primitive cubic
calc = GPAW('FeC-rGO_GroundState.gpw').fixed_density(
    nbands=16,
    symmetry='off',
    kpts={'path': 'GXMGRX', 'npoints': 60},
    convergence={'bands': 8})

# P3
bs = calc.band_structure()
bs.plot(filename='FeC-rGO_BandStructure.png', show=True, emax=10.0)
# opt = QuasiNewton(system, trajectory='h2.emt.traj')
# opt.run(fmax=0.05)
