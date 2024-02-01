"""
### Abstract ###
    URL: https://wiki.fysik.dtu.dk/gpaw/tutorialsexercises/electronic/transport/transport.html
    Theme: Electron Transport

### Calculated Model ###
    Target: Graphene (supercell(4*4)) - Ferrocene - Graphene (supercell(4*4))
    Model: Tight-Binding
    Scattering region: Graphene-FeC-Graphene
    Principal layer: Graphene

### Comments###
    This code calculates Graphene-FeC-Graphene system's Hamiltonian
    to calculate electric conductivity of Graphene-FeC-Graphene based on Non-Static Green Function method.
"""

# import libraries
from ase import Atoms
from gpaw import GPAW, Mixer, FermiDirac, PW
from gpaw.xc import XC
from gpaw.lcao.tools import (get_lcao_hamiltonian,
                             get_lead_lcao_hamiltonian)
import pickle as pickle
import numpy as np
import math

#####################
# Scattering region #
#####################

# Setup the Atoms for the scattering region.

"""Experimental Parameters"""
dist_C_C            = 1.4 # C-C distance
dist_C_H            = 1.104
dist_Cp_Fe          = 1.66 # Cp-Fe distance

theta               = 2*math.pi/5
radius_Cp           = (dist_C_C/2)/math.sin(theta/2)

dist_C_C_graphene     = 1.42
len_graphene_unitcell = dist_C_C_graphene*math.sqrt(3)
dist_Plane_to_Cp      = 3.3

cellsize            = 30

"""
Graphene unit cell is like a diamond which contains 2 Carbon atoms
On the oblique coordinate of an unitcell, 
C1 is located at (1/3, 1/3, 0), C2 is at (2/3, 2/3, 0)
They're translated to (dist_C_C_G, 0, 0), (2*dist...) in Rectangle coordinate
"""

relative_loc_C1 = np.array([dist_C_C_graphene, 0, 0])
relative_loc_C2 = np.array([2*dist_C_C_graphene, 0, 0])
upper_plane_shift = np.array([0, 0,  (dist_Plane_to_Cp + dist_Cp_Fe)])
lower_plane_shift = np.array([0, 0, -(dist_Plane_to_Cp + dist_Cp_Fe)])
unitcell_vec_1  = np.array([0.5*math.sqrt(3)*len_graphene_unitcell, -0.5*len_graphene_unitcell, 0])
unitcell_vec_2  = np.array([0.5*math.sqrt(3)*len_graphene_unitcell, 0.5*len_graphene_unitcell,  0])
unitcell_vec_3  = upper_plane_shift - lower_plane_shift
supercellsize_1 = 4
supercellsize_2 = 4
origin_shift_1  = -2
origin_shift_2  = -2

def rotX_np(theta, input):
    return np.dot(np.array([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]]), input)

def rotY_np(theta, input):
    return np.dot(np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]]), input)

def rotZ_np(theta, input):
    return np.dot(np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]]), input)

rotX_theta_FeC = np.pi/2

system = Atoms('C10H10FeC32C32', 
               positions=[

                # Ferrocene's Cp ring C
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(0*theta), radius_Cp*math.sin(0*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(1*theta), radius_Cp*math.sin(1*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(2*theta), radius_Cp*math.sin(2*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(3*theta), radius_Cp*math.sin(3*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(4*theta), radius_Cp*math.sin(4*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(0*theta), radius_Cp*math.sin(0*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(1*theta), radius_Cp*math.sin(1*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(2*theta), radius_Cp*math.sin(2*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(3*theta), radius_Cp*math.sin(3*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([radius_Cp*math.cos(4*theta), radius_Cp*math.sin(4*theta), dist_Cp_Fe])),                  

                  # Ferrocene's Cp ring H
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(0*theta), (dist_C_H+radius_Cp)*math.sin(0*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(1*theta), (dist_C_H+radius_Cp)*math.sin(1*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(2*theta), (dist_C_H+radius_Cp)*math.sin(2*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(3*theta), (dist_C_H+radius_Cp)*math.sin(3*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(4*theta), (dist_C_H+radius_Cp)*math.sin(4*theta), -dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(0*theta), (dist_C_H+radius_Cp)*math.sin(0*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(1*theta), (dist_C_H+radius_Cp)*math.sin(1*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(2*theta), (dist_C_H+radius_Cp)*math.sin(2*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(3*theta), (dist_C_H+radius_Cp)*math.sin(3*theta), dist_Cp_Fe])),
                    rotX_np(rotX_theta_FeC, np.array([(dist_C_H+radius_Cp)*math.cos(4*theta), (dist_C_H+radius_Cp)*math.sin(4*theta), dist_Cp_Fe])),

                  # Ferrocene's Fe
                    rotX_np(rotX_theta_FeC, np.array([0.0, 0.0, 0.0])),

                # Graphene Carbon Atoms
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

                

                # Lower Graphene Carbon Atoms
                # Each atom's location is described as the sum of
                # 1. atom's relative location in unitcell
                # 2. unitcell's relative location in supercell
                # 3. supercell's relative location in the system
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

               ],
               cell = [
                   supercellsize_1*unitcell_vec_1, 
                   supercellsize_2*unitcell_vec_2, 
                   (4/3)*unitcell_vec_3, 
               ],
               pbc = [True, True, True],

               charges = [
                  -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 2.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0, 0.0 
               ]
               )

system.center()

from ase.visualize import view
from ase.io import write
view(system)
write('model_DFT_FeC-graphene_OFF.png', system)

# Attach a GPAW calculator
calc = GPAW(h=0.2,
            xc='PBE',
            basis='szp(dzp)',
            occupations=FermiDirac(width=0.1),
            # kpts= {'size' : (4, 4, 4)},
            kpts={'density': 3.5, 'even': True},
            # mode = PW(300),
            mode='lcao',
            txt='FeC-graphene_OFF_lcao_scat_1.txt',
            mixer=Mixer(0.02, 5, weight=100.0),
            # symmetry={'point_group': False, 'time_reversal': False}
            )
system.calc = calc

system.get_potential_energy()  # Converge everything!
Ef = system.calc.get_fermi_level()

"""
### lead (Principal layer) ###
    Atoms: Graphene
    Hamiltonian:
    H_lead = (
        H_l   V
        V†    H_l
    )

### Scattering Region ###
    Atoms: G-FeC-G
    Hamiltonian:
    H_scat = (
        H_l 0      0
        0   H_scat 0
        0   0      H_l
    )
"""

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
# Only use first kpt, spin, as there are no more
if type(H_skMM) != None and type(S_kMM) != None:
    H, S = H_skMM[0, 0], S_kMM[0]
    H -= Ef * S
    # remove_pbc(system, H, S, 0)

    # Dump the Hamiltonian and Scattering matrix to a pickle file
    pickle.dump((H.astype(complex), S.astype(complex)), open('scat_hs.pickle', 'wb'), 2)

########################
# Left principal layer #
########################

# Use upper graphene in the lead, so only take those from before
system = system[21:60].copy()

# Attach a GPAW calculator
calc = GPAW(h=0.3,
            xc='PBE',
            basis='szp(dzp)',
            occupations=FermiDirac(width=0.1),
            # kpts=(4, 1, 1),  # More kpts needed as the x-direction is shorter
            kpts={'density': 3.5, 'even': True},
            mode='lcao',
            txt='FeC-graphene_OFF_lcao_llead_1.txt',
            mixer=Mixer(0.02, 5, weight=100.0),
            symmetry={'point_group': False, 'time_reversal': False})
system.calc = calc

system.get_potential_energy()  # Converge everything!
Ef = system.calc.get_fermi_level()

ibz2d_k, weight2d_k, H_skMM, S_kMM = get_lead_lcao_hamiltonian(calc)
# Only use first kpt, spin, as there are no more
if type(H_skMM) != None and type(S_kMM) != None:
    H, S = H_skMM[0, 0], S_kMM[0]
    H -= Ef * S

    # Dump the Hamiltonian and Scattering matrix to a pickle file
    pickle.dump((H, S), open('lead1_hs.pickle', 'wb'), 2)

    #########################
    # Right principal layer #
    #########################
    # This is identical to the left prinicpal layer so we don't have to do anything
    # Just dump the same Hamiltonian and Scattering matrix to a pickle file
    pickle.dump((H, S), open('lead2_hs.pickle', 'wb'), 2)
