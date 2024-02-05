from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton

from gpaw import GPAW, PW, FermiDirac, Mixer
from gpaw.xc import XC

import math
import numpy as np

"""Experimental Parameters"""
SystemName = "FeC"

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

# Setup the atoms
# Fe atom of Ferrocene is located at center (0, 0, 0)
system = Atoms('C10H10Fe', 
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

calc = GPAW(h=0.3,
            xc='PBE',
            basis='szp(dzp)',
            occupations=FermiDirac(width=0.1),
            kpts={'density': 3.5, 'even': True},
            mode='lcao',
            txt=SystemName+"_"+"StrucureOpt.txt",
            mixer=Mixer(0.02, 5, weight=100.0),
            )
system.calc = calc

from ase.visualize import view
from ase.io import write
view(system)
write('FeC-graphene_BeforeOpt.png', system)

opt = QuasiNewton(system, trajectory='FeC-graphene.lcao.traj')
opt.run(fmax=0.05)

view(system)
write('FeC-graphene_AfterOpt.png', system)
