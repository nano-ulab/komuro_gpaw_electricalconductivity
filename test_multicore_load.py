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
from ase import parallel
from gpaw import GPAW, Mixer, FermiDirac, PW, mpi
from gpaw.xc import XC
from gpaw.lcao.tools import (get_lcao_hamiltonian,
                             get_lead_lcao_hamiltonian)
import pickle as pickle
import numpy as np
import math

from ase.transport.calculators import TransportCalculator
import pylab

from ase import units

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
            parallel = {'sl_auto': True}
            # symmetry={'point_group': False, 'time_reversal': False},
            )
Ef_scat = -4.267072483455023

H_skMM, S_kMM = get_lcao_hamiltonian(calc)
# Only use first kpt, spin, as there are no more
h, s = H_skMM[0, 0], S_kMM[0]
h -= Ef_scat * s
pickle.dump((h, s), open('test.pickle', 'wb'), 2)
