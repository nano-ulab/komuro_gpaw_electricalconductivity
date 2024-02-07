##################
#     Setup      #
##################

# import libraries

import pickle as pickle
import numpy as np
import math
import sys

from ase.visualize import view
from ase.io import read, write, cif
from ase import Atoms
from ase import parallel

from gpaw import GPAW, Mixer, FermiDirac, PW, mpi
from gpaw.xc import XC
from gpaw.lcao.tools import (get_lcao_hamiltonian, get_lead_lcao_hamiltonian)


# Change Current directory to root dir of this file
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import glob

def main(cif_files_folder_name):
  
    NumAssignedCores = mpi.world.size
    IsDrawSystem = False
    IsSaveSystemFig = False

    # Setup the Atoms for the scattering region.
    
    cif_file_paths = glob.glob(os.path.join(".", cif_files_folder_name, "*.cif"))

    for cif_file_path in cif_file_paths:

            
        system = read(cif_file_path)

        # add charge in Fe atom

        system_charge = 0.0
        if mpi.rank == 0:
            if "c+1.0" in cif_file_path:
                print("setting charges to system...")
                charges = []
                for i, atom in enumerate(system.get_chemical_symbols()):
                    if atom == "Fe":
                        charges.append(1.0)
                    else:
                        charges.append(0.0)
                system.set_initial_charges(charges)
                system_charge = 1.0

            else : 
                system_charge = 0.0
            print(system.get_initial_charges())

        ##################
        #  Calculation   #
        ##################

        # Scattering region-------------------------

        # Attach a GPAW calculator

        savename = os.path.splitext(os.path.basename(cif_file_path))[0]

        filepath_scat_gpw = os.path.join(os.path.dirname(cif_file_path), savename+"_scat.gpw")
        filepath_scat_txt = os.path.join(os.path.dirname(cif_file_path), savename+"_scat.txt")
        filepath_llead_gpw = os.path.join(os.path.dirname(cif_file_path), savename+"_llead.gpw")
        filepath_llead_txt = os.path.join(os.path.dirname(cif_file_path), savename+"_llead.txt")

        calc = GPAW(h=0.3,
                    xc='PBE',
                    basis='szp(dzp)',
                    occupations=FermiDirac(width=0.1),
                    kpts={'density': 3.5, 'even': True},
                    mode='lcao',
                    txt=filepath_scat_txt,
                    mixer=Mixer(0.02, 5, weight=100.0),
                    symmetry={'point_group': False, 'time_reversal': False},
                    charge = system_charge
                    )
        system.calc = calc

        system.get_potential_energy()  # Converge everything!

        calc.write(filepath_scat_gpw)

        # Left lead layer-----------------------------

        # Use upper graphene in the lead, so only take those from before
        system = system[21:60].copy()

        # Attach a GPAW calculator
        calc = GPAW(h=0.3,
                    xc='PBE',
                    basis='szp(dzp)',
                    occupations=FermiDirac(width=0.1),
                    kpts={'density': 3.5, 'even': True},
                    mode='lcao',
                    txt=filepath_llead_txt,
                    mixer=Mixer(0.02, 5, weight=100.0),
                    symmetry={'point_group': False, 'time_reversal': False}
                    )
        system.calc = calc

        system.get_potential_energy()  # Converge everything!

        calc.write(filepath_llead_gpw)


# ShellScript Interface
import sys

args = sys.argv
if len(args) == 2: # args[0]はpythonファイル名自体
    cif_files_folder_name = str(args[1])
    main(cif_files_folder_name)
else:
    print(args)
    try:
        raise ValueError("!!! ERROR : cif_files_folder_name must be given as args !!!")
    except ValueError as e:
        print(e)
    # SystemName = "FeC-graphene_ON"
    # Voltage_range = np.arange(-0.5, 0.5, 0.01)
