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

    """Experimental Parameters"""
    dist_C_C            = 1.4 # C-C distance
    dist_C_H            = 1.104
    dist_Cp_Fe          = 1.66 # Cp-Fe distance

    theta               = 2*math.pi/5
    radius_Cp           = (dist_C_C/2)/math.sin(theta/2)

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
    upper_plane_shift = np.array([0, 0,  (dist_Plane_to_Cp + dist_Cp_Fe)])
    lower_plane_shift = np.array([0, 0, -(dist_Plane_to_Cp + dist_Cp_Fe)])
    unitcell_vec_1  = np.array([0.5*math.sqrt(3)*len_graphene_unitcell, -0.5*len_graphene_unitcell, 0])
    unitcell_vec_2  = np.array([0.5*math.sqrt(3)*len_graphene_unitcell, 0.5*len_graphene_unitcell,  0])
    unitcell_vec_3  = upper_plane_shift - lower_plane_shift
    supercellsize_1 = 4
    supercellsize_2 = 4
    origin_shift_1  = -2
    origin_shift_2  = -2

    
    # Setup the Atoms for the scattering region.
    
    cif_file_paths = glob.glob(os.path.join(".", cif_files_folder_name, "*.cif"))
    
    for cif_file_path in cif_file_paths:
            
        system = read(cif_file_path)

        # add charge in Fe atom

        system_charge = 0.0
        if "c+1.0" in cif_file_path:
            # print("setting charges to system...")
            # charges = []
            # for i, atom in enumerate(system.get_chemical_symbols()):
            #     if atom == "Fe":
            #         charges.append(1.0)
            #     else:
            #         charges.append(0.0)
            # system.set_initial_charges(charges)
            system_charge = 1.0

        else : 
            system_charge = 0.0
        print(system_charge)

        # Add L/R leads
        # Use upper graphene in the lead, so only take those from before
        llead = system[:32].copy()
        # G-FeC-GのGを複製し，横方向にずらしてくっつけてleadにする！
        width = len_graphene_unitcell
        llead.positions = llead.positions + np.array([4*width*np.cos(-np.pi/3), 4*width*np.sin(-np.pi/3), 0])

        rlead = system[32:64].copy()
        # G-FeC-GのGを複製し，横方向にずらしてくっつけてleadにする！
        width = len_graphene_unitcell
        rlead.positions = rlead.positions + np.array([4*width*np.cos(-np.pi/3), 4*width*np.sin(-np.pi/3), 0])

        scat_lead_chem_symbols = system.get_chemical_symbols()
        scat_lead_chem_symbols.extend(["C"]*64)

        scat_lead_positions = np.vstack((system.positions, llead.positions))
        scat_lead_positions = np.vstack((scat_lead_positions, rlead.positions))
        system_scat_lead = Atoms(symbols = scat_lead_chem_symbols, positions = scat_lead_positions, cell = system.get_cell()*[[1, 1, 1],[2, 2, 2],[1, 1, 1]])

        # Scattering region-------------------------

        # Attach a GPAW calculator

        savename = os.path.splitext(os.path.basename(cif_file_path))[0]

        view(system)

        # Left lead layer-----------------------------

        view(system_scat_lead)

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
