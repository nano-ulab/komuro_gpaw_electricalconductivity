from ase.transport.calculators import TransportCalculator
import numpy as np
import pickle
import matplotlib.pyplot as plt

from ase import units

from gpaw.lcao.tools import (get_lcao_hamiltonian, get_lead_lcao_hamiltonian, lead_kspace2realspace)
from gpaw import restart

# Change Current directory to root dir of this file
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import glob


def main(cif_files_folder_name, Voltage_range):
    # load calculated results

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

    cif_file_paths = glob.glob(os.path.join(".", cif_files_folder_name, "*.cif"))

    for cif_file_path in cif_file_paths:

        savename = os.path.splitext(os.path.basename(cif_file_path))[0]

        filepath_scat_gpw = os.path.join(os.path.dirname(cif_file_path), savename+"_scat.gpw")
        print(filepath_scat_gpw)
        filepath_llead_gpw = os.path.join(os.path.dirname(cif_file_path), savename+"_llead.gpw")
        print(filepath_llead_gpw)
        filepath_rlead_gpw = os.path.join(os.path.dirname(cif_file_path), savename+"_rlead.gpw")
        print(filepath_rlead_gpw)

        if os.path.isfile(filepath_scat_gpw):
            
            scat, scat_calc = restart(filepath_scat_gpw)

            Ef_scat = scat.calc.get_fermi_level()
            H_skMM_scat, S_kMM_scat = get_lcao_hamiltonian(scat_calc)
            # Only use first kpt, spin, as there are no more

            H_scat, S_scat = H_skMM_scat[0, 0], S_kMM_scat[0]
            H_scat -= Ef_scat * S_scat

        else:
            try: raise ValueError("!!! ERROR : " + filepath_scat_gpw + " is missing!")
            except ValueError as e: print(e)
            return None


        if os.path.isfile(filepath_llead_gpw):

            llead, llead_calc = restart(filepath_llead_gpw)

            Ef_llead = llead.calc.get_fermi_level()
            tmp, tmp2, H_skMM_llead, S_kMM_llead = get_lead_lcao_hamiltonian(llead_calc, direction='z')

            # Only use first kpt, spin, as there are no more

            H_llead, S_llead = H_skMM_llead[0, 0], S_kMM_llead[0]
            H_llead -= Ef_llead * S_llead


        else:
            try: raise ValueError("!!! ERROR : " + filepath_llead_gpw + " is missing!")
            except ValueError as e: print(e)
            return None

       
        if os.path.isfile(filepath_rlead_gpw):

            rlead, rlead_calc = restart(filepath_rlead_gpw)

            Ef_rlead = rlead_calc.get_fermi_level()
            tmp, tmp2, H_skMM_rlead, S_kMM_rlead = get_lead_lcao_hamiltonian(rlead_calc, direction='z')
            # Only use first kpt, spin, as there are no more

            H_rlead, S_rlead = H_skMM_rlead[0, 0], S_kMM_rlead[0]
            H_rlead -= Ef_rlead * S_rlead

        else:
            try: raise ValueError("!!! ERROR : " + filepath_rlead_gpw + " is missing!")
            except ValueError as e: print(e)
            return None
        
        fileloc = os.path.join(".", cif_files_folder_name, savename)

        # Set TranportCalculator for calculation
        tcalc = TransportCalculator(h=H_scat, h1=H_llead, h2=H_rlead,  # hamiltonian matrices
                                    s=S_scat, s1=S_llead, s2=S_rlead,  # overlap matrices
                                    align_bf=1)        # align the Fermi levels


        # Calculate the conductance (the energy zero corresponds to the Fermi level)


        tcalc.set(energies=[Voltage_range])
        # for i in range(len(voltage_range)):
        #     G = tcalc.get_transmission()[i]
        #     print(f'Conductance: {G:.2f} 2e^2/h')

        # Determine the basis functions of the two Hydrogen atoms and subdiagonalize
        Fe_nbf = 3
        C_nbf = 3
        H_nbf = 2
        Electrode_N = 32*1    # Number of Electrode atoms on each side in the scattering region
        Electrode_nbf = C_nbf  # number of bf per Electrode atom (basis=szp)
        FeC_nbf = 10*C_nbf + 10* H_nbf + 1* Fe_nbf   # number of bf per H atom (basis=szp)
        bf_H1 = Electrode_nbf * Electrode_N
        bfs = range(bf_H1, bf_H1 + FeC_nbf) # Bridging layer's basis funcs

        h_rot, s_rot, eps_n, vec_jn = tcalc.subdiagonalize_bfs(bfs)
        for n in range(len(eps_n)):
            print("bf %i corresponds to the eigenvalue %.2f eV" % (bfs[n], eps_n[n]))

        # Switch to the rotated basis set
        tcalc.set(h=h_rot, s=s_rot)

        # plot the transmission function
        tcalc.set(energies=Voltage_range)
        T = tcalc.get_transmission()
        plt.plot(tcalc.energies, T)
        plt.title("Transmission function of " + savename)
        plt.savefig(fileloc+"_"+"TransmissionFunc.png")
        plt.close()

        # # ... and the projected density of states (pdos) of the FeC molecular orbitals
        # tcalc.set(pdos=bfs)
        # pdos_ne = tcalc.get_pdos()
        # plt.title("Projected density of states of " + SystemName)
        # for i in range(len(pdos_ne)):
        #     plt.plot(tcalc.energies, pdos_ne[i], label=str(i))
        # plt.savefig(SystemName+"_"+"PDOS.png")
        # plt.close()

        # Plot current correspond to Vb
        current = tcalc.get_current(Voltage_range, T = 300.)
        current_mods = 2.*units._e**2/units._hplanck*current

        plt.title("I-V curve of " + savename)
        plt.plot(Voltage_range, 2.*units._e**2/units._hplanck*current)
        plt.xlabel("U [V]")
        plt.ylabel("I [A]")
        plt.savefig(fileloc+"_"+"IV.png")
        plt.close()

        # log10
        plt.title("logI-V curve of " + savename)
        plt.plot(Voltage_range, np.log10(np.abs(current_mods)))
        plt.xlabel("U [V]")
        plt.ylabel("I [A]")
        plt.savefig(fileloc+"_"+"logIV.png")
        plt.close()

        import csv

        IV_data = np.array([Voltage_range, current_mods]).T
        with open(fileloc+"_"+"IV.csv", "w") as IV_file:
            writer = csv.writer(IV_file)
            writer.writerows(IV_data.tolist())

        # Cut the coupling to the anti-bonding orbital.
        # print('Cutting the coupling to the renormalized molecular state at %.2f eV' % (
        #     eps_n[1]))
        # h_rot_cut, s_rot_cut = tcalc.cutcoupling_bfs([bfs[1]])
        # tcalc.set(h=h_rot_cut, s=s_rot_cut)
        # plt.plot(tcalc.energies, tcalc.get_transmission())
        # plt.title('Transmission without anti-bonding orbital')
        # plt.show()

        # Cut the coupling to the bonding-orbital.
        # print('Cutting the coupling to the renormalized molecular state at %.2f eV' % (
        #     eps_n[0]))
        # tcalc.set(h=h_rot, s=s_rot)
        # h_rot_cut, s_rot_cut = tcalc.cutcoupling_bfs([bfs[0]])
        # tcalc.set(h=h_rot_cut, s=s_rot_cut)
        # plt.plot(tcalc.energies, tcalc.get_transmission())
        # plt.title('Transmission without bonding orbital')
        # plt.show()


# ShellScript Interface
import sys

args = sys.argv
if len(args) == 5:
    cif_files_folder_name = str(args[1])
    V_lowest = float(args[2])
    V_highest = float(args[3])
    V_delta = float(args[4])
    Voltage_range = np.arange(V_lowest, V_highest, V_delta)
    main(cif_files_folder_name, Voltage_range)
else:
    try:
        raise ValueError("!!! ERROR : SystemName, V_lowest, V_highest, V_step are must be given as args !!!")
    except ValueError as e:
        print(e)
    # SystemName = "FeC-graphene_ON"
    # Voltage_range = np.arange(-0.5, 0.5, 0.01)
