from ase.transport.calculators import TransportCalculator
import numpy as np
import pickle
import matplotlib.pyplot as plt

from ase import units

# Principal layer size
# Uncomment this line if going back to gpawtransport again
# pl = 4 * 9 # 9 is the number of bf per Pt atom (basis=szp), see below

# Read in the hamiltonians
h, s = pickle.load(open('scat_hs.pickle', 'rb'))
# Uncomment this line if going back to gpawtransport again
# h, s = h[pl:-pl, pl:-pl], s[pl:-pl, pl:-pl]
h1, s1 = pickle.load(open('lead1_hs.pickle', 'rb'))
h2, s2 = pickle.load(open('lead2_hs.pickle', 'rb'))

tcalc = TransportCalculator(h=h, h1=h1, h2=h2,  # hamiltonian matrices
                            s=s, s1=s1, s2=s2,  # overlap matrices
                            align_bf=1)        # align the Fermi levels

# Calculate the conductance (the energy zero corresponds to the Fermi level)

voltage_range = np.arange(-0.5, 0.5, 0.01)

tcalc.set(energies=[voltage_range])
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
tcalc.set(energies=voltage_range)
T = tcalc.get_transmission()
plt.plot(tcalc.energies, T)
plt.title('Transmission function')
plt.savefig('FeC-graphene_ON_TransimissonFunc.png')
plt.close()

# ... and the projected density of states (pdos) of the FeC molecular orbitals
tcalc.set(pdos=bfs)
pdos_ne = tcalc.get_pdos()
for i in range(len(pdos_ne)):
    plt.plot(tcalc.energies, pdos_ne[i], label=str(i))
    plt.title('Projected density of states')
    # plt.legend()
    # plt.savefig('FeC-graphene_ON_PDOS_'+str(i)+'.png')
    # plt.close()
plt.savefig('FeC-graphene_ON_PDOS_FeC.png')
plt.close()

# Plot current correspond to Vb
current = tcalc.get_current(voltage_range, T = 300.)
plt.plot(voltage_range, 2.*units._e**2/units._hplanck*current)
plt.xlabel('U [V]')
plt.ylabel('I [A]')
plt.savefig('FeC-graphene_ON_IV.png')
plt.close()

# log10
plt.plot(voltage_range, np.log10(2.*units._e**2/units._hplanck*current))
plt.xlabel('U [V]')
plt.ylabel('I [A]')
plt.savefig('FeC-graphene_ON_logIV.png')
plt.close()

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
