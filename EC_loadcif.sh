NumCalculatingCore=4

cif_files_folder_name=cif_files

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P ${NumCalculatingCore} python EC_SCF_loadcif.py ${cif_files_folder_name}

python EC_TransmissionCalc_loadcif.py ${cif_files_folder_name} ${V_low} ${V_high} ${V_delta}
