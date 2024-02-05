SystemName=FeC-graphene_ON

rotX_theta_FeC=0
rotY_theta_FeC=0
rotZ_theta_FeC=0
relativepos=Overlap
# relativepos=Shift
# relativepos=Bridge

systemname_mod=${SystemName}_relativeposition_rotX${rotX_theta_FeC}_$rotY_${rotY_theta_FeC}_rotZ$str${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P 4 python EC_SCF_FeC-graphene.py ${systemname_mod}_ ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} 
${relativepos}
python EC_TransmissionCalc.py ${systemname_mod} ${V_low} ${V_high} ${V_delta} 
