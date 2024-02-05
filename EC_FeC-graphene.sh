SystemName=FeC-graphene_ON
rotX_theta_FeC=0.0
rotY_theta_FeC=0.0
rotZ_theta_FeC=0.0
relativepos=Overlap

systemname_mod=${SystemName}_${relativepos}_rotX${rotX_theta_FeC}_rotY${rotY_theta_FeC}_rotZ${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P 4 python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} ${relativepos} 

python EC_TransmissionCalc.py ${systemname_mod} ${V_low} ${V_high} ${V_delta} 
