SystemName=FeC-graphene_ON

rotX_theta_FeC=0
rotY_theta_FeC=0
rotZ_theta_FeC=0

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P 4 python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC}

python EC_TransmissionCalc.py ${SystemName} ${V_low} ${V_high} ${V_delta} 
