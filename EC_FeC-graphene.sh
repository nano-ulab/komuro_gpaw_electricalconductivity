NumCalculatingCore=4

SystemName=FeC-graphene_ON
rotX_theta_FeC=0
rotY_theta_FeC=0
rotZ_theta_FeC=0
relativepos=Overlap

systemname_mod=${SystemName}_${relativepos}_rotX${rotX_theta_FeC}_rotY${rotY_theta_FeC}_rotZ${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P ${NumCalculatingCore} python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} ${relativepos} 

python EC_Transmission_FeC-graphene.py ${systemname_mod} ${V_low} ${V_high} ${V_delta}

SystemName=FeC-graphene_OFF
rotX_theta_FeC=90
rotY_theta_FeC=0
rotZ_theta_FeC=0
relativepos=Overlap

systemname_mod=${SystemName}_${relativepos}_rotX${rotX_theta_FeC}_rotY${rotY_theta_FeC}_rotZ${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P ${NumCalculatingCore} python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} ${relativepos} 

python EC_Transmission_FeC-graphene.py ${systemname_mod} ${V_low} ${V_high} ${V_delta}

SystemName=FeC-graphene_ON
rotX_theta_FeC=0
rotY_theta_FeC=0
rotZ_theta_FeC=0
relativepos=Shift

systemname_mod=${SystemName}_${relativepos}_rotX${rotX_theta_FeC}_rotY${rotY_theta_FeC}_rotZ${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P ${NumCalculatingCore} python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} ${relativepos} 

python EC_Transmission_FeC-graphene.py ${systemname_mod} ${V_low} ${V_high} ${V_delta}

SystemName=FeC-graphene_OFF
rotX_theta_FeC=90
rotY_theta_FeC=0
rotZ_theta_FeC=0
relativepos=Shift

systemname_mod=${SystemName}_${relativepos}_rotX${rotX_theta_FeC}_rotY${rotY_theta_FeC}_rotZ${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P ${NumCalculatingCore} python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} ${relativepos} 

python EC_Transmission_FeC-graphene.py ${systemname_mod} ${V_low} ${V_high} ${V_delta}

SystemName=FeC-graphene_ON
rotX_theta_FeC=0
rotY_theta_FeC=0
rotZ_theta_FeC=0
relativepos=Bridge

systemname_mod=${SystemName}_${relativepos}_rotX${rotX_theta_FeC}_rotY${rotY_theta_FeC}_rotZ${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P ${NumCalculatingCore} python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} ${relativepos} 

python EC_Transmission_FeC-graphene.py ${systemname_mod} ${V_low} ${V_high} ${V_delta}

SystemName=FeC-graphene_OFF
rotX_theta_FeC=90
rotY_theta_FeC=0
rotZ_theta_FeC=0
relativepos=Bridge

systemname_mod=${SystemName}_${relativepos}_rotX${rotX_theta_FeC}_rotY${rotY_theta_FeC}_rotZ${rotZ_theta_FeC}

V_low=-0.5
V_high=0.5
V_delta=0.01

gpaw -P ${NumCalculatingCore} python EC_SCF_FeC-graphene.py ${SystemName} ${rotX_theta_FeC} ${rotY_theta_FeC} ${rotZ_theta_FeC} ${relativepos} 

python EC_Transmission_FeC-graphene.py ${systemname_mod} ${V_low} ${V_high} ${V_delta}

