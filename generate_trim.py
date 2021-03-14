import pandas as pd 

elevator_continuation=pd.read_csv('./bifurcation_data/elevator_full.csv',header=None)
aileron_continuation=pd.read_csv('./bifurcation_data/aileron_full.csv',header=None)
rudder_continuation=pd.read_csv('./bifurcation_data/rudder_full.csv',header=None)
eta_continuation=pd.read_csv('./bifurcation_data/eta_full.csv',header=None)

names=['V','alpha','beta','p','q','r','phi','theta']
elevator_continuation.columns=names+['elevator']
elevator_continuation=elevator_continuation[(elevator_continuation['elevator']<25) & (elevator_continuation['elevator']>-25)]
elevator_continuation['type']='elevator'

aileron_continuation.columns=names+['aileron']
aileron_continuation=aileron_continuation[(aileron_continuation['aileron']<35) & (aileron_continuation['aileron']>-35)]
aileron_continuation['type']='aileron'

rudder_continuation.columns=names+['rudder']
rudder_continuation=rudder_continuation[(rudder_continuation['rudder']<30) & (rudder_continuation['rudder']>-30)]
rudder_continuation['type']='rudder'

eta_continuation.columns=names+['eta']
eta_continuation=eta_continuation[(eta_continuation['eta']<1) & (eta_continuation['eta']>0)]
eta_continuation['type']='eta'

trim=pd.concat([elevator_continuation,aileron_continuation,rudder_continuation,eta_continuation], axis=0, ignore_index=True)
trim['elevator'].fillna(value=-7.2157,inplace=True)
trim['aileron'].fillna(value=0.,inplace=True)
trim['rudder'].fillna(value=0.,inplace=True)
trim['eta'].fillna(value=0.1836,inplace=True)

trim=trim[names+['elevator','aileron','rudder','eta','type']]
trim.to_csv('./input/bifurcation.csv')