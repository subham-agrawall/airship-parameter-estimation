import os
import pandas as pd 
trim=pd.read_csv("./input/bifurcation.csv")
col_names=trim.columns
folder_name='./data/'
data_number=0
for index,row in trim.iterrows():
	if row['type']!='eta':
		if (row['type']!='elevator') or (row['type']=='elevator' and index%3==0): 
			arg1,arg2=str(),str()
			for i in range(8):
				arg1+=(str(float(row[col_names[i]]))+',')
			for i in range(8,12):
				arg2+=(str(float(row[col_names[i]]))+',')
			arg1=arg1[:-1]
			arg2=arg2[:-1]
			arg3=folder_name+'data_'+str(row['type'])+'_'+str(data_number)
			print('initial condition: %s' %arg1)
			print('controls: %s' %arg2)
			print('file name: %s' %arg3)
			os.system('python airship.py '+arg1+' '+arg2+' '+arg3)
			print('\n')
			data_number+=1