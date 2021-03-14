import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd  
from constants import *
import time
import sys

def elevator_deflection(t):
	mean_angle=mean_elevator
	return mean_angle
 
def aileron_deflection(t):
	mean_angle=mean_aileron
	return mean_angle

def rudder_deflection(t):
	mean_angle=mean_rudder
	return mean_angle

def read_data_table(data_table,plot=False):
	x=list(data_table.index)
	y=data_table.values
	# print(data_table.columns)
	parameters_func=interp1d(x,y,axis=0,kind='linear')
	if plot==True:
		plt.figure(1)
		plot_number=1
		for colname,col in data_table.iteritems():
			plt.subplot(6,4,plot_number)
			plt.plot(col)
			plt.ylabel(colname)
			plot_number+=1
	return parameters_func

def aerodynamic_coef(par_function,input_long,input_lat,alpha):
	parameters=par_function(alpha)
	# lateral parameters
	par_lat=np.reshape(np.matrix(parameters[0:15]),(3,5))
	input_lat=np.reshape(np.matrix(input_lat),(5,1))
	lat_coef=np.matmul(par_lat,input_lat)
	# longitudinal parameters
	par_long=np.reshape(np.matrix(parameters[15:24]),(3,3))
	input_long=np.reshape(np.matrix(input_long),(3,1))
	long_coef=np.matmul(par_long,input_long)
	return long_coef,lat_coef

def system(t,Y):
	V,alpha,beta,p,q,r,phi,theta=Y
	del_e=elevator_deflection(t)
	del_a=aileron_deflection(t)
	del_r=rudder_deflection(t)
	input_lat=[beta,(p*b)/(2*V),(r*b)/(2*V),del_a,del_r]
	input_long=[1.,(q*c)/(2*V),del_e]
	long_coef,lat_coef=aerodynamic_coef(par_func,input_long,input_lat,alpha)
	CD,CL,Cm=long_coef[0,0],long_coef[1,0],long_coef[2,0]
	CY,Cr,Cn=lat_coef[0,0],lat_coef[1,0],lat_coef[2,0]
	
	Q=0.5*rho*V*S

	Sy=np.cos(alpha*dtr)*np.cos(beta*dtr)*np.sin(theta*dtr)-\
		np.sin(beta*dtr)*np.sin(phi*dtr)*np.cos(theta*dtr)-\
		np.sin(alpha*dtr)*np.cos(beta*dtr)*np.cos(phi*dtr)*np.cos(theta*dtr)
	
	Cmcy=np.sin(theta*dtr)*np.sin(alpha*dtr)+\
		np.cos(alpha*dtr)*np.cos(phi*dtr)*np.cos(theta*dtr)
	
	Smcy=np.sin(theta*dtr)*np.cos(alpha*dtr)*np.sin(beta*dtr)+\
		np.sin(phi*dtr)*np.cos(theta*dtr)*np.cos(beta*dtr)-\
		np.sin(alpha*dtr)*np.sin(beta*dtr)*np.cos(phi*dtr)*np.cos(theta*dtr)
	
	V_dot=(1/mx)*(Tm*eta*np.cos(alpha*dtr)*np.cos(beta*dtr)-CD*Q*V-(m*g-B)*Sy)
	
	alpha_dot=q-(1/np.cos(beta*dtr))*((p*np.cos(alpha*dtr)+\
		r*np.sin(alpha*dtr))*np.sin(beta*dtr)+\
		(1/(V*mz))*(Tm*eta*np.sin(alpha*dtr)+CL*Q*V-((m*g)-B)*Cmcy))
	
	beta_dot=(1/(my*V))*(-Tm*eta*np.cos(alpha*dtr)*np.sin(beta*dtr)+CY*Q*V+\
		(m*g-B)*Smcy)+(p*np.sin(alpha*dtr)-r*np.cos(alpha*dtr))

	p_dot=((Jy-Jz)/Jx)*q*r + (Jxz/Jx)*p*q + ((Q*V*b*Cr)/Jx)-\
		(B*bz*Smcy/Jx)

	q_dot=(1/Jy)*((Jz-Jx)*p*r+Jxz*((r**2)-(p**2))+Q*V*c*Cm+Tm*eta*dz*np.cos(alpha*dtr)*np.cos(beta*dtr)-\
		B*bz*np.sin(theta*dtr))

	r_dot=((Jx-Jy)/Jz)*p*q + (Jxz/Jz)*q*r + (1/Jz)*Q*V*b*Cn

	phi_dot=p + q*np.tan(theta*dtr)*np.sin(phi*dtr) +\
		r*np.tan(theta*dtr)*np.cos(phi*dtr)
	
	theta_dot=q*np.cos(phi*dtr)-r*np.sin(phi*dtr)

	derivatives=[V_dot,alpha_dot,beta_dot,p_dot,q_dot,r_dot,phi_dot,theta_dot]
	return derivatives

def solver(system,y0,t0,t1,dt):
	y,t=[[] for i in range(8)],[]
	r=ode(system).set_integrator('dopri5',method='bdf')
	r.set_initial_value(y0,t0)
	while r.successful() and r.t<t1:
		r.integrate(r.t+dt)
		for i in range(8):
			y[i].append(r.y[i])
		t.append(r.t)
	return y,t

def generate_data(x,t,states,plot=False):
	x=np.transpose(np.array(x))
	data=pd.DataFrame(x,index=t,columns=states)
	# data['elevator(deg)']=[elevator_deflection(i) for i in t]
	# data['aileron(deg)']=[aileron_deflection(i) for i in t]
	# data['rudder(deg)']=[rudder_deflection(i) for i in t]
	# data['gamma(deg)']=data['alpha(deg)']-data['theta(deg)']
	if plot==True:
		plt.figure(2)
		plot_number=1
		for colname,col in data.iteritems():
			plt.subplot(8,1,plot_number)
			plt.plot(col)
			plt.ylabel(colname)
			plot_number+=1
	return data

def aero_coef_signals(data,par_function,coef,plot=False):
	CD,CL,Cm,CY,Cr,Cn,pb,qc,rb=[],[],[],[],[],[],[],[],[]
	for _,row in data.iterrows():
		V,alpha,beta,p,q,r=row['V(m/s)'],row['alpha(deg)'],row['beta(deg)'],row['p(deg/s)'],row['q(deg/s)'],row['r(deg/s)']
		del_e,del_a,del_r=row['elevator(deg)'],row['aileron(deg)'],row['rudder(deg)']
		input_lat=[beta,(p*b)/(2*V),(r*b)/(2*V),del_a,del_r]
		input_long=[1.,(q*c)/(2*V),del_e]
		long_coef,lat_coef=aerodynamic_coef(par_func,input_long,input_lat,alpha)
		CD.append(long_coef[0,0]);CL.append(long_coef[1,0]);Cm.append(long_coef[2,0])
		CY.append(lat_coef[0,0]);Cr.append(lat_coef[1,0]);Cn.append(lat_coef[2,0])
		qc.append((q*c)/(2*V));pb.append((p*b)/(2*V));rb.append((r*b)/(2*V))
	data['pb'],data['qc'],data['rb']=pb,qc,rb
	coefficients=[CD,CL,Cm,CY,Cr,Cn]
	for i in range(6):
		data[coef[i]]=coefficients[i]
	if plot==True:
		plt.figure(3)
		plot_number=1
		for i in range(6):
			plt.subplot(6,1,plot_number)
			plt.plot(data[coef[i]])
			plt.ylabel(coef[i])
			plot_number+=1
	return data

def plot_input(data,data_inp,plot=False):
	if plot==True:
		plt.figure(4)
		plot_number=1
		for i in data_inp:
			plt.subplot(7,1,plot_number)
			plt.plot(data[i])
			plt.ylabel(i)
			plot_number+=1

if __name__=='__main__':
	data_table=pd.read_csv('./input/airship_datatable.csv',index_col='alpha')
	par_func=read_data_table(data_table)
	if len(sys.argv)>1:
		initial_conditions=[float(i) for i in sys.argv[1].split(',')]
		arg2=[float(i) for i in sys.argv[2].split(',')]
		mean_elevator,mean_aileron,mean_rudder,eta=arg2[0],arg2[1],arg2[2],arg2[3]
	else:
		initial_conditions=[13.9256, 0.595, 0., 0., 0., 0., 0., 1.0415]
		mean_elevator=-1.5636
		mean_aileron=0.
		mean_rudder=0.
		eta=0.1836
	
	try:
		x,t=solver(system,initial_conditions,0.,5000,timestep)
	except Exception as e: 
		print(e)
		try:
			x,t=solver(system,initial_conditions,0.,2500,timestep)
		except Exception as e: 
			print(e)
			try:
				x,t=solver(system,initial_conditions,0.,1000,timestep)
			except Exception as e:
				print(e)
				try:
					x,t=solver(system,initial_conditions,0.,500,timestep)
				except Exception as e:
					print(e)
					try:
						x,t=solver(system,initial_conditions,0.,100,timestep)
					except Exception as e:
						print(e)

	states=["V(m/s)","alpha(deg)","beta(deg)","p(deg/s)","q(deg/s)","r(deg/s)","phi(deg)","theta(deg)"]
	data=generate_data(x,t,states,plot=True)
	coef=['CD_act','CL_act','Cm_act','CY_act','Cr_act','Cn_act']
	# data=aero_coef_signals(data,par_func,coef)
	data_input=['qc','elevator(deg)','beta(deg)','pb','rb','aileron(deg)','rudder(deg)']
	# plot_input(data,data_input)
	if len(sys.argv)>1:
		file_name=str(sys.argv[3]+'.csv')
	else:
		file_name='./data_trial'+'.csv'
	data.to_csv(file_name)
	# plt.savefig(str(sys.argv[3])+'.png')
	# plt.close()
	plt.show()