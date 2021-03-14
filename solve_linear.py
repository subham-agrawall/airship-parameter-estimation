import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from constants import * 

def derivative(vector,timestep):
	# Centrally pivoted five-point algorithm
	derivative=[]
	for i,value in enumerate(vector):
		if i==0:
			temp=(-25*value+48*vector[i+1]-36*vector[i+2]+16*vector[i+3]-3*vector[i+4])/(12*timestep)
		elif i==1:
			temp=(-3*vector[i-1]-10*value+18*vector[i+1]-6*vector[i+2]+vector[i+3])/(12*timestep)
		elif i==len(vector)-2:
			temp=(3*vector[i+1]+10*value-18*vector[i-1]+6*vector[i-2]-vector[i-3])/(12*timestep)
		elif i==len(vector)-1:
			temp=(25*value-48*vector[i-1]+36*vector[i-2]-16*vector[i-3]+3*vector[i-4])/(12*timestep)			
		else:
			temp=(-8*vector[i-1]+vector[i-2]-vector[i+2]+8*vector[i+1])/(12*timestep)
		derivative.append(temp)
	return derivative

def add_derivatives(data,timestep):
	data['V_dot']=derivative(list(data['V(m/s)']),timestep)
	data['alpha_dot']=derivative(list(data['alpha(deg)']),timestep)
	data['beta_dot']=derivative(list(data['beta(deg)']),timestep)
	data['p_dot']=derivative(list(data['p(deg/s)']),timestep)
	data['q_dot']=derivative(list(data['q(deg/s)']),timestep)
	data['r_dot']=derivative(list(data['r(deg/s)']),timestep)	
	return data

def calculate_aero_coeff(data,names,plot=False):
	CD,CL,Cm,CY,Cr,Cn=[],[],[],[],[],[]
	for _,row in data.iterrows():
		V,alpha,beta,p,q,r,phi,theta=row['V(m/s)'],row['alpha(deg)'],row['beta(deg)'],\
			row['p(deg/s)'],row['q(deg/s)'],row['r(deg/s)'],row['phi(deg)'],row['theta(deg)']
		
		Q=0.5*rho*V*S

		Sy=np.cos(alpha*dtr)*np.cos(beta*dtr)*np.sin(theta*dtr)-\
		np.sin(beta*dtr)*np.sin(phi*dtr)*np.cos(theta*dtr)-\
		np.sin(alpha*dtr)*np.cos(beta*dtr)*np.cos(phi*dtr)*np.cos(theta*dtr)
	
		Cmcy=np.sin(theta*dtr)*np.sin(alpha*dtr)+\
			np.cos(alpha*dtr)*np.cos(phi*dtr)*np.cos(theta*dtr)
	
		Smcy=np.sin(theta*dtr)*np.cos(alpha*dtr)*np.sin(beta*dtr)+\
			np.sin(phi*dtr)*np.cos(theta*dtr)*np.cos(beta*dtr)-\
			np.sin(alpha*dtr)*np.sin(beta*dtr)*np.cos(phi*dtr)*np.cos(theta*dtr)
		
		cd=(1./(Q*V))*(-row['V_dot']*mx+Tm*eta*np.cos(alpha*dtr)*np.cos(beta*dtr)\
			-(m*g-B)*Sy)
	
		cl=(mz/Q)*((q-row['alpha_dot'])*np.cos(beta*dtr)-p*np.cos(alpha*dtr)*np.sin(beta*dtr)-\
			r*np.sin(alpha*dtr)*np.sin(beta*dtr)-(1/(V*mz))\
			*(Tm*eta*np.sin(alpha*dtr)-((m*g)-B)*Cmcy))
		
		cy=(my*V*(row['beta_dot']-p*np.sin(alpha*dtr)+r*np.cos(alpha*dtr))+\
			Tm*eta*np.cos(alpha*dtr)*np.sin(beta*dtr)-(m*g-B)*Smcy)*(1./(Q*V))

		cr=(1./(Q*V*b))*(row['p_dot']*Jx-(Jy-Jz)*q*r-Jxz*p*q+\
			B*bz*Smcy)

		cm=(1./(Q*V*c))*(Jy*row['q_dot']-(Jz-Jx)*p*r-Jxz*((r**2)-(p**2))\
			-Tm*eta*dz*np.cos(alpha*dtr)*np.cos(beta*dtr)+B*bz*np.sin(theta*dtr))

		cn=(1./(Q*V*b))*(row['r_dot']*Jz -(Jx-Jy)*p*q -Jxz*q*r)
		
		CD.append(cd);CL.append(cl);Cm.append(cm)
		CY.append(cy);Cr.append(cr);Cn.append(cn)

	coef=[CD,CL,Cm,CY,Cr,Cn]
	for i in range(6):
		data[names[i]]=coef[i]
	if plot==True:
		plt.figure(1)
		plot_number=1
		for i in range(6):
			plt.subplot(6,1,plot_number)
			plt.plot(data[names[i]])
			plt.ylabel(names[i])
			plot_number+=1
	return data

def verify_aero_coeff(data,names):
	plt.figure(2)
	plot_number=1
	for i in range(6):
		plt.subplot(6,1,plot_number)
		plt.plot(data[names[i]])
		plt.plot(data[names[i]+"_act"],'--r')
		plt.ylabel(names[i])
		plot_number+=1

if __name__=='__main__':
	data=pd.read_csv("./data_trial.csv",index_col='Unnamed: 0')
	print("loaded data")
	data=add_derivatives(data,timestep)
	print("added derivatives")
	# system ##########(check these values before running the script)
	eta=0.1836
	################################################################
	names=['CD','CL','Cm','CY','Cr','Cn']
	data=calculate_aero_coeff(data,names,True)
	print("calculated derivatives")
	verify_aero_coeff(data,names)
	plt.show()