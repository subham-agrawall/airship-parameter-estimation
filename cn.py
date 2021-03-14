from constants import *
from keras.models import Sequential,Model,model_from_json
from keras.layers import Input,Dense,dot,concatenate
from keras import backend as K
from keras.optimizers import Adam,RMSprop
from keras.layers.advanced_activations import PReLU
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import glob
import copy 
import os

# fix random seed for reproducibility
seed=1
np.random.seed(seed)

# Generate dataset
path ='data_combined'
allFiles = glob.glob(os.path.join(path, "*.csv"))
dataset = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    del df['Unnamed: 0']
    list_.append(df)

dataset = pd.concat(list_)
dataset=dataset.drop_duplicates()
dataset=dataset.sample(frac=1)
dataset=dataset.reset_index()
print(len(dataset))

# Create input and ouput
coef_name=['Cn_act']
alpha_name=['alpha(deg)']
states_name=['beta(deg)','pb','rb','aileron(deg)','rudder(deg)']
alpha_input=np.transpose(np.array([dataset[i] for i in alpha_name]))
states_input=np.transpose(np.array([dataset[i] for i in states_name]))
coef_output=np.transpose(np.array([dataset[i] for i in coef_name]))

# Create keras model
alpha = Input(shape=(1,), name='alpha')
x1=Dense(5, activation='softmax')(alpha)
# x1=Dense(3, activation='softmax')(x1)
x1=Dense(1, activation='linear',name='cn_b')(x1)

x2=Dense(5, activation='softmax')(alpha)
# x2=Dense(3, activation='softmax')(x2)
x2=Dense(1, activation='linear',name='cn_p')(x2)

x3=Dense(5, activation='softmax')(alpha)
# x3=Dense(3, activation='softmax')(x3)
x3=Dense(1, activation='linear',name='cn_r')(x3)

x4=Dense(5, activation='softmax')(alpha)
# x4=Dense(3, activation='softmax')(x4)
x4=Dense(1, activation='linear',name='cn_da')(x4)

x5=Dense(5, activation='softmax')(alpha)
# x5=Dense(3, activation='softmax')(x5)
x5=Dense(1, activation='linear',name='cn_dr')(x5)

x=concatenate([x1,x2,x3,x4,x5],axis=-1)
states = Input(shape=(5,), name='states')
output=dot([x, states],axes=1,name='output')
model = Model(inputs=[alpha,states], outputs=[output])

epochs=1000
learning_rate= 0.0001
decay_rate = learning_rate / epochs
adam=Adam(lr=learning_rate)
model.compile(loss='mean_squared_error',optimizer='adam')
print(model.summary())
history=model.fit({'alpha':alpha_input,'states':states_input},{'output':coef_output},epochs=epochs,batch_size=10000,verbose=2)

# # Prediction
# prediction=model.predict({'alpha':alpha_input, 'states':states_input})
# plt.figure(1)
# plt.plot(range(len(dataset)),dataset['Cn_act'],'--r')
# plt.plot(list(prediction))
# plt.show()

# Creating estimates dataset
data_table=pd.read_csv('./input/airship_datatable.csv',index_col='alpha')
max_values=dataset.max()
min_values=dataset.min()
for i in data_table.index:
	if min_values['alpha(deg)']<i:
		alpha_min=i
		break

for i in data_table.index[::-1]:
	if max_values['alpha(deg)']>i:
		alpha_max=i
		break

print(alpha_min,alpha_max)
estimates=pd.DataFrame(index=range(-15,19,1),columns=['cn_b','cn_p','cn_r','cn_da','cn_dr'])

alpha_values=np.transpose(np.array([estimates.index]))

layer_name = 'cn_b'
cn_b_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
cn_b_output = cn_b_layer_model.predict([alpha_values,states_input])
estimates['cn_b']=cn_b_output.reshape(len(estimates),1)

layer_name = 'cn_p'
cn_p_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
cn_p_output = cn_p_layer_model.predict([alpha_values,states_input])
estimates['cn_p']=cn_p_output.reshape(len(estimates),1)

layer_name = 'cn_r'
cn_r_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
cn_r_output = cn_r_layer_model.predict([alpha_values,states_input])
estimates['cn_r']=cn_r_output.reshape(len(estimates),1)

layer_name = 'cn_da'
cn_da_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
cn_da_output = cn_da_layer_model.predict([alpha_values,states_input])
estimates['cn_da']=cn_da_output.reshape(len(estimates),1)

layer_name = 'cn_dr'
cn_dr_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
cn_dr_output = cn_dr_layer_model.predict([alpha_values,states_input])
estimates['cn_dr']=cn_dr_output.reshape(len(estimates),1)

plot_number=1
for colname,col in estimates.iteritems():
	plt.subplot(5,1,plot_number)
	plt.plot(data_table[colname],'--r')
	plt.plot(col)
	plt.ylabel(colname)
	plot_number+=1

# plt.show()
plt.savefig('./result6/cn.eps')

# serialize model to JSON
model_json = model.to_json()
with open("./models/cn.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("./models/cn.h5")
print("Saved model to disk")

# load json and create model
json_file = open('./models_1/cn.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./models_1/cn.h5")
print("Loaded model from disk")
model=loaded_model