#from evaluation_metrics import *
from sklearn.svm import SVR
#from evaluation_metrics import *
import numpy as np 
from sklearn import preprocessing
import os
from scipy.io import loadmat,savemat
from scipy.stats import iqr
import pandas as pd
import matplotlib.pyplot as plt
from evaluation_metrics import *
'''
def convert_to_float(data):

	for idx1, elements in enumerate(data) :


		for idx2,items in enumerate(elements):
			if not isinstance(items,float) :
				pass 
				#print (data[idx1,idx2])
		print (data[0,410:430])
		input ('Here')
'''
def svr_comprehensive(train_data,train_label,val_data,val_label,test_data,test_label):


	assert len(train_data) == len (train_label)
	assert len(valid_data) == len (valid_label)
	assert len(test_data)  == len(test_label)

	train_data[train_data == -np.inf] =0.0
	train_data[train_data == np.inf]  =0.0
	train_data[np.isnan(train_data)] = 0.0
	#train_data=np.array(train_data).astype('float')
	
	valid_data[valid_data == -np.inf] =0.0
	valid_data[valid_data == np.inf]  =0.0
	valid_data[np.isnan(valid_data)] = 0.0	
	
	test_data[test_data == -np.inf] =0.0
	test_data[test_data == np.inf]  =0.0
	test_data[np.isnan(test_data)] = 0.0
	

	scaler = preprocessing.StandardScaler().fit(train_data)
	train_data_scaled = scaler.transform(train_data)
	val_data_scaled   = scaler.transform(val_data)
	test_data_scaled  = scaler.transform(test_data)

	
	reg1=SVR(kernel='linear',C=0.01,max_iter=100000)
	reg2=SVR(kernel='linear',C=0.1,max_iter=100000)
	reg3=SVR(kernel='linear',C=1,max_iter=100000)
	reg4=SVR(kernel='linear',C=10,max_iter=500000)
	reg5=SVR(kernel='linear',C=100,max_iter=100000)
	
	reglist=[reg1,reg2,reg3,reg4,reg5]

	bestreg=reglist[0]
	ctr_reg=0
	best_ctr=ctr_reg


	# Running it with the first regression as an example. It will be repeated again
	bestreg.fit(train_data_scaled,train_label)
	best_val_predicted_label=bestreg.predict(val_data_scaled)
	best_mse=mean_error(val_label,best_val_predicted_label)


	# The training errors are just for curiosity. Not that much important to the current code
	best_train_reg=bestreg
	best_train_predicted_label=bestreg.predict(train_data_scaled)
	best_train_mse=mean_error(train_label,best_train_predicted_label)
	

	for reg in reglist:

		reg.fit(train_data_scaled,train_label)
		val_predicted_label=reg.predict(val_data_scaled)
		train_predicted_label=reg.predict(train_data_scaled)

		mse=mean_error(val_label,val_predicted_label)
		train_mse=mean_error(train_label,train_predicted_label)


	
	
		# This is the actual criteria for selecting the regressor as we are checking on the validation accuracy 
		if mse < best_mse:
			bestreg=reg
			best_mse=mse
			best_ctr= ctr_reg
			best_val_predicted_label=val_predicted_label
			 

		# Checking which classifier gives the absolute best error for training data
		if train_mse < best_train_mse:
			best_train_mse=train_mse
			best_train_reg=reg

		ctr_reg=ctr_reg+1

	
	train_predicted_label=bestreg.predict(train_data_scaled)
	corresponding_train_error=mean_error(train_label,train_predicted_label)
	x=bestreg.coef_
	
	test_predicted_label=bestreg.predict(test_data_scaled)
	test_mse=mean_error(test_label,test_predicted_label)
	test_pcc=pcc(test_label,test_predicted_label)

	
	return test_mse,test_pcc,[test_predicted_label,test_label]



fold_dir= 'kfold_reg/'
data_list= sorted (np.array([name for name in os.listdir(fold_dir) if name.startswith('data')]))


for idx, fold in enumerate(data_list):

	data= np.load(os.path.join(fold_dir,fold),allow_pickle=True).item ()

	train_data, train_label = data['train_data'], data['train_label']
	valid_data, valid_label = data['valid_data'], data['valid_label']
	test_data,  test_label = data['test_data'], data['test_label']

	assert len (train_data) == len(train_label)
	assert len (valid_data) == len (valid_label)
	assert len (test_data)  == len (test_label)

	print (train_data.shape, train_label.shape)

	#train_data= convert_to_float(train_data)
	res_mse, res_pcc , test_result = svr_comprehensive (train_data, train_label, valid_data, valid_label, test_data, test_label)
	'''
	save_filename= fold_dir + 'result_' +str(idx) + '.mat'
	savemat(save_filename, dict (x=test_result[0],y=test_result[1]))
	input ('Now here')
	'''
	save_filename= fold_dir + 'plot_' +str(idx) + '.png'

	y_pred, y_true = test_result
	y_pred = y_pred * 50
	y_true = y_true * 50
	plt.scatter (y_true,y_pred)
	plt.xlabel('True Values')
	plt.ylabel('Predicted Values')
	plt.title ('Scatter Plot for Regressor')
	plt.savefig(save_filename)
	plt.close()
	#input ('Now heres')
print ('Done')