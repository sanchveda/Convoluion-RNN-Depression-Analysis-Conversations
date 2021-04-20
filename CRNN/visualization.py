import numpy as np 

import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.metrics import auc,roc_curve,mean_squared_error


def plot_confusion_matrix(confusion_mat,name,save_name,normalize=False):

	if normalize:
		confusion_mat=confusion_mat.astype('float')/confusion_mat.sum(axis=1) [:,np.newaxis]

	ax=sns.heatmap(confusion_mat, annot=True , center=0, annot_kws={"size":20} )
	ax.set_title(name)

	#plt.show()
	plt.savefig(save_name+".png",dpi=300)
	plt.close()


def compute_roc_binary(y_test,y_score,pred_score,roc_text):
	
	probs=np.zeros(len(pred_score))
	for idx,score in enumerate(y_score):

		if pred_score[idx] == 0:
			probs[idx]= y_score[idx]
		else:
			probs[idx]= 1 - y_score[idx]

	print (probs)
	input ('')
	assert len(pred_score) == len(probs)

	fpr,tpr,_= roc_curve(y_test,probs,pos_label=0)
	roc_auc  = auc (fpr,tpr)
	
	plt.plot(fpr,tpr,color='darkorange',lw=2, label='ROC curve (area = %0.2f)' %roc_auc)
	plt.plot([0,1],[0,1], color='navy', lw=2 , linestyle='--')
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title("ROC Curve")
	plt.legend(loc="lower right")
	plt.savefig(roc_text+".png")
	plt.close()
	

def plot_loss_functions(read_filename,write_filename,train=False):

	

	if len(read_filename) > 1:
		train_filename,test_filename=read_filename
		train_data=np.load(train_filename,allow_pickle=True)
		test_data= np.load(test_filename,allow_pickle=True)

		train_data=np.mean(train_data,axis=1)

		plt.xlabel('Number of Epochs')
		plt.ylabel('Loss')
		plt.title('Loss over epochs')
		plt.plot(train_data, label='Traning Loss')
		plt.plot(test_data, label='Validation Loss')
		plt.legend(loc="upper left")
		plt.savefig (write_filename+'.png')
		plt.close()

		return


	data= np.load (read_filename,allow_pickle=True)


	if not train:
		print (data)
		plt.xlabel('Number of epochs')
		plt.ylabel('Validataion Loss')
		plt.title('Validataion Loss over Epochs')
		plt.plot(data,label='Validataion Loss')
		plt.legend(loc='upper left')
		plt.savefig(write_filename+'.png')
	else:
		
		plt.xlabel('Number of epochs')
		plt.ylabel('Training Loss')
		plt.title('Training Loss over Epochs')
		
		data=np.mean(data,axis=1)
		
		#data= data.flatten()
		

		plt.plot(data,label='Training Loss')
		plt.legend(loc="upper left")
		plt.savefig(write_filename+'.png')

	return 


def plot_probabilities(data,filename,text,mode='line'):

	if mode == 'line':

		plt.xlabel('Subjects')
		plt.ylabel('Probabilities')
		plt.title(text)
		plt.plot(data)
		plt.savefig(filename)
		plt.close()
	elif mode == 'bar':

		plt.xlabel('Subjects')
		plt.ylabel('Probabilities')
		plt.title(text)
		plt.scatter(list(range(len(data))),data)
		plt.savefig(filename)
		plt.close()
	return
def plot_heatmap():
	return


'''

raw_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan2/cnn_pool/result_valid/cnn_pool4/'
save_dir= './plot_dir/'

final_raw_dir=[os.path.join(raw_dir,s) for s in os.listdir(raw_dir)]


for elements in final_raw_dir:

	
		
	test_loss,test_score,train_loss,train_score=sorted ([j for j in os.listdir(elements) if j.startswith('CRNN')])
	
	test_filename= elements + '/' + test_loss
	train_filename= elements + '/' + train_loss

	names= elements.split('/')

	test_out_name=save_dir + names[-2] + names[-1] + '_test'
	train_out_name= save_dir + names[-2] + names[-1] + '_train'
	print (test_loss,train_out_name)
	#plot_loss_functions(test_filename, test_out_name)
	#plot_loss_functions(train_filename, train_out_name,train=True)
	plot_loss_functions([train_filename,test_filename], train_out_name)
	input ('Here')
'''

