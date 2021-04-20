import numpy as np
import os
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from scipy import stats

from visualization import *

#test_data=np.load('./data/test_data_1.npy',allow_pickle=True).item()

test_data= np.load ("./kfold_data/test_data_reliable_0.npy",allow_pickle=True).item()
test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_data['mother_name'],\
                                                                                    test_data['child_name'],\
                                                                                    test_data['label'],\
                                                                                    test_data['start_list'],\
                                                                                    test_data['end_list']





'''
test_indices=np.array([test_mother_name[i][6:8] == '02' for i in range (len(test_mother_name)) ])
test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_mother_name[test_indices],\
                                                                                test_child_name[test_indices],\
                                                                                test_label_list[test_indices],\
                                                                                test_start_list[test_indices],\
                                                                                test_end_list[test_indices]

assert test_mother_name.shape == test_child_name.shape == test_label_list.shape == test_start_list.shape == test_end_list.shape
'''
print (test_mother_name.shape)
input ('Here')
'''
print (np.unique(test_mother_name[test_label_list==1]).shape)
print (np.unique(test_mother_name).shape)
input ('')
'''
def fusion_metric(probs):

	x=np.argmax(probs,1)

	flag=0
	i=0
	count=0

	current_subject=test_mother_name[0]
	fused_pred=np.empty(0)
	fused_confidence=np.empty((0))
	#print  (valid_mother_name[:23])
	
	fused_real=np.empty(0)
	y_score=np.empty(0)
	video_tracker=0

	while True:

		
		if i == probs.shape[0]:

			break

		if test_mother_name[i] == current_subject :
			#print (valid_mother_name[i],probs[i],i)
			
			#chunk_probs=np.vstack([chunk_probs,probs[i]]) if chunk_probs.size else probs[i]
			if i== probs.shape[0]-1:
				
				chunk_probs=probs[video_tracker:]
				mean_score=np.mean(chunk_probs,axis=0)
				print (chunk_probs)
				pred=np.argmax(chunk_probs,1)
				print (pred)
				
				mode=stats.mode(pred)[0]
				mode=np.argmax(mean_score)
				#print (video_tracker,i)
				fused_pred=np.vstack([fused_pred,mode]) if fused_pred.size else mode
				fused_real=np.vstack([fused_real,test_label_list[video_tracker]]) if fused_real.size else test_label_list[video_tracker]
				y_score=np.vstack([y_score,mean_score[mode]]) if y_score.real.size else mean_score[mode]
				count+=1
				

			pass
		else:

				
			print (video_tracker,i)
			
			chunk_probs=probs[video_tracker:i]
			#chunk_probs=chunk_probs>0.8
			mean_score=np.mean(chunk_probs,axis=0)
			pred=np.argmax(chunk_probs,1)
			
			#print (pred)
			#confidence=np.max(chunk_probs,1)
			mode=stats.mode(pred)[0]
			mode=np.argmax(mean_score)
			#print (made)
			#input ('')
			#print (mean_score[mode],mean_score,test_label_list[video_tracker],mode)
			#input ('')
			fused_pred=np.vstack([fused_pred,mode]) if fused_pred.size else mode
			fused_real=np.vstack([fused_real,test_label_list[video_tracker]]) if fused_real.size else test_label_list[video_tracker]
			y_score=np.vstack([y_score,mean_score[mode]]) if y_score.real.size else mean_score[mode]

			#print (valid_mother_name[:i],np.argmax(probs[:i],1),valid_label_list[:i])
			video_tracker=i
			count+=1
			current_subject=test_mother_name[i]

			continue

		i+=1

	fused_pred=fused_pred.ravel().reshape(-1,1)
	fused_real=fused_real.ravel().reshape(-1,1)
	#print (fused_pred,fused_real)

	
	assert len(fused_pred) == len(fused_real)
	#print (fused_real.shape)
	#print ("Count is ",count)
	#input ('')

	
	return fused_pred,fused_real,y_score


root_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan/test_depression_siamese/'


save_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan/confusion/confusion_depression_siamese/'
results=os.listdir(root_dir)
print (results,len(results))
input ('')
for i in range (len(results)):

	test_dict=np.load(root_dir+results[i],allow_pickle=True).item()

	score=test_dict['score']
	probs=np.array(test_dict['probs'])

	print (probs.shape[0])

	if probs.shape[0] != test_mother_name.shape[0]:
		continue

	print (probs.shape)
	#input ('')
	fused_pred,fused_real,y_score=fusion_metric(probs)
	print (fused_pred.shape,fused_real.shape,y_score.shape)
	

	#print (np.sum(fused_real==1),np.sum(fused_real==0))
	
	acc=accuracy_score(fused_real,fused_pred)
	conf_mat=confusion_matrix(fused_real,fused_pred)
	f1=f1_score(fused_real,fused_pred,pos_label=0,average='binary')
	report= classification_report(fused_real,fused_pred)
	

	print (results[i])
	print ("Balanced Accuracy %0.3f" % acc)
	print ("FI score = %0.3f" %f1)
	print (conf_mat)
	print (report)
	
	print (save_dir+results[i]+"_norm")
	input ('')	
	#plot_confusion_matrix(conf_mat,"Mother",save_dir+results[i]+"_norm",normalize=True)
	#plot_confusion_matrix(conf_mat,"Mother",save_dir+results[i],normalize=False)
	#compute_roc_binary(fused_real,y_score,save_dir+results[i]+"ROC")
	
	
	
