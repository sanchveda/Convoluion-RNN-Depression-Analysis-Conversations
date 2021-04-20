"This file only does error analysis from Nov 9 results"

import os 
import numpy as np 

from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.metrics import confusion_matrix,f1_score,classification_report
from scipy import stats



test_data=np.load('./data/test_data_1.npy',allow_pickle=True).item()

test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_data['mother_name'],\
                                                                                    test_data['child_name'],\
                                                                                    test_data['label'],\
                                                                                    test_data['start_list'],\
                                                                                    test_data['end_list']
root_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan/test_depression_cnn_rnn/'
save_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan/confusion_siamese_psi/'

error_truth_path='../data_analysis/ground_truth_data/error_truth.npy'

full_data=	np.load(error_truth_path,allow_pickle=True).item()
fam_id_full	=	full_data['fam_id']
phq9_full	=	full_data['phq9']
lag_full		=	full_data['lag']
group_full	=	full_data['group']

assert len(fam_id_full)==len(phq9_full) == len(lag_full) == len(group_full)

def analyse_error(true_score,predicted_score,names,f1_score):

	assert len(true_score) == len(predicted_score) == len(names)
	names=np.array([name[:4] for name in names]).reshape(-1,1)
	
	"analysis of Error"
	indices1 = (true_score==0) & (predicted_score==1) # Actually depressed but classified as non-depressed
	indices2 = (true_score==1) & (predicted_score==0) # Actually non-depressed but predicted as depressed

	print (names[indices1],names[indices2])
	phq_value_depressed=[]
	lag_depressed=[]
	phq_value_normal=[]
	lag_normal=[]
	for idx,val in enumerate(np.unique(names[indices1])):

		index= fam_id_full==val
		phq_value_depressed.append(phq9_full[index])
		lag_depressed.append(lag_full[index])

	for idx,val in enumerate(np.unique(names[indices2])):

		index= fam_id_full == val
		phq_value_normal.append(phq9_full[index])
		lag_normal.append(lag_full[index])

	phq_value_depressed=np.array(phq_value_depressed)
	phq_value_normal= np.array (phq_value_normal)
	lag_depressed=np.array(lag_depressed)
	lag_normal=	np.array(lag_normal)


	print ('*******************************************************')
	print ("F1 Score =",f1_score,lag_depressed,lag_normal)
	print ("phq9 = ", phq_value_depressed,phq_value_normal)
	input ('')
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
	fam_id=[]
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
				#print (video_tracker,i)
				fused_pred=np.vstack([fused_pred,mode]) if fused_pred.size else mode
				fused_real=np.vstack([fused_real,test_label_list[video_tracker]]) if fused_real.size else test_label_list[video_tracker]
				y_score=np.vstack([y_score,mean_score[mode]]) if y_score.real.size else mean_score[mode]
				count+=1
				
				fam_id.append(test_mother_name[video_tracker])
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
			#print (mean_score[mode],mean_score,test_label_list[video_tracker],mode)
			#input ('')
			fused_pred=np.vstack([fused_pred,mode]) if fused_pred.size else mode
			fused_real=np.vstack([fused_real,test_label_list[video_tracker]]) if fused_real.size else test_label_list[video_tracker]
			y_score=np.vstack([y_score,mean_score[mode]]) if y_score.real.size else mean_score[mode]

			#print (valid_mother_name[:i],np.argmax(probs[:i],1),valid_label_list[:i])
			fam_id.append(test_mother_name[video_tracker])
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

	
	return fused_pred,fused_real,y_score,fam_id




results=os.listdir(root_dir)

for i in range (len(results)):

	test_dict=np.load(root_dir+results[i],allow_pickle=True).item()

	score=test_dict['score']
	probs=np.array(test_dict['probs'])

	print (probs.shape[0])

	if probs.shape[0] != test_mother_name.shape[0]:
		continue

	print (probs.shape)
	#input ('')
	fused_pred,fused_real,y_score,fam_id=fusion_metric(probs)
	print (fused_pred.shape,fused_real.shape,y_score.shape,len(fam_id))
	

	#print (np.sum(fused_real==1),np.sum(fused_real==0))
	
	acc=accuracy_score(fused_real,fused_pred)
	conf_mat=confusion_matrix(fused_real,fused_pred)
	f1=f1_score(fused_real,fused_pred,average='macro')
	report= classification_report(fused_real,fused_pred)
	

	print (results[i])
	print ("Balanced Accuracy %0.3f" % acc)
	print ("FI score = %0.3f" %f1)
	print (conf_mat)
	print (report)
	
	analyse_error(fused_real,fused_pred,fam_id,f1)


	input ('')
	
	#plot_confusion_matrix(conf_mat,"Mother+Child",save_dir+results[i]+"_norm",normalize=True)
	#plot_confusion_matrix(conf_mat,"Mother+Child",save_dir+results[i],normalize=False)
	#compute_roc_binary(fused_real,y_score,save_dir+results[i]+"ROC")
