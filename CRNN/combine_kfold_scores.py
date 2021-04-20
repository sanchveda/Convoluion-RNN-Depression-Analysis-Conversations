import numpy as np
import os
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.metrics import confusion_matrix,f1_score,classification_report,roc_auc_score
from scipy import stats
import shutil

import cv2
from visualization import *


raw_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan2/cnn_pool/result_valid_siamese/'
data_path='./kfold_full/'

final_raw_dir=[os.path.join(raw_dir,s) for s in os.listdir(raw_dir)]

final_raw_dir=[name for name in final_raw_dir if name.split('/')[-1].startswith('siamese')]

print (final_raw_dir)
input('Here')
#final_raw_dir=final_raw_dir[:5]

'''
print (final_raw_dir)
input ('Here is the code')
'''
def fusion_metric(probs,test_mother_name,test_label_list):

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

		
		if i == probs.shape[0]: #Terminating condition. That is if all the segments are already read

			break

		if test_mother_name[i] == current_subject :
			#print (valid_mother_name[i],probs[i],i)
			
			#chunk_probs=np.vstack([chunk_probs,probs[i]]) if chunk_probs.size else probs[i]
			if i== probs.shape[0]-1:
				
				chunk_probs=probs[video_tracker:]
				mean_score=np.mean(chunk_probs,axis=0)
				#print (chunk_probs)
				pred=np.argmax(chunk_probs,1)
				#print (pred)
				
				mode=stats.mode(pred)[0]
				#mode=np.argmax(mean_score)
				#print (video_tracker,i)
				fused_pred=np.vstack([fused_pred,mode]) if fused_pred.size else mode
				fused_real=np.vstack([fused_real,test_label_list[video_tracker]]) if fused_real.size else test_label_list[video_tracker]
				y_score=np.vstack([y_score,mean_score[mode]]) if y_score.real.size else mean_score[mode]
				count+=1
				

			pass
		else:

				
			#print (video_tracker,i)
			
			chunk_probs=probs[video_tracker:i]
			#chunk_probs=chunk_probs>0.8
			mean_score=np.mean(chunk_probs,axis=0)
			pred=np.argmax(chunk_probs,1)
			
			#print (pred)
			#confidence=np.max(chunk_probs,1)
			mode=stats.mode(pred)[0]
			#mode=np.argmax(mean_score)
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


def report_classification_info(probs,test_mother_name,test_label_list,test_start_list,test_end_list):

	assert len(probs) == len(test_label_list)
	x=np.argmax(probs,1)
	
	flag=0
	i=0
	count=0

	current_subject=test_mother_name[0]
	
	#print  (valid_mother_name[:23])
	
	
	video_tracker=0
	information_list=[]
	while True:

		
		if i == probs.shape[0]: #Terminating condition. That is if all the segments are already read

			break

		if test_mother_name[i] == current_subject :
			#print (valid_mother_name[i],probs[i],i)
			
			#chunk_probs=np.vstack([chunk_probs,probs[i]]) if chunk_probs.size else probs[i]
			if i== probs.shape[0]-1:
				
				chunk_probs=probs[video_tracker:]
				chunk_label=test_label_list[video_tracker:]
				chunk_start=test_start_list[video_tracker:]
				chunk_end= test_end_list[video_tracker:]

				mean_score=np.mean(chunk_probs,axis=0)
				#print (chunk_probs)
				pred=np.argmax(chunk_probs,1)
				#print (pred)
				

				#condition= pred != chunk_label
				condition= pred == chunk_label

				info={}
				info['name'] = test_mother_name[video_tracker]
				info['start'] = chunk_start[condition]
				info['end'] = chunk_end[condition]
				info['true_label']= chunk_label[condition]
				info['pred_label']= pred[condition]
				information_list.append(info)

				
				count+=1
				

			pass
		else:

				
			#print (video_tracker,i)
			
			chunk_probs=probs[video_tracker:i]
			chunk_label=test_label_list[video_tracker:i]
			chunk_start=test_start_list[video_tracker:i]
			chunk_end= test_end_list[video_tracker:i]

			#print (test_mother_name[video_tracker],current_subject,chunk_probs.shape)
			
			#chunk_probs=chunk_probs>0.8
			mean_score=np.mean(chunk_probs,axis=0)
			pred=np.argmax(chunk_probs,1)
			
			#condition= pred != chunk_label
			condition= pred == chunk_label

			info={}
			info['name'] = test_mother_name[video_tracker]
			info['start'] = chunk_start[condition]
			info['end'] = chunk_end[condition]
			info['true_label']= chunk_label[condition]
			info['pred_label']= pred[condition]

			information_list.append(info)

			
			
			

			#print (valid_mother_name[:i],np.argmax(probs[:i],1),valid_label_list[:i])
			video_tracker=i
			count+=1
			current_subject=test_mother_name[i]

			continue

		i+=1


	
	return information_list


def file_info (filepath):


	test_data=np.load(filepath,allow_pickle=True).item()

	test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_data['mother_name'],\
                                                                                    test_data['child_name'],\
                                                                                    test_data['label'],\
                                                                                    test_data['start_list'],\
                                                                                    test_data['end_list']


	
	
	return (test_mother_name,test_label_list,test_start_list,test_end_list)

def process_epochs(root_dir,filelist,test_data_info):

	#print (root_dir, filelist)
	#input ('here')
	filelist=[root_dir + string for string in filelist]
	"filelist has the result filenames"
	test_mother_name,test_label_list,test_start_list,test_end_list=test_data_info
	
	#result=[]
	best_acc=0.0
	best_conf_mat=np.empty(0)
	best_y_score=np.empty(0)
	best_real= np.empty(0)
	best_pred= np.empty(0)
	best_auc_roc=0.0
	best_f1_score=0.0
	best_info_list=[]
	for i in range (len(filelist)):
		#print (i,filelist[i].split('_')[-1:])
		data=np.load(filelist[i],allow_pickle=True).item()
		valid_score,valid_loss,probs=data['valid_score'],data['valid_loss'],data['probs']


		fused_pred,fused_real,y_score=fusion_metric(probs,test_mother_name,test_label_list)

		#information_list=report_classification_info(probs,test_mother_name,test_label_list,test_start_list,test_end_list)

		
		acc=accuracy_score(fused_pred,fused_real)
		c_matrix=confusion_matrix(fused_real,fused_pred)
		auc_roc=roc_auc_score(fused_real,y_score)
		f1=f1_score(fused_real,fused_pred,pos_label=0,average='binary')

		if f1> best_f1_score :
			best_acc= acc
			best_auc_roc= auc_roc

			best_conf_mat = c_matrix
			best_y_score= y_score
			best_f1_score= f1
			best_real=fused_real.copy()
			best_pred=fused_pred.copy()

			#best_info_list=information_list
	
	return best_acc,best_auc_roc,best_f1_score,best_conf_mat,best_y_score,best_real,best_pred,0#,best_info_list

def main_result_analysis(plot=False):
	conf_mat=np.zeros((2,2))
	y_score=[]
	real_score=[]
	pred_score=[]
	avg_f1_score=0.0
	avg_accuracy=0.0

	f1_score_list=[]
	Accuracy_list=[]

	scores=[]

	param=[]

	for idx, fold in enumerate(final_raw_dir):
		print ('Evaluating %s' %fold)

		results= os.listdir(fold)


		data_path= './kfold_full/valid_data_reliable_'+str(idx)+'.npy'
		test_data_info=file_info(data_path)
		
		final_roc_auc=0.0
		final_accuracy=0.0
		final_f1_score=0.0
		final_conf_mat=np.empty(0)
		final_y_score=np.empty(0)
		final_real=np.empty(0)
		final_pred=np.empty(0)
		
		best_results=results[0]

		for i in range (len(results)):

			folder_dir=fold+'/'+results[i]+'/'
			all_files=os.listdir(folder_dir)

			dropout_rate,learn_rate,decay_rate=results[i].split('_')
			dropout_rate,learn_rate,decay_rate=float(dropout_rate),float(learn_rate),float(decay_rate)


			result_files=sorted([j for j in all_files if j.startswith('result')]) #Stores the names for all the epochs

			loss_files=sorted ([j for j in all_files if j.startswith('CRNN')])


			try :
				train_losses=np.load(folder_dir+loss_files[2],allow_pickle=True)
			except:
				print ("All files are not there. Re-run this configuration")




			best_acc,best_auc_roc,best_f1_score,best_conf_mat,best_y_score,best_real,best_pred,_=process_epochs(folder_dir,result_files,test_data_info) 
			
			if best_f1_score> final_f1_score:

				final_roc_auc= best_auc_roc
				final_accuracy= best_acc
				final_f1_score= best_f1_score
				final_conf_mat= best_conf_mat
				final_y_score= best_y_score
				final_real= best_real
				final_pred= best_pred

				best_results= str(dropout_rate)+'_'+str(learn_rate)+"_"+str(decay_rate)

			#report= classification_report(fused_real,fused_pred)
			'''
			print ("Best results ", results[i])
			input ('Here is something you will like')
			'''
			'''

			print (results[i])
			print ("Balanced Accuracy %0.3f" % acc)
			print ("FI score = %0.3f" %f1)
			print (conf_mat)
			'''
			#print (report)
			
			#print (save_dir+results[i]+"_norm")
			
			#plot_confusion_matrix(conf_mat,"Mother",save_dir+results[i]+"_norm",normalize=True)
			#plot_confusion_matrix(conf_mat,"Mother",save_dir+results[i],normalize=False)
			#compute_roc_binary(fused_real,y_score,save_dir+results[i]+"ROC")

		f1_score_list.append (final_f1_score)
		Accuracy_list.append (final_accuracy)
		avg_f1_score+=final_f1_score
		avg_accuracy+=final_accuracy
		
		conf_mat = conf_mat + final_conf_mat
		y_score.extend (final_y_score)
		real_score.extend(final_real)
		pred_score.extend(final_pred)
		
		scores.append(final_y_score)
		param.append(best_results)
		#print (final_roc_auc,final_accuracy,final_f1_score,final_conf_mat)
		#input ('Here')


	avg_f1_score/=5
	avg_accuracy/=5

	c_report=classification_report(real_score,pred_score)
	print (conf_mat)
	print (len(y_score),len(real_score),len(pred_score))
	print (pred_score)
	print (avg_f1_score,avg_accuracy)

	a,b=[real_score,y_score]
	print (len(a))
	print (y_score,pred_score)
	

	print ("Accuracy", Accuracy_list)
	print ("f1_score", f1_score_list)
	print (c_report)
	print (param)
	print (len(y_score))
	input ('Herer')
	if plot:
		save_dir='./confusion_plot_feb2_lstm/'
		plot_confusion_matrix(conf_mat,"Mother",save_dir+"norm_f1",normalize=True)
		compute_roc_binary(real_score,y_score,pred_score,save_dir+"norm_f1"+"ROC")
		np.save('scores_normal.npy',[real_score,y_score,pred_score])
		
	return final_y_score,param,scores

main_result_analysis()
#y_score,params,scores=main_result_analysis()
#print (params)
#input ('')

#-----------------------------------------Plotting Loss plots
configuration_list=[#'0.3_0.00001_0.0',
                    #'0.3_0.0001_0.0',
                    #'0.0_0.00001_0.0',
                    #'0.0_0.0001_0.0',
                    #'0.3_0.00001_0.9',
                    #'0.3_0.0001_0.9',
                    #'0.0_0.00001_0.9',
                    #'0.0_0.0001_0.9',
                    #'0.5_0.0001_0.0',
                    #'0.8_0.0001_0.0',
                    #'0.8_0.00001_0.0',
                    '0.0_0.000001_0.0',
                    '0.0_0.00000001_0.0'
                    ] # Uncomment this for siamese

print (configuration_list)

configuration_list= os.listdir(raw_dir)

"Now we are going to begin printing the plots"
#raw_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan2/cnn_pool/result_test_siamese/'
output_dir='./plot_mar3/'
#str1="cnn_pool"
str1="siamese_fold"
#str1=""
for i in range (1,5):

	#folder_name=  raw_dir + str1 + str(i)
	if i > 1:
		break

	#print (folder_name)
	
	for idx,params in enumerate(configuration_list):
		
		#file_dir= raw_dir + str1 + str(i)+ '/' + params + '/'
		file_dir= raw_dir + params + '/'
		#print (file_dir)
		#input ('Code is here')
		
		try:
			test_loss,test_score,train_loss,train_score=sorted ([j for j in os.listdir(file_dir) if j.startswith('CRNN')])
		except:
			print ('This one is not present',params)
			continue
		

		test_filename= file_dir +  test_loss
		train_filename= file_dir +  train_loss

		names=file_dir.split('/')
		
		test_out_name= output_dir + "fold_" + str(i) +"_" +names[-3] + '_' + names[-2] + '_test'
		train_out_name= output_dir + "fold_" + str(i) + "_"+ names[-3] + '_'+ names[-2] + '_train'
		
		#print (train_out_name,test_out_name)
		plot_loss_functions([train_filename,test_filename],train_out_name)
		
		print ('Plotting is done')
		input ('here')

#-----------------------------------------------------------------------------


'''
#--------Plotting probabilities------------
output_dir='./plot_bars_pool/'
str1='cnn_pool'
str1='cnn_pool_siamese'
#str1='crnn'
#str1='crnn_siamese'
mode='max'
for idx in range (len(params)):

	if mode == 'max':
		filename= output_dir + str1 + str(idx) + '_' + mode + '.png'
		text= 'Decision Function scores with Max Fusion'
	elif mode=='avg':
		filename= output_dir + str1 + str(idx) + '_' + mode +  '.png'
		text= 'Decision Function Scores with Average Fusion'
	plot_probabilities(scores[idx],filename,text,'bar')
'''
#main_result_analysis()
#loss_analysis()
'''
print (os.listdir(data_path))
#params= main_result_analysis()

conf_mat=np.zeros((2,2))
y_score=[]
real_score=[]
pred_score=[]
avg_f1_score=0.0
avg_accuracy=0.0

f1_score_list=[]
Accuracy_list=[]

param=[]
info_list=[]
for idx, fold in enumerate(final_raw_dir):
	print ('Evaluating %s' %fold)

	results= os.listdir(fold)
	

	data_path= './kfold_full/valid_data_reliable_'+str(idx)+'.npy'
	test_data_info=file_info(data_path)
	
	final_roc_auc=0.0
	final_accuracy=0.0
	final_f1_score=0.0
	final_conf_mat=np.empty(0)
	final_y_score=np.empty(0)
	final_real=np.empty(0)
	final_pred=np.empty(0)
	final_info_list=np.empty(0)
	best_results=results[0]

	for i in range (len(results)):

		folder_dir=fold+'/'+results[i]+'/'
		all_files=os.listdir(folder_dir)

		dropout_rate,learn_rate,decay_rate=results[i].split('_')
		dropout_rate,learn_rate,decay_rate=float(dropout_rate),float(learn_rate),float(decay_rate)


		result_files=sorted([j for j in all_files if j.startswith('result')]) #Stores the names for all the epochs

		loss_files=sorted ([j for j in all_files if j.startswith('CRNN')])


		try :
			train_losses=np.load(folder_dir+loss_files[2],allow_pickle=True)
		except:
			print ("All files are not there. Re-run this configuration")




		best_acc,best_auc_roc,best_f1_score,best_conf_mat,best_y_score,best_real,best_pred,best_info_list=process_epochs(folder_dir,result_files,test_data_info) 
		
		if best_f1_score> final_f1_score:

			final_roc_auc= best_auc_roc
			final_accuracy= best_acc
			final_f1_score= best_f1_score
			final_conf_mat= best_conf_mat
			final_y_score= best_y_score
			final_real= best_real
			final_pred= best_pred
			final_info_list=best_info_list

			best_results= str(dropout_rate)+'_'+str(learn_rate)+"_"+str(decay_rate)

		#report= classification_report(fused_real,fused_pred)

		#print (report)
		
		#print (save_dir+results[i]+"_norm")
		
		#plot_confusion_matrix(conf_mat,"Mother",save_dir+results[i]+"_norm",normalize=True)
		#plot_confusion_matrix(conf_mat,"Mother",save_dir+results[i],normalize=False)
		#compute_roc_binary(fused_real,y_score,save_dir+results[i]+"ROC")

	f1_score_list.append (final_f1_score)
	Accuracy_list.append (final_accuracy)
	avg_f1_score+=final_f1_score
	avg_accuracy+=final_accuracy
	
	conf_mat = conf_mat + final_conf_mat
	y_score.extend (final_y_score)
	real_score.extend(final_real)
	pred_score.extend(final_pred)
	
	param.append(best_results)
	info_list.append(final_info_list)
	#print (final_roc_auc,final_accuracy,final_f1_score,final_conf_mat)
	#input ('Here')


avg_f1_score/=5
avg_accuracy/=5
c_report=classification_report(real_score,pred_score)
print (conf_mat)
print (len(y_score),len(real_score),len(pred_score))
print (pred_score)
print (avg_f1_score,avg_accuracy)


print ("Accuracy", Accuracy_list)
print ("f1_score", f1_score_list)
print (c_report)
print (param)
'''
'''
info= info_list[0]

read_data_path= "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/normalized_raw_final/"
output_video_directory='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan2/misclassified/fold0/label1_pred1/'

for elements in info:

	name= elements['name']
	start_list= elements['start']
	end_list= elements['end']
	label= elements['true_label']
	pred = elements['pred_label']
	
	assert len(start_list) == len(label) == len(pred) == len(end_list)
	condition=  (label==1) & (pred==1)

	if start_list.size and condition.any(): #If there are any misclassified segments
	
		

		
		for idx in range (len(start_list)): #Iterating over every misclassified segments
			#Video level
			start=start_list[idx]
			end= end_list[idx]
			img_array=[]

			#-------------------------------Reading the frames ------------------------------------------#
			for jj in range (start,end):
				#Frame level
				data_name = os.path.join (read_data_path,name,'%s_%s_%05d.jpg' %(name,"norm",jj))
				img = cv2.imread(data_name)
				height,width,layers= img.shape
				size= (width,height)
				img_array.append(img)
			#--------------------------------------------#


			#----------Writing the video--------------------------------#
			outname=name+"_"+str(idx)+'_'+str(float(start)/1800)+'_'+str(float(end)/1800)+".avi"
			out = cv2.VideoWriter(output_video_directory+outname,cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
			
			for i in range(len(img_array)):
				out.write(img_array[i])
			out.release()
			#-----------------------------------------------------------#
			
			print (outname)

'''