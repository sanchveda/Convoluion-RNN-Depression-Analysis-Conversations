import numpy as np 
from scipy.io import loadmat
import os 
from scipy import stats
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from sklearn.metrics import confusion_matrix,f1_score,classification_report,roc_auc_score
import matplotlib.pyplot as plt 
root_zface_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/'



zface_filenames=[]
for name in  os.listdir(root_zface_dir):

	if name.endswith('fit.mat'):
		zface_filenames.append(str(name))



zface_filenames=np.array(zface_filenames)

def zface_info():


	return
directory='./kfold_full/valid_data_reliable_1.npy'

valid_data= np.load(directory,allow_pickle=True).item()

valid_mother_name,valid_child_name,valid_label_list,valid_start_list,valid_end_list=valid_data['mother_name'],\
                                                                                    valid_data['child_name'],\
                                                                                    valid_data['label'],\
                                                                                    valid_data['start_list'],\
                                                                                    valid_data['end_list']

def zface_script():
	ntracked_list=[]
	duration_list=[]
	lag_list=[]

	item_list_tracker=[]
	for idx, mother in enumerate(valid_mother_name):


		res= zface_filenames[zface_filenames == mother+'_fit.mat'].item()
		
		if  res in item_list_tracker:
			pass
		else:
			item_list_tracker.append(res)

			mat=loadmat(os.path.join(root_zface_dir,res))
			zface_data = mat['fit']
			isTracked_m  = zface_data[0]['isTracked']
		
			#print (len(isTracked_m))	
		
		
		segment_info= isTracked_m[valid_start_list[idx]:valid_end_list[idx]]
		
		n_tracked_count = np.sum(segment_info == 0)

		seq_len=[]
		start_index=[]
		end_index=[]
		seq=0
		'''
		if n_tracked_count == len(segment_info):
			seq_len.append(len(segment_info))
		else:
		'''
		for i,frame in enumerate(segment_info):
			
			#Handle the initial case 
			if i==0 and segment_info[i]==0:
				seq=1
				start_index.append(i)
				continue

			if i==0 and segment_info[i] ==1:
				continue

			#Handle when a non-tracked frame starts from the middle
			if segment_info[i] == 0 and segment_info[i-1] == 1:  
				seq=1 #Start the counter
				start_index.append(i)
				continue

			#Hhandle when non-tracked frame ends in middle
			if segment_info[i] == 1 and segment_info[i-1] == 0:
				seq_len.append(seq)
				end_index.append(i-1)
				seq=0 #Reset thhe counter
				continue
			
			#Handle when two consecutive zeros appear
			if segment_info[i] == 0 and segment_info[i-1] == 0:
				seq=seq+1
				continue	

		if seq > 0:
			seq_len.append(seq)
			end_index.append(i)  #Please note that i does not go to 900 . I remains at 899. This is differeent from C++
		
		end_index.insert(0,0) #Put a zero in the beginning of the list. TThis is just to calculate the lag lengths
		end_index=np.array(end_index)
		start_index=np.array(start_index)


		lag_len= start_index - end_index [:-1] + 1  # This will be give the lags between the non-tracked  frames
		#assert len(end_index) == len(start_index)
		'''
		check_sum =np.sum(np.array(seq_len))
		if check_sum != n_tracked_count:
			print (segment_info,n_tracked_count,seq_len,mother,valid_start_list[idx])
			input ('check_sum')
		
		
		for sequences in seq_len:
			if sequences == 0:
				print (segment_info,n_tracked_count,seq_len,mother,valid_start_list[idx])
				input ('Zeeros')
		'''
		'''
		for j,lex in enumerate(start_index):

			if end_index[j]-start_index[j]+1 != seq_len[j]:
				print (segment_info,n_tracked_count,seq_len,mother,valid_start_list[idx],start_index,end_index)
				input ('Here')
		'''
		print (mother,valid_start_list[idx],n_tracked_count,seq_len,start_index,end_index,lag_len)
		
		duration_list.append(seq_len)
		lag_list.append(lag_len)
		ntracked_list.append(n_tracked_count)

	assert len(duration_list) == len(lag_list) == len(ntracked_list) == len(valid_label_list)


def fusion_metric(probs,test_mother_name,test_label_list,tracking_info):

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

	mother_duration,mother_lag,mother_ntracked=tracking_info

	true_pos_dist=[]
	false_pos_dist=[]
	true_neg_dist=[]
	false_neg_dist=[]

	true_pos_duration_dist=[]
	false_pos_duration_dist=[]
	true_neg_duration_dist=[]
	false_neg_duration_dist=[]
	while True:

		
		if i == probs.shape[0]: #Terminating condition. That is if all the segments are already read

			break

		if test_mother_name[i] == current_subject :
			#print (valid_mother_name[i],probs[i],i)
			
			#chunk_probs=np.vstack([chunk_probs,probs[i]]) if chunk_probs.size else probs[i]
			if i== probs.shape[0]-1:
				
				chunk_probs=probs[video_tracker:]
				chunk_label=test_label_list[video_tracker:]
				mean_score=np.mean(chunk_probs,axis=0)
				#print (chunk_probs)
				pred=np.argmax(chunk_probs,1)
				track = mother_ntracked[video_tracker:]
				duration= mother_duration[video_tracker:]

				# Conditioning on thee four elements of confusion matrix for the length of non tracked 
				true_pos= (pred == 0 )& (chunk_label ==0)
				false_neg= (pred == 1) & (chunk_label ==0)
				true_neg= (pred==1) & (chunk_label==1)
				false_pos= (pred==0) & (chunk_label==1) 
				
				
				true_pos_count= track [true_pos]
				false_neg_count= track[false_neg]
				true_neg_count= track[true_neg]
				false_pos_count= track[false_pos]

				true_pos_duration= duration[true_pos]
				false_neg_duration= duration[false_neg]
				true_neg_duration= duration[true_neg]
				false_pos_duration= duration[false_pos]

				for ele in true_pos_duration:
					#print (true_pos_duration[iii])
					for items in ele:
						true_pos_duration_dist.extend(ele)

				for ele in false_neg_duration:
					for items in ele:
						false_neg_duration_dist.extend(ele)
				for ele in true_neg_duration:
					for items in ele:
						true_neg_duration_dist.extend(ele)

				for ele in false_pos_duration:
					for items in ele:
						false_pos_duration_dist.extend(ele)

				if len(true_pos_count):
					true_pos_dist.extend(true_pos_count)
				if len(false_neg_count):
					false_neg_dist.extend(false_neg_count)
				if len(false_pos_count):
					false_pos_dist.extend(false_pos_count)
				if len(true_neg_count):
					true_neg_dist.extend(true_neg_count)
				
				mode=stats.mode(pred)[0]
				#mode=np.argmax(mean_score)
				#print (video_tracker,i)
				track = mother_ntracked[video_tracker:i]
				fused_pred=np.vstack([fused_pred,mode]) if fused_pred.size else mode
				fused_real=np.vstack([fused_real,test_label_list[video_tracker]]) if fused_real.size else test_label_list[video_tracker]
				y_score=np.vstack([y_score,mean_score[mode]]) if y_score.real.size else mean_score[mode]
				count+=1
				

			pass
		else:

				
			#print (video_tracker,i)
			
			chunk_probs=probs[video_tracker:i]
			chunk_label=test_label_list[video_tracker:i]
			#chunk_probs=chunk_probs>0.8
			mean_score=np.mean(chunk_probs,axis=0)
			pred=np.argmax(chunk_probs,1)
			track = mother_ntracked[video_tracker:i]
			duration= mother_duration[video_tracker:i]

			

			# Conditioning on thee four elements of confusion matrix for the length of non tracked 
			true_pos= (pred == 0 )& (chunk_label ==0)
			false_neg= (pred == 1) & (chunk_label ==0)
			true_neg= (pred==1) & (chunk_label==1)
			false_pos= (pred==0) & (chunk_label==1) 
			
			
			true_pos_count= track [true_pos]
			false_neg_count= track[false_neg]
			true_neg_count= track[true_neg]
			false_pos_count= track[false_pos]

			true_pos_duration= duration[true_pos]
			false_neg_duration= duration[false_neg]
			true_neg_duration= duration[true_neg]
			false_pos_duration= duration[false_pos]

			for ele in true_pos_duration:
				#print (true_pos_duration[iii])
				for items in ele:
					true_pos_duration_dist.extend(ele)

			for ele in false_neg_duration:
				for items in ele:
					false_neg_duration_dist.extend(ele)
			for ele in true_neg_duration:
				for items in ele:
					true_neg_duration_dist.extend(ele)

			for ele in false_pos_duration:
				for items in ele:
					false_pos_duration_dist.extend(ele)
				#print (true_pos_duration)

				#nput ('Here ')
			#print (len(true_pos_duration_dist),len(false_pos_duration_dist))
			#input ('out')
			if len(true_pos_count):
				true_pos_dist.extend(true_pos_count)
			if len(false_neg_count):
				false_neg_dist.extend(false_neg_count)
			if len(false_pos_count):
				false_pos_dist.extend(false_pos_count)
			if len(true_neg_count):
				true_neg_dist.extend(true_neg_count)

			
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
	#print (len(false_pos_dist),len(true_pos_dist),len(false_neg_dist),len(true_neg_dist))
	#input ('Herer')
	
	return fused_pred,fused_real,y_score,[true_pos_dist,false_neg_dist,true_neg_dist,false_pos_dist],[true_pos_duration_dist,false_neg_duration_dist,true_neg_duration_dist,false_pos_duration_dist]

def process_epochs(root_dir,filelist,test_data_info):

	#print (root_dir, filelist)
	#input ('here')
	filelist=[root_dir + string for string in filelist]
	"filelist has the result filenames"
	test_mother_name,test_label_list,test_start_list,test_end_list,mother_duration,mother_lag,mother_ntracked=test_data_info
	
	#result=[]
	best_acc=0.0
	best_conf_mat=np.empty(0)
	best_y_score=np.empty(0)
	best_real= np.empty(0)
	best_pred= np.empty(0)
	best_auc_roc=0.0
	best_f1_score=0.0
	best_info_list=[]
	best_duration_list=[]
	for i in range (len(filelist)):
		#print (i,filelist[i].split('_')[-1:])
		data=np.load(filelist[i],allow_pickle=True).item()
		valid_score,valid_loss,probs=data['valid_score'],data['valid_loss'],data['probs']


		fused_pred,fused_real,y_score,track_info,duration_info=fusion_metric(probs,test_mother_name,test_label_list,[mother_duration,mother_lag,mother_ntracked])

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

			best_info_list=track_info
			best_duration_list= duration_info
	
	return best_acc,best_auc_roc,best_f1_score,best_conf_mat,best_y_score,best_real,best_pred,best_info_list,best_duration_list

def file_info (filepath):


	test_data=np.load(filepath,allow_pickle=True).item()

	test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list,mother_duration,mother_lag,mother_ntracked=test_data['mother_name'],\
                                                                                    test_data['child_name'],\
                                                                                    test_data['label'],\
                                                                                    test_data['start_list'],\
                                                                                    test_data['end_list'],\
                                                                                    test_data['mother_duration'],\
                                                                                    test_data['mother_lag'],\
                                                                                    test_data['mother_ntracked']



	
	
	return (test_mother_name,test_label_list,test_start_list,test_end_list,mother_duration,mother_lag,mother_ntracked)

raw_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/Sanchayan2/cnn_pool/result_valid_siamese/siamese_fold1/'
data_path='./kfold_full/'
final_raw_dir=[os.path.join(raw_dir,s) for s in os.listdir(raw_dir)]
#final_raw_dir=final_raw_dir[:5]


'''
#----ntracked - number of non-tracked frames
#----duration_list = duration of non tracked frames
#----lag _list = duration of the lag between the non tracked frames , i.e. the tracked frarmes
valid_data['mother_duration'] = np.array(duration_list)
valid_data['mother_lag']= np.array(lag_list)
valid_data['mother_ntracked']= np.array(ntracked_list)

#np.save('valid_data_for_plot_2.npy',valid_data)
'''
valid_data=np.load('valid_data_for_plot_2.npy',allow_pickle=True).item()
print (valid_data.keys())

assert len (valid_data['mother_name']) == len(valid_data['mother_lag']) 
#print (len(item_list))


results= os.listdir(raw_dir)
	
idx=1
data_path= 'valid_data_for_plot_'+str(idx)+'.npy'


test_data_info=file_info(data_path)

final_roc_auc=0.0
final_accuracy=0.0
final_f1_score=0.0
final_conf_mat=np.empty(0)
final_y_score=np.empty(0)
final_real=np.empty(0)
final_pred=np.empty(0)

best_results=results[0]
best_info=[]
best_duration_info=[]
for i in range (len(results)):

	folder_dir=raw_dir+'/'+results[i]+'/'
	
	all_files=os.listdir(folder_dir)
	
	dropout_rate,learn_rate,decay_rate=results[i].split('_')
	dropout_rate,learn_rate,decay_rate=float(dropout_rate),float(learn_rate),float(decay_rate)


	result_files=sorted([j for j in all_files if j.startswith('result')]) #Stores the names for all the epochs

	loss_files=sorted ([j for j in all_files if j.startswith('CRNN')])
	

	try :
		train_losses=np.load(folder_dir+loss_files[2],allow_pickle=True)
	except:
		print ("All files are not there. Re-run this configuration")




	best_acc,best_auc_roc,best_f1_score,best_conf_mat,best_y_score,best_real,best_pred,track_info,duration_info=process_epochs(folder_dir,result_files,test_data_info) 

	if best_f1_score> final_f1_score:

		final_roc_auc= best_auc_roc
		final_accuracy= best_acc
		final_f1_score= best_f1_score
		final_conf_mat= best_conf_mat
		final_y_score= best_y_score
		final_real= best_real
		final_pred= best_pred
		best_info=track_info
		best_duration_info=duration_info
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

true_pos_dist,false_neg_dist,true_neg_dist,false_pos_dist=best_info
true_pos_duration_dist,false_neg_duration_dist,true_neg_duration_dist,false_pos_duration_dist=duration_info

print (len(true_pos_dist),len(false_neg_dist),len(true_neg_dist),len(false_pos_dist))
#print (true_pos_dist,len(true_pos_dist))
print (len(true_pos_duration_dist),len(false_neg_duration_dist),len(true_neg_duration_dist),len(false_pos_duration_dist))
#input ('Here')
'''
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
'''

def do_plot(text_name,var_name,save_dir=''):

	print (text_name)

	weights= np.ones_like(var_name)/float(len(var_name))

	
	plt.hist (np.array(var_name),weights=weights)
	plt.title(text_name)
	plt.xlabel('Number of non-tracked frames')
	plt.ylabel('Number of Segments')
	plt.savefig(save_dir+text_name+'_var2.png')
	plt.close()
plot=True
text=['True_Positive','False_Negative','True_Negative','False_Positive']
var= [true_pos_duration_dist,false_neg_duration_dist,true_neg_duration_dist,false_pos_duration_dist]
var2=[true_pos_dist,false_neg_dist,true_neg_dist,false_pos_dist]
save_dir='./plot_mar3/'

for idx, text_name in enumerate(text):


	do_plot(text[idx],var2[idx],save_dir)



'''
if plot:
	save_dir='./confusion_plot_feb2_lstm/'
	plot_confusion_matrix(conf_mat,"Mother",save_dir+"norm_f1",normalize=True)
	compute_roc_binary(real_score,y_score,pred_score,save_dir+"norm_f1"+"ROC")
	np.save('scores_normal.npy',[real_score,y_score,pred_score])
'''