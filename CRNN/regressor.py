#from evaluation_metrics import *
from sklearn.svm import SVR
#from evaluation_metrics import *
import numpy as np 
from sklearn import preprocessing
import os
from scipy.io import loadmat
from scipy.stats import iqr
import pandas as pd

zface_dir ='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/'
au_dir	='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/AU_input/formatted/occurrence/'
au_intensity_dir = '/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/AU_input/formatted/intensity/'
life_code_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/LIFE/LIFE Coding Stop Frame Constructs/PSI Task/TXT Files/StartFrame/'
covarep_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/covarep/'
opensmile_egemaps_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/opensmile_eGeMAPSv01a/'
opensmile_vad_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/opensmile_vad_opensource/'
opensmile_prosody_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/opensmile_prosodyAcf/'
volume_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/audio_processed/volume'

zface_list=np.array([name for name in os.listdir(zface_dir) if name.endswith('fit.mat')])
au_list=np.array([name for name in os.listdir(au_dir) if name.endswith('.mat')])
au_intensity_list=np.array([name for name in os.listdir(au_intensity_dir) if name.endswith ('.mat')])
covarep_list=np.array([name for name in os.listdir(covarep_dir) if name.endswith('.hdf')])
opensmile_egemaps_list=np.array([name for name in os.listdir(opensmile_egemaps_dir) if name.endswith('.hdf')])
opensmile_vad_list=np.array([name for name in os.listdir(opensmile_vad_dir) if name.endswith('.hdf')])
opensmile_prosody_list= np.array([name for name in os.listdir(opensmile_prosody_dir) if name.endswith('.hdf')])
volume_list=np.array([name for name in os.listdir(volume_dir) if name.endswith ('.hdf')])


def does_zface_exist(filename):
	name = zface_list[zface_list==filename]
	if len(name):
		return True
	return False 
def does_AU_exist (filename):
	name = au_list[au_list==filename]
	if len(name):
		return True
	return False
def does_covarep_exist(filename):
	name =covarep_list[covarep_list==filename]

	if len(name):
		return True
	return False
def does_au_intensity_exist(filename):
	'''
	for intensity_files in au_intensity_list:
		

		family_subject,task,video = intensity_files.split('.mat')[0].split('_')

		print (family,subject,video)
		input ('')
	'''
	name = au_intensity_list[au_intensity_list==filename]
	
	if len(name):
		return True
	return False
def does_opensmile_egemaps_exist (filename):
	name = opensmile_egemaps_list[opensmile_egemaps_list==filename]
	

	if len(name):
		return True
	return False
def does_opensmile_prosody_exist (filename):
	name = opensmile_prosody_list[opensmile_prosody_list==filename]
	
	
	if len(name):
		return True
	return False
def does_opensmile_vad_exist (filename):
	name = opensmile_vad_list[opensmile_vad_list==filename]
	
	if len(name):
		return True
	return False
def does_volume_exist (filename):
	name= volume_list [ volume_list == filename]
	if len(name):
		return True 
	return False 

def svr_comprehensive(train_data,train_label,val_data,val_label,test_data,test_label):

	scaler = preprocessing.StandardScaler().fit(train_data)
	train_data_scaled = scaler.transform(train_data)
	val_data_scaled   = scaler.transform(val_data)
	test_data_scaled  = scaler.transform(test_data)

	reg1=SVR(kernel='linear',C=0.01)
	reg2=SVR(kernel='linear',C=0.1)
	reg3=SVR(kernel='linear',C=1)
	reg4=SVR(kernel='linear',C=10)
	reg5=SVR(kernel='linear',C=100)
	
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
	print(x)
	test_predicted_label=bestreg.predict(test_data_scaled)
	test_mse=mean_error(test_label,test_predicted_label)
	test_pcc=pcc(test_label,test_predicted_label)

	print("Test mean-squared-errror %f" % (test_mse))
	
	return test_mse,test_pcc,x

def read_zface_features (zface_filename,start_list,end_list):

	res_vector=np.empty(0)

	mat=loadmat(os.path.join(zface_dir,zface_filename))
	zface_data = mat['fit']

	no_frames=zface_data.shape[1]
	isTracked_m  = zface_data[0]['isTracked']
	headPose_m   = zface_data[0]['headPose']
	pts_3d_m= zface_data[0]['pts_3d']
	pts_2d_m= zface_data[0]['pts_2d']
	pdmPars_m = zface_data[0]['pdmPars']
	no_pdm_parameters = 30
	
	isTracked = np.zeros(no_frames)
	pts_3d= np.zeros((no_frames,512*3))
	pts_2d= np.zeros((no_frames,49*2))
	headPose = np.zeros((no_frames,3) )
	pdmPars = np.zeros((no_frames,no_pdm_parameters) )
	

	for ii in range (no_frames):
		isTracked[ii] = isTracked_m[ii][0]
		if isTracked[ii] != 0:
			headPose[ii]  = headPose_m[ii].reshape(1,3)[0]
			pdmPars[ii]   = pdmPars_m[ii].reshape(1,no_pdm_parameters)[0]
			pts_3d[ii]	  = pts_3d_m[ii].ravel()
			pts_2d[ii]	  = pts_2d_m[ii].ravel()

	#print (zface_filename,no_frames,start_list,end_list)
	pdmPars = pdmPars[:,:15]
	
	if no_frames < 10000:
		return res_vector
	
	vector=np.concatenate((pdmPars,headPose),axis=1)   #Use this line to add as many zface_features as you want for thee raw zface vector
	
	for idx, (start, end) in enumerate(zip(start_list,end_list)):

		amp_vector= vector[start:end,:]
		vel_vector= amp_vector[1:,:] - amp_vector[:-1,:]
		acc_vector= vel_vector[1:,:] - vel_vector[:-1,:]

		amp_vector= amp_vector [2:,:]
		vel_vector= vel_vector [1:,:]

		
		amp_stats = compute_statistics (amp_vector)
		vel_stats = compute_statistics (vel_vector)
		acc_stats = compute_statistics (acc_vector)

		feature_vector= np.hstack ([amp_stats,vel_stats,acc_stats])

		res_vector= np.vstack ([res_vector,feature_vector]) if res_vector.size else feature_vector
		
		
	
	
	return res_vector

def read_AUs (AU_filename,start_list,end_list):

	res_vector=np.empty(0)

	mat=loadmat(os.path.join(au_dir,AU_filename))
	au_data = mat['occurrence']

	no_frames=len(au_data)
	#print  (AU_filename,no_frames,start_list,end_list)

	if no_frames <10000:
		return res_vector,no_frames
	
	vector = au_data.copy()

	for idx, (start,end) in enumerate(zip(start_list,end_list)):
		amp_vector= vector[start:end,:]
		vel_vector= amp_vector[1:,:] - amp_vector[:-1,:]
		#acc_vector= vel_vector[1:,:] - vel_vector[:-1,:]

		amp_vector= amp_vector[1:,:]
		vel_vector= vel_vector[:,:]

		amp_stats = compute_statistics (amp_vector)
		vel_stats = compute_statistics (vel_vector)
		#acc_stats = compute_statistics (acc_vector)

		feature_vector= np.hstack ([amp_stats,vel_stats])

		res_vector= np.vstack ([res_vector,feature_vector]) if res_vector.size else feature_vector

	return res_vector, no_frames

def read_AU_intensity(AU_filename,start_list,end_list):

	res_vector=np.empty(0)
	mat= loadmat(os.path.join(au_intensity_dir,AU_filename))
	#au_data= mat['intensity']

	au6= mat['AU6'][0][0][0][0].reshape(-1,1)
	au12=mat['AU12'][0][0][0][0].reshape(-1,1)

	au10=mat['AU10'][0][0][0][0].reshape(-1,1)
	au14=mat['AU14'][0][0][0][0].reshape(-1,1)

	assert len(au6) == len(au12) == len(au10) == len(au14)
	
	no_frames=len(au6)
	if no_frames < 10000:
		return res_vector,no_frames

	vector= np.concatenate((au6,au10,au12,au14),axis=1)

	for idx, (start,end) in enumerate(zip(start_list,end_list)):
		amp_vector= vector [start : end , :]

		feature_vector= compute_statistics (amp_vector)

		res_vector= np.vstack ([res_vector,feature_vector]) if res_vector.size else feature_vector
	
	
	return res_vector,no_frames

def read_covarep(covarep_filename,start_list,end_list,start_time_in_seconds=0.0,end_time_in_seconds=0.0):

	res_vector=np.empty(0)
	data = pd.read_hdf(os.path.join(covarep_dir,covarep_filename), 'df')
	
	covarep_vowelSpace= data['vowelSpace']
	covarep_MCEP_0= data['MCEP_0']
	covarep_MCEP_1= data['MCEP_1']
	covarep_VAD= data['VAD']
	covarep_f0 = data['f0']
	covarep_NAQ = data['NAQ']
	covarep_QOQ= data['QOQ']
	covarep_MDQ= data['MDQ']
	covarep_peakSlope= data['peakSlope']
	covarep_F1= data['F1']
	covarep_F2= data['F2']
	size= data.shape[1] * 5 
	
	if len (data) < 50000:
		return res_vector

	start_time_in_seconds = start_list /  30.0 
	end_time_in_seconds = end_list / 30.0 



	median_vector= [covarep_vowelSpace,covarep_MCEP_0,covarep_MCEP_1,covarep_VAD,covarep_f0,covarep_NAQ,covarep_QOQ,covarep_MDQ,covarep_peakSlope,covarep_F1,covarep_F2]
	iqr_vector= [covarep_f0,covarep_NAQ,covarep_QOQ,covarep_MDQ,covarep_peakSlope,covarep_F1,covarep_F2]
	#resampled_data = signal.resample(data, no_frames)
	#input ('here')
	for idx, (start,end) in enumerate(zip(start_time_in_seconds,end_time_in_seconds)):
		
		med_list=[]
		for items in median_vector:
			vec= np.median (items[start:end],axis=0)
			med_list.append(vec)

		item_med_vec= np.array(med_list)
		iqr_list=[]
		for items in iqr_vector:
			vec= iqr(items[start:end],axis=0)
			
			iqr_list.append(vec)
		item_iqr_vec= np.array(iqr_list)

		feature_vector= np.hstack([item_med_vec,item_iqr_vec])
			
		res_vector= np.vstack ([res_vector,feature_vector]) if res_vector.size else feature_vector   
	return  res_vector


def read_opensmile(openface_file_list,start_list,end_list,start_time_in_seconds=0.0,end_time_in_seconds=0.0):
	opensmile_lld,opensmile_csv,opensmile_prosody,opensmile_vad= openface_file_list

	res_vector=np.empty(0)

	data_lld= pd.read_hdf(os.path.join(opensmile_egemaps_dir,opensmile_lld),'df')
	data_csv= pd.read_hdf(os.path.join(opensmile_egemaps_dir,opensmile_csv),'df')
	data_prosody= pd.read_hdf(os.path.join(opensmile_prosody_dir,opensmile_prosody),'df')
	data_vad=   pd.read_hdf(os.path.join(opensmile_vad_dir,opensmile_vad),'df')

	start_time_in_seconds = start_list /  30.0 
	end_time_in_seconds = end_list / 30.0 
	#print (data_vad)
	#input ('Here')
	
	for idx, (start,end) in enumerate(zip(start_time_in_seconds,end_time_in_seconds)):

		csv_vector= np.median(data_csv [start:end],axis=0)

		lld_vector_med= np.median (data_lld [start:end],axis=0)
		lld_vector_iqr= iqr (data_lld[start:end],axis=0)
		
		prosody_med= np.median(data_prosody[start:end],axis=0)

		vad_med= np.median(data_vad[start:end], axis=0)

		#print (csv_vector.shape,lld_vector_med.shape,lld_vector_iqr.shape,prosody_med.shape,vad_med.shape)
		feature_vector= np.hstack ([csv_vector,lld_vector_med,lld_vector_iqr,prosody_med,vad_med])
		#print (data_lld.shape,data_csv.shape,data_prosody.shape,data_vad.shape)
		res_vector= np.vstack([res_vector,feature_vector]) if res_vector.size else feature_vector
		
	return res_vector

def read_volume(volume_filename,start_list,end_list,start_time_in_seconds=0.0,end_time_in_seconds=0.0):

	res_vector=np.empty(0)

	data = pd.read_hdf(os.path.join(volume_dir,volume_filename),'df')
	start_time_in_seconds = start_list /  30.0 
	end_time_in_seconds = end_list / 30.0 

	for idx, (start,end) in enumerate(zip(start_time_in_seconds,end_time_in_seconds)):

		vol_med= np.median(data[start:end],axis=0)
		vol_iqr= iqr(data[start:end],axis=0)

		feature_vector= np.hstack ([vol_med,vol_iqr]) 

		res_vector= np.vstack([res_vector,feature_vector]) if res_vector.size else feature_vector
		
	return res_vector

def compute_statistics(vector):


	max_vector=np.max(vector,axis=0)
	mean_vector=np.mean(vector,axis=0)
	std_vector=np.std(vector,axis=0)
	iqr_vector=iqr (vector,axis=0)
	median_vector= np.median(vector,axis=0)


	stats_vec= np.hstack([max_vector,mean_vector,std_vector,iqr_vector,median_vector])
	return stats_vec


def expand_filenames(family_list,filelist):

	result_filename=[]
	result_zface_filename=[]
	result_au_filename=[]
	result_au_intensity_filename=[]
	result_covarep_filename=[]
	
	result_opensmile_filenames_lld=[]
	result_opensmile_filenames_csv=[]
	result_opensmile_prosody=[]
	result_opensmile_vad=[]
	
	result_volume_filename=[]
	result_information=[]   # This is regarding the gender  and the subejct id 

	resuult_gender=[]
	count=0
	for idx, item in enumerate(family_list):
		for names in filelist:

			first_term= names.split('_')[0]
			family,subject= first_term[:4], first_term[4:]


			zface_filename= family + str(subject) +'_02_01_fit.mat'
			au_filename=    family + str(subject + '_02_01_au_out.mat')
			au_intensity_filename= family + str(subject) + '_02_01.mat'
			covarep_filename= 'TPOT_' + family + '_' + str(subject) + '_2.hdf'
			opensmile_filename_lld= 'TPOT_' + family + '_' + str(subject) + '_2_lldcsvoutput.hdf'
			opensmile_filename_csv= 'TPOT_'  + family + '_' + str(subject) + '_2_csvoutput.hdf'

			opensmile_prosody_file= 'TPOT_'  + family + '_' + str(subject) + '_2_csvoutput.hdf'
			opensmile_vad_file    = 'TPOT_'  + family + '_' + str(subject) + '_2_csvoutput.hdf'
			volume_file 		  = 'TPOT_'  + family + '_' + str(subject) + '_2.hdf'

			information= family + '_' + str(subject) #+'_' + str(gender_list[idx])

			

			

			zface_exist= does_zface_exist(zface_filename)
			au_exist=    does_AU_exist(au_filename)		
			au_intensity_exist= does_au_intensity_exist (au_intensity_filename)
			covarep_exist= does_covarep_exist (covarep_filename)
			opensmile_lld_exist= does_opensmile_egemaps_exist (opensmile_filename_lld)
			opensmile_csv_exist= does_opensmile_egemaps_exist (opensmile_filename_csv)
			opensmile_prosody_exist= does_opensmile_prosody_exist (opensmile_prosody_file)
			opensmile_vad_exist =  does_opensmile_vad_exist (opensmile_vad_file)
			volume_exist        =  does_volume_exist (volume_file)
			
			
			if family == item and zface_exist and au_exist and au_intensity_exist and covarep_exist and opensmile_lld_exist and opensmile_csv_exist  and opensmile_prosody_exist  and opensmile_vad_exist and volume_exist: 
				result_filename.append (names)
				result_zface_filename.append(zface_filename)
				result_au_filename.append (au_filename)
				result_covarep_filename.append(covarep_filename)
				result_information.append(information)	
				
				result_opensmile_filenames_lld.append(opensmile_filename_lld)
				result_opensmile_filenames_csv.append(opensmile_filename_csv)
				result_opensmile_prosody.append(opensmile_prosody_file)
				result_opensmile_vad.append(opensmile_vad_file)
				
				result_volume_filename.append(volume_file)
				result_au_intensity_filename.append(au_intensity_filename)
	
	assert len (result_filename) == len (result_zface_filename) == len(result_au_filename) == len(result_covarep_filename) == len(result_information) == len(result_volume_filename)
				

	result_filename= np.array(result_filename)
	result_zface_filename= np.array(result_zface_filename)
	result_au_filename= np.array(result_au_filename)
	result_covarep_filename= np.array(result_covarep_filename)
	rersult_information= np.array(result_information)
	result_opensmile_filenames= [np.array(result_opensmile_filenames_lld),np.array(result_opensmile_filenames_csv),np.array(result_opensmile_prosody),np.array(result_opensmile_vad)]
	result_volume_filename= np.array(result_volume_filename)
	result_au_intensity_filename=np.array(result_au_intensity_filename)

	print (result_filename)
	input ('Here')
	result=[result_filename,
			result_zface_filename,
			result_au_filename,
			result_au_intensity_filename,
			result_covarep_filename,
			result_opensmile_filenames,
			result_volume_filename,
			result_information]

	
	return result 

def prepare_features (filename , start_list, end_list , panas_score):
	total_data=np.empty(0)
	total_label=np.empty(0)
	total_info=np.empty(0)

	unique_filenames= sorted(np.unique(filename))
	for idx,names in enumerate(unique_filenames): 

		first_term= names.split('_')[0]
		family,subject= first_term[:4], first_term[4:]

		condition= filename == names
		chunk_families = filename [ condition ] 
		event_start = start_list [ condition]
		event_end  =  end_list [condition]
		chunk_panas = panas_score[condition]
		
		zface_filename= family + str(subject) +'_02_01_fit.mat'
		au_filename=    family + str(subject + '_02_01_au_out.mat')
		au_intensity_filename= family + str(subject) + '_02_01.mat'
		covarep_filename= 'TPOT_' + family + '_' + str(subject) + '_2.hdf'
		opensmile_filename_lld= 'TPOT_' + family + '_' + str(subject) + '_2_lldcsvoutput.hdf'
		opensmile_filename_csv= 'TPOT_'  + family + '_' + str(subject) + '_2_csvoutput.hdf'

		opensmile_prosody_file= 'TPOT_'  + family + '_' + str(subject) + '_2_csvoutput.hdf'
		opensmile_vad_file    = 'TPOT_'  + family + '_' + str(subject) + '_2_csvoutput.hdf'
		volume_file 		  = 'TPOT_'  + family + '_' + str(subject) + '_2.hdf'

		information= family + '_' + str(subject) #+'_' + str(gender_list[idx])

		zface_exist= does_zface_exist(zface_filename)
		au_exist=    does_AU_exist(au_filename)		
		au_intensity_exist= does_au_intensity_exist (au_intensity_filename)
		covarep_exist= does_covarep_exist (covarep_filename)
		opensmile_lld_exist= does_opensmile_egemaps_exist (opensmile_filename_lld)
		opensmile_csv_exist= does_opensmile_egemaps_exist (opensmile_filename_csv)
		opensmile_prosody_exist= does_opensmile_prosody_exist (opensmile_prosody_file)
		opensmile_vad_exist =  does_opensmile_vad_exist (opensmile_vad_file)
		volume_exist        =  does_volume_exist (volume_file)



		if zface_exist and au_exist and au_intensity_exist and covarep_exist and opensmile_lld_exist and opensmile_csv_exist  and opensmile_prosody_exist  and opensmile_vad_exist and volume_exist: 
		
			zface_vector = read_zface_features(zface_filename,event_start,event_end)  #  This will be a construct * dim vector
			au_vector, no_frames=read_AUs(au_filename,event_start,event_end)
			au_intensity_vector, no_frames = read_AU_intensity ( au_intensity_filename, event_start,event_end)
			covarep_vector= read_covarep(covarep_filename,event_start,event_end)
			opensmile_vector= read_opensmile([opensmile_filename_lld,opensmile_filename_csv,opensmile_prosody_file,opensmile_vad_file],event_start,event_end)
			volume_vector = read_volume(volume_file,event_start,event_end)
			

			if not len(zface_vector) or not len(au_vector) or not len(covarep_vector):
				#print (au_filelist[idx])
				#print (zface_vector)
				#print (au_vector)
				continue
			combined_vector= np.concatenate((zface_vector,au_vector,au_intensity_vector,covarep_vector,opensmile_vector,volume_vector),axis=1)
			
			total_data=np.vstack([total_data,combined_vector]) if total_data.size else combined_vector
			

			total_label=np.hstack ([total_label,chunk_panas]) if total_label.size else chunk_panas
			total_info= np.hstack([total_info,chunk_families]) if total_info.size else chunk_families
			print(idx, total_data.shape,total_label.shape,total_info.shape)
			
	
	assert len(total_data) == len(total_label) == len(total_info)

	
	return total_data,total_label,total_info

fold_dir= 'kfold_reg/'

test_list= sorted ([name for name in os.listdir(fold_dir) if name.startswith('test')])
valid_list= sorted ([name for name in os.listdir(fold_dir) if name.startswith('valid')])
train_list= sorted ([name for name in os.listdir(fold_dir) if name.startswith('train')])

for idx, fold in enumerate(train_list):

	if idx < 3:
		continue 
	train_data=np.load(os.path.join(fold_dir,train_list[idx]),allow_pickle=True).item()
	valid_data=np.load(os.path.join(fold_dir,valid_list[idx]),allow_pickle=True).item()
	test_data=np.load(os.path.join(fold_dir,test_list[idx]),allow_pickle=True).item()

	train_mother_name=train_data['mother_name']
	train_child_name=train_data['child_name']
	train_panas_score=train_data['label']
	train_start_list=train_data['start_list']
	train_end_list=train_data['end_list']
	train_family=train_data['family'] 

	valid_mother_name=valid_data['mother_name']
	valid_child_name=valid_data['child_name']
	valid_panas_score=valid_data['label']
	valid_start_list=valid_data['start_list']
	valid_end_list=valid_data['end_list']
	valid_family=valid_data['family']

	test_mother_name=test_data['mother_name']
	test_child_name=test_data['child_name']
	test_panas_score=test_data['label']
	test_start_list=test_data['start_list']
	test_end_list=test_data['end_list']
	test_family=test_data['family'] 
 



	assert len(train_mother_name)== len(train_panas_score)
	assert len(valid_mother_name) == len(valid_panas_score)
	assert len (test_mother_name) == len(test_panas_score)
	
	
	print (len(test_mother_name),len(valid_mother_name),len(train_mother_name))


	#train_set=expand_filenames(train_family,train_mother_name)
	
	#valid_set=expand_filenames(valid_family,valid_mother_name)
	#test_set =expand_filenames(test_family,valid_mother_name)
	
	train_data, train_label, train_information = prepare_features(train_mother_name,train_start_list,train_end_list,train_panas_score )
	valid_data, valid_label, valid_information = prepare_features(valid_mother_name, valid_start_list,valid_end_list,valid_panas_score)
	test_data, test_label ,test_information =prepare_features(test_mother_name,test_start_list,test_end_list,test_panas_score)


	total_data={}
	total_data['train_data']=train_data
	total_data['valid_data']=valid_data
	total_data['test_data']=test_data


	total_data['train_label']=train_label
	total_data['valid_label']=valid_label
	total_data['test_label']=test_label
	
	total_data['train_info']=train_information
	total_data['valid_info']=valid_information
	total_data['test_info']=test_information
	

	filename= fold_dir + "data_fold_" + str(idx) + ".npy" 
	np.save(filename,total_data)

	print ("Fold"+str(idx)+"Done")
	
	#input ('')
#print (fold_list)
print ('Done')