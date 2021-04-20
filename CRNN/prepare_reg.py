import os
import numpy as np 
import pandas as pd 
from scipy.io import loadmat


from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

raw_dir= "/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/normalized_raw_final/"
zface_dir ='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=ground_truth_data/TPOT/20191111_Converted_pipeline_output/'

raw_list= np.array([name for name in os.listdir(raw_dir)])
zface_list= np.array([name for name in os.listdir(zface_dir) if name.endswith('fit.mat')])

#print (len(zface_list))
#input ('here')

def convert_bstring_to_normal(inputlist):

    outputlist=[]
    for elements in inputlist:

        if isinstance(elements,np.bytes_):
            outputlist.append(elements.decode('utf-8'))
        else:

            outputlist=inputlist
    
    return outputlist

def group_to_category(scores):

	scores=np.array(scores)

	for i in range (len(scores)):
		if scores[i] == 1:
			scores[i]=0
		elif scores[i]==2:
			scores[i]=1

	return np.array(scores,dtype=int)


def phq_score_to_category(scores):

	scores=np.array(scores)

	for i in range(len(scores)):
		if scores[i] >=20:
			scores[i]=4
		elif scores[i] >=15:
			scores[i]=3
		elif scores[i] >= 10:
			scores[i] = 2
		elif scores[i] >=5:
			scores[i] = 1
		else:
			scores[i] = 0


	return np.array(scores,dtype=int)
def panas_convert(names,scores):
	#print (scores)
	names=np.array(names)
	blank_indices= np.where (scores==' ')
	
	names=np.delete(names,blank_indices)
	scores=np.delete(scores,blank_indices)

	assert len(names) == len(scores)
	

	scores= np.array(scores).astype('float')

	new_array= scores / 50
	
	return names,new_array

def does_exist(name,name_list,length_list):

	flag=0
	length=0
	for i in range(len(name_list)):

		elements=name_list[i]

		if name == elements:
			flag=flag+1
			length=length_list[i]
	if flag:
		#print ("flag", flag, "Name", name)
		#input ('')
	
		return length

	else:
		return 0

def segment_real (total_frames,eliminate_time,slide=0,window_size=0,fps=30,mode='slice'):
    ##--------Variable descriptions ####
    #total_frames : length of the video in terms of frames
    # eliminated_time : Argument passed in minutes. The time we want to exclude from beginning and end of the video. Give 0 in case you don't want to eliminate any
    # window_size: Argument passed in seconds. 
    # slide : Argument passed in seconds to calculate how many seconds to slide
    # mode : can be 'slice' or 'slide'. Use slice for getting equal interval slices. Else use slide for sliding window.

    eliminated_frames=int(eliminate_time*60*fps)

    start=eliminated_frames
    end=total_frames-eliminated_frames

    actual_frames=end-start
    
    window_length=window_size*fps
    index_list=[]
    slide_length=slide*fps

    if mode=='slice':
        factor =int (actual_frames/window_length)
        
        for i in range (factor):
            index_list.append([start,start+window_length])
            start=start+window_length
    else:
        while start < end :
            #print (start,start+window_length)
            if start+window_length < end:
                index_list.append([start,start+window_length])
            
            start=start+slide_length

    #print (len(index_list))
    #print (end)
    
    return index_list

def read_zface(zface_filename):
	mat=loadmat(os.path.join(zface_dir,zface_filename))
	zface_data = mat['fit']

	no_frames=zface_data.shape[1]

	return no_frames

def expand_data(file_list,label_list):
	mother_name=np.empty(0)
	child_name= np.empty(0)
	start_list=np.empty(0)
	end_list=np.empty(0)
	panas_score=np.empty(0)
	family_list=[]

	for idx, family in enumerate(file_list):

		family= str(family)[2:]

		mother_video_name= family + str(2) + '_02_01' 
		child_video_name=  family + str(1) + '_02_01'

		mother_exist = raw_list[raw_list==mother_video_name]
		child_exist = raw_list[raw_list==child_video_name]
		
		if len (mother_exist) and len(child_exist) :
			#print  (mother_video_name,child_video_name)
			mother_zface= mother_video_name + '_fit.mat'
			child_zface= child_video_name + '_fit.mat'
			try:
				mother_frames,child_frames=read_zface(mother_zface),read_zface(child_zface)
			except:
				continue

			if mother_frames < 20000 or mother_frames > 50000 or child_frames <20000 or child_frames >50000:
				continue 

			mother_seg= np.array(segment_real(mother_frames,2.5,window_size=30))
			#child_seg = segment_real(child_frames,2.5,window_size=30)

			mother_split=np.full(len(mother_seg),mother_video_name)
			child_split=np.full(len(mother_seg), child_video_name)
			label_aug=np.full(len(mother_seg),label_list[idx])

			start,end = mother_seg[:,0] , mother_seg[:,1]	
		
			mother_name=np.hstack([mother_name,mother_split]) if mother_name.size else mother_split
			child_name=np.hstack([child_name,child_split]) if child_name.size else child_split
			start_list=np.hstack([start_list,start]) if start_list.size else start
			end_list=np.hstack([end_list,end]) if end_list.size else end
			panas_score=np.hstack([panas_score,label_aug]) if panas_score.size else label_aug

			family_list.append(family)

			#mother_list.extend()		
	assert len(mother_name) == len(child_name)  == len(start_list) == len(end_list) == len(panas_score)
			
	return [mother_name,child_name,start_list,end_list,panas_score,family_list]
			

tpot_filename='../tpot-actual.csv'

tpot_data=pd.read_csv(tpot_filename)

families= tpot_data['FamId']
parent_psi_pos= np.array(tpot_data['m_2t12pp'])
child_psi_pos= np.array(tpot_data['c_7z12pp'])

group= parent_psi_pos.copy()
#group=np.concatenate((parent_psi_pos,child_psi_pos),axis=1)


assert len(families) == len(parent_psi_pos) == len(child_psi_pos)
'''
"Creating 5 folds for families .Uncomment only when you need to create new folds"

kf=StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for idx,(train_index,test_index) in enumerate(kf.split(families,group)):
	x_train =	families[train_index]
	x_test  =	families[test_index]
	y_train =	group[train_index]
	y_test  =	group[test_index]
	
	x_train,x_valid,y_train,y_valid= train_test_split(x_train,y_train, test_size=0.2)

	d={}
	d['x_train'] = x_train
	d['x_test'] = x_test
	d['y_train']= y_train
	d['y_test'] = y_test
	d['x_valid']= x_valid
	d['y_valid']= y_valid
	np.save ('./kfold_reg/fold_'+str(idx)+'.npy',d)

'''

fold_dir= 'kfold_reg/'
fold_list= sorted ([name for name in os.listdir(fold_dir) if name.startswith('fold')])

name_list_filename='../data_analysis/ground_truth_data/name_list.npy'
length_list_filename='../data_analysis/ground_truth_data/length_list.npy'
name_list=np.load(name_list_filename,allow_pickle=True)
length_list=np.load(length_list_filename,allow_pickle=True)

assert len(name_list) == len(length_list)


for idx, folds in enumerate(fold_list):
	d=np.load(os.path.join(fold_dir,folds),allow_pickle=True).item()
	x_train,y_train = d['x_train'],d['y_train']
	x_valid,y_valid = d['x_valid'],d['y_valid']
	x_test,y_test  = d['x_test'],d['y_test']

	x_train,y_train= panas_convert (x_train,y_train)
	x_valid,y_valid= panas_convert (x_valid,y_valid)
	x_test,y_test = panas_convert  (x_test,y_test)
	
	train_mother_name,train_child_name,train_start_list,train_end_list,train_panas_score,train_family=expand_data(x_train,y_train)
	valid_mother_name,valid_child_name,valid_start_list,valid_end_list,valid_panas_score,valid_family=expand_data(x_valid,y_valid)
	test_mother_name,test_child_name,test_start_list,test_end_list,test_panas_score,test_family	     =expand_data(x_test,y_test)

	train_data={}
	train_data['mother_name']=train_mother_name
	train_data['child_name']=train_child_name
	train_data['label']=train_panas_score
	train_data['start_list']=train_start_list
	train_data['end_list']=train_end_list
	train_data['family'] = train_family


	valid_data={}
	valid_data['mother_name']=valid_mother_name
	valid_data['child_name']=valid_child_name
	valid_data['label']=valid_panas_score
	valid_data['start_list']=valid_start_list
	valid_data['end_list']=valid_end_list
	valid_data['family'] = valid_family

	test_data={}
	test_data['mother_name']=test_mother_name
	test_data['child_name']=test_child_name
	test_data['label']=test_panas_score
	test_data['start_list']=test_start_list
	test_data['end_list']=test_end_list
	test_data['family'] = test_family

	train_filename= './kfold_reg/train_data_' + str(idx) + '.npy'
	valid_filename= './kfold_reg/valid_data_' + str(idx) + '.npy'
	test_filename = './kfold_reg/test_data_' + str(idx) + '.npy'

	np.save(train_filename,train_data)
	np.save(valid_filename,valid_data)
	np.save(test_filename,test_data)


	print ('Done for fold %d' %(idx))

print ('Done for all folds')
