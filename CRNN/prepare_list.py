import numpy as np 
import random
import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
def convert_bstring_to_normal(inputlist):

    outputlist=[]
    for elements in inputlist:

        if isinstance(elements,np.bytes_):
            outputlist.append(elements.decode('utf-8'))
        else:

            outputlist=inputlist
    
    return outputlist


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
    # eliminated_time : Arguement passed in minutes. The time we want to exclude from beginning and end of the video
    # window_size: Argument passed in seconds
    # slide : Argument passed in seconds to calculate how many seconds to slide
    # mode : can be 'slice' or 'slide'

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


def prepare_phq_data(phq_data,name_list,length_list):
	family_id,subject_id,scores=phq_data['family_id'],phq_data['subject_id'],phq_data['phq_score']
	scores=phq_score_to_category(scores)

	name1_list=[]
	name2_list=[]
	score_list=[]
	mother_length=[]
	child_length=[]
	task_id=[]
	family_list=[]

	for i in range(len(family_id)):


		mother_t1=family_id[i]+'2'+'_01'+'_01'
		mother_t2=family_id[i]+'2'+'_02'+'_01'
		child_t1=family_id[i]+'1'+'_01'+'_01'
		child_t2=family_id[i]+'1'+'_02'+'_01'

		score=scores[i]

		mother_t1_length=does_exist(mother_t1,name_list,length_list)
		child_t1_length=does_exist(child_t1,name_list,length_list)
		mother_t2_length=does_exist(mother_t2,name_list,length_list)
		child_t2_length= does_exist(child_t2,name_list,length_list)

		if mother_t1_length>0 and child_t1_length>0 and mother_t1_length>20000 and child_t1_length>20000:
			name1_list.append(mother_t1)
			name2_list.append(child_t1)
			score_list.append(score)
			mother_length.append(mother_t1_length)
			child_length.append(child_t1_length)
			task_id.append(1)
			family_list.append(family_id[i])

		if mother_t2_length>0 and child_t2_length>0 and mother_t2_length > 20000 and child_t2_length>20000 and mother_t2_length<50000:
			name1_list.append(mother_t2)
			name2_list.append(child_t2)
			score_list.append(score)
			mother_length.append(mother_t2_length)
			child_length.append(child_t2_length)
			#print (mother_t2,child_t2)

			task_id.append(2)
			family_list.append(family_id[i])
		else :
			if mother_t2_length>50000:
				print (mother_t2,child_t2)


	assert len(name1_list)==len(name2_list)==len(score_list)==len(mother_length)==len(child_length) == len(family_list)

	truth={}
	name1_list=np.array(name1_list)
	name2_list=np.array(name2_list)
	score_list=np.array(score_list)
	mother_length=np.array(mother_length)
	child_length=np.array(child_length)
	task_id=np.array(task_id)
	family_list=np.array(family_list)

	truth['name1']=name1_list
	truth['name2']=name2_list
	truth['score']=score_list
	truth['mother_length']=mother_length
	truth['child_length']=child_length
	truth['task_id']=task_id
	truth['family_id']=family_list

	return name1_list,name2_list,score_list,mother_length,child_length,task_id,family_list


def synchronize (filepath):

	data=pd.read_csv(filepath)

	key1=np.array(data['key1'])
	key2=np.array(data['key2'])
	shift=np.array(data['shift'])


	assert len(key1) == len(key2) == len(shift)

	shift_list=[]
	family_list=[]
	subject_list=[]
	task_list=[]
	for ii in range (key1.shape[0]):

		if key1[ii].startswith('TPOT'):
			family,subject,task=key1[ii].split('_')[1:]
			

			if (task  == '1') or (task == '2'):
				family_list.append(family)
				task_list.append(task)
				shift_list.append(shift[ii])
			else:
				pass #For neutral tasks
				

	assert len(family_list) == len(task_list) == len(shift_list)

	synchronize={}
	synchronize['family_list'] = family_list
	synchronize['task_list'] = task_list
	synchronize['shift_list'] = shift_list

	np.save ('synchronization.npy',synchronize)

	return

synchronize_filepath='synchronization.csv'

synchronize(synchronize_filepath)

raw_dir='/run/user/1435515045/gvfs/smb-share:server=istanbul.psychology.pitt.edu,share=raw_data/TPOT/Video_Data/normalized_raw_frames/'

# Counting the number of frames for each file. This code is just to be run once for creating 
# name_list and length_list which will useful on many sections
# Uncomment if you don't have the name_list and the length_list
'''
raw_list=os.listdir(raw_dir)
name_list=[]
length_list=[]
for files in raw_list:

	frame_count=len(os.listdir(raw_dir+files+'/'))
	print (files,frame_count)
	name_list.append(files)
	length_list.append(frame_count)

assert len(name_list)== len(length_list)
print (len(name_list),len(length_list))
np.save('name_list.npy',name_list)
np.save('length_list.npy',length_list)
'''

name_list=np.load('name_list.npy',allow_pickle=True)
length_list=np.load('length_list.npy',allow_pickle=True)


assert len(name_list) == len(length_list)

#names=convert_bstring_to_normal(np.load('name_list.npy',allow_pickle=True,encoding='latin1'))
phq_data=np.load('phq_data.npy',allow_pickle=True,encoding='latin1').item()


families=np.array(phq_data['family_id'])
score=np.array(phq_score_to_category(phq_data['phq_score']))


X_train, X_test, y_train, y_test = train_test_split(families, score,
                                                    stratify=score, 
                                                    test_size=0.15)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
														stratify=y_train,
														test_size=0.20)

#Remember this is not the final training set . This is just the splitting of the families. The actual data is the splits which need to be shuffled again

name1,name2,score,mother_length,child_length,task_id,family_id=prepare_phq_data(phq_data,name_list,length_list)
assert len(name1)==len(name2)
print (len(name1)+len(name2))
for ii in range (len(name1)):
	print (name1[ii],name2[ii])


task =2


def expand_data(data_split,name1,name2,score,mother_length,child_length,task_id,family_id,task=2):
	count =0

	mother_name=np.empty(0)
	child_name=np.empty(0)
	label_list=np.empty(0)
	start_list=np.empty(0)
	end_list=np.empty(0)

	checklist=[]
	for ii in range (len(data_split)):
		for jj in range (len(family_id)):
			

			if (data_split[ii] == family_id[jj]) and (task_id[jj]  == task) :
				#print (family_id[jj],name1[jj])
				
				index_list=np.array(segment_real(mother_length[jj],2.5,window_size=30))
				count=count+ len (index_list)
				
				mother_split=np.full(len(index_list),name1[jj])
				
				child_split=np.full(len(index_list),name2[jj])
				
		
				label_aug=np.full(len(index_list),score[jj])
				start,end= index_list[:,0], index_list[:,1]
				
				assert len(start) == len(end) == len(mother_split)
				if np.abs(mother_length[jj]-child_length[jj]) > 30:
					pass
					#print (name1[jj],mother_length[jj],child_length[jj])
				
				mother_name=np.hstack([mother_name,mother_split]) if mother_name.size else mother_split	
				child_name=np.hstack([child_name,child_split]) if child_name.size else child_split
				label_list=np.hstack([label_list,label_aug]) if label_list.size else label_aug
				start_list=np.hstack([start_list,start]) if start_list.size else start
				end_list=np.hstack([end_list,end]) if end_list.size else end
			

	assert count == len(mother_name) == len(label_list) == len(start_list) == len(end_list)

	
	return mother_name,child_name,label_list,start_list,end_list


train_mother_name,train_child_name,train_label_list,train_start_list,train_end_list=expand_data(X_train,
																								name1,
																								name2,
																								score,
																								mother_length,
																								child_length,
																								task_id,
																								family_id,
																								task,
																								)


valid_mother_name,valid_child_name,valid_label_list,valid_start_list,valid_end_list=expand_data(X_valid,
																								name1,
																								name2,
																								score,
																								mother_length,
																								child_length,
																								task_id,
																								family_id,
																								task
																								)


print (valid_label_list)
input ('')

test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=expand_data(X_test,
																							name1,
																							name2,
																							score,
																							mother_length,
																							child_length,
																							task_id,
																							family_id,
																							task
																							)

print (test_label_list)
input ('')
def shuffle_wrapper(data):
	result_list=data.copy()
	random.shuffle(result_list)
	return result_list

'''
fail=[]
for ii in range(len(train_mother_name)):
	flag=0  
	for jj in range (len(valid_mother_name)):
		#print (names, valid_mother_name[valid_mother_name==names])
		if train_mother_name[ii] == valid_mother_name[jj]:
			print (train_mother_name[ii],valid_mother_name[jj])
			flag = flag +1

			fail.append(train_mother_name[ii])

print (len(fail),len(valid_mother_name))
input ('')
'''
train_mother_name,train_child_name,train_label_list,train_start_list,train_end_list=zip(*shuffle_wrapper(list(zip(train_mother_name,
																											train_child_name,
																											train_label_list,
																											train_start_list,
																											train_end_list))))
'''
valid_mother_name,valid_child_name,valid_label_list,valid_start_list,valid_end_list=zip(*shuffle_wrapper(list(zip(valid_mother_name,
																											valid_child_name,
																											valid_label_list,
																											valid_start_list,
																											valid_end_list))))

'''
'''
test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=zip(*shuffle_wrapper(list(zip(test_mother_name,
																											test_child_name,
																											test_label_list,
																											test_start_list,
																											test_end_list))))

'''


#X_train_name1=final_list['name1'][final_list['task_id']==2 and final_list['family_id'] in [0,1]]

train_data={}
train_data['mother_name']=train_mother_name
train_data['child_name']=train_child_name
train_data['label']=train_label_list
train_data['start_list']=train_start_list
train_data['end_list']=train_end_list


valid_data={}
valid_data['mother_name']=valid_mother_name
valid_data['child_name']=valid_child_name
valid_data['label']=valid_label_list
valid_data['start_list']=valid_start_list
valid_data['end_list']=valid_end_list

test_data={}
test_data['mother_name']=test_mother_name
test_data['child_name']=test_child_name
test_data['label']=test_label_list
test_data['start_list']=test_start_list
test_data['end_list']=test_end_list

np.save('train_data.npy',train_data)
np.save('valid_data.npy',valid_data)
np.save('test_data.npy',test_data)

print ('Done')
