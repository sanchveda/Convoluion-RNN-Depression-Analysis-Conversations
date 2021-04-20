import numpy as np 


filename = 'final_list_test0.npy'

final_list= np.load(filename,allow_pickle=True)


test_data=np.load('test_data.npy',allow_pickle=True).item()

test_mother_name,test_child_name,test_label_list,test_start_list,test_end_list=test_data['mother_name'],\
                                                                                    test_data['child_name'],\
                                                                                    test_data['label'],\
                                                                                    test_data['start_list'],\
                                                                                    test_data['end_list']


