import numpy as np 

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,roc_auc_score,cohen_kappa_score
from sklearn.metrics import auc,roc_curve,mean_squared_error
import matplotlib.pyplot as plt
from scipy import stats

def accuracy(y_true,y_pred):

	return accuracy_score(y_true,y_pred)

def f1score(y_true,y_pred):

	return f1_score(y_true,y_pred,average='weighted')

def auc_score(y_true,y_pred):

	return roc_auc_score(y_true,y_pred)


#-----------------This is for a multiclass setting-----------------#
def compute_roc(n_classes,y_test,y_score):
	fpr=dict()
	tpr=dict()
	roc_auc=dict()
	

	for i in range (n_classes):
		
		fpr[i],tpr[i], _ =roc_curve(y_test[:,i],y_score[:,1])
		
		roc_auc[i]=auc(fpr[i],tpr[i])



	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	for idx in range (n_classes):		
		titletext=('Roc Curve for Class %d' % idx)
		plt.figure()
		lw = 2
		plt.plot(fpr[idx], tpr[idx], color='darkorange',
		         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[idx])
		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(titletext)
		plt.legend(loc="lower right")
		plt.show()
		#plt.savefig("Class "+str(idx)+".png")

def compute_roc_binary(y_test,y_score,roc_text):
	
	fpr,tpr,_= roc_curve(y_test,y_score)
	roc_auc  = auc (fpr,tpr)
	
	plt.plot(fpr,tpr,color='darkorange',lw=2, label='ROC curve (area = %0.2f)' %roc_auc)
	plt.plot([0,1],[0,1], color='navy', lw=2 , linestyle='--')
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(roc_text)
	plt.legend(loc="lower right")
	plt.savefig(roc_text+".png")
	plt.close()


def mean_error(y_true,y_pred):

	return mean_squared_error(y_true,y_pred)

		
def pcc(y_true,y_pred):
	return stats.pearsonr(y_true,y_pred)

def kappa(y_true,y_pred):
	return cohen_kappa_score(y_pred,y_true)

def kappa_custom(label,pred,NO_CLASSES):

	n_items = pred.shape[0]
	n_raters = 2.0
	Q = NO_CLASSES

	w_kl = np.eye(Q,dtype = float)
	r_ik = np.zeros((n_items,Q))
	r_i = np.zeros(n_items)

	for ii in range(n_items):
		r_i[ii] = n_raters
		r_ik[ii,int(pred[ii])] += 1.0
		r_ik[ii,int(label[ii])] += 1.0

	r_il = r_ik	
	p_o = 0.0

	for ii in range(n_items):
		for kk in range(Q):
			p_o += ( r_ik[ii,kk]*(r_ik[ii,kk] - 1.0) )/( r_i[ii]*(r_i[ii] - 1.0) )

	p_o = p_o/n_items
	n_gk = np.zeros((int(n_raters),Q))
	n_g = np.zeros(int(n_raters))
	p_gk = np.zeros((int(n_raters),Q))


	for kk in range(Q):
		n_gk[0,kk] = (np.sum(label == kk))
		n_gk[1,kk] = (np.sum(pred == kk))
		
	n_g[0] = np.sum(n_gk[0,:])
	n_g[1] = np.sum(n_gk[1,:])

	# pdb.set_trace()

	for gg in range(int(n_raters)):
		for kk in range(Q):
			p_gk[gg,kk] = n_gk[gg,kk]/n_g[gg]

	p_plus_k = np.sum(p_gk,axis=0)/n_raters
	s2_kl = np.zeros((Q,Q))

	for kk in range(Q):
		for ll in range(Q):
			# for gg in range(int(n_raters)):
			# 	s2_kl += (p_gk[gg,kk]*p_gk[gg,ll]  -  n_raters*p_plus_k[kk]*p_plus_k[ll])
			for gg in range(int(n_raters)):
				s2_kl += (p_gk[gg,kk]*p_gk[gg,ll])    
			s2_kl = s2_kl - n_raters*p_plus_k[kk]*p_plus_k[ll]

	s2_kl = (s2_kl/(n_raters*1.0-1.0))
	pdb.set_trace()



	p_c = 0.0
	for kk in range(Q):
		for ll in range(Q):
			p_c += w_kl[kk,ll]*(p_plus_k[kk]*p_plus_k[ll]  -  s2_kl[kk,ll]/n_raters)

	kappa = ( (p_o - p_c)/(1.0 - p_c) )
	return kappa



	

	