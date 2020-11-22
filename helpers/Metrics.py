from sklearn.metrics import *
import numpy as np

#standard identical-count statistics
def accuracy(y_pred,y_true,task):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))

	if task=='binary-classification':
		y_pred = np.where(y_pred<0.5,0,1)
	elif task=='multi-classification':
		y_pred = np.argmax(y_pred,axis=1)
	else:
		assert False,'Unsupported task '+task

	return accuracy_score(y_pred=y_pred,y_true=y_true)

#???
def balanced_accuracy(y_pred,y_true,task):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))

	if task=='binary-classification':
		y_pred = np.where(y_pred<0.5,0,1)
	elif task=='multi-classification':
		y_pred = np.argmax(y_pred,axis=1)
	else:
		assert False,'Unsupported task '+task

	return balanced_accuracy_score(y_pred=y_pred,y_true=y_true)

#area under PR curve
def AUPR(y_pred,y_true,task):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))
	return 0.

#area under ROC curve
def AUROC(y_pred,y_true,task):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))

	if task=='binary-classification':
		return roc_auc_score(y_score=y_pred,y_true=y_true)
	return 0.

#???
def EIS(y_pred,y_true,task):
	assert len(y_pred)==len(y_true),'Inconsistent length, %d != %d' %(len(y_pred),len(y_true))
	return 0.
