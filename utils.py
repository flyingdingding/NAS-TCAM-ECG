from scipy.io import loadmat
import torch
import pandas as pd
import numpy as np
import time
import math
import os
import random

def standard(x):
	x = x.astype(np.float32)
	return (x - np.mean(x).astype(np.float32))/np.std(x).astype(np.float32)

def test_complement(data):
	data = standard(data)
	length = data.shape[1]
	if length > 7168:
		seed = random.randint(3584, length - 3584)
		data = data[:, seed-3584:seed+3584]
	else:
		data = np.pad(data, ((0,0),(math.floor((7168-length)/2),math.ceil((7168-length)/2))), 'wrap')#'constant'
	return data

class ECG_dataset(torch.utils.data.Dataset):
	def __init__(self, start, end):
		super(ECG_dataset, self).__init__()

		self.data_list = np.load('data_list.npy')[start:end]

		self.info = np.array(pd.read_csv('/home/liuzuhao/torch/12ECG/TrainingSet/REFERENCE.csv'))
		
	def __getitem__(self, index):
		
		signal = test_complement(loadmat(os.path.join('/home/liuzuhao/torch/12ECG/TrainingSet',
							self.info[self.data_list[index,0],0]+'.mat'))['ECG']['data'][0][0])
		label = self.data_list[index,1]
		return signal, label

	def __len__(self):
		return self.data_list.shape[0]

def evaluate(predict, label):
	""" calculate precision, recall, F1-score and accuracy
	"""
	precision = np.zeros([9], np.float32)
	recall = np.zeros([9], np.float32)
	F1 = np.zeros([9], np.float32)
	num_class = 9
	predict = np.argmax(predict, axis = -1)
	accuracy = np.sum((predict == label).astype(np.float32)) / predict.size
	predict = (np.arange(num_class)==predict[:,None]).astype(np.float32)
	label = (np.arange(num_class)==label[:,None]).astype(np.float32)
	for i in range(num_class):
		fp = np.dot((predict[:,i]==1).astype(np.float32),(label[:,i]==0).astype(np.float32))
		fn = np.dot((predict[:,i]==0).astype(np.float32),(label[:,i]==1).astype(np.float32))
		tp = np.dot((predict[:,i]==1).astype(np.float32),(label[:,i]==1).astype(np.float32))
		tn = np.dot((predict[:,i]==0).astype(np.float32),(label[:,i]==0).astype(np.float32))
		precision[i] = tp / (tp + fp + 1e-10)
		recall[i] = tp / (tp + fn + 1e-10)
		F1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i] + 1e-10)
	return F1, precision, recall, accuracy



