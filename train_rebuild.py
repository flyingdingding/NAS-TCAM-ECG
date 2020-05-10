import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import math
import time
import random
from network_rebuild import RebuildNet
from utils import ECG_dataset, evaluate

def test_rebuild(net, highest_F1, valid_loader, batch_size):
	net.eval()
	
	all_prediction = np.zeros([3200, 9], dtype = np.float32)
	all_label = np.zeros([3200], dtype = np.int32)
	criterion = nn.CrossEntropyLoss()
	total_loss = 0
	total_num = 0
	
	for (val_data, val_lab) in valid_loader:
		val_data, val_lab = val_data.cuda(), val_lab.cuda().long()
		
		logits = net(val_data)
		logits = F.softmax(logits, dim = -1)
		loss = criterion(logits, val_lab)
		all_prediction[total_num*batch_size:(total_num+1)*batch_size] = logits.cpu().detach().numpy()
		all_label[total_num*batch_size:(total_num+1)*batch_size] = val_lab.cpu().detach().numpy()
		total_loss += loss.item()
		total_num += 1
	F1, precision, recall, accuracy = evaluate(all_prediction, all_label)
	print("Average F1 score: {:.4f}, Average precision: {:.4f}, Average recall: {:.4f}, Accuracy: {:.4f}"\
			.format(np.mean(F1), np.mean(precision), np.mean(recall), accuracy))
	for each_category in range(9):
		print(
			"Class {:1d}: F1 score {:.4f}, Precision {:.4f}, Recall {:.4f}".format(
			each_category, F1[each_category], precision[each_category], recall[each_category]))
	if np.mean(F1) > highest_F1:
		highest_F1 = np.mean(F1)
		print("xxx highest F1 is {:.4f} xxx".format(highest_F1))
		torch.save(net.state_dict(),'CP_highest.pth')
	return highest_F1
		

def train_rebuild(net, epochs = 100, batch_size = 16, lr = 1e-4, gpu = True, workers = 4):

	print('''
		Starting training:
		Epochs: {}
		Batch size: {}
		Learning rate: {}
		CUDA: {}
		'''.format(epochs, batch_size, lr, str(gpu)))

	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	criterion = nn.CrossEntropyLoss()
	highest_F1 = 0
	train_data = ECG_dataset(0, 3200)
	valid_data = ECG_dataset(3200, 6400)
	train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True)
	valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True)
	highest_F1 = 0
	net.train()                                            
	for epoch in range(epochs):
		if epoch % 3 == 0:
			highest_F1 = test_rebuild(net, highest_F1, valid_loader, batch_size)
		print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
		net.train()
		total_loss = 0
		total_num = 0
	
		for (trn_data, trn_lab) in train_loader:
			trn_data, trn_lab = trn_data.cuda(), trn_lab.cuda().long()
			optimizer.zero_grad()
			logits = net(trn_data)
			loss = criterion(logits, trn_lab)
			total_loss += loss.item()
			total_num += 1
			loss.backward()
			optimizer.step()
		print('Epoch finished ! Loss: {}'.format(total_loss / total_num))

if __name__ == '__main__':
	TAM_Index = np.load('TAM.npy')
	CAM_Index = np.load('CAM.npy')
	net  = RebuildNet(TAM_Index, CAM_Index)
	cudnn.benchmark = True
	net.cuda()
	train_rebuild(net)
