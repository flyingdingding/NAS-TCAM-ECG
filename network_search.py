import torch
import torch.nn as nn
from torch import sigmoid
import torch.nn.functional as F
from torch.nn.functional import relu
from primitives import PRIMITIVES, OPS
import math


class MixedOp(nn.Module):

	def __init__(self, input_C, output_C, PRIMITIVE, OPS):
		super(MixedOp, self).__init__()

		self.ops = nn.ModuleList()

		for primitive in PRIMITIVE:
			op = OPS[primitive](input_C, output_C)
			self.ops.append(op)

	def forward(self, x, weights, function, ops_number):
		for i in range(ops_number):
			if i == 0:
				output = weights[i] * function(self.ops[i](x))
			else:
				output.add_(weights[i] * function(self.ops[i](x)))
		return output



class Cell(nn.Module):
	def __init__(self, C, operation_number, ops_number, cell_type = 'TAM'):
		super(Cell, self).__init__()
		self.operation_number = operation_number
		self.ops_number = ops_number
		self.cell_type = cell_type
		channel_list = [32]*(operation_number+1)
		channel_list[0] = C
		channel_list[-1] = 1
		self.mixop_list = nn.ModuleList()
		for i in range(operation_number):
			for j in range(i+1):
				self.mixop_list.append(MixedOp(channel_list[j], channel_list[i+1], PRIMITIVES, OPS))
		self.function_list = ['relu']*math.factorial(operation_number)
		self.function_list[-operation_number:] = ['sigmoid']*operation_number
	def forward(self, input_x, weights):
		x = input_x
		if self.cell_type == 'TAM':
			pass
		else:
			batch_size, channels, length = x.size()
			x = x.mean(dim=2)
			x = torch.unsqueeze(x, dim = 1)
		total_x = list()
		total_index = 0
		add_x = x
		for i in range(self.operation_number):
			total_x.append(add_x)
			now_x = 0
			for j in range(i+1):
				now_x = torch.add(self.mixop_list[total_index](total_x[j],
							weights[total_index],
							globals()[self.function_list[total_index]],
							self.ops_number), now_x)
				total_index += 1
			add_x = now_x
		x = torch.div(add_x,self.operation_number)
		if self.cell_type == 'TAM':
			return torch.mul(input_x, x)
		else:
			return torch.mul(input_x, x.view(batch_size, channels, 1))



class TCAM_P(nn.Module):
	def __init__(self, C):
		super(TCAM_P, self).__init__()
		self.TAM = Cell(C, 3, 6, cell_type = 'TAM')
		self.CAM = Cell(1, 3, 8, cell_type = 'CAM')
	def forward(self, x, TAM_weights, CAM_weights):
		x = torch.max(self.TAM(x, TAM_weights), self.CAM(x, CAM_weights))
		return x


class TCAM_S(nn.Module):
	def __init__(self, C):
		super(TCAM_S, self).__init__()
		self.TAM = Cell(C, 3, 6, cell_type = 'TAM')
		self.CAM = Cell(1, 3, 8, cell_type = 'CAM')
	def forward(self, x, TAM_weights, CAM_weights):
		x = self.TAM(x, TAM_weights)
		x = self.CAM(x, CAM_weights)
		return x
	
class Convblock(nn.Module):
	def __init__(self, C_in, C_out):
		super(Convblock, self).__init__()
		
		self.conv1  = nn.Sequential(nn.Conv1d(C_in, C_out, 19, padding=9),
								nn.BatchNorm1d(C_out),
								nn.ReLU())
								
		self.dropout = nn.Dropout(p = 0.2)	
					
		self.conv2 = nn.Sequential(nn.Conv1d(C_out, C_out, 3, padding=1),
								nn.BatchNorm1d(C_out),
								nn.ReLU())
								
		self.attention = TCAM_P(C_out)
		
		self.down = nn.Sequential(nn.Conv1d(C_out, C_out, 19, stride = 2, padding=9),
								nn.BatchNorm1d(C_out),
								nn.ReLU())
								
	def forward(self, x, TAM_weights, CAM_weights):
		x = self.conv1(x)
		x = self.dropout(x)
		x = self.conv2(x)
		x = self.attention(x, TAM_weights, CAM_weights)
		x = self.down(x)
		return x
		
		

class SearchNet(nn.Module):
	def __init__(self):
		super(SearchNet, self).__init__()
		
		channel_number = [12, 16, 16, 32, 32, 64, 64, 128, 128, 256]	
		self.layers = nn.ModuleList()		
		for i in range(9):
			self.layers.append(Convblock(channel_number[i], channel_number[i+1]))				
		self.linear1 = nn.Linear(3584, 100)					
		self.linear2 = nn.Linear(100, 9)
		self.initialize_alphas()	

	def forward(self, x):

		TAM_weights = F.softmax(self.alphas_TAM, dim = -1)
		CAM_weights = F.softmax(self.alphas_CAM, dim = -1)	
		
		for i in range(9):
			x = self.layers[i](x, TAM_weights[i], CAM_weights[i])

		x = torch.flatten(x, start_dim = 1)
		x = F.relu(self.linear1(x))
		x = self.linear2(x)
		return x, TAM_weights, CAM_weights


	def initialize_alphas(self):
		self.alphas_TAM = nn.Parameter(1e-3*torch.randn(9, 6, 8).cuda())
		self.alphas_CAM = nn.Parameter(1e-3*torch.randn(9, 6, 8).cuda())		

