import torch
import torch.nn as nn


PRIMITIVES = [
	'none',
	'deep_conv3',
	'deep_conv5',
	'deep_conv9',
	'deep_conv15',
	'deep_conv19',
	'deep_conv25',
	'deep_conv29',
	
]	


OPS = {
	'none': lambda input_C, output_C: Zero(input_C, output_C),

	'deep_conv3' : lambda input_C, output_C: nn.Sequential(
									nn.Conv1d(input_C, output_C, 3, padding = 1),
									nn.BatchNorm1d(output_C)),
							
	'deep_conv5' : lambda input_C, output_C: nn.Sequential(
									nn.Conv1d(input_C, output_C, 5, padding = 2),
									nn.BatchNorm1d(output_C)),
						
	'deep_conv9' : lambda input_C, output_C: nn.Sequential(
									nn.Conv1d(input_C, output_C, 9, padding = 4),
									nn.BatchNorm1d(output_C)),
						
	'deep_conv15' : lambda input_C, output_C: nn.Sequential(
									nn.Conv1d(input_C, output_C, 15, padding = 7),
									nn.BatchNorm1d(output_C)),
						
	'deep_conv19' : lambda input_C, output_C: nn.Sequential(
									nn.Conv1d(input_C, output_C, 19, padding = 9),
									nn.BatchNorm1d(output_C)),

	'deep_conv25' : lambda input_C, output_C: nn.Sequential(
									nn.Conv1d(input_C, output_C, 25, padding = 12),
									nn.BatchNorm1d(output_C)),

	'deep_conv29' : lambda input_C, output_C: nn.Sequential(
									nn.Conv1d(input_C, output_C, 29, padding = 14),
									nn.BatchNorm1d(output_C)),
	
}

class Zero(nn.Module):
  """ Because of the change of channel in search module, we use convolution to change the channel number
  """
  def __init__(self, input_C, output_C):
    super(Zero, self).__init__()
    self.conv = nn.Conv1d(input_C, output_C, 3, padding = 1)
    

  def forward(self, x):
    output = torch.mul(self.conv(x), 0.)
    return output

