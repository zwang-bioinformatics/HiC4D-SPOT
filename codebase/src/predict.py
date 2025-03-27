# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Predict using the HiC4D-SPOT model 
# In args_mega, select appropriate number before running the script. More detail on args_mega.py

import os
import sys
import json
import pickle
import datetime
import argparse
import importlib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F

from models.hic4d_spot import *
from utils.scheduling import *
from utils.load_dataset import *

torch.backends.cudnn.enabled =  True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

torch.jit.enable_onednn_fusion(True)

print(f"Process ID: {os.getpid()}", flush=True)

##### Arguments #####
sys.path.append('/home/bshrestha/HiC4D-SPOT/args/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', type=str, help='id of the argument file')
args_id = parser.parse_args()
args_id = args_id.id
module_name = f'args_{args_id}'
args = importlib.import_module(module_name).get_args()


def test(model, device, test_loader):
	print("Predicting...", flush=True)
	model.eval()
	loss_sum = 0.0
	predictions = []
	with torch.no_grad():
		for i, X in enumerate(test_loader):
			start_time = datetime.datetime.now()
			output, loss, max_pixel_loss = model(X.to(device))
			# print("Inference time: ", datetime.datetime.now() - start_time, flush=True)

			predictions.append(output.cpu().detach().numpy())
			loss_sum = loss_sum + loss.item()

	return predictions, loss_sum/i

def time_swap(data):
    print("Performing Time-Swap", flush=True)
    # Shape: (1230, 6, 1, 50, 50) if patch_size = 1 | (1230, 6, 4, 25, 25) if patch_size = 2, = (number_of_sub_matrices, number_of_time_points, patch_size*patch_size, sub_mat_n//patch_size, sub_mat_n//patch_size)
    # I want to swap the data between 2nd and 4th time points
    data_perturbed = np.copy(data)
    data_perturbed[:,1,:,:,:] = data[:,5,:,:,:]
    data_perturbed[:,5,:,:,:] = data[:,1,:,:,:]
    
    return data_perturbed
    

def main():

	torch.manual_seed(args['seed'])

	# print some basic information
	print(f"PWD: {os.getcwd()}", flush=True)
	print("Arguments: ", json.dumps(args, indent=4), flush=True)
 
	os.makedirs(args['output_predict_dir'], exist_ok=True)
	# check if output_predict_file exists
	if os.path.exists(f"{args['output_predict_file']}"):
		print(f"Output file {args['output_predict_file']} already exists. Exiting...", flush=True)
		print("Exiting...", flush=True)
		sys.exit(0)

	# load data
	if args['dummy']:
		# Make dummy data 
		print("Using Dummy data", flush=True)
		dat_test = np.random.rand(100, 8, 1, 50, 50).astype(np.float32)
	else:
		dat_test = load_input_data(loc=args['input_data'], partition='predict', args=args).astype(np.float32) # Shape of dat_test: (number_of_sub_matrices, number_of_time_points, sub_mat_n, sub_mat_n)

		dat_test = np.expand_dims(dat_test, axis=4)			# Shape: (1230, 6, 50, 50, 1) = (number_of_sub_matrices, number_of_time_points, sub_mat_n, sub_mat_n, 1)
		dat_test = patch_image(dat_test, args['patch_size'])	# Shape: (1230, 6, 50, 50, 1) if patch_size = 1 | (1230, 6, 25, 25, 4) if patch_size = 2, = (number_of_sub_matrices, number_of_time_points, sub_mat_n//patch_size, sub_mat_n//patch_size, patch_size*patch_size)
		dat_test = np.transpose(dat_test, [0,1,4,2,3])		# Shape: (1230, 6, 1, 50, 50) if patch_size = 1 | (1230, 6, 4, 25, 25) if patch_size = 2, = (number_of_sub_matrices, number_of_time_points, patch_size*patch_size, sub_mat_n//patch_size, sub_mat_n//patch_size)


	if (args['augmentation']):
		if args['aug_type'] == 'time_swap':
			dat_test = time_swap(dat_test)
	print("Input data", flush=True)
	print("Test data: ", dat_test.shape, flush=True)

	test_loader = torch.utils.data.DataLoader(torch.from_numpy(dat_test), batch_size=1, shuffle=False)

	# check if CUDA is available
	use_cuda = not args['no_cuda'] and torch.cuda.is_available()
	device = torch.device("cuda:"+str(args['GPU_index']) if use_cuda else "cpu")

	# info from data itself
	input_dim = dat_test.shape[2]
	img_size = dat_test.shape[3]
	imgSizes = (input_dim, img_size, img_size)
	channel = input_dim
	total_length = dat_test.shape[1]

	# load network
	model = HiC4D_SPOT(input_dim=input_dim, device=device)
	model.load_state_dict(torch.load(args['best_model'], map_location='cpu'))
	model.to(device)
	print("Loading model. Done!")

	frames, loss_test = test(model, device, test_loader)
	frames2 = patch_image_back(np.transpose(np.concatenate(frames, axis=0), [0,1,3,4,2]), args['patch_size'])	# Shape: (1230, 6, 50, 50, 1) if patch_size = 1 | (1230, 6, 25, 25, 4) if patch_size = 2, = (number_of_sub_matrices, number_of_time_points, sub_mat_n, sub_mat_n, 1)
	frames2 = np.squeeze(frames2)	# Shape: (1230, 6, 50, 50) if patch_size = 1 | (1230, 6, 25, 25) if patch_size = 2, = (number_of_sub_matrices, number_of_time_points, sub_mat_n, sub_mat_n)
	frames2 = np.clip(frames2, 0, 1)	# clip to [0, 1]
	print(loss_test, frames2.shape)

	# save predictions
	np.save(args['output_predict_file'], frames2)
	print("Predictions saved to: ", args['output_predict_file'], flush=True)


if __name__ == '__main__':
	main()
