# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: Train the HiC4D-SPOT model
# In args_mega, select appropriate number before running the script. More detail on args_mega.py

import os
import sys
import json
import pickle
import datetime
import argparse
import numpy as np
import importlib

print(f"Process ID: {os.getpid()}", flush=True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512,expandable_segments:False"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:1024,expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.set_num_threads(20)

from models.hic4d_spot import *
from utils import *
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

##### Arguments #####
sys.path.append('/home/bshrestha/HiC4D-SPOT/args/')
parser = argparse.ArgumentParser()
parser.add_argument('-id', type=str, help='id of the argument file')
args_id = parser.parse_args()
args_id = args_id.id
module_name = f'args_{args_id}'
args = importlib.import_module(module_name).get_args()

# get current learning rate
def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


# training 
def train(model, device, train_loader, optimizer):
	print("Training started", flush=True)
	model.train()
	loss_sum = 0.0

	for i, X in enumerate(train_loader):
		if i == (len(train_loader) - 1):
			continue
		optimizer.zero_grad()
		output, loss, max_pixel_loss = model(X.to(device))
		loss.backward()
		optimizer.step()

		loss_sum = loss_sum + loss.detach().cpu().numpy()
	return loss_sum/i


# validation
def validate(model, device, validation_loader):
	print("Validation started", flush=True)
	model.eval()
	loss_sum = 0.0
	with torch.no_grad():
		for i, X in enumerate(validation_loader):
			output, loss, max_pixel_loss = model(X.to(device))
			loss_sum = loss_sum + loss.item()

	return loss_sum/i

def main():
    
	torch.manual_seed(args['seed'])

	# print some basic information
	print("Arguments: ", json.dumps(args, indent=4), flush=True)

	if args['dummy']:
		# Make dummy data 
		print("Using Dummy data", flush=True)
		dat_train = np.random.rand(100, 8, 1, 54, 54).astype(np.float32)
		dat_validate = np.random.rand(10, 8, 1, 54, 54).astype(np.float32)
	else:
		# Load data
		dat_train = load_input_data(loc=args['input_data'], partition='training', args=args).astype(np.float32)
		dat_validate = load_input_data(loc=args['input_data'], partition='validation', args=args).astype(np.float32)

		# from n*s*h*w to n*s*h*w*1
		dat_train = np.expand_dims(dat_train, axis=4)
		dat_validate = np.expand_dims(dat_validate, axis=4)

		# patch reshape
		# from n*s*50*50*1 to n*s*25*25*4 and to n*s*4*25*25: here n=number of samples, s=number of time points, 50*50=patch size, 4=number of channels, 25*25=patch size/2
		dat_train = patch_image(dat_train, args['patch_size'])
		dat_validate = patch_image(dat_validate, args['patch_size'])
		dat_train = np.transpose(dat_train, [0,1,4,2,3])
		dat_validate = np.transpose(dat_validate, [0,1,4,2,3])
	
	print("Input data", flush=True)
	print("Training data: ", dat_train.shape, flush=True)
	print("Validation data: ", dat_validate.shape, flush=True)

	train_loader = torch.utils.data.DataLoader(torch.from_numpy(dat_train), batch_size=args['batch_size'], shuffle=True)
	validation_loader = torch.utils.data.DataLoader(torch.from_numpy(dat_validate), batch_size=1, shuffle=False)

	# check if CUDA is available
	use_cuda = not args['no_cuda'] and torch.cuda.is_available()
	device = torch.device("cuda:"+str(args['GPU_index']) if use_cuda else "cpu")

	# info from data itself
	input_dim = dat_train.shape[2]
	img_size = dat_train.shape[3]
	imgSizes = (input_dim, img_size, img_size)
	channel = input_dim
	total_length = dat_train.shape[1]
	
	# network
	model = HiC4D_SPOT(input_dim=input_dim, device=device).to(device)
	print(model, flush=True)
	print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)

	optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
	
	# reducing learning rate when loss from validation has stopped improving
	# scheduler = ReduceLROnPlateau(optimizer, 'min', factor=args['lr_factor'], patience=args['lr_patience'], min_lr=args['lr_min'])
	
	min_val_loss = float('inf')
	patience, counter = 25, 0
	for epoch in range(1, args['epochs']+1):
		epoch_start_time = datetime.datetime.now()
  
		loss_train = train(model, device, train_loader, optimizer) 
		loss_validate = validate(model, device, validation_loader)

		# scheduler.step(loss_validate)	
		lr_current = get_lr(optimizer)  

		print(f"Epoch: {epoch}, LR: {lr_current}, Training loss: {loss_train}, Validation loss: {loss_validate}, Time: {datetime.datetime.now()}", flush=True)

		os.makedirs(args['output_model_dir'], exist_ok=True)
		
		# save the model
		if loss_validate < min_val_loss:
			min_val_loss = loss_validate
			counter = 0
			torch.save(model.state_dict(), args['best_model'])
		else:
			counter += 1
			if counter > patience: break

		print(f"Time for single epoch: {datetime.datetime.now() - epoch_start_time}", flush=True)
	print("Training completed!", flush=True)


if __name__ == '__main__':
	main()
