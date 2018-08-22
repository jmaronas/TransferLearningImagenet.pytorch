import torch
if torch.__version__ != '0.4.0':
	raise RuntimeError('PyTorch version must be 0.4.0')
import torch.nn as nn
import torchvision
import torchvision.transforms as trans

import argparse
import os
import logging
import time

from utils import anneal_lr, dataset_specs, return_model, select_optimizer, train_inception,train,test,create_dir, save_checkpoint,load_checkpoint,return_logits,save_logits

#TODO  : resume training from a model
def parse_args():

	parser = argparse.ArgumentParser(description='Preactivation Models on ImageNet by Juan Maron~as Molano. Resarcher at PRHLT Universidad Politecnica de Valencia jmaronasm@prhlt.upv.es . Enjoy!')
	'''String Variables'''
	parser.add_argument('--net', type=str,choices=['densenet-169','densenet-121','resnet-18','resnet-50','resnet-101','inception_v3'],required=True,help='which model to train')
	parser.add_argument('--dataset', type=str,choices=['birds','cars'],required=True,help='which model to train')
	parser.add_argument('--t', action='store_true',help='test model')

	parser.add_argument('--epochs', type=int,nargs='+',default=None,required=False,help='list of epochs to use [ 100 10 10 ] will train with 100 epochs 10 epochs and 10 epochs with the learning rates given in --lr argument')
	parser.add_argument('--lr', type=float,nargs='+',default=None,required=False,help='list of learning rates [0.1 0.01 0.001] will use provided learning rates with 100 epochs 10 epochs and 10 epochs')
	parser.add_argument('--mmu', type=float,nargs='+',default=None,required=False,help='Provide a list as in --lr and --epochs. If not given use 0.9')
	parser.add_argument('--batch', type=int,required=False,default=100,help='batch size')
	parser.add_argument('--anneal', action='store_true',required=False,default=False,help='Perform Linear Annealing on last epochs')
	parser.add_argument('--lr_conv_scale', type=float,required=False,default=10,help='learning rate scale factor for convolutional updates. Divide learning rate by this factor to train the convolutional part')
	parser.add_argument('--train_conv_after', type=int,required=False,default=10,help='Number of epochs to train the random part before training all the parameter (random and pretrained)')
	parser.add_argument('--workers', type=int,required=False,default=None,help='How many threads used for data loading. If not given use as maximum as given by nproc linux command')
	parser.add_argument('--model_dir', type=str,required=False,default=None,help='Provide directory to search for the model to perform test')
	parser.add_argument('--s_logits', action='store_true',required=False,default=False,help='Save the logits of the model. Only done when --t is given')

	args=parser.parse_args()
	return args

args=parse_args()

#data specs
out_shape,dir_extend,number_samples = dataset_specs(args.dataset)

#create the model
net=return_model(args.net)

#transfer learning
if 'densenet' in args.net:
	net.classifier=nn.Linear(net.classifier.in_features,out_shape,bias=True)
else:
	net.fc=nn.Linear(net.fc.in_features,out_shape,bias=True)

#auxiliary output only in inception
if args.net=='inception_v3':
	net.AuxLogits.fc=nn.Linear(net.AuxLogits.fc.in_features,out_shape,bias=True)
	folder_size='400x400/'
	im_x,im_y=360,360
else:
	folder_size='300x300/'
	im_x,im_y=224,224

#dataset
root='/home/jmaronasm/data/'+dir_extend+folder_size
root_train=root+"train/"
root_valid=root+"valid/"
root_test=root+"test/"

#pytorch dataset
normalize=trans.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
train_transforms=trans.Compose([trans.RandomHorizontalFlip(),trans.RandomResizedCrop(im_x),trans.ToTensor(),normalize])
test_transforms=trans.Compose([trans.CenterCrop(im_x),trans.ToTensor(),normalize])

dataset_train=torchvision.datasets.ImageFolder(root_train,transform=train_transforms)
dataset_valid=torchvision.datasets.ImageFolder(root_valid,transform=test_transforms)
dataset_test=torchvision.datasets.ImageFolder(root_test,transform=test_transforms)

#dataloader
workers = (int)(os.popen('nproc').read()) if args.workers==None else args.workers
trainLoader=torch.utils.data.DataLoader(dataset_train,batch_size=args.batch,shuffle=True,num_workers=workers)
testLoader=torch.utils.data.DataLoader(dataset_test,batch_size=100,num_workers=workers)
validLoader=torch.utils.data.DataLoader(dataset_valid,batch_size=100,num_workers=workers)

#algorithm for train or test
if args.t:
	net.cuda()
	net.eval()
        net=torch.nn.DataParallel(net)
        net.load_state_dict(load_checkpoint(args.model_dir))
        torch.backends.cudnn.benchmark=True

	if args.s_logits:
                save_logits(net,number_samples,args.dataset,args.net,*[trainLoader,validLoader,testLoader])
		exit(0)

	mc_loss,tot_samples=test(testLoader,net)

	print "ACC test[{:.3f}%] {:.3f} of {}".format(float(mc_loss)/tot_samples*100,mc_loss,tot_samples)
	
else:

	net.cuda()
	net=torch.nn.DataParallel(net)
	torch.backends.cudnn.benchmark=True

	#training
	if args.epochs==None or args.lr==None:
		print "Must provide learning parameters: epochs and lr. Check python main.py -h"
		exit(-1)

	if args.mmu==None:
		mmu_t=[0.9]*len(args.epochs)
	else:
		mmu_t=args.mmu
	assert len(mmu_t)==len(args.epochs) and len(mmu_t)==len(args.lr)
	
	#create directory and log file
	directory=create_dir(args.dataset,args.net)
	logging.basicConfig(filename=directory+'train.log',level=logging.INFO)
	logging.info("Logger for model {}, dataset {}".format(args.net,args.dataset))
	logging.info("Training specificacitions: epochs [{}] lr [{}] mmu [{}] batch {} optimizer {} Linear anneal {} learning rate factor for conv part {} train conv after {} epochs".format(args.epochs,args.lr,mmu_t,args.batch,'SGD',args.anneal,args.lr_conv_scale,args.train_conv_after))

	#training criterion and parameters to optimize
	criterion = nn.CrossEntropyLoss()
	parameters_conv=list()
	parameters_fc=list()
	for p in net.named_parameters():
		if p[0] in ['module.fc.weight','module.fc.bias','module.AuxLogits.fc.weight','module.AuxLogits.fc.bias','module.classifier.weight','module.classifier.bias']:
			parameters_fc+=[p[1]]
		else:
			parameters_conv+=[p[1]]

	#avoid using a conditional branch in the loop
	train_set=(train,train_inception) 
	train_sel = 1 if args.net == 'inception_v3' else 0

	#useful variables for training loop and display
	total_ep=0
	after_total_ep=args.train_conv_after
	factor_over_pretrain=args.lr_conv_scale
	
	best_validation_acc=0
	
	#training loop
	for ind,epoch,lr,mmu in zip(range(len(args.epochs)),args.epochs,args.lr,mmu_t):

		if ind == len(args.epochs)-1 and args.anneal:
                        activate_anneal=True
                        lr_init=lr
                        epochs_N=epoch
		else:
			activate_anneal=False
			lr_new=lr

		optim_fc=select_optimizer(parameters_fc,lr=lr_new,mmu=mmu,optim='SGD')

		for e in range(epoch):
			if activate_anneal:
				lr_new=anneal_lr(lr_init,epochs_N,e)
				optim_fc=select_optimizer(parameters_fc,lr=lr_new,mmu=mmu,optim='SGD')

			if total_ep>=after_total_ep:
				optim_conv=select_optimizer(parameters_conv,lr=lr_new/factor_over_pretrain,mmu=mmu,optim='SGD')
			else:
				optim_conv=select_optimizer(parameters_conv,lr=0.0,mmu=mmu,optim='SGD') #TODO check performance, I suppose that with lr=0.0 updates are not performed

			current_time=time.time()
			ce_train,acc_train,tot_train=train_set[train_sel](trainLoader,net,criterion,[optim_fc,optim_conv])
			acc_valid,tot_valid=test(validLoader,net)
			acc_test,tot_test=test(testLoader,net)

			acc_train=float(acc_train)/float(tot_train)*100
			acc_valid=float(acc_valid)/float(tot_valid)*100
			acc_test=float(acc_test)/float(tot_test)*100
			
			total_ep+=1
			print "Epoch: {} lr fc: {:5f} lr conv: {:5f}\t CE train {:.3f}\t ACC train[{:.3f}] of {}\t ACC valid[{:.3f}] of {}\t ACC test[{:.3f}] of {}\t spend {:3f} minutes".format(total_ep,lr_new,lr_new/factor_over_pretrain,ce_train,acc_train,tot_train,acc_valid,tot_valid,acc_test,tot_test,(time.time()-current_time)/60.)

			logging.info("Epoch: {} lr fc: {:5f} lr conv: {:5f}\t CE train {:.3f}\t ACC train[{:.3f}] of {}\t ACC valid[{:.3f}] of {}\t ACC test[{:.3f}] of {}\t spend {:3f} minutes".format(total_ep,lr_new,lr_new/factor_over_pretrain,ce_train,acc_train,tot_train,acc_valid,tot_valid,acc_test,tot_test,(time.time()-current_time)/60.))

			#save best model over validation  
			is_best=False
			if acc_valid>=best_validation_acc:
				is_best=True
				best_validation_acc=acc_valid

			save_checkpoint({
		            'net': args.net,
		            'state_dict': net.state_dict(),
		            'best_prec_valid': best_validation_acc,				
			}, is_best,directory)



