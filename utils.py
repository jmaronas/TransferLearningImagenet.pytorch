import torch
import torchvision.models as models

import os
import errno
import shutil
import numpy

def anneal_lr(lr_init,epochs_N,e):
        lr_new=-(lr_init/epochs_N)*e+lr_init
        return lr_new

def dataset_specs(dataset):
	if dataset=='birds':
		num_train=5994
		num_valid=1200
		num_test=4594
		return 200,'birds_stanford_2011/database_pytorch_format/',[num_train,num_valid,num_test,200]
	if dataset=='cars':
		num_train=8144
		num_valid=1960
		num_test=6081
		return 196,'cars_stanford/database_pytorch_format/',[num_train,num_valid,num_test,196]

def return_model(net):
	if net=='resnet-18':
		net = models.resnet18(pretrained=True)		
	elif net=='resnet-50':
		net = models.resnet50(pretrained=True)
	elif net=='resnet-101':
		net = models.resnet101(pretrained=True)
	elif net=='densenet-169':
		net = models.densenet169(pretrained=True)
	elif net=='densenet-121':
		net = models.densenet121(pretrained=True)
	elif net=='inception_v3':
		net = models.inception_v3(pretrained=True)
	else:
		print ("Provide a valid network, check python main.py -h")
		exit(-1)
	return net

def select_optimizer(parameters,lr=0.0,mmu=0.0,optim='SGD'):
	if optim=='SGD':
		optimizer = torch.optim.SGD(parameters,lr=lr,momentum=mmu)
	elif optim=='ADAM':
		optimizer = torch.optim.Adam(parameters,lr=lr)

	return optimizer


def accuracy(out,t):
	return (out.max(1)[1]==t).sum()


def train_inception(dataLoader,net,criterion,optimizer):
	net.train()
	acc_loss=0.0
	mc_loss=0.0
	tot_samples=0.0
	for x,t in dataLoader:
		x,t=x.cuda(),t.cuda()
		predict1,predict2=net(x)
		Loss=criterion(predict1,t)+criterion(predict2,t)
		Loss.backward()

		for _ in optimizer:
			_.zero_grad()
			_.step()
			_.zero_grad()
		mc_loss+=accuracy(predict1,t)
		acc_loss+=Loss.data
		tot_samples+=t.size(0)
	return acc_loss,mc_loss,tot_samples

def train(dataLoader,net,criterion,optimizer):
	net.train()
	acc_loss=0.0
	mc_loss=0.0
	tot_samples=0.0
	for x,t in dataLoader:
		x,t=x.cuda(),t.cuda()
		predict=net(x)
		Loss=criterion(predict,t)
		Loss.backward()
		for _ in optimizer:
			_.step()
			_.zero_grad()

		mc_loss+=accuracy(predict,t)
		acc_loss+=Loss.data
		tot_samples+=t.size(0)

	return acc_loss,mc_loss,tot_samples


def test(dataLoader,net):
	net.eval()
	mc_loss=0.0
	tot_samples=0.0
	for x,t in dataLoader:
		with torch.no_grad():
			x,t=x.cuda(),t.cuda()
			predict=net(x)
		
			mc_loss+=accuracy(predict,t)
			tot_samples+=t.size(0)
	return mc_loss,tot_samples


def create_dir(dataset,model_name):
	#for saving the model and logging

	counter=0
	directory="./PreTrainImageNet/"+dataset+"/"+model_name+"/"+str(counter)+"/"	
	while True:
		if os.path.isdir(directory):
			counter+=1
			directory="./PreTrainImageNet/"+dataset+"/"+model_name+"/"+str(counter)+"/"	
		else:
			break
	
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    torch.save(state, directory+filename)
    if is_best:
	shutil.copyfile(directory+filename, directory+'model_best.pth.tar')

def load_checkpoint(directory):
	if directory==None:
		print ("Provide a file, check python main.py -h")
		exit(-1)
		#only python 3 raise  FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), directory)

	if os.path.isfile(directory+'model_best.pth.tar'):
		checkpoint=torch.load(directory+'model_best.pth.tar')
		return checkpoint['state_dict']
	else:
		print ("File {} not found".format(directory+'model_best.pth.tar'))
		exit(-1)
		#only python 3 raise  FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), directory)



def return_logits(x,t,net):
        with torch.no_grad():
             x,t=x.cuda(),t.cuda()
             predict=net(x)

             mc_loss=accuracy(predict,t)
             tot_samples=t.size(0)
        return predict,mc_loss,tot_samples


def save_logits(net,number_samples,dataset,net_name,*args):
	print ("Computing and saving logits...")
	dir_="./npy_files/"
	if not os.path.exists(dir_):
		os.makedirs(dir_)

	n_train,n_valid,n_test,n_classes=number_samples

	logit_prediction_train=numpy.zeros((n_train,n_classes),dtype=numpy.float32)
        logit_true_train=numpy.zeros((n_train,),dtype=numpy.int64)
        logit_prediction_valid=numpy.zeros((n_valid,n_classes),dtype=numpy.float32)
        logit_true_valid=numpy.zeros((n_valid,),dtype=numpy.int64)
        logit_prediction_test=numpy.zeros((n_test,n_classes),dtype=numpy.float32)
        logit_true_test=numpy.zeros((n_test,),dtype=numpy.int64)

	lista=[(logit_prediction_train,logit_true_train),(logit_prediction_test,logit_true_test),(logit_prediction_valid,logit_true_valid)]
	lista_str=[("_logit_prediction_train","_true_train"),("_logit_prediction_test","_true_test"),("_logit_prediction_valid","_true_valid")]
	stats=list()
	for idx,loader in enumerate(args): #train loader, test loader, valid loader
	        logits,true=lista[idx]		
		name_logit,name_true=lista_str[idx]
		mc_loss=0.0
		tot_samples=0.0
		for x,t in loader:
			x,t=x.cuda(),t.cuda()
			logit,mc,t_s=return_logits(x,t,net)
			logits[int(tot_samples):int(tot_samples)+int(t_s),:]=logit.cpu().numpy()
			true[int(tot_samples):int(tot_samples)+int(t_s)]=t.cpu().numpy()

			mc_loss+=mc
			tot_samples+=t_s
		
		
		numpy.save(dir_+dataset+"_"+net_name+name_logit,logits)
	        numpy.save(dir_+dataset+"_"+net_name+name_true,true)
		stats.append((float(mc_loss)/float(tot_samples)*100.,tot_samples))

	if len(stats)==2:
		print "ACC train[{:.3f}] of {}\t ACC test[{:.3f}] of {}".format(stats[0][0],stats[0][1],stats[1][0],stats[1][1])
	else:
		print "ACC train[{:.3f}] of {}\t ACC test[{:.3f}] of {}\t ACC valid[{:.3f}] of {}".format(stats[0][0],stats[0][1],stats[1][0],stats[1][1],stats[2][0],stats[2][1])



