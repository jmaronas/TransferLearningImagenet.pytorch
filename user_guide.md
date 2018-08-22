# User Guide: how can you make modifications to the code to use your own database
You can check the functionality of the code by executing python main.py -h
The only changes you must do to this code are incorporating the database you want to train. As ImageNet is tipically pretrained on range -1,1 I do not recommend to change the input of your image to other range so as long as it is saved in png format everything should work fine. (the final transformation would rescale to range 0 1 and my normalization would rescale to range -1 1)

## Datasets in Pytorch

Pytorch provide a wonderfull interface to upload any dataset. Assume our root_dir is ```/home/username/data/mydatasetfolder/``` you should put inside this folder one folder per each subset on your database, typically: train, test, validation... Please check https://pytorch.org/docs/stable/torchvision/datasets.html You can always create your own data loader, however this is something that for big datasets might be a wrong idea as they would not fit in memory, as example check https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py for the MNIST.

## Modifications

You can find the functions on the utils.py file. Before touching this file add to the parser the name of your dataset: mydataset. As example on main.py:

```
parser.add_argument('--dataset', type=str,choices=['birds','cars','mydataset'],required=True,help='which model to train')
```

If you want to train other pretrained models add to the parser the name of the new model, as example for loading resnet-164 add:

```
parser.add_argument('--net', type=str,choices=['densenet-169','densenet-121','resnet-18','resnet-50','resnet-101','inception_v3','resnet-164',required=True,help='which model to train') 
```

### dataset_specs(dataset)
Modify this function to add your dataset. You must add the number of training images, test images, validation images and the number of classes. For example if your dataset has 400 classes:

```
if dataset=='mydataset'
  num_train=100
  num_test=100
  num_valid=100
  return 400,'mydatasetfolder/',[num_train,num_valid,num_test,400]
```

### return_model(net):    

Modify this function to add other pretrained models, as example:
```
if net=='resnet-164':                                                                                                            
     net = models.resnet164(pretrained=True)                                                                             
```

In this example I assume the model is under the torchvision package, however you can return any model you want from other repositories as long as they are in pytorch and you manage correctly which part of the model you must substitute to fit to your dataset. This means that if your model is trained on image-net, the classification layers are projections from an arbitrary space to a 1000 space. You must replace this classification layer to project to the number of classes of your dataset. The code do that for you but we will see some special cases if you incorporate new networks, and how should you proceed.

## Modifications to main.py

This is probably the most variable part as it depends on how the model was created, and which names where given to the different layers. You can check that just by printing the model parameters and check its names.

As example the fully connected part of the desnenet has different name from the resnet and so this code manage that:

```
#transfer learning                                                                                                                      
if 'densenet' in args.net:                                                                                                              
        net.classifier=nn.Linear(net.classifier.in_features,out_shape,bias=True)                                                        
else:                                                                                                                                   
        net.fc=nn.Linear(net.fc.in_features,out_shape,bias=True)
```

Moreover, the inception model has different input shape and it has an auxiliary output. This code take control for that: 
```
#auxiliary output only in inception                                                                                                     
if args.net=='inception_v3':                                                                                                            
        net.AuxLogits.fc=nn.Linear(net.AuxLogits.fc.in_features,out_shape,bias=True)                                                    
        folder_size='400x400/'                                                                                                          
        im_x,im_y=360,360                                                                                                               
else:                                                                                                                                   
        folder_size='300x300/'                                                                                                          
        im_x,im_y=224,224
```


Here you should manage any auxiliary output your model could have (in this case we have one more auxiliary output and we change the projection from 1000 to out_shape), and some additional information. im_x and im_y refers to the shape of the input image to these models. In this code the only one that differs is the inception. folder_size is the name where the dataset is saved as in this case I reshaped the images to 400x400 for train the inception and 300x300 for the rest. From this size we take crops of the necessary shapes (the ones given by im_x and im_y). Pytorch give you transformations so you can avoid doing this preprocessing. I personally prefer to do that because I can choose the padding I use to make the images same shape (this can be avoid if your images have all of them the same shape and just use the reshape transformation from pytorch). You should take in account that if a preprocessing is going to be done always (such as reshape) it is better to do a preprocessing stage than do it everytime we load an image, but that is up to yo ;D.

The final directory is placed in this part:

```
root='/home/username/data/'+dir_extend+folder_size                                                                                     
root_train=root+"train/"                                                                                                                
root_valid=root+"valid/"                                                                                                                
root_test=root+"test/"
```

Where dir_extend is returned when calling dataset_specs (in this example would be 'mydatasetfolder/'). Remark that you can ommit subdirectory folder_size or add more subdirectories if you want, but you should do it here.

There are two list named fc_parameters and conv_parameters that store which parameteres correspond to pre trained parameters (conv_parameters) and which parameters correspond to random parameters (fc_parameters) (remember you can choose to update differently the pretrained parameters and the random parameters). This code is ready to incorporate to these lists the parameters depending on they name the model assigns to them (for example densenet has classifier as the name for the classification layer and resnet has fc, so we search for this names when choosing to which list we place them). If you incorporate a new model take care of this. The part of the code that manages this is:

```
for p in net.named_parameters():
                if p[0] in ['module.fc.weight','module.fc.bias','module.AuxLogits.fc.weight','module.AuxLogits.fc.bias','module.classifier.weight','module.classifier.bias']:
                        parameters_fc+=[p[1]]
                else:
                        parameters_conv+=[p[1]]
```

## Data augmentation
For changing data augmentation please change:

```
normalize=trans.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])                                                                                  
train_transforms=trans.Compose([trans.RandomHorizontalFlip(),trans.RandomResizedCrop(im_x),trans.ToTensor(),normalize])                 
test_transforms=trans.Compose([trans.CenterCrop(im_x),trans.ToTensor(),normalize])
```


I do not recommend changing normalize as it reescale range to -1 1. The rest of transformations are optional, please refer to torchvision website. Here I only provide a minimal example.

## Avoid using validation set

If you want to avoid using validation set just do this. From the save logit call change this:
```
save_logits(net,number_samples,args.dataset,args.net,*[trainLoader,validLoader,testLoader])
```

to this:
```
save_logits(net,number_samples,args.dataset,args.net,*[trainLoader,testLoader])
```
and the code would do the rest for you.

When training, if you do not have validation set you should just comment this line (the code will show  you a 0% accuracy on validation:

```
acc_valid,tot_valid=test(validLoader,net)
```

One more point. This code save the best model on each epoch based on the validation accuracy. If you do not want to use validation accuracy make the neccessary changes to this part of the code:

```
#save best model over validation  
is_best=False
if acc_valid>=best_validation_acc:
        is_best=True
        best_validation_acc=acc_valid
```




