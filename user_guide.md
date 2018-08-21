# User Guide: how can you make modifications to the code to use your own database
You can check the functionality of the code by executing python main.py -h
The only changes you must do to this code is incorporating the database you want to train. As ImageNet is tipically pretrained on range -1,1 I do not recommend to change the input of your image to other range so as long as it is saved in png format everything should work fine.

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

Modify this function to add other pretrained models, as e
```
if net=='resnet-164':                                                                                                            
     net = models.resnet164(pretrained=True)                                                                             
```

In this example I assume the model is under the torchvision package, however you can return any model you want from other repositories as long as they are in pytorch and you manage correctly which part of the model you must substitute to fit to your dataset.

## Modifications to main.py

This is probably the most variable part as it depends on how the model was created, and which names where given to the different layers. You can check that just by printint the model parameters and check its names.

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


Here you should place any auxiliary output your model could have, with additional information. im_x and im_y refers to the shape of the input image to this models. In this code the only one that differs is the inception. folder_size is the name where the dataset is saved as in this case the images are reshape to 400x400 for the inception and 300x300 for the rest. From this size we take crops of the necessary shapes. The final directory is placed in this part:

```
root='/home/username/data/'+dir_extend+folder_size                                                                                     
root_train=root+"train/"                                                                                                                
root_valid=root+"valid/"                                                                                                                
root_test=root+"test/"
```

Where dir_extend is returned when calling dataset_specs (in this example would be 'mydatasetfolder/'). Remark that you can ommit subdirectory folder_size or add more subdirectories if you want, but you should do it here.

## Data augmentation
For changing data augmentation please change:

```
normalize=trans.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])                                                                                  
train_transforms=trans.Compose([trans.RandomHorizontalFlip(),trans.RandomResizedCrop(im_x),trans.ToTensor(),normalize])                 
test_transforms=trans.Compose([trans.CenterCrop(im_x),trans.ToTensor(),normalize])
```


I do not recommend changing normalize as it reescale range to -1 1. The rest of transformations are optional, please refer to torchvision website. Here I only provide a minimal example.
