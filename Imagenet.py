import torchvision.datasets as dsets
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import glob
def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self,ImageInfo,Transform,loader,DataScale=0):
        ImagePathList = []
        ImageLabelList = []
        F = open(ImageInfo)
        lines = F.readlines()
        for line in lines:
            ImagePath = line.split(',')[0]
            ImageLabel = line.split(',')[1].split('\n')[0]
            ImagePathList.append(ImagePath)
            ImageLabelList.append(ImageLabel)
        self.ImagePathList = ImagePathList
        self.ImageLabelList = ImageLabelList
        if not DataScale == 0:
            self.ImagePathList = ImagePathList[:DataScale]
            self.ImageLabelList = ImageLabelList[:DataScale]
        self.loader = loader
        self.Transform = Transform
        self.current_path = ''
    def __getitem__(self, index):
        ImagePath = self.ImagePathList[index]
        self.current_path = ImagePath
        # PATH = '/home/pubuser/TJY/TIANCHI/TrainData/00058/af1894bce72d6981f6854afa60178981.jpg'
        label = float(self.ImageLabelList[index])
        label = torch.tensor(int(label), dtype=torch.long)
        try:
            img = self.loader(ImagePath)
            img = self.Transform(img)
        except:
            self.FError.write(ImagePath)
        img = img[0:3,:]
        if not img.shape == torch.Size([3,244,244]):
            img = img[0:3,:]
        return img, label
        # return img, label, ImagePath,index

    def __len__(self):
        return len(self.ImagePathList)

def Imagenet(imputsize,BATCH_SIZE,shuffle,NEC,NC,NV,TrainDataSourceFile,ValDataSourceFile):
    ## NC: number of classes
    ## NEC: number of samples in each class
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    transform = transforms.Compose([
    transforms.transforms.Resize(imputsize),
    transforms.RandomHorizontalFlip(p=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std),
])
    if not os.path.exists(TrainDataSourceFile):
        TrainDataSourceFile = Ge_Train_Dataloader(NEC,NC)
    if not os.path.exists(ValDataSourceFile):
        ValDataSourceFile = Ge_Val_Dataloader(NV)
    ValData = MyDataset(ValDataSourceFile, transform, default_loader)
    TrainDataloader = torch.utils.data.DataLoader(dataset=ValData, batch_size=BATCH_SIZE, shuffle=True)
    ValDataloader = torch.utils.data.DataLoader(dataset=ValData, batch_size=BATCH_SIZE, shuffle=shuffle)
    return TrainDataloader,ValDataloader,TrainDataSourceFile,ValDataSourceFile

def Ge_Train_Dataloader(NEC,NC):
    ## NC: number of classes
    ImageInfoFolder = 'ImageInfo'
    os.makedirs(ImageInfoFolder,exist_ok=True)
    DataSource = '/home/jinyu/ImageNet_train'
    DataSourceFile = os.path.join(ImageInfoFolder, 'TrainData_' + str(NEC) + '_' + str(NC) + '.txt')
    LabelInfo = {}
    F = open(DataSourceFile,'w')
    F.close()
    F = open(DataSourceFile,'a')
    ClassNameList = os.listdir(DataSource)
    ClassNameList = np.sort(ClassNameList)
    for id,ClassName in enumerate(ClassNameList):
        LabelInfo[ClassName] = id

    for ClassName in ClassNameList[:NC]:
        TrainImageNameList = glob.glob(os.path.join(DataSource,ClassName)+'/*.JPEG')
        TrainImageNameList = np.random.permutation(TrainImageNameList)
        for ImageName in TrainImageNameList[:NEC]:
            ImageLabel = LabelInfo[ClassName]
            F.write(ImageName+','+str(ImageLabel)+'\n')
    return DataSourceFile

def Ge_Val_Dataloader(NV):
    ## NC: number of classes
    ImageInfoFolder = 'ImageInfo'
    os.makedirs(ImageInfoFolder,exist_ok=True)
    Datalog = os.path.join(ImageInfoFolder, 'ValData_' + str(NV)+'.txt')
    ImageInfoTxt = '/home/jinyu/ImageNet_val/validation_label.txt'
    F = open(Datalog,'w')
    F.close()
    F = open(Datalog,'a')

    for count,line in enumerate(open(ImageInfoTxt).readlines()):
        if count > NV:
            break
        F.write('/home/jinyu/ImageNet_val/'+line.split(' ')[0] + ',' + line.split(' ')[1].split('\n')[0] + '\n')


    return Datalog