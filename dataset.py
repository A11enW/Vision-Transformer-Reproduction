import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MNISTTrainDataset(Dataset):
    def __init__(self, image, label, indicies):
        self.images = image
        self.labels = label
        self.indicies = indicies
        self.transfor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15), #rotate the image by 15 degree, 15 is enough to avoid the loss of number
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ]) #transforms的作用是打开图片，对图片进行一定增强作用，包括图片旋转，转换成Tensor等操作

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28,28)).astype(np.uint8) #single image
        label = self.labels[idx] #single
        index = self.indicies[idx]
        image = self.transfor(image) #对图片进行一定变换

        return {"image":image, "label":label, "index":index}

##以上为训练集的构成
##以下为验证集的构成


class MNISTValDataset(Dataset): #Val→Validate
    def __init__(self, image, label, indicies):
        self.images = image
        self.labels = label
        self.indicies = indicies
        self.transfor = transforms.Compose([
            #transforms.ToPILImage(), #don't need to open the image
            #transforms.RandomRotation(15) #Rotation is unnecessary
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ]) #transforms的作用是打开图片，对图片进行一定增强作用，包括图片旋转，转换成Tensor等操作

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28,28)).astype(np.uint8) #single image
        label = self.labels[idx] #single
        index = self.indicies[idx]
        image = self.transfor(image) #对图片进行一定变换

        return {"image":image, "label":label, "index":index}

##测试集数据从cargo中下载得到，只需要提交结果，系统会给出准确率,不需要label

class MNISTSubmissionDataset(Dataset): #Val→Validate
    def __init__(self, image, indicies):
        self.images = image
        #self.labels = label
        self.indicies = indicies
        self.transfor = transforms.Compose([
            #transforms.ToPILImage(), #don't need to open the image
            #transforms.RandomRotation(15) #Rotation is unnecessary
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ]) #transforms的作用是打开图片，对图片进行一定增强作用，包括图片旋转，转换成Tensor等操作

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((28,28)).astype(np.uint8) #single image
        #label = self.labels[idx] #single
        index = self.indicies[idx]
        image = self.transfor(image) #对图片进行一定变换

        return {"image":image, "index":index}