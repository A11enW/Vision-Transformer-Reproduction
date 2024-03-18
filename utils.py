#作用是读取数据集

import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import MNISTTrainDataset, MNISTValDataset, MNISTSubmissionDataset

import numpy as np
from torch.utils.data import DataLoader, Dataset

def get_loaders(train_df_dir, test_df_dir, submission_df_dir, batch_size):
    train_df = pd.read_csv(train_df_dir) #此处的train包含两个部分，后续用sklearn的方法把他分成train和val
    test_df = pd.read_csv(test_df_dir) #验证一个过程，负责提交的submission
    submission = pd.read_csv(submission_df_dir)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=17)
    #waiting for check

    train_dataset = MNISTTrainDataset(train_df.iloc[:,1:].values.astype(np.uint8), #图片
                                      train_df.iloc[:,0].values, #标签
                                      train_df.index.values)
    #iloc function belongs to pandas, index location  对数据进行位置索引，从而提取出对应数据,左到右不到

    val_dataset = MNISTValDataset(val_df.iloc[:,1:].values.astype(np.uint8), #图片
                                  val_df.iloc[:,0].values, #标签
                                  val_df.index.values)

    test_dataset = MNISTSubmissionDataset(test_df.iloc[:,1:].values.astype(np.uint8), #图片
                                          test_df.index.values) #submission组没有标签


    ##用Data load 装进来
    train_dataloader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size= batch_size, shuffle=True) #shuffle验证集也需要打乱
    test_dataloader = DataLoader(dataset=test_dataset, batch_size= batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader