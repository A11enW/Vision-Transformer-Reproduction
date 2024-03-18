import torch #common-used
import torch.nn as nn #common-used
from torch import optim #common-used
import timeit
from tqdm import tqdm #common-used
from utils import get_loaders
from model import Vit

print("开始运行了")

#Hyper parameter
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50 #数据集不大，训练次数不用太多

BATCH_SIZE =32 #GPU越好越大
TRAIN_DF_DIR = "./dataset/train.csv"
TEST_DF_DIR = "./dataset/test.csv"
SUBMISSION_DF_DIR = "./dataset/sample_submission.csv"

#Model Parameters
IN_CHANNELS = 1 #能以表格形式表示，通道数肯定是1
IMG_SIZE = 28 #由数据决定
PATCH_SIZE = 4 #希望切四个小方块
EMBED_DIM = (PATCH_SIZE **2) * IN_CHANNELS #小方块的大小
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) **2
DROPOUT = 0.001

#Transformer Parameter
NUM_HEADS = 8
ACTIVATION = "gelu" #NLP领域最佳，especially for transformer. 能够避免梯度消失
NUM_ENCODERS = 768
NUM_CLASS = 10 #数字0~9

#Training Parameter
LEARNING_RATE = 1e-4 #可以根据计算机性能进行优化
ADAM_WEIGHT_DECAY = 0 #原文的加速方法
ADAM_BETAS = (0.9, 0.999)

train_dataloader, val_dataloader, test_dataloader = get_loaders(TRAIN_DF_DIR, TEST_DF_DIR, SUBMISSION_DF_DIR, BATCH_SIZE)

model = Vit(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES, DROPOUT,
            NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASS).to(device)  #模型需要转换成CUDA状态

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=ADAM_BETAS, lr=LEARNING_RATE, weight_decay=ADAM_WEIGHT_DECAY)

print("预处理没问题")

#记录一下训练时间
start = timeit.default_timer()

for epoch in range(EPOCHS):
    model.train()
    train_labels = []
    train_preds = [] #predicts
    train_runningloss = 0

    for idx, img_label in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        #enumerate用于返回枚举的对象
        #tqdm是进度条库，在长循环中进行进度提示，只需要封装任意迭代器. position控制进度条在终端的位置,leave控制进度条完成后是否保留
        img = img_label["image"].float().to(device) #.to(device)转换成cuda的形式
        #label 在dataset.py的时候转换成map形式了 return {"image":image, "label":label, "index":index}
        label = img_label["label"].type(torch.uint8).to(device)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(label.cpu().detach()) #.detach()什么意思
        train_preds.extend((y_pred_label.cpu().detach())) #这个过程在CPU上更快

        loss = criterion(y_pred, label)

        optimizer.zero_grad() #optimizer清零
        loss.backward()#反向传播
        optimizer.step() #optimizer进入下一个步骤


        train_runningloss += loss.item()
    train_loss = train_runningloss / (idx + 1) #index是什么东西啊？？？？？

    #调成验证模式
    model.eval()
    val_labels = []
    val_preds = []
    val_runningloss = 0
    with torch.no_grad():#梯度不更新
        for idx, img_label in enumerate(tqdm(val_dataloader, position=0, leave=True)):
            img = img_label["image"].float().to(device)  # .to(device)转换成cuda的形式
            # label 在dataset.py的时候转换成map形式了 return {"image":image, "label":label, "index":index}
            label = img_label["label"].type(torch.uint8).to(device)
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)

            val_labels.extend(label.cpu().detach())  # .detach()什么意思
            val_preds.extend((y_pred_label.cpu().detach()))  # 这个过程在CPU上更快

            loss = criterion(y_pred, label) #不更新weight，不optimize
            val_runningloss += loss.item()
        val_loss = val_runningloss / (idx + 1)

    #看一下结果
    print("-"*30)
    print(f"Train Loss Epoch {epoch+1} : {train_loss:.4f}")
    print(f"Validate Loss Epoch {epoch + 1} : {val_loss:.4f}")
    #看一下训练集的准确率是多少
    print(
        f"Train Accuracy Epoch {epoch+1} : {sum(1 for x, y in zip(train_preds, train_labels) if x == y) /len(train_labels):.4f}" #看不懂如何实现的，但似乎很常用
    )
    # 同理看一下验证集的准确率是多少
    print(
        f"Validate Accuracy Epoch {epoch + 1} : {sum(1 for x, y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}"
        # 看不懂如何实现的，但似乎很常用
    )
    print("-"*30)

stop = timeit.default_timer()
print(f"Training Time : {stop-start:.2f} s")
