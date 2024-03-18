

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, patch_size, embed_dim, num_patches, dropout):
        ##输入图片的通道数,想要小方块的大小,切完之后会产生多少个小方块,小方块的数目
        ##dropout：为了防止过拟合而设置。nn.dropout(p=0.3)表示每个神经元有0.3可能性不激活
            super(PatchEmbedding, self).__init__()
            #super用于调用父类的方法，参考：https://zhuanlan.zhihu.com/p/636105836?utm_id=0
            self.patcher = nn.Sequential(
                #Sequential 允许我们构建序列化的模块。就把Sequential当作list来看
                #nn.sequential(), 一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
                #keras中也有类似的类，tensorflow中没有
                nn.Conv2d(in_channels=in_channel, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
                #nn.Conv2d 二维卷积, 参考：https://blog.csdn.net/qq_50001789/article/details/120381140
                nn.Flatten(2)
                #把切完的小方块拉平, (512,768,14,14)→(512,768,196)
            )

            self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim), requires_grad=True)
            #cls_token 是进行带有注意力机制的汇总， 和其他 token 进行计算
            #cls_token加在哪个位置对结果没有影响，计算是并行的 参考：https://zhuanlan.zhihu.com/p/550515118
            #requires_grad 表达的含义是，这一参数是否保留（或者说持有，即在前向传播完成后，是否在显存中记录这一参数的梯度，而非立即释放）梯度，等待优化器执行optim.step()更新参数。
            self.position_embedding = nn.Parameter(torch.randn(size=(1,num_patches+1,embed_dim)), requires_grad=True)
            self.dropout = nn.Dropout(p=dropout)
    #以上代码实现从(512,3,224,224)到(512,768,196) step1的前半完成


    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        #expand -1表示不改变维度，只有第0维可以新增维度 参考:https://blog.csdn.net/weixin_43135178/article/details/120602026

        x = self.patcher(x).permute(0, 2, 1)
        #permute用于坐标变换，类似转置,021就是第三坐标和第二坐标交换位置 参考:https://blog.csdn.net/weixin_41377182/article/details/120808310

        x = torch.cat([x, cls_token], dim=1)
        x = x + self.position_embedding
        x = self.dropout(x)
        return x

    # Step1的后半+Step2 完成


#构建Vit的全部流程

class Vit(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, dropout,
                 num_heads, activation, num_encoders, num_classes):
        #第一行针对transformer，第二行针对后续的模型
        super(Vit, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                                   activation=activation,
                                                   batch_first=True, norm_first=True)
        #encoder_layer就是transformer
        #nhead=num_heads指有多少注意力机制 参考notion
        #Encoder层的参数都在这里定义好了,作为激活装置
        self.encoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        #把激活装置送入Transformer中激活

        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim), #归一化处理, Nor = Normalization
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )
        #最后再走一个MLP

    def forward(self, x):
        x = self.patch_embedding(x) #把输入图片切成小方块
        x = self.encoder_layer(x)
        x = self.MLP(x[:,0,:]) #MLP的输入是二维的，需要舍弃一个
        return x

    #得益于Torch把TransformerEncoderLayer已经封装好了，所以程序本身很简单



