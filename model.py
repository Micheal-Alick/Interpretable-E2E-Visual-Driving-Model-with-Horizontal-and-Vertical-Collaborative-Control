import torch  # PyTorch 主库，用于张量和自动微分
import torch.nn as nn  # PyTorch 神经网络模块别名
import torchvision.models as models  # torchvision 中的预训练模型集合
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights  # 导入 EfficientNet-B0 及其权重枚举


activation = {}  # 用于存储中间激活值的字典（可通过 hook 捕获）

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()  # 将输出 detach 后保存，避免计算图被保留
    return hook  # 返回一个可注册为 hook 的函数，用于捕获指定层的激活

class NvidiaModelTransferLearning(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False):
        super().__init__()  # 初始化父类

        # EfficientNet 主干网络（提取特征用）
        weights =  EfficientNet_B0_Weights.IMAGENET1K_V1  # 使用 ImageNet 预训练权重版本
        efficientnet = efficientnet_b0(weights=weights)  # 加载 EfficientNet-B0 模型并应用权重

        feature_dim = 1280  # EfficientNet-B0 的输出特征维度
        
        # 去掉分类头，保留特征提取部分
        self.conv_layers = efficientnet.features  # EfficientNet 的卷积特征提取模块
        self.avgpool = efficientnet.avgpool  # EfficientNet 的全局平均池化层
        
        
        # 根据参数选择是否冻结特征层（不更新权重）
        if freeze_features:
            for param in self.conv_layers.parameters():
                param.requires_grad = False  # 冻结参数，节省显存和计算
        
        # 回归头（用于预测方向盘角度）
        self.regressor = nn.Sequential(
            nn.Flatten(),  # 展平特征图为向量
            nn.Dropout(0.4),  # 丢弃 40% 神经元以降低过拟合风险
            
            nn.Linear(feature_dim, 256),  # 全连接层：feature_dim -> 256
            nn.BatchNorm1d(256),  # 批归一化，提高训练稳定性
            nn.ReLU(),  # ReLU 激活函数
            nn.Dropout(0.3),  # 丢弃 30%
            
            nn.Linear(256, 100),  # 全连接层：256 -> 100
            nn.BatchNorm1d(100),  # 批归一化
            nn.ReLU(),  # ReLU 激活
            nn.Dropout(0.2),  # 丢弃 20%
            
            nn.Linear(100, 50),  # 全连接层：100 -> 50
            nn.ReLU(),  # ReLU 激活
            nn.Dropout(0.1),  # 丢弃 10%
            
            nn.Linear(50, 1)  # 输出层：50 -> 1，预测单个连续值（转向角度）
        )
    
    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积特征提取器
        x = self.avgpool(x)  # 全局平均池化，得到固定尺寸特征
        x = self.regressor(x)  # 通过回归头得到预测
        return x.squeeze()  # 去掉多余维度，返回标量或一维张量
    
    

class NvidiaModel(nn.Module):
    def __init__(self, pretrained=True, freeze_features=False):
        super().__init__()  # 初始化父类

        # 加载预训练的 ResNet18 作为特征提取器
        resnet = models.resnet18(pretrained=pretrained)  # 如果 pretrained=True，则加载 ImageNet 权重
        
        # 去掉最后的分类层，只保留特征提取部分
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])  # 取除最后一层外的所有子模块并顺序化
        
        # 根据参数选择是否冻结特征层
        if freeze_features:
            for param in self.conv_layers.parameters():
                param.requires_grad = False  # 冻结特征层参数
        
        # 自定义回归头，用于转向角预测（单回归头）
        self.regressor = nn.Sequential(
            # 展平层
            nn.Flatten(),  # 展平操作
            nn.Dropout(0.5),  # 丢弃 50%

            # 全连接层1
            nn.Linear(512, 256),  # ResNet18 输出维度 512 -> 256
            nn.BatchNorm1d(256),  # 批归一化
            nn.ReLU(),  # ReLU 激活
            nn.Dropout(0.3),  # 丢弃 30%
            
            # 全连接层2
            nn.Linear(256, 100),  # 256 -> 100
            nn.BatchNorm1d(100),  # 批归一化
            nn.ReLU(),  # ReLU 激活
            nn.Dropout(0.3),  # 丢弃 30%
            
            # 全连接层3
            nn.Linear(100, 50),  # 100 -> 50
            nn.BatchNorm1d(50),  # 批归一化
            nn.ReLU(),  # ReLU 激活
            nn.Dropout(0.2),  # 丢弃 20%
            
            # 全连接层4
            nn.Linear(50, 10),  # 50 -> 10
            nn.ReLU(),  # ReLU 激活
            
            # 最终输出层
            nn.Linear(10, 1)  # 最终输出：10 -> 1，用于回归预测转向角
        )
        
    def forward(self, x):
        x = self.conv_layers(x)  # 通过 ResNet18 的特征提取部分
        x = self.regressor(x)  # 通过回归头得到预测
        return x.squeeze()  # 去除多余维度并返回