import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiTaskNvidiaModel(nn.Module):
    """多任务模型：共享 ResNet 主干 + 多个头。

    结构说明：
    - backbone (ResNet18 convs) -> shared_head -> 128-d 特征
    - traffic light 分类头 (tl_head)：从 shared 特征输出 raw logits (num_tl_classes)
    - is_stopped 二分类头 (stop_head)：从 shared 特征输出 1 个 logit
    - steer_head：使用 shared 特征 (+ 可选 prev_speed)
    - throttle / brake：在 steer 输入基础上，额外拼接 traffic light 的 softmax 概率向量和 is_stopped 的 sigm 概率（concat fusion）

    返回形式：字典，包括 'steer','throttle','brake','tl_logits','stop_logit'
    """

    def __init__(self, pretrained=True, freeze_features=False, use_speed_input=True, num_tl_classes=4):
        super().__init__()
        self.use_speed_input = use_speed_input
        self.num_tl_classes = int(num_tl_classes)

        # 1) Backbone: ResNet18 去掉最后全连接层
        resnet = models.resnet18(pretrained=pretrained)
        self.conv_layers = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_features:
            for p in self.conv_layers.parameters():
                p.requires_grad = False

        # 2) Shared projection head
        self.shared_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # 3) Auxiliary heads (traffic light classification and stop detection)
        self.tl_head = nn.Linear(128, self.num_tl_classes)
        self.stop_head = nn.Linear(128, 1)  # 输出 logit，训练时用 BCEWithLogitsLoss

        # 4) Task heads
        head_in_dim = 128 + (1 if self.use_speed_input else 0)

        self.steer_head = nn.Sequential(
            nn.Linear(head_in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

        # throttle/brake 接受拼接特征：shared(+speed) + tl_probs + stop_prob
        ext_dim = head_in_dim + self.num_tl_classes + 1
        self.throttle_head = nn.Sequential(nn.Linear(ext_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.brake_head = nn.Sequential(nn.Linear(ext_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x, prev_speed_kmh=None):
        """前向：x (B,C,H,W)，prev_speed_kmh 可选 (B,) 标量向量。"""
        b = self.conv_layers(x)
        shared = self.shared_head(b)  # B x 128

        # auxiliary predictions from shared features
        tl_logits = self.tl_head(shared)         # B x num_tl_classes (raw logits)
        stop_logit = self.stop_head(shared).squeeze(1)  # B

        # 将 shared 特征扩展为任务输入，必要时拼接速度
        task_feat = shared
        if self.use_speed_input:
            if prev_speed_kmh is None:
                raise ValueError("use_speed_input=True 时，forward 需要提供 prev_speed_kmh")
            speed_feat = (prev_speed_kmh.view(-1, 1) / 100.0).to(task_feat.dtype)
            task_feat = torch.cat([task_feat, speed_feat], dim=1)

        # steering: 先计算 raw logit（未激活），再使用 tanh 约束输出
        steer_logit = self.steer_head(task_feat).squeeze(1)
        steer = torch.tanh(steer_logit)

        # 计算附加概率特征并拼接到 throttle/brake 的输入中
        tl_probs = F.softmax(tl_logits, dim=1)            # B x num_tl
        stop_prob = torch.sigmoid(stop_logit).unsqueeze(1)  # B x 1

        ext = torch.cat([task_feat, tl_probs, stop_prob], dim=1)

        throttle = torch.sigmoid(self.throttle_head(ext)).squeeze(1)
        brake = torch.sigmoid(self.brake_head(ext)).squeeze(1)

        return {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "tl_logits": tl_logits,
            "stop_logit": stop_logit,
            "steer_logit": steer_logit,
        }
