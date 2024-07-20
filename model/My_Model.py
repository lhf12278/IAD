import torch
import os
import torch.nn as nn
import torch
import re
from math import sqrt
# from torch_geometric.nn import GCNConv
import torch.nn as nn
import numpy as np
import cv2
import copy
import torch.nn.functional as F
from collections import defaultdict
def load_model_with_max_number(folder_path):
    # 获取文件夹中所有模型文件的列表
    model_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]

    # 如果没有找到任何模型文件，可以进行错误处理或返回默认模型
    if not model_files:
        raise FileNotFoundError("未找到任何模型文件")

    # 从模型文件名中提取数字，并找到最大的数字
    max_number = -1
    max_model_file = None

    for model_file in model_files:
        # 从文件名中提取数字部分
        try:
            number = int(model_file.split('_')[1].split('.')[0])
            if number > max_number:
                max_number = number
                max_model_file = model_file
        except ValueError:
            continue

    # 检查是否找到了具有最大数字的模型文件
    if max_model_file is not None:
        # 构建最终的模型文件路径
        max_model_path = os.path.join(folder_path, max_model_file)
        # 导入最大数字的模型
        model = torch.load(max_model_path)
        # os.remove(max_model_path)
        return model
    else:
        return None



def update_memory(T1,T2,weight):

    # 定义权重

    # 将类中心特征张量化为列表
    T1_features = [T1[label]['features'] for label in T1.keys()]
    T2_features = [T2[label]['features'] for label in T2.keys()]

    # 将类中心特征堆叠为张量
    T1_tensor = torch.stack(T1_features, dim=0)
    T2_tensor = torch.stack(T2_features, dim=0)

    # 计算加权平均特征（利用广播）
    merged_features = (T1_tensor * weight) + (T2_tensor * (1 - weight))

    # 创建新的类中心特征字典
    merged_T = defaultdict(lambda: {'features': torch.zeros(3), 'count': 0})

    # 更新合并后的类中心特征，不包括 count 字段
    for idx, label in enumerate(T1.keys()):
        merged_T[label]['features'] = merged_features[idx]
        merged_T[label]['count'] = T1[label]['count']  # 保持原始 count 值

    return merged_T


def compute_class_centers(data,device):
    class_centers = defaultdict(lambda: {'features': torch.zeros(384).to(device), 'count': 0})
    for sample in data:
        # print(sample,'======')
        label = sample['label'].item()  # 将标签张量转换为整数
        features = sample['features']
        class_centers[label]['features'] += features.to(device)
        class_centers[label]['count'] += 1
    for label in class_centers:
        if class_centers[label]['count']==0:
            print(class_centers[label]['count'])
        class_centers[label]['features'] /= class_centers[label]['count']
    return class_centers



def cos_simility_loss(feat, target_feat, target):
    feat_normalized = feat / torch.norm(feat, dim=1, keepdim=True)
    target_feat_normalized = target_feat / torch.norm(target_feat, dim=1, keepdim=True)
    # 计算点积矩阵
    cos_sim_matrix =torch.mm(feat_normalized, target_feat_normalized.t())

    # 计算余弦相似度矩阵
    # cos_sim_matrix = dot_products / torch.ger(M_norms, Q_norms)  # 每一对对应行向量之间的余弦相似度
    cos_sim = torch.diag(cos_sim_matrix)
    mse_loss = nn.MSELoss()
    target_num = torch.tensor([target]).cuda()
    LOSS = torch.mean(mse_loss(cos_sim, target_num))

    return LOSS




def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def colorful_spectrum_mix(img1, img2, alpha, ratio=1.0):
    """Input image size: ndarray of [B, C, H, W]"""
    lam_noise = torch.FloatTensor(np.random.uniform(0, alpha, size=(img1.shape[0],img1.shape[1], 1, 1))).cuda()
    lam_peo = torch.FloatTensor(np.random.uniform(alpha, 1, size=(img1.shape[0],img1.shape[1], 1, 1))).cuda()
    assert img1.shape == img2.shape
    c, h, w = img1.shape[1:]
    h_crop = int(h * np.sqrt(ratio))
    w_crop = int(w * np.sqrt(ratio))
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)


    img1_fft = torch.fft.fftn(img1, dim=(2, 3))
    img2_fft = torch.fft.fftn(img2, dim=(2, 3))
    img1_abs, img1_pha = torch.abs(img1_fft), torch.angle(img1_fft)
    img2_abs, img2_pha = torch.abs(img2_fft), torch.angle(img2_fft)
    img1_abs = torch.fft.fftshift(img1_abs, dim=(2, 3))
    img2_abs = torch.fft.fftshift(img2_abs, dim=(2, 3))
    img1_abs_ = torch.clone(img1_abs)
    img2_abs_ = torch.clone(img2_abs)


    img1_abs[:,:, h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam_noise * img2_abs_[:,:, h_start:h_start + h_crop, w_start:w_start + w_crop] + \
        lam_peo * img1_abs_[:,:, h_start:h_start + h_crop, w_start:w_start + w_crop]
    img1_abs = torch.fft.ifftshift(img1_abs, dim=(2, 3))
    img21 = torch.fft.ifftn(img1_abs * torch.exp(1j * img1_pha), dim=(2, 3)).real
    return img21


class Attack(nn.Module):
    def __init__(self, alpha = None,input_channel=3,output_channel=64,latent_dim=32):
        super(Attack, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, latent_dim, 3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, output_channel, 3, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, input_channel, 3, padding=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.alpha = alpha

    def forward(self, x):
        B, C, H, W = x.shape
        matrax = torch.full((B, C, H, W), 0.5, requires_grad=True).cuda()
        new_matrax = self.decoder(self.encoder(matrax))
        att_img = colorful_spectrum_mix(x, new_matrax, self.alpha)
        return att_img


class Bn(nn.Module):
    def __init__(self, inplanes):
        super().__init__()

        self.bottleneck = nn.BatchNorm1d(inplanes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.bottleneck(x)
        return x
class CrossEntropy(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
		super(CrossEntropy, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.use_gpu = use_gpu
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		device = "cuda"
		target1 = targets.to(device)
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros(log_probs.size()).scatter_(1, target1.unsqueeze(1).data.cpu(), 1)
		if self.use_gpu: targets = targets.to(torch.device('cuda'))
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (- targets * log_probs).mean(0).sum()
		return loss

def KL_loss(generate_vic,target_vic):
    kld_loss = nn.KLDivLoss(reduction='batchmean')
    distribution1 = torch.log_softmax(generate_vic, dim=1)
    distribution2 = nn.functional.softmax(target_vic, dim=1)
    loss = kld_loss(distribution1, distribution2)
    return loss

class KL_defen(nn.Module):
    def __init__(self, part_layer):
        super(KL_defen, self).__init__()
        self.part_layer = part_layer
        self.KL_loss =  KL_loss
        self.in_planes = 768

    def forward(self, x_feat,x_feat_noise):

        part_token1 = x_feat[:, 0:1]
        part_token1_noise = x_feat_noise[:, 0:1]

        new_x_feat1 = torch.cat((part_token1,x_feat_noise[:,1:,]),dim=1)

        new_x_feat1_noise = torch.cat((part_token1_noise, part_token1[:, 1:, ]), dim=1)

        new_x_feat1_after = self.part_layer(new_x_feat1)[:,0]

        new_x_feat1_noise_after = self.part_layer(new_x_feat1_noise)[:,0]


        kl_1_token = self.KL_loss(new_x_feat1_after,x_feat[:,0])
        kl_1_token_noise = self.KL_loss(new_x_feat1_noise_after, x_feat_noise[:,0])

        kl_loss = (kl_1_token +  kl_1_token_noise)/2


        return kl_loss,new_x_feat1_after,new_x_feat1_noise_after




class DG_Net(nn.Module):
    def __init__(self,model,num_classes):#####  ,GGLM1,GGLM2
        super(DG_Net, self).__init__()
        self.E = model
        self.KL_defen = KL_defen(model.layer1)

        self.num_classes = num_classes
        self.in_planes = 384


        self.classifier_final1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_final1.apply(weights_init_classifier)


        self.BN_fin1 = nn.BatchNorm1d(self.in_planes)
        self.BN_fin1.bias.requires_grad_(False)
        self.BN_fin1.apply(weights_init_kaiming)

        self.gap = nn.AdaptiveAvgPool1d(1)


    def forward(self, x, x_noise):

        global_x = self.E(x)
        global_x_noise = self.E(x_noise)

        if self.training:

            kl_loss, feat1, feat1_noise = self.KL_defen(global_x, global_x_noise)

            x_1 = global_x[:, 0]
            x_1_noise = global_x_noise[:, 0]


            score_fin1 = self.classifier_final1(self.BN_fin1(x_1))
            score_fin1_noise = self.classifier_final1(self.BN_fin1(x_1_noise))

            feat_list = [x_1, x_1_noise,feat1,feat1_noise]###
            score_list = [score_fin1, score_fin1_noise]


            return feat_list, score_list,kl_loss
        else:
            result = global_x[:, 0]
            return result
