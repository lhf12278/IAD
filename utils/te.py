a = 1  # A显示的次数
b = 2  # B显示的次数
c = 0

my_list = [1, 2, 3, 4, 5]
first_two_values = my_list[0:2]  # 或者使用 my_list[:2]
print(first_two_values)

total_epochs =  500 # 总的显示次数
current_a = 0
current_b = 0
current_c = 0
s_a = 1
s_b = 1

for epoch in range(1, total_epochs + 1):

    if current_a == a and current_b == b:
        current_a = 0
        current_b = 0
    if current_a < a:
        module = 0
        module_epoch = s_a
        s_a += 1
        current_a += 1

    else:
        if current_b < b:
            module = 1
            module_epoch = s_b
            s_b += 1
            current_b += 1
        elif current_a == a or current_b == b:
            current_a = 0
            current_b = 0

    print(epoch,module,module_epoch)

import torch
import torch.nn as nn
def cos1(feat, target_feat, target):
    M_norms = torch.norm(feat, dim=1)  # M的每个行向量的范数
    Q_norms = torch.norm(target_feat, dim=1)  # Q的每个行向量的范数
    # 计算点积矩阵
    dot_products = torch.mm(feat, target_feat.t())  # Q和M的点积矩阵

    # 计算余弦相似度矩阵
    cos_sim_matrix = dot_products / torch.ger(Q_norms, M_norms)  # 每一对对应行向量之间的余弦相似度
    cos_sim = torch.diag(cos_sim_matrix)
    mse_loss = nn.MSELoss()
    target_num = torch.tensor([target]).cuda()
    LOSS = torch.mean(mse_loss(cos_sim, target_num))

    return LOSS


def cos2(feat, target_feat, target):
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

feat = torch.randn(4, 128).cuda()  # 示例输入特征
target_feat = torch.randn(4, 128).cuda()  # 示例目标特征
target_similarity = 0.8  # 示例目标余弦相似度
loss1 = cos1(feat, target_feat, target_similarity)
loss2 = cos2(feat, target_feat, target_similarity)
# print(loss1,loss2)


# def memory(old_feat,feat,labels):



import numpy as np
from collections import defaultdict

# class CustomIterator:
#     def __init__(self, batch_size, num_samples, num_features, num_classes):
#         self.batch_size = batch_size
#         self.num_samples = num_samples
#         self.num_features = num_features
#         self.num_classes = num_classes
#         self.current_index = 0
#         self.data = self.generate_data()
#
#     def generate_data(self):
#         data = []
#         for _ in range(self.num_samples):
#             features = np.random.rand(self.num_features)
#             features = np.round(features, 5)
#             label = np.random.randint(0, self.num_classes)
#             data.append((features, label))
#         return data
#
#     def compute_class_centers(self):
#         class_centers = defaultdict(lambda: np.zeros(self.num_features))
#         class_counts = defaultdict(int)
#
#         for features, label in self.data:
#             class_centers[label] += features
#             class_counts[label] += 1
#
#         for label in class_centers:
#             class_centers[label] /= class_counts[label]
#
#         return class_centers
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.current_index >= self.num_samples:
#             raise StopIteration
#
#         batch_data = self.data[self.current_index:self.current_index + self.batch_size]
#         self.current_index += self.batch_size
#
#         batch_features, batch_labels = zip(*batch_data)
#         batch_features = np.stack(batch_features, axis=0)
#         batch_labels = np.array(batch_labels)
#
#         return batch_features, batch_labels
#
# # 用法示例：
# batch_size = 2
# num_samples = 10
# num_features = 3
# num_classes = 5
#
# custom_iterator = CustomIterator(batch_size, num_samples, num_features, num_classes)
#
# for batch_features, batch_labels in custom_iterator:
#     # 在这里使用每个批次的特征向量和标签
#     print(f"Batch Features Shape: {batch_features}")
#     print(f"Batch Labels: {batch_labels}")
#
# # 计算类中心
# class_centers = custom_iterator.compute_class_centers()
# print(class_centers)
# for label, center in class_centers.items():
#     print()
#     print(f"Class {label} Center: {center}")





import numpy as np
import torch
from collections import defaultdict

class CustomIterator:
    def __init__(self, batch_size, num_samples, num_features, num_classes):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_classes = num_classes
        self.current_index = 0
        self.data = self.generate_data()

    def generate_data(self):
        data = []
        for _ in range(self.num_samples):
            features = torch.rand(self.num_features)  # 生成随机特征张量
            label = torch.randint(0, self.num_classes, (1,), dtype=torch.int64)  # 生成随机标签张量
            data.append({'features': features, 'label': label})
        return data

    def compute_class_centers(self):
        class_centers = defaultdict(lambda: {'features': torch.zeros(self.num_features), 'count': 0})
        print((self.data))
        for sample in self.data:
            # print(sample)
            label = sample['label'].item()  # 将标签张量转换为整数
            features = sample['features']
            class_centers[label]['features'] += features
            class_centers[label]['count'] += 1

        for label in class_centers:
            class_centers[label]['features'] /= class_centers[label]['count']

        return class_centers

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration

        batch_data = self.data[self.current_index:self.current_index + self.batch_size]
        self.current_index += self.batch_size

        batch_features = torch.stack([sample['features'] for sample in batch_data], dim=0)
        batch_labels = torch.stack([sample['label'] for sample in batch_data], dim=0)

        return batch_features, batch_labels

# 用法示例：
batch_size = 2
num_samples = 10
num_features = 3
num_classes = 5

custom_iterator = CustomIterator(batch_size, num_samples, num_features, num_classes)

# for batch_features, batch_labels in custom_iterator:
#     # 在这里使用每个批次的特征张量和标签张量
#     print(f"Batch Features Shape: {batch_features}")
#     print(f"Batch Labels: {batch_labels}")

# 计算类中心
class_centers = custom_iterator.compute_class_centers()
print(class_centers,'++++')
# for label, center in class_centers.items():
#     print(f"Class {label} Center Features: {center['features']} (Count: {center['count']})")


batch_features = torch.randn(10,5)
N=10
batch_labels = torch.randint(0, 2, (N,), dtype=torch.int64)

# 合成特征和标签字典并遍历
# print(batch_labels)



def compute_class_centers(data):
    class_centers = defaultdict(lambda: {'features': torch.zeros(5), 'count': 0})
    for sample in data:
        # print(sample,'======')
        label = sample['label'].item()  # 将标签张量转换为整数
        features = sample['features']
        class_centers[label]['features'] += features
        class_centers[label]['count'] += 1
    for label in class_centers:
        class_centers[label]['features'] /= class_centers[label]['count']
    return class_centers

data_dict = []
feat = []
# for features, label in zip(batch_features, batch_labels):
#     data_dict.append({'features': features, 'label': label})
# center_feat = compute_class_centers(data_dict)
#
#     feat.append(['0']['features'])
# print(data_dict)
# new_feat = torch.cat(feat,dim=0)
# print(new_feat)


import torch
from collections import defaultdict

# 假设有类中心特征 T 和输入的 batch 数据 M
# T = defaultdict(lambda: {'features': torch.zeros(N)})
# T[0]['features'] = torch.tensor([0.5048, 0.2275, 0.2309])
# T[1]['features'] = torch.tensor([0.5511, 0.2424, 0.1799])
# T[2]['features'] = torch.tensor([0.6679, 0.4342, 0.2864])
# T[3]['features'] = torch.tensor([0.2587, 0.4014, 0.3349])
# T[4]['features'] = torch.tensor([0.9750, 0.0525, 0.2276])

id = torch.tensor([1, 1, 2, 2, 3])  # 假设标签张量 id
M = torch.rand((5, N))  # 假设输入的 batch 数据 M，大小为 (B, N)

# 根据标签从类中心特征中取出特征
selected_features = [class_centers[label.item()]['features'] for label in id]

# 使用 torch.stack 将选定的特征拼接在一起，大小与 M 一致
concatenated_features = torch.stack(selected_features, dim=0)

# 打印拼接后的特征矩阵

print(class_centers)
print(concatenated_features,'=======',id)


import torch
from collections import defaultdict

# 假设有两个类中心特征 T1 和 T2
T1 = defaultdict(lambda: {'features': torch.zeros(3), 'count': 0})
T1[0]['features'] = torch.tensor([0.5048, 0.2275, 0.2309])
T1[1]['features'] = torch.tensor([0.5511, 0.2424, 0.1799])
T1[2]['features'] = torch.tensor([0.6679, 0.4342, 0.2864])
T1[3]['features'] = torch.tensor([0.2587, 0.4014, 0.3349])
T1[4]['features'] = torch.tensor([0.9750, 0.0525, 0.2276])

T2 = defaultdict(lambda: {'features': torch.zeros(3), 'count': 0})
T2[0]['features'] = torch.tensor([0.5048, 0.2275, 0.2309])
T2[1]['features'] = torch.tensor([0.5511, 0.2424, 0.1799])
T2[2]['features'] = torch.tensor([0.6679, 0.4342, 0.2864])
T2[3]['features'] = torch.tensor([0.2587, 0.4014, 0.3349])
T2[4]['features'] = torch.tensor([0.9750, 0.0525, 0.2276])

# 定义权重
weight = 0.5

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

# 打印合并后的类中心特征，不包括 count 字段
print(merged_T,'===============')





