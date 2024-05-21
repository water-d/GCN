# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
NHID = 16
weight_decay = 5e-4
learning_rate = 0.001
Dropout=0.5

import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GCN
import utils_graph
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans

adj, features, labels, idx_train, idx_val, idx_test = utils_graph.load_data()

#print(features.shape)

model = GCN(features.shape[1],NHID,10,Dropout)
optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
#print(model.parameters())
model = model.cuda()
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()

# 添加全局变量以记录历史数据
train_losses = []
val_accuracies = []

def visualize_with_tsne(features, labels):
    # 首先确保特征已经移动到CPU并从计算图中分离
    features = features.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()  # 确保标签也在CPU上，并转为numpy

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=33)
    embeddings = tsne.fit_transform(features)  # Ensure the data is on CPU and detached from gradients

    # 获取唯一的标签
    unique_labels = np.unique(labels)  # 使用numpy的unique方法

    # 创建图表
    plt.figure(figsize=(10, 8))
    for label in unique_labels:
        idxs = np.where(labels == label)  # 使用numpy的where方法查找
        plt.scatter(embeddings[idxs, 0], embeddings[idxs, 1], label=f'Class {label}', alpha=0.5)

    # 添加图例和标题
    plt.legend()
    plt.title("t-SNE visualization of GCN node embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_val = utils_graph.accuracy(output[idx_val], labels[idx_val])
    loss_train.backward()
    optimizer.step()

    # 记录损失和准确率
    train_losses.append(loss_train.item())
    val_accuracies.append(acc_val.item())

    if epoch % 100 == 0:
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_val: {:.4f}'.format(acc_val.item()))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = utils_graph.accuracy(output[idx_test], labels[idx_test])
    
    # 计算聚类标签
    cluster_labels = torch.argmax(output[idx_test], dim=1).cpu().detach().numpy()
    true_labels = labels[idx_test].cpu().detach().numpy()
    
    # 计算轮廓系数
    silhouette_avg = silhouette_score(output[idx_test].cpu().detach().numpy(), cluster_labels)

    # 计算其他指标
    homogeneity = homogeneity_score(true_labels, cluster_labels)
    completeness = completeness_score(true_labels, cluster_labels)
    v_measure = v_measure_score(true_labels, cluster_labels)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    ami = adjusted_mutual_info_score(true_labels, cluster_labels)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "silhouette= {:.4f}".format(silhouette_avg),
          "homogeneity= {:.4f}".format(homogeneity),
          "completeness= {:.4f}".format(completeness),
          "v_measure= {:.4f}".format(v_measure),
          "ari= {:.4f}".format(ari),
          "ami= {:.4f}".format(ami))

def plot_performance():
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()
    plt.show()

t_total = time.time()
for epoch in range(20500):  # 20500
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
test()
plot_performance()
