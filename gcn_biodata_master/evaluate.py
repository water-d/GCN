import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score, fowlkes_mallows_score
import seaborn as sns
import umap

# 加载数据
features = np.load('D:\\Download\\GCN\\biological_cluster\\GCN_cluster\\CST-main\\generated_data\\V1_Breast_Cancer_Block_A_Section_1\\features.npy')
coordinates = np.load('D:\\Download\\GCN\\biological_cluster\\GCN_cluster\\CST-main\\generated_data\\V1_Breast_Cancer_Block_A_Section_1\\coordinates.npy')

gcn_labels = np.loadtxt('D:\\Download\\GCN\\biological_cluster\\GCN_cluster\\CST-main\\results\\V1_Breast_Cancer_Block_A_Section_1\\lambdaI0.3\\types.txt', usecols=1, dtype=np.int32)

# 加载原始标签数据
metadata = pd.read_csv('D:\\Download\\GCN\\biological_cluster\\GCN_cluster\\CST-main\\dataset\\V1_Breast_Cancer_Block_A_Section_1\\metadata .tsv', sep='\t')
ground_truth_labels = metadata['fine_annot_type'].values  # 使用 fine_annot_type 列作为原始标签数据

# 定义聚类函数
def perform_clustering(features, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)

    labels_kmeans = kmeans.fit_predict(features)
    labels_hierarchical = hierarchical.fit_predict(features)

    return labels_kmeans, labels_hierarchical

labels_kmeans, labels_hierarchical = perform_clustering(features)

# 定义评估函数
def evaluate_clustering(features, labels_kmeans, labels_hierarchical, gcn_labels, ground_truth_labels):
    scores = pd.DataFrame({
        'Method': ['KMeans', 'Hierarchical', 'GCN'],
        'Adjusted Rand Score': [
            adjusted_rand_score(ground_truth_labels, labels_kmeans),
            adjusted_rand_score(ground_truth_labels, labels_hierarchical),
            adjusted_rand_score(ground_truth_labels, gcn_labels)
        ],
        'Homogeneity Score': [
            homogeneity_score(ground_truth_labels, labels_kmeans),
            homogeneity_score(ground_truth_labels, labels_hierarchical),
            homogeneity_score(ground_truth_labels, gcn_labels)
        ],
        'Completeness Score': [
            completeness_score(ground_truth_labels, labels_kmeans),
            completeness_score(ground_truth_labels, labels_hierarchical),
            completeness_score(ground_truth_labels, gcn_labels)
        ],
        'V-Measure': [
            v_measure_score(ground_truth_labels, labels_kmeans),
            v_measure_score(ground_truth_labels, labels_hierarchical),
            v_measure_score(ground_truth_labels, gcn_labels)
        ],
        'Adjusted Mutual Information': [
            adjusted_mutual_info_score(ground_truth_labels, labels_kmeans),
            adjusted_mutual_info_score(ground_truth_labels, labels_hierarchical),
            adjusted_mutual_info_score(ground_truth_labels, gcn_labels)
        ],
        'Fowlkes-Mallows Index': [
            fowlkes_mallows_score(ground_truth_labels, labels_kmeans),
            fowlkes_mallows_score(ground_truth_labels, labels_hierarchical),
            fowlkes_mallows_score(ground_truth_labels, gcn_labels)
        ]
    })
    return scores

scores = evaluate_clustering(features, labels_kmeans, labels_hierarchical, gcn_labels, ground_truth_labels)
print("Clustering Performance Evaluation:\n", scores)

# 可视化聚类结果
def visualize_clustering(coordinates, labels, title):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.savefig(f'D:\\Download\\GCN\\biological_cluster\\GCN_cluster\\CST-main\\results\\V1_Breast_Cancer_Block_A_Section_1\\{title}.png')
    plt.show()
'''
visualize_clustering(coordinates, labels_kmeans, 'KMeans Clustering')
visualize_clustering(coordinates, labels_hierarchical, 'Hierarchical Clustering')
visualize_clustering(coordinates, gcn_labels, 'GCN Clustering')
'''
