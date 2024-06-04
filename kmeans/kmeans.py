import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def kmeans(X, n_clusters, dim):
    # 随机选择 n_clusters 个数据点作为初始质心
    centers = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    # 存储每个数据点的聚类标签
    labels = np.zeros(X.shape[0], dtype=int)

    while True:
        # 计算每个数据点到各个质心的距离
        distances = np.zeros((X.shape[0], n_clusters))
        for i in range(X.shape[0]):
            for j in range(n_clusters):
                for k in range(dim):
                    distances[i, j] += (X[i][k] - centers[j][k]) ** 2
                distances[i][j] = np.sqrt(distances[i][j])

        # 为每个数据点分配最近的质心标签
        new_labels = np.argmin(distances, axis=1)

        # 如果标签没有变化,算法收敛
        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        # 更新每个质心的位置
        for i in range(n_clusters):
            centers[i] = X[labels == i].mean(axis=0)

    return labels, centers


def calculate_accuracy(labels: numpy.ndarray):
    not_equals_count = 0
    counter = Counter(labels[0:60])
    commons = []
    most_common = counter.most_common(1)
    most_common = most_common[0][0]
    commons.append(most_common)
    for i in range(60):
        if labels[i] != most_common:
            not_equals_count += 1

    counter = Counter(labels[60:120])
    most_common = counter.most_common(1)
    most_common = most_common[0][0]
    commons.append(most_common)
    for i in range(60, 120):
        if labels[i] != most_common:
            not_equals_count += 1

    counter = Counter(labels[120:180])
    most_common = counter.most_common(1)
    most_common = most_common[0][0]
    commons.append(most_common)
    for i in range(120, 180):
        if labels[i] != most_common:
            not_equals_count += 1
    print(not_equals_count)
    print(labels.size)
    return 1 - not_equals_count / labels.size,commons


def calculate_inertia(X: numpy.ndarray, centers: numpy.ndarray,commons, dim):
    sum = 0
    for i in range(60):
        for j in range(dim):
            sum += (X[i][j] - centers[commons[0]][j]) ** 2
    for i in range(60, 120):
        for j in range(dim):
            sum += (X[i][j] - centers[commons[1]][j]) ** 2
    for i in range(120, 180):
        for j in range(dim):
            sum += (X[i][j] - centers[commons[2]][j]) ** 2
    return sum


if __name__ == '__main__':
    df = pd.read_csv('kmeans.csv', header=None)
    df.columns = ['label'] + ['Popularity'] + [f'Score {i}' for i in range(2, 11)]
    true_labels = df['label'].to_numpy()
    # 提取除label列
    cols = [col for col in df.columns if col != 'label']
    X = df[cols].to_numpy()
    # 运行K-Means算法
    labels, centers = kmeans(X, n_clusters=3, dim=10)
    accuracy,commons = calculate_accuracy(labels)
    # 计算所有数据点到各自质心距离的平方和
    inertia = calculate_inertia(X,centers,commons,10)
    print(true_labels)
    print(labels)
    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 9], X[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Score 10')
    plt.ylabel('Score 2')
    plt.title(f'Accuracy:{accuracy:.2f} , Inertia:{inertia:.2f}')
    plt.show()
