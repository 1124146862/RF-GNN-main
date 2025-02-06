import pandas as pd


import pandas as pd
from sklearn.model_selection import train_test_split

def load_data_SGER():
    # 读取SGER CSV文件
    path = '/home/gehongfei/project/TabGNN/dataset/SGER.csv'

    df = pd.read_csv(path)

    # 检查读取的数据框前几行，确保数据读取正确
    # print(df.head())

    # 确保 'kredit' 列存在
    if 'kredit' not in df.columns:
        print("Error: 'kredit' column not found.")
        return None, None, None, None, None, None

    # 目标变量
    y = df['kredit']
    # 特征
    X = df.drop(columns=['kredit'])

    # 先划分 训练集 (70%) 和 临时集 (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 再划分 验证集 (10%) 和 测试集 (20%) -> 这里 X_temp 是 30%，我们按 1:2 的比例划分
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

    return X_train, X_valid, X_test, y_train, y_valid, y_test

# 调用函数并获取数据
# X_train, X_valid, X_test, y_train, y_valid, y_test = load_data_SGER()
# print(f"Train set: {X_train.shape}, {y_train.shape}")
# print(f"Validation set: {X_valid.shape}, {y_valid.shape}")
# print(f"Test set: {X_test.shape}, {y_test.shape}")


# X, y = load_data()
# print(X.head())  # 打印前几行特征数据
# print(y.head())  # 打印前几行目标变量数据


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix

def build_adjacency_matrix(X, y, n_estimators=100, threshold=None):
    """
    使用随机森林计算样本的接近度（Proximity），并构建邻接矩阵
    :param X: 特征矩阵 (DataFrame 或 numpy array)
    :param y: 目标变量 (Series 或 numpy array)
    :param n_estimators: 随机森林树的数量
    :param threshold: 邻接矩阵的阈值（默认使用均值）
    :return: 邻接矩阵 (numpy array)
    """

    # 1. 训练随机森林分类器
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X, y)

    # 2. 获取叶子节点索引
    leaf_nodes = rf.apply(X)  # 形状: (样本数, 树的数量)

    # 3. 计算接近度矩阵 (Proximity Matrix)
    n_samples = X.shape[0]
    proximity_matrix = np.zeros((n_samples, n_samples))

    # 遍历所有树，计算样本共同落入叶子节点的次数
    for tree_idx in range(n_estimators):
        leaf_ids = leaf_nodes[:, tree_idx]
        for i in range(n_samples):
            for j in range(i, n_samples):
                if leaf_ids[i] == leaf_ids[j]:  # 如果两个样本落在同一叶子节点
                    proximity_matrix[i, j] += 1
                    if i != j:
                        proximity_matrix[j, i] += 1  # 确保对称

    # 归一化：除以树的总数，得到接近度
    proximity_matrix /= n_estimators

    # 4. 转换为邻接矩阵 (Adjacency Matrix)
    if threshold is None:
        threshold = np.mean(proximity_matrix)  # 设定阈值为均值

    adjacency_matrix = (proximity_matrix >= threshold).astype(int)

    return adjacency_matrix

# 示例调用：
# X, y = load_data()  # 先调用你的数据加载函数

