from utils import load_data_SGER, build_adjacency_matrix
import pandas as pd
from GCN import GraphSAGE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

# 调用函数并获取数据
X_train, X_valid, X_test, y_train, y_valid, y_test = load_data_SGER()

# 检查数据划分比例
print(f"Train set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_valid.shape}, {y_valid.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")

# 使用训练集 + 验证集作为 X 和 y
X_combined = pd.concat([X_train, X_valid], axis=0)
y_combined = pd.concat([y_train, y_valid], axis=0)

# 构建邻接矩阵
Adj_Mtx = build_adjacency_matrix(X_combined, y_combined, n_estimators=100, threshold=0.2)

# 假设Adj_Mtx是已经计算出来的邻接矩阵
# 需要将其转换成PyTorch Geometric的格式
edge_index = torch.tensor(Adj_Mtx.nonzero(), dtype=torch.long).t().contiguous()  # 获取邻接矩阵的边
node_features = torch.tensor(X_combined.values, dtype=torch.float)  # 特征

# 假设任务是分类，输出的标签也可以作为目标进行训练
labels = torch.tensor(y_combined.values, dtype=torch.long)

# 创建Data对象，将数据输入模型
data = Data(x=node_features, edge_index=edge_index, y=labels)

# 定义GraphSAGE模型
model = GraphSAGE(in_channels=node_features.shape[1], hidden_channels=64, out_channels=len(labels.unique()))

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 定义训练过程
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # 前向传播
    loss = F.cross_entropy(out, data.y)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 优化参数
    return loss.item()


# 训练模型
for epoch in range(200):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# 预测
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    _, pred = out.max


