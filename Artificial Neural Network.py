import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 读取数据
X = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/features_counts_1_13.csv').drop(columns=['Tensile Strength (MPa)'], axis=1)
y = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/features_counts_1_13.csv')['Tensile Strength (MPa)']

# 设置随机划分次数
n_splits = 10
r2_scores = []
best_models = []

# 多次随机划分
for i in range(n_splits):
    print(f"正在进行第 {i+1}/{n_splits} 次随机划分...")
    
    # 使用不同的随机种子进行数据分割
    random_seed = 42 + i  # 每次使用不同的随机种子
    
    # 数据分割
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=random_seed+1)
    
    # 定义Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 数据标准化
        ('mlp', MLPRegressor(random_state=42, max_iter=10000))  # 多层感知机回归器
    ])
    
    # ANN的超参数搜索空间
    parameters = {
        'mlp__hidden_layer_sizes': [(15, 49, 15), (10, 15, 10)],  # 隐藏层神经元数量
        'mlp__activation': ['relu', 'tanh'],  # 激活函数
        'mlp__alpha': [0.001, 0.01, 0.05, 0.1, 0.5],  # L2正则化参数
        'mlp__learning_rate_init': [0.01]  # 初始学习率
    }
    
    # 使用GridSearchCV进行超参数搜索和交叉验证
    grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_trainval, y_trainval)
    
    # 获取最佳模型和最佳参数
    best_mlp = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # 合并训练集和验证集
    X_trainval_combined = np.concatenate([X_train, X_valid], axis=0)
    y_trainval_combined = np.concatenate([y_train, y_valid], axis=0)
    
    # 使用最佳参数重新构建模型
    best_mlp_retrained = MLPRegressor(
        hidden_layer_sizes=best_params['mlp__hidden_layer_sizes'],
        activation=best_params['mlp__activation'],
        alpha=best_params['mlp__alpha'],
        learning_rate_init=best_params['mlp__learning_rate_init'],
        random_state=42,
        max_iter=10000
    )
    
    # 数据标准化
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval_combined)
    X_test_scaled = scaler.transform(X_test)
    
    # 在训练集+验证集上训练模型
    best_mlp_retrained.fit(X_trainval_scaled, y_trainval_combined)
    
    # 计算模型在测试集上的R²
    r2_test = best_mlp_retrained.score(X_test_scaled, y_test)
    r2_scores.append(r2_test)
    best_models.append(best_mlp_retrained)
    
    print(f"第 {i+1} 次划分 - 测试集 R²: {r2_test:.4f}")
    print(f"最佳参数: {best_params}\n")

# 计算平均R²和标准差
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)

print("="*50)
print(f"{n_splits}次随机划分的平均测试集 R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"R² 范围: {min(r2_scores):.4f} - {max(r2_scores):.4f}")

# 找到性能最好的模型
best_model_idx = np.argmax(r2_scores)
best_model = best_models[best_model_idx]

print(f"最佳模型来自第 {best_model_idx+1} 次划分，R² = {r2_scores[best_model_idx]:.4f}")

# 保存最佳模型
joblib.dump(best_model, 'best_mlp_model.joblib')
print("最佳模型已保存为 'best_mlp_model.joblib'")

# 可选：保存所有模型的信息
results_df = pd.DataFrame({
    'split': range(1, n_splits+1),
    'r2_score': r2_scores
})
results_df.to_csv('model_evaluation_results.csv', index=False)
print("所有划分的结果已保存为 'model_evaluation_results.csv'")
