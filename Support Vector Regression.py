import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import joblib

# 读取数据
X = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/features_counts_1_13.csv').drop(columns=['Tensile Strength (MPa)'], axis=1)
y = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/features_counts_1_13.csv')['Tensile Strength (MPa)']

# 设置随机划分次数
n_splits = 10
r2_scores = []
best_models = []
all_results = []

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
        ('scaler', StandardScaler()),
        ('svm', SVR())
    ])
    
    # SVR的超参数搜索空间
    parameters = {
        'svm__C': [100, 500, 1000, 2000, 5000],
        'svm__gamma': [0.001, 0.01, 0.1, 0.3, 0.4, 0.5, 1.0],
        'svm__kernel': ['rbf']  # 可以添加其他核函数如['rbf', 'linear', 'poly']
    }
    
    # 使用GridSearchCV进行超参数搜索和交叉验证
    grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_trainval, y_trainval)
    
    # 获取最佳模型和最佳参数
    best_svr = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # 计算最佳参数组合在5折交叉验证中的R²标准差
    best_index = grid_search.best_index_
    cv_scores = []
    for fold in range(5):
        split_score = grid_search.cv_results_[f'split{fold}_test_score'][best_index]
        cv_scores.append(split_score)
    cv_std = np.std(cv_scores)
    
    # 合并训练集和验证集
    X_trainval_combined = pd.concat([X_train, X_valid], axis=0)
    y_trainval_combined = pd.concat([y_train, y_valid], axis=0)
    
    # 使用最佳参数重新构建模型
    best_svr_retrained = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVR(
            C=best_params['svm__C'],
            gamma=best_params['svm__gamma'],
            kernel=best_params.get('svm__kernel', 'rbf')
        ))
    ])
    
    # 在训练集+验证集上训练模型
    best_svr_retrained.fit(X_trainval_combined, y_trainval_combined)
    
    # 计算模型在测试集上的R²
    r2_test = best_svr_retrained.score(X_test, y_test)
    r2_scores.append(r2_test)
    best_models.append(best_svr_retrained)
    
    # 记录每次划分的结果
    result = {
        'split': i+1,
        'best_params': best_params,
        'cv_r2_mean': grid_search.best_score_,
        'cv_r2_std': cv_std,
        'test_r2': r2_test
    }
    all_results.append(result)
    
    print(f"第 {i+1} 次划分 - 交叉验证 R²: {grid_search.best_score_:.4f} ± {cv_std:.4f}")
    print(f"第 {i+1} 次划分 - 测试集 R²: {r2_test:.4f}")
    print(f"最佳参数: {best_params}\n")

# 计算平均R²和标准差
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
mean_cv_r2 = np.mean([r['cv_r2_mean'] for r in all_results])

print("="*60)
print(f"{n_splits}次随机划分的结果统计:")
print(f"平均交叉验证 R²: {mean_cv_r2:.4f}")
print(f"平均测试集 R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"测试集 R² 范围: {min(r2_scores):.4f} - {max(r2_scores):.4f}")

# 找到性能最好的模型
best_model_idx = np.argmax(r2_scores)
best_model = best_models[best_model_idx]
best_result = all_results[best_model_idx]

print(f"最佳模型来自第 {best_model_idx+1} 次划分")
print(f"测试集 R² = {r2_scores[best_model_idx]:.4f}")
print(f"对应参数: {best_result['best_params']}")

# 保存最佳模型
joblib.dump(best_model, 'best_svr_model.joblib')
print("最佳模型已保存为 'best_svr_model.joblib'")

# 保存所有划分的结果
results_df = pd.DataFrame(all_results)
results_df.to_csv('svr_model_evaluation_results.csv', index=False)
print("所有划分的结果已保存为 'svr_model_evaluation_results.csv'")

# 使用最佳模型进行预测并保存结果
train_predictions = best_model.predict(X_trainval_combined)
test_predictions = best_model.predict(X_test)

# 保存训练集预测结果
train_results = pd.DataFrame({
    'Actual': y_trainval_combined,
    'Predicted': train_predictions
})
train_results.to_csv('E:/PAPER_WRITE/paper/practice1/code/data/train_svr.csv', index=False)

# 保存测试集预测结果
test_results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': test_predictions
})
test_results.to_csv('E:/PAPER_WRITE/paper/practice1/code/data/test_svr.csv', index=False)

print("预测结果已保存")
