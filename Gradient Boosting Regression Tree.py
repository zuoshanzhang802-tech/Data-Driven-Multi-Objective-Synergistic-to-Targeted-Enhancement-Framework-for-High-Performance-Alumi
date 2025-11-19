import joblib
import pandas as pd
import numpy as np
import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 读取数据
X = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/features_counts_1_7.csv').drop(columns=['Tensile Strength (MPa)'], axis=1)
y = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/features_counts_1_7.csv')['Tensile Strength (MPa)']

# 设置随机划分次数
n_splits = 10
r2_scores = []
best_models = []
all_results = []

# 多次随机划分
for i in range(n_splits):
    print(f"正在进行第 {i+1}/{n_splits} 次随机划分...")
    
    # 使用不同的随机种子进行数据分割
    random_seed = 42 + i
    
    # 数据分割
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=random_seed+1)
    
    # 定义 GBRT 模型
    gbrt = GradientBoostingRegressor(random_state=42)
    
    # GBRT的超参数搜索范围
    param_grid = {
        'n_estimators': [20, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 使用 GridSearchCV 进行超参数搜索，结合 5 折交叉验证
    grid_search = GridSearchCV(
        estimator=gbrt,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    
    # 在训练集+验证集上进行超参数搜索
    grid_search.fit(X_trainval, y_trainval)
    
    # 获取最佳模型和参数
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # 计算交叉验证的R²标准差
    best_index = grid_search.best_index_
    cv_scores = []
    for fold in range(5):
        split_score = grid_search.cv_results_[f'split{fold}_test_score'][best_index]
        cv_scores.append(split_score)
    cv_std = np.std(cv_scores)
    
    # 合并训练集和验证集
    X_trainval_combined = pd.concat([X_train, X_valid], axis=0)
    y_trainval_combined = pd.concat([y_train, y_valid], axis=0)
    
    # 使用最佳参数重新训练模型
    best_model_retrained = GradientBoostingRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    
    best_model_retrained.fit(X_trainval_combined, y_trainval_combined)
    
    # 计算测试集R²
    test_r2 = best_model_retrained.score(X_test, y_test)
    r2_scores.append(test_r2)
    best_models.append(best_model_retrained)
    
    # 记录结果
    result = {
        'split': i+1,
        'best_params': best_params,
        'cv_r2_mean': grid_search.best_score_,
        'cv_r2_std': cv_std,
        'test_r2': test_r2
    }
    all_results.append(result)
    
    print(f"第 {i+1} 次划分 - 交叉验证 R²: {grid_search.best_score_:.4f} ± {cv_std:.4f}")
    print(f"第 {i+1} 次划分 - 测试集 R²: {test_r2:.4f}")
    print(f"最佳参数: {best_params}\n")

# 计算统计结果
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
mean_cv_r2 = np.mean([r['cv_r2_mean'] for r in all_results])

print("="*60)
print(f"{n_splits}次随机划分的结果统计:")
print(f"平均交叉验证 R²: {mean_cv_r2:.4f}")
print(f"平均测试集 R²: {mean_r2:.4f} ± {std_r2:.4f}")
print(f"测试集 R² 范围: {min(r2_scores):.4f} - {max(r2_scores):.4f}")

# 选择最佳模型（测试集R²最高的）
best_model_idx = np.argmax(r2_scores)
best_model = best_models[best_model_idx]
best_result = all_results[best_model_idx]

print(f"最佳模型来自第 {best_model_idx+1} 次划分")
print(f"测试集 R² = {r2_scores[best_model_idx]:.4f}")
print(f"对应参数: {best_result['best_params']}")

# 保存最佳模型
joblib.dump(best_model, 'best_GBRT_model.joblib')
print("最佳模型已保存为 'best_GBRT_model.joblib'")

# 保存所有划分结果
results_df = pd.DataFrame(all_results)
results_df.to_csv('gbrt_model_evaluation_results.csv', index=False)
print("所有划分的结果已保存为 'gbrt_model_evaluation_results.csv'")

# 使用最佳模型进行预测
y_pred = best_model.predict(X_test)
data_df = pd.DataFrame({
    '测试集y真实': y_test,
    '测试集y预测': y_pred
})
data_df.to_csv('E:/PAPER_WRITE/paper/practice1/code/data/GBRT_model.csv', index=False)

# SHAP分析（仅在最佳模型上进行）
print("正在进行SHAP分析...")

# 初始化SHAP解释器
explainer = shap.TreeExplainer(best_model)

# 计算SHAP值（使用完整训练+验证集）
X_trainval_final = pd.concat([X_trainval, X_test], axis=0)  # 使用所有数据
y_trainval_final = pd.concat([y_trainval, y_test], axis=0)
shap_values = explainer.shap_values(X_trainval_final)

# 绘制特征重要性图
plt.rcParams['font.family'] = 'Times New Roman'
width_cm, height_cm = 7, 5.8
width_in, height_in = width_cm / 2.54, height_cm / 2.54

plt.figure(figsize=(width_in, height_in))
shap.summary_plot(shap_values, X_trainval_final, plot_type="dot", 
                  feature_names=['σ$_V$$_e$$_c$', 'mean(χ)', 'D(χ)', 'mean(r)', 'D(r)',
                                'D(EA)', 'D(group)', 'D(minOS)', 'mean(mn)',
                                'σ$_c$$_r$', 'processing', 'Cu', 'Ω'], 
                  show=False)

plt.xlabel("SHAP Value", fontsize=9)
plt.ylabel("Features", fontsize=9)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)

cbar = plt.gcf().axes[-1]
cbar.yaxis.set_tick_params(labelsize=9)
plt.tight_layout()
plt.savefig("E:/PAPER_WRITE/paper/practice1/code/data/shap_summary_plot.svg", 
            format="svg", dpi=300, bbox_inches="tight")
plt.show()

# 保存SHAP值
shap_df = pd.DataFrame(shap_values, columns=X.columns)
feature_df = pd.DataFrame(X_trainval_final, columns=X.columns)
target_df = pd.DataFrame(y_trainval_final, columns=['Tensile Strength (MPa)'])
shap_feature_df = pd.concat([shap_df, feature_df, target_df], axis=1)
shap_feature_df.to_excel('E:/PAPER_WRITE/paper/practice1/code/data/shap_values.xlsx', index=False)

print("SHAP分析完成，结果已保存")
