import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import RFE

data = pd.read_excel('E:/PAPER_WRITE/paper/practice1/code/data/data.xlsx')
X = data.drop(columns=['Tensile Strength (MPa)', 'density'], axis=1)
y = data['Tensile Strength (MPa)']

scores_random_state = []
for random_state in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    scores_n_features = []
    for n_features_to_select in range(1, 11, 1):

        select = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=n_features_to_select)
        select.fit(X_train, y_train)

        # 对训练集进行特征选择
        X_train_selected = select.transform(X_train)

        # 对测试集进行相同的特征选择
        X_test_selected = select.transform(X_test)

        # 使用选定的特征训练最终模型
        final_model = RandomForestRegressor(n_estimators=100, random_state=42)
        final_model.fit(X_train_selected, y_train)
        # 在测试集上进行预测
        y_pred = final_model.predict(X_test_selected)
        y_pred_train = final_model.predict(X_train_selected)
        if n_features_to_select==7 and random_state == 1:
            pd.DataFrame({'y_Experiment_train': y_train, 'y_pred_train': y_pred_train}).to_csv('E:/PAPER_WRITE/paper/practice1/code/data/train_rf.csv', index=False)
            pd.DataFrame({'y_Experiment_test': y_test, 'y_pred_test': y_pred}).to_csv('E:/PAPER_WRITE/paper/practice1/code/data/test_rf.csv', index=False)
        # 计算决定系数（R²）
        r2 = r2_score(y_test, y_pred)
        # 计算mean_squared_error
        r2 = mean_squared_error(y_test, y_pred)
        #print(f"测试集上的决定系数 (R²): {r2}")
        scores_n_features.append(r2)
        if n_features_to_select==7 and random_state == 1:
            # 获取被选中的特征的掩码
            selected_features_mask = select.support_

            # 获取被选中的特征的名字
            selected_feature_names = X_train.columns[selected_features_mask]

            # 打印被选中的特征的名字
            print("Selected feature names:", selected_feature_names)
            data_get = data[selected_feature_names].copy()
            data_get['Tensile Strength (MPa)'] = data['Tensile Strength (MPa)']
            data_get.to_csv(f'E:/PAPER_WRITE/paper/practice1/code/data/features_counts_{random_state}_{n_features_to_select}.csv', index=False)


    print('scores_n_features=', scores_n_features)
    scores_random_state.append(scores_n_features)
    print('scores_random_state=', scores_random_state)
print([sum(col) / len(col) for col in zip(*scores_random_state)])
pd.DataFrame([sum(col) / len(col) for col in zip(*scores_random_state)]).to_csv(
    'E:/PAPER_WRITE/paper/practice1/code/data/scores_mean.csv', index=False)

pd.DataFrame(scores_random_state).to_csv('E:/PAPER_WRITE/paper/practice1/code/data/scores_histogram.csv', index=False)

