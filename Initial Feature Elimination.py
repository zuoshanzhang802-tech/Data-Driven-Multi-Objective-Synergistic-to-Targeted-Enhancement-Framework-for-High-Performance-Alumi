import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
data = pd.read_excel('E:/PAPER_WRITE/paper/practice1/code/data/properties_df.xlsx')


data_correlation = data[['Vec_std','X_mean','X_std/mean*100',
                         'atomic_radius_mean','atomic_radius_std/mean*100','electron_affinity_std/mean*100','group_std/mean*100',
                         'max_oxidation_state_std','min_oxidation_state_std/mean*100',
                         'mendeleev_no_mean','mendeleev_no_std','electrical_resistivity_std/mean*100','molar_volume_mean',
                         'thermal_conductivity_mean','melting_point_std',
                         'atomic_radius_calculated_std','enthalpy','Processing_number',
                         'Ag','Al','Cr','Cu','Fe','Mg','Mn','Ni','Sc','Ti','Zn','Zr', 'oumiga', 'Tensile Strength (MPa)','density',
                         ]].copy()

correlation_matrix = data_correlation.corr()
print(correlation_matrix.iloc[:30, :30])
# 绘制热图
plt.figure(figsize=(12, 12))  # 设置图形大小
sns.heatmap(correlation_matrix.iloc[-10:, -10:],
            annot=True,  # 显示数值
            fmt=".2f",  # 数值格式
            cmap='coolwarm',  # 颜色映射
            square=True,  # 单元格为正方形
            linewidths=0.5,  # 单元格边框宽度
            annot_kws={'size': 12})  # 调整字体大小)
# 添加标题
plt.title('Pearson Correlation Heatmap')
# 保存图像
plt.savefig('E:/PAPER_WRITE/paper/practice1/code/data/pearson_correlation_heatmap.png', dpi=600)
plt.show()


data_correlation[['Tensile Strength (MPa)', 'density']] = data[['Tensile Strength (MPa)', 'density']]
data_correlation.to_excel('E:/PAPER_WRITE/paper/practice1/code/data/data.xlsx', index=False)
