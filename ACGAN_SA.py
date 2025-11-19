import re

import matplotlib.pyplot as plt
import pandas as pd
import sys

sys.path.append("E:/python/ACGAN_HEAS")  # 替换为你的目标文件夹路径
from elements_properties import EntropyOfMixingCalculator
from elements_properties import EnthalpyOfMixingCalculator
from elements_properties import R_Analyzer
from elements_properties import get_density
from ACGAN import Generator
import torch
from joblib import load

def objective_function(noise, label, generator, gbrt_model):
    #print('noise', noise)
    #print('label.shape', label)
    # 将噪声和标签拼接，作为生成器的输入
    #input = torch.cat([noise, label], dim=1)
    # 生成器生成数据
    generated_data = generator(noise, label)
    # 将生成的数据转换为numpy格式，供GBRT模型预测
    generated_data_np = generated_data.detach().cpu().numpy()
    df = pd.read_csv('E:/PAPER_WRITE/paper/practice1/code/data/data_ACGAN.csv')

    features = df.iloc[:, :14].values.astype(np.float32)
    #labels = df.iloc[:, 13].values.astype(np.int64)
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    generated_features = (generated_data_np + 1) / 2 * (max_val - min_val) + min_val
    #print('generated_features', generated_features)
    generated_features_df = pd.DataFrame(generated_features, columns=['Processing_number','Ag','Al','Cr','Cu','Fe',
                                                                      'Mg','Mn','Ni','Sc','Si', 'Ti','Zn','Zr'])
    generated_features_df_drop_Processing_number = generated_features_df.drop(columns=['Processing_number'], axis=1)
    print(generated_features_df_drop_Processing_number.columns)

    def combine_columns(row):
        # 遍历每一列和对应的值
        pairs = []
        for col, value in zip(generated_features_df.columns, row):
            if value != 0:  # 如果值为0，剔除该列
                pairs.append(f"{col}{value}")  # 将列名和值直接连接，不使用冒号和逗号
        return ' '.join(pairs)  # 用空格连接所有非0的列

    # 对每一行应用该函数
    result = generated_features_df.apply(combine_columns, axis=1)
    generated_features_df['Composition'] = result

    # 定义一个函数，将科学计数法转换为普通十进制形式
    def convert_scientific_to_decimal(text):
        # 使用正则表达式找到所有科学计数法的模式
        pattern = r'([a-zA-Z]+)(\d+\.\d+e[-+]\d+|\d+e[-+]\d+)'
        matches = re.findall(pattern, text)

        # 替换科学计数法为普通十进制形式
        for match in matches:
            element, num = match
            # 转换为十进制，并保留足够的小数位数
            decimal_num = "{:.10f}".format(float(num)).rstrip('0').rstrip('.')
            text = text.replace(f"{element}{num}", f"{element}{decimal_num}")

        return text

    # 应用转换函数
    generated_features_df['Composition'] = generated_features_df['Composition'].apply(convert_scientific_to_decimal)
    generated_features_df.to_excel('E:/PAPER_WRITE/paper/practice1/code/data/shiyan.xlsx', index=False)
    '''
    做出所有特征描述符的数据
    1.通用特征
    '''
    properties_df = pd.DataFrame()
    properties_list = ['Vec','X','atomic_radius','atomic_radius','electron_affinity','group','min_oxidation_state',
                       'mendeleev_no','atomic_radius_calculated', 'melting_point']
    for prop in properties_list:
        vec = R_Analyzer('E:/PAPER_WRITE/paper/practice1/code/data/shiyan.xlsx',
                         'E:/python/ACGAN_HEAS/properties/elements_properties.xlsx',
                         'Composition', prop)
        result = vec.calculate_properties()
        properties_df[f'{prop}_mean'] = result[0]
        properties_df[f'{prop}_std'] = result[1]
        properties_df[f'{prop}_std/mean*100'] = result[2]
    '''
    2.
    混合熵
    '''
    entropy = EntropyOfMixingCalculator('E:/PAPER_WRITE/paper/practice1/code/data/shiyan.xlsx', 'Composition')
    entropy_result = entropy.calculate_entropy_of_mixing()
    properties_df['entropy'] = entropy_result

    '''
    3.
    混合焓
    '''
    enthalpy = EnthalpyOfMixingCalculator('E:/PAPER_WRITE/paper/practice1/code/data/shiyan.xlsx', 'Composition',
                                          'E:/python/ACGAN_HEAS/properties/混合焓对.xlsx')
    enthalpy_result = enthalpy.calculate_enthalpy_of_mixing()
    properties_df['enthalpy'] = enthalpy_result
    new_samples_properties_GBRT_df = properties_df[['Vec_std', 'X_mean', 'X_std/mean*100', 'atomic_radius_mean',
                                                   'atomic_radius_std/mean*100', 'electron_affinity_std/mean*100',
                                                   'group_std/mean*100',
                                                   'min_oxidation_state_std/mean*100', 'mendeleev_no_mean',
                                                   'atomic_radius_calculated_std']
    ]
    new_samples_properties_GBRT_df.loc[:, ['Processing_number', 'Cu']] = pd.read_excel('E:/PAPER_WRITE/paper/practice1/code/data/shiyan.xlsx')[['Processing_number', 'Cu']]
    oumiga = properties_df['melting_point_mean'] * properties_df['entropy'] / abs(properties_df['enthalpy'])
    new_samples_properties_GBRT_df.loc[:, ['oumiga']] = oumiga
    new_samples_properties_GBRT_df.loc[:, 'Processing_number'] = new_samples_properties_GBRT_df['Processing_number'].round().astype(int)
    print('new_samples_properties_GBRT_df=', new_samples_properties_GBRT_df)
    density = get_density('E:/PAPER_WRITE/paper/practice1/code/data/shiyan.xlsx', 'Composition',
                          'E:/python/ACGAN_HEAS/properties/elements_properties.xlsx')
    fake_density = density.calculate_all_densities()['density']
    #print('fake_density====', fake_density.values)
    # 使用GBRT模型预测y值
    y = gbrt_model.predict(new_samples_properties_GBRT_df)

    print('强度等于', y, '密度等于', fake_density.values, '比值=', y/fake_density.values)
    return y[0]/(fake_density.values), generated_features_df.values[0], fake_density.values[0], y[0]  # 返回预测值y[0]/(fake_density.values)

# 初始化生成器
generator = Generator(100, 13, 3)  # 假设Generator是生成器的类定义
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# 加载梯度增强决策树模型
best_GBRT_model = load('best_GBRT_model.joblib')

import numpy as np

# 初始温度
initial_temp = 200.0
# 冷却速率
cooling_rate = 0.95
# 迭代次数
max_iterations = 1000
# 最佳解记录
best_noise = None
best_label = None
best_y = -np.inf
# 初始解
'''
current_noise = np.random.randn(1, 100)  # 100维标准正态分布噪声
current_label = np.random.randint(0, 3)  # 0到8的整数标签
'''
current_noise = torch.randn(1, 100)
current_label = torch.randint(2, 3, (1,))
print('current_label', current_label)

current_y, Composition, Density, TS = objective_function(current_noise, current_label, generator, best_GBRT_model)
#print('current_y', current_y)
best_y = current_y
best_noise = current_noise.clone()
best_label = current_label.clone()

temperature = initial_temp
temperature1 = initial_temp

y_list = []
composition_list = []
Density_list = []
TS_list = []
current_y_list = []
candidate_y_list = []
acceptance_prob_list = []
temperature1_list = []
iteration_list = []
# **********************************************************
import numpy as np

accept_list = []
k_list = []
for i in range(max_iterations):
    k_list.append(i)
    temperature1 = initial_temp / (np.log(i + 10)/np.log(10))
    temperature1_list.append(temperature1)
d_temperature1 = temperature1_list[0] - temperature1_list[-1]

suofang1 = [(x-temperature1_list[-1])/d_temperature1*100+0.01 for x in temperature1_list]

accept = [np.exp(-1/x) for x in suofang1]

for iteration in range(max_iterations):
    print(f'第{iteration}次循环')
    iteration_list.append(iteration)
    # 生成候选解
    # 噪声扰动：添加高斯噪声
    noise_perturbation = torch.randn(current_noise.shape).mul(0.5)
    #print('noise_perturbation')
    candidate_noise = current_noise + noise_perturbation
    # 标签扰动：有一定概率改变标签
    if np.random.rand() < 0.001:  # 10%的概率改变标签
        candidate_label = current_label#torch.randint(0, 3, (1,))
    else:
        candidate_label = current_label
    #print(candidate_noise.dtype, candidate_label.dtype)
    # 计算候选解的目标函数值
    candidate_y, Composition, Density, TS = objective_function(candidate_noise, candidate_label, generator,
                                     best_GBRT_model)
    print('candidate_y=', candidate_y[0], 'current_y=', current_y[0])
    current_y_list.append(current_y[0])
    # 计算差值
    delta = 1*(candidate_y - current_y)
    #print('current_y=', current_y)
    '''
    增加条件判断语句，如果生成的样本符合要求，就输出这个样本。
    '''
    if current_y[0] > 0:
        y_list.append(candidate_y[0])
        composition_list.append(Composition)
        Density_list.append(Density)
        TS_list.append(TS)
    # 如果候选解更好，接受它
    if delta > 0:
        current_noise = candidate_noise
        current_label = candidate_label
        current_y = candidate_y

        acceptance_prob = np.exp(delta / suofang1[iteration])
        acceptance_prob_list.append(acceptance_prob[0])

        if candidate_y > best_y:
            best_y = candidate_y
            best_noise = candidate_noise.clone()
            best_label = candidate_label
    # 否则，以概率接受
    else:
        acceptance_prob = np.exp(delta / suofang1[iteration])
        acceptance_prob_list.append(acceptance_prob[0])
        print('接受率=', acceptance_prob)
        if np.random.rand() < acceptance_prob:

            current_noise = candidate_noise
            current_label = candidate_label
            current_y = candidate_y

    # 降低温度
    temperature *= cooling_rate
    print('temperature = ', temperature)
    temperature1 = initial_temp / (np.log(iteration + 10)/np.log(10))

    print('temperature1 = ', temperature1)

    # 输出当前状态
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Current y: {current_y}, Best y: {best_y}")
    candidate_y_list.append(candidate_y[0])

# 最佳噪声和标签
best_sample = generator(best_noise, best_label)
print(f"Best 比值: {best_y}")

all_df = pd.DataFrame()
all_df['T/D'] = y_list
all_df['Composition'] = composition_list
all_df['Density'] = Density_list
all_df['TS'] = TS_list
all_df['current_y'] = current_y_list
all_df['candidate_y'] = candidate_y_list
print(temperature1_list)
all_df['temperature1'] = temperature1_list
all_df['acceptance_prob'] = acceptance_prob_list
all_df['iteration'] = iteration_list


all_df.to_csv(f'E:/PAPER_WRITE/paper/practice1/code/data/all_df_T.csv')

