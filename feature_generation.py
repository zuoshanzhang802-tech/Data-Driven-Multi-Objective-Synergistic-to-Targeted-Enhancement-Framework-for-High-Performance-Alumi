import numpy as np
import pandas as pd
from pymatgen.core import Element, Composition, Lattice, Structure, Molecule

'''
1.读取excel数据
'''
print('1.读取数据集——data_clean_junyun.xlsx')
data = pd.read_excel('E:/PAPER_WRITE/paper/practice1/code/data/data_clean_junyun.xlsx')
ele_name_list = ['Ag', 'Al', 'Cr', 'Cu', 'Fe', 'Li', 'Mg', 'Mn', 'Ni', 'Sc', 'Si', 'Ti', 'Zn', 'Zr']
'''
2.读取原子摩尔质量数据
'''
print('2.读取摩尔质量数据（来自pymatgen数据库）')
moar_mass_list = []
for element in ele_name_list:
    si = Element(element)
    moar_mass_list.append(si.atomic_mass*0+1) #********************日后更改
print('原子摩尔质量数据——', moar_mass_list)
'''
3.把质量比转化成摩尔比
'''
print('3.把质量占比转化成摩尔占比')

ele_wt = data[ele_name_list].values
moar_mass_np = np.array(moar_mass_list)
for i in range(ele_wt.shape[0]):
    ele_wt[i] *= moar_mass_np
    ele_wt[i] /= sum(ele_wt[i])
print('转化成摩尔占比后的数据为=', ele_wt)
'''
4.把摩尔占比数据转化成dataframe数据，并加上列名字
'''
print('4.把摩尔占比数据转化成dataframe数据，并加上列名字')
data_get_features = pd.DataFrame(data=ele_wt, columns=ele_name_list)

'5.获取特征'
print('5.获取特征')
import sys

sys.path.append("E:/python/ACGAN_HEAS")  # 替换为你的目标文件夹路径
from elements_properties import EntropyOfMixingCalculator
from elements_properties import EnthalpyOfMixingCalculator
from elements_properties import R_Analyzer
from elements_properties import get_density

'''做出所有特征描述符的数据'''
'''1.通用特征'''
properties_df = pd.DataFrame()
properties_list = ['Vec', 'X', 'Z', 'atomic_mass', 'atomic_radius', 'average_cationic_radius',
                   'electron_affinity', 'group', 'ionization_energy', 'max_oxidation_state', 'min_oxidation_state',
                   'row', 'mendeleev_no', 'electrical_resistivity', 'molar_volume', 'thermal_conductivity',
                   'boiling_point', 'melting_point', 'liquid_range', 'density_of_solid', 'atomic_radius_calculated',
                   '汽化热', '熔化热', '标准摩尔生成焓']
for prop in properties_list:
    vec = R_Analyzer('E:/PAPER_WRITE/paper/practice1/code/data/data_clean_junyun.xlsx',
                     'E:\python\ACGAN_HEAS\properties\elements_properties.xlsx',
                     'Composition', prop)
    result = vec.calculate_properties()
    properties_df[f'{prop}_mean'] = result[0]
    properties_df[f'{prop}_std'] = result[1]
    properties_df[f'{prop}_std/mean*100'] = result[2]

'''2.混合熵'''
entropy = EntropyOfMixingCalculator('E:/PAPER_WRITE/paper/practice1/code/data/data_clean_junyun.xlsx', 'Composition')
entropy_result = entropy.calculate_entropy_of_mixing()
properties_df['entropy'] = entropy_result

'''3.混合焓'''
enthalpy = EnthalpyOfMixingCalculator('E:/PAPER_WRITE/paper/practice1/code/data/data_clean_junyun.xlsx', 'Composition',
                                      'E:\python\ACGAN_HEAS\properties\混合焓对.xlsx')
enthalpy_result = enthalpy.calculate_enthalpy_of_mixing()
properties_df['enthalpy'] = enthalpy_result
'''4.密度'''
density = get_density('E:/PAPER_WRITE/paper/practice1/code/data/data_clean_junyun.xlsx', 'Composition',
                      'E:/python/ACGAN_HEAS/properties/elements_properties.xlsx')

print('6.把所有的特征描述符和原子摩尔比和工艺（独热编码）和目标值放在一起，组成一个dataframe')
properties_df[['Ag', 'Al', 'Cr', 'Cu', 'Fe', 'Li', 'Mg', 'Mn', 'Ni', 'Sc', 'Si', 'Ti', 'Zn', 'Zr']] = data_get_features[
    ['Ag', 'Al', 'Cr', 'Cu', 'Fe', 'Li', 'Mg', 'Mn', 'Ni', 'Sc', 'Si', 'Ti', 'Zn', 'Zr']]

data['Processing'] = data['Processing'].astype(str)
data_onehot = pd.get_dummies(data, columns=['Processing'], dtype=int)
properties_df['Tensile Strength (MPa)'] = data['Tensile Strength (MPa)']
properties_df['density'] = density.calculate_all_densities()['density']


# 使用factorize方法对文本进行编码
labels, unique_texts = pd.factorize(data['Processing'])

# 将编码结果添加到DataFrame中
properties_df['Processing_number'] = labels
properties_df.to_excel('E:/PAPER_WRITE/paper/practice1/code/data/properties_df.xlsx', index=False)

