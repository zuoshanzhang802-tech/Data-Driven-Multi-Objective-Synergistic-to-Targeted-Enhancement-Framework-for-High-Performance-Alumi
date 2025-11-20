import re
import pandas as pd
import math

# 这个可以是任何原子属性描述符 (composition_file_path, vec_file_path, composition_column_name, radius_column_name)


class R_Analyzer:
    def __init__(self, composition_file_path, vec_file_path, composition_column_name, radius_column_name):
        self.composition_file_path = composition_file_path
        self.vec_file_path = vec_file_path
        self.composition_column_name = composition_column_name
        self.radius_column_name = radius_column_name
        self.data = pd.read_excel(self.composition_file_path)
        self.Composition = self.data[self.composition_column_name]
        self.data_han = pd.read_excel(self.vec_file_path)
        self.vec_dict = self.data_han.set_index('Element')[self.radius_column_name].to_dict()

    def parse_composition(self, formula):
        """
        解析化学式，提取元素及其摩尔分数
        """
        # 使用正则表达式匹配元素和其后的数字（如果有）
        pattern = r'([A-Z][a-z]*)(?:(\d+(?:\.\d+)?))'
        matches = re.findall(pattern, formula)

        # 如果没有找到匹配项，尝试匹配没有数字的元素
        if not matches:
            matches = re.findall(r'([A-Z][a-z]*)', formula)

        # 初始化元素及其数量列表
        elements = []
        counts = []

        # 处理匹配结果
        for match in matches:
            if len(match) == 2:
                element, count = match
                elements.append(element)
                counts.append(float(count) if count else 1)
            else:
                elements.append(match[0])
                counts.append(1)

        # 计算摩尔分数
        total_count = sum(counts)
        mole_fractions = {element: count / total_count for element, count in zip(elements, counts)}
        return mole_fractions

    def calculate_properties(self):
        result_list = []
        result_2_list = []

        for composition in self.Composition:
            mole_fractions = self.parse_composition(composition)
            key_list = list(mole_fractions.keys())
            weigh_ele = []
            property_ele = []
            for element in key_list:
                property_ele.append(self.vec_dict.get(element, 0))  # 使用get方法避免KeyError
            for element in key_list:
                weigh_ele.append(mole_fractions[element])
            result = sum(a * b for a, b in zip(property_ele, weigh_ele))
            add_list = [x - result for x in property_ele]
            squared_list = [x**2 for x in add_list]
            result_2 = math.sqrt(sum(a * b for a, b in zip(weigh_ele, squared_list)))
            result_list.append(result)
            result_2_list.append(result_2)
            Atomic_size_difference = [a / b for a, b in zip(result_2_list, result_list)]
            results_paper = [a * 100 for a in Atomic_size_difference]

        return result_list, result_2_list, results_paper


# 混合熵
class EntropyOfMixingCalculator:
    """
    计算混合熵的类
    输入：Excel文件路径和化学式所在的列名
    输出：混合熵的计算结果
    """

    def __init__(self, file_path, column_name):
        """
        初始化类
        :param file_path: Excel文件路径
        :param column_name: 化学式所在的列名
        """
        self.file_path = file_path
        self.column_name = column_name
        self.data = self._load_data()

    def _load_data(self):
        """
        加载Excel文件中的数据
        """
        try:
            data = pd.read_excel(self.file_path)
            return data
        except Exception as e:
            raise FileNotFoundError(f"无法加载文件 {self.file_path}: {e}")

    def parse_composition(self, formula):
        """
        解析化学式，提取元素及其摩尔分数
        """
        # 使用正则表达式匹配元素和其后的数字（如果有）
        pattern = r'([A-Z][a-z]*)(?:(\d+(?:\.\d+)?))'
        matches = re.findall(pattern, formula)

        # 如果没有找到匹配项，尝试匹配没有数字的元素
        if not matches:
            matches = re.findall(r'([A-Z][a-z]*)', formula)

        # 初始化元素及其数量列表
        elements = []
        counts = []

        # 处理匹配结果
        for match in matches:
            if len(match) == 2:
                element, count = match
                elements.append(element)
                counts.append(float(count) if count else 1)
            else:
                elements.append(match[0])
                counts.append(1)

        # 计算摩尔分数
        total_count = sum(counts)
        mole_fractions = {element: count / total_count for element, count in zip(elements, counts)}
        print('摩尔分数=', mole_fractions)
        return mole_fractions

    def calculate_entropy_of_mixing(self):
        """
        计算混合熵
        """
        if self.column_name not in self.data.columns:
            raise ValueError(f"列名 '{self.column_name}' 不存在于数据中")

        # 获取化学式列
        compositions = self.data[self.column_name]

        # 初始化结果列表
        results = []

        # 遍历化学式并计算混合熵
        for composition in compositions:
            mole_fractions = self.parse_composition(composition)

            # 提取摩尔分数
            percent_list = list(mole_fractions.values())

            # 计算对数值
            log_percent_list = [math.log(x) for x in percent_list]

            # 计算混合熵
            R = 8.314  # 单位是 J/(mol·K)
            entropy = -R * sum(a * b for a, b in zip(percent_list, log_percent_list))
            results.append(entropy)

        return results


# 混合焓，需要的是混合焓对.xlsx
class EnthalpyOfMixingCalculator:
    def __init__(self, main_file_path, composition_column, enthalpy_file_path):
        self.main_file_path = main_file_path
        self.composition_column = composition_column
        self.enthalpy_file_path = enthalpy_file_path
        self.main_data = pd.read_excel(main_file_path)
        self.por_values = pd.read_excel(enthalpy_file_path)
        self.por_values = self.por_values.set_index(self.por_values.columns[0])  # 设置行索引

    def parse_composition(self, formula):
        pattern = r'([A-Z][a-z]*)(?:(\d+(?:\.\d+)?))'
        matches = re.findall(pattern, formula)
        if not matches:
            matches = re.findall(r'([A-Z][a-z]*)', formula)

        elements = []
        counts = []
        for match in matches:
            if len(match) == 2:
                element, count = match
                elements.append(element)
                counts.append(float(count) if count else 1)
            else:
                elements.append(match[0])
                counts.append(1)

        total_count = sum(counts)
        mole_fractions = {element: count / total_count for element, count in zip(elements, counts)}
        return mole_fractions

    def calculate_enthalpy_of_mixing(self):
        composition_column = self.composition_column
        por_values = self.por_values
        data = self.main_data

        results = []
        for i in range(len(data)):
            formula = data[composition_column].iloc[i]
            Percent = self.parse_composition(formula)
            Percent_values_list = list(Percent.values())
            Percent_keys_list = list(Percent.keys())

            han_list = []
            for i in range(len(Percent) - 1):
                a = Percent_values_list[i]
                for j in range(i + 1, len(Percent)):
                    b = Percent_values_list[j]
                    hang = Percent_keys_list[i]
                    lie = Percent_keys_list[j]
                    han = por_values.loc[hang, lie]
                    han_list.append(a * b * han)

            result = 4 * sum(han_list)
            results.append(result)

        return results              ##


# 解析化学式，输出的是一个dataframe格式
class CompositionParser:
    def __init__(self, excel_file, formula_column):
        """
        初始化CompositionParser类
        :param excel_file: Excel文件路径
        :param formula_column: 化学式所在的列名
        """
        self.excel_file = excel_file
        self.formula_column = formula_column

    def parse_composition(self, formula_alloys):
        """
        解析化学式，返回元素及其摩尔分数
        """
        pattern = r'([A-Z][a-z]*)(?:(\d+(?:\.\d+)?))'
        matches = re.findall(pattern, formula_alloys)
        if not matches:
            matches = re.findall(r'([A-Z][a-z]*)', formula_alloys)

        elements = []
        counts = []
        for match in matches:
            if len(match) == 2:
                element, count = match
                elements.append(element)
                counts.append(float(count) if count else 1)
            else:
                elements.append(match[0])
                counts.append(1)

        total_count = sum(counts)
        mole_fractions = {element: count / total_count for element, count in zip(elements, counts)}
        return mole_fractions

    def calculate_jiexi(self):
        """
        从Excel文件中读取化学式，并返回解析后的数据
        """
        # 读取Excel文件
        data = pd.read_excel(self.excel_file)

        # 提取化学式列
        formulas = data[self.formula_column].tolist()

        # 解析每个化学式
        data_jiexi = []
        for formula in formulas:
            mole_fracs = self.parse_composition(formula)
            data_jiexi.append(mole_fracs)

        # 转换为DataFrame，并用0填充缺失值
        df = pd.DataFrame(data_jiexi).fillna(0)

        return df


import pandas as pd
import numpy as np
import re


class get_density:
    def __init__(self, main_file_path, composition_column, elements_file_path):
        # 读取主文件和元素文件
        self.main_df = pd.read_excel(main_file_path, dtype={composition_column: str})
        self.elements_df = pd.read_excel(elements_file_path,
                                         dtype={'Element': str, 'atomic_mass': float, 'density_of_solid': float})

        # 保存列名
        self.composition_column = composition_column

        # 创建元素符号到摩尔质量和密度的映射
        self.element_map = self.elements_df.set_index('Element').to_dict(orient='index')

    def parse_composition(self, formula):
        # 解析化学式（如 "Mg0.029 Zr0.0014" 转换为 {'Mg': 0.029, 'Zr': 0.0014}）
        pattern = r'([A-Z][a-z]*)(\d+(?:\.\d+)?)'  # 匹配元素符号和小数或整数
        matches = re.findall(pattern, formula)

        # 如果没有匹配到小数部分，自动填充为1
        elements = {}
        for match in matches:
            element, count = match
            elements[element] = float(count)

        return elements

    def calculate_alloy_density(self, formula):
        # 计算合金密度
        elements = self.parse_composition(formula)
        #print('合金解析式：', elements)
        total_mass = 0.0
        total_volume = 0.0

        for element, mole_fraction in elements.items():
            if element not in self.element_map:
                raise ValueError(f"未在元素文件中找到元素 '{element}'")

            atomic_mass = self.element_map[element]['atomic_mass']
            density_solid = self.element_map[element]['density_of_solid']

            mass_contribution = mole_fraction * atomic_mass  # 质量贡献
            volume_contribution = mass_contribution / density_solid  # 体积贡献

            total_mass += mass_contribution
            total_volume += volume_contribution
            #print('质量=', total_mass)
            #print('体积等于=', total_volume)
        density = total_mass / total_volume  # 合金密度 (g/cm³)
        return density

    def calculate_all_densities(self):
        # 计算所有合金的密度并返回结果
        density_list = []
        for formula in self.main_df[self.composition_column]:
            try:
                density = self.calculate_alloy_density(formula)
            except Exception as e:
                density = np.nan  # 如果计算失败，用 NaN 表示
                print(f"计算 '{formula}' 密度时出错: {str(e)}")
            density_list.append(density)
        # 将结果添加到主 DataFrame
        self.main_df['density'] = density_list
        return self.main_df

'''
# 示例用法
if __name__ == "__main__":
    # 初始化类
    dens = EntropyOfMixingCalculator(
        file_path='E:/PAPER_WRITE/paper/practice1/code/data/shiyan.xlsx',
        column_name='Composition',

    )

    # 计算所有合金的密度
    result_df = dens.calculate_entropy_of_mixing()

    # 打印结果
    print(result_df)
'''
