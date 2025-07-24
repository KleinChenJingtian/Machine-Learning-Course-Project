import pandas as pd

# 读取 Excel 文件
excel_file = pd.ExcelFile('D:/junior_year_1st/数据分析/项目/数据采集/车300重采-2320条.xlsx')

# 读取每个表名的数据
df = excel_file.parse('Sheet1')

# 对能源类型为电力的油耗填充为 0
df.loc[df['能源类型'] == '电力', '油耗'] = 0
print("规则 1：将能源类型为电力的油耗填充为 0，影响行数：", len(df[df['能源类型'] == '电力']))

# 对能源类型为空值，但油耗不为空值的数据，将能源类型填充为油电
df.loc[(df['能源类型'].isnull() & df['油耗'].notnull()), '能源类型'] = '油电'
print("规则 2：将能源类型为空但油耗不为空的数据填充为油电，影响行数：", len(df[(df['能源类型'].isnull() & df['油耗'].notnull())]))

# 筛选出列缺失值数量大于等于 4 的数据的索引
delete_row = df.isnull().sum(axis=1) >= 4
# 删除数据
df = df[~delete_row]
print("规则 3：删除缺失字段大于等于 4 的数据，删除行数：", len(df[delete_row]))

# 处理能源类型与油耗都为空的数据
null_energy_and_fuel = df[(df['能源类型'].isnull()) & (df['油耗'].isnull())]
for index, row in null_energy_and_fuel.iterrows():
    brand = row['品牌']
    series = row['系列']
    # 在原数据集中查找品牌和系列相同且能源类型和油耗不为空的数据
    filled_data = df[((df['品牌'] == brand) & (df['系列'] == series)) & (df['能源类型'].notnull() & df['油耗'].notnull())]
    if not filled_data.empty:
        df.at[index, '能源类型'] = filled_data['能源类型'].iloc[0]
        df.at[index, '油耗'] = filled_data['油耗'].iloc[0]
print("规则 4：填充能源类型与油耗都为空的数据，填充行数：", len(null_energy_and_fuel))

# 将其保存为 xlsx 文件
df.to_excel('D:/junior_year_1st/数据分析/项目/数据采集/车300重采-2320条_清洗后.xlsx', index=False)

