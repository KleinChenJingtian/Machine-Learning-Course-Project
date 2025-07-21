import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('数据集.xlsx')
print(df.info())
print(df.head())

# 1. 缺失值处理函数
def fill_missing_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    return df

def fill_missing_categorical(df, cols):
    for col in cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode())
    return df


# 2. 数据类型转换和缺失值处理整合函数
def convert_and_fill_missing(df, col):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isnull().sum() > 0:
        if df[col].isnull().sum() / len(df) < 0.05:
            df = df.dropna(subset=[col])
        else:
            # 根据具体列的逻辑进行估算填充
            if col == 'price':
                if 'newCarPrice' in df.columns and 'valueLoss' in df.columns:
                    df[col] = df[col].fillna(df['newCarPrice'] - df['valueLoss'])
            elif col == 'newCarPrice':
                if 'price' in df.columns and 'valueLoss' in df.columns:
                    df[col] = df[col].fillna(df['price'] + df['valueLoss'])
            elif col == 'valueLoss':
                if 'newCarPrice' in df.columns and 'price' in df.columns:
                    df[col] = df[col].fillna(df['newCarPrice'] - df['price'])
            elif col =='resaleValueRate':
                if 'price' in df.columns and 'newCarPrice' in df.columns:
                    df[col] = df[col].fillna((df['price'] / df['newCarPrice']) * 100)
    return df


# 数据清洗
# 1. 缺失值处理
# 检查缺失值
print("\n缺失值检查：")
print(df.isnull().sum())

numeric_cols = ['mileAge', 'price', 'newCarPrice', 'valueLoss','resaleValueRate']
df = fill_missing_numeric(df, numeric_cols)

categorical_cols = ['region', 'brand','series', 'bodyType','seat', 'fuelType', 'color', 'warranty']
df = fill_missing_categorical(df, categorical_cols)


# 2. 异常值检测
# 检查保值率和折价额是否有异常值
print("\n异常值检查：")
# 检查保值率是否大于1（100%）或折价额小于0
outliers = df[(df['resaleValueRate'] > 1) | (df['valueLoss'] < 0)]
print(outliers)

# 除了直接删除保值率超过100%的行，还可以考虑对异常值进行修正
# 对于保值率大于1的情况，如果确定是数据录入错误，根据同品牌同系列车型的平均保值率进行修正
# 先计算每个品牌 - 系列组合的平均保值率
brand_series_avg_holding_rate = df.groupby(['brand','series'], as_index=False)['resaleValueRate'].mean()
brand_series_avg_holding_rate_dict = brand_series_avg_holding_rate.to_dict()

for index, row in df.iterrows():
    brand = row['brand']
    series = row['series']
    if row['resaleValueRate'] > 1:
        if (brand, series) in brand_series_avg_holding_rate_dict:
            df.at[index,'resaleValueRate'] = brand_series_avg_holding_rate_dict[(brand, series)]

# 对于折价额小于0的情况，如果是少数异常值，可能是数据错误，可直接删除
df = df[df['valueLoss'] >= 0]

# 使用箱线图可视化检查数值型变量的异常值
for col in numeric_cols:
    plt.boxplot(df[col])
    plt.title(f'{col}')
    plt.show()

# 对于分类变量中的异常值检测，可以通过检查类别频率来判断
# 如果某个类别出现的频率极低，可能是异常值（根据业务逻辑判断）
for col in categorical_cols:
    value_counts = df[col].value_counts()
    top_counts = value_counts.nlargest(10)  # 获取出现频率最高的10个类别
    total_count = len(df)
    for value in value_counts.index:
        if value not in top_counts.index:
            ratio = value_counts[value] / total_count
            if ratio < 0.01:
                print(f"在列 {col} 中，类别 {value} 可能是异常值，其出现频率为 {ratio}")

# 3. 数据格式统一
# 确保日期格式一致
if'regDate' in df.columns:
    df['regDate'] = pd.to_datetime(df['regDate'], format='%Y/%m', errors='coerce')


# 对于日期列，检查日期范围是否合理，不能晚于当前日期（假设当前日期为2024 - 12 - 09）
current_date = pd.Timestamp('2024-12-09')
if'regDate' in df.columns:
    df = df[df['regDate'] <= current_date]

# 如果里程列的数据类型为其他类型（如字符串），可能需要转换为数值型
if df['mileAge'].dtype == 'object':
    df['mileAge'] = pd.to_numeric(df['mileAge'], errors='coerce')

# 对于保值率和折价额，如果数据类型不是数值型，转换为数值型
for col in ['resaleValueRate', 'valueLoss']:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col].replace('%', '', regex=True), errors='coerce')

# 检查车辆所属地列，如果有大写、小写混合或特殊字符的情况，可以统一格式
if 'region' in df.columns:
    df['region'] = df['region'].str.title()  # 将首字母大写

# 对于品牌列，如果有名称不规范的情况，可进行标准化
if 'brand' in df.columns:
    brand_mapping = {
        'Tesla': '特斯拉',
        'Honda': '本田'
    }
    df['brand'] = df['brand'].replace(brand_mapping)

# 4. 数据类型转换
cols = ['price', 'newCarPrice', 'valueLoss','resaleValueRate']
for col in cols:
    df = convert_and_fill_missing(df, col)


# 5. 去重
# 检查重复记录
print("\n重复记录检查：")
duplicated_count = df.duplicated().sum()
print(duplicated_count)

if duplicated_count > 0:
    # 查看重复的行内容以便分析（可选择）
    duplicated_rows = df[df.duplicated(keep=False)]
    print("重复的行示例：")
    print(duplicated_rows.head())

    # 删除重复记录
    df = df.drop_duplicates()

# 6. 特征工程
# 计算折旧比
if 'valueLoss' in df.columns and 'newCarPrice' in df.columns:
    df['折旧比'] = df['valueLoss'] / df['newCarPrice']
    # 处理新车价格为0的情况，避免除零错误
    df.loc[df['newCarPrice'] == 0, '折旧比'] = np.nan

# 计算车辆使用时长（基于上牌时间）并添加为新特征
if'regDate' in df.columns:
    current_date = pd.Timestamp('2024 - 12 - 09')
    df['使用时长'] = (current_date - df['regDate']).dt.days

# 对数值型特征进行标准化（这里使用Z - score标准化）
numeric_features = ['mileAge', 'price', 'newCarPrice', 'valueLoss','resaleValueRate', '折旧比']
for feature in numeric_features:
    if pd.api.types.is_numeric_dtype(df[feature]):
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / std

# 计算每年平均折旧额（假设使用时长以年为单位，这里简化计算为使用时长/365）
if 'valueLoss' in df.columns and '使用时长' in df.columns:
    df['每年平均折旧额'] = df['valueLoss'] / (df['使用时长'] / 365)
    # 处理使用时长为0的情况，避免除零错误
    df.loc[df['使用时长'] == 0, '每年平均折旧额'] = np.nan

# 计算里程与使用时长的比值，反映车辆使用的强度
if 'mileAge' in df.columns and '使用时长' in df.columns:
    df['里程使用强度'] = df['mileAge'] / df['使用时长']
    # 处理使用时长为0的情况，避免除零错误
    df.loc[df['使用时长'] == 0, '里程使用强度'] = np.nan

# 根据保值率和使用时长创建新特征，例如保值率随使用时长的变化率
if'resaleValueRate' in df.columns and '使用时长' in df.columns:
    # 计算保值率随使用时长的变化率，这里简单使用差分计算
    df['保值率变化率'] = df['resaleValueRate'].diff() / df['使用时长'].diff()
    # 处理第一个值为NaN的情况，可根据业务需求填充合适的值或者直接保留NaN
    df['保值率变化率'] = df['保值率变化率'].bfill()

# 7. 分类变量编码
# 对分类变量（如品牌、车型、地区等）进行编码
df['brand'] = df['brand'].astype('category').cat.codes
df['series'] = df['series'].astype('category').cat.codes

# 使用独热编码（One - Hot Encoding）对车辆所属地进行编码
if 'region' in df.columns:
    vehicle_location_dummies = pd.get_dummies(df['region'], prefix='车辆所属地')
    df = pd.concat([df, vehicle_location_dummies], axis=1)
    df = df.drop('region', axis=1)


# 查看清洗后的数据基本信息
print("\n清洗后数据基本信息：")
print(df.info())
print("\n清洗后数据头部查看：")
print(df.head())


# 导出清洗后的数据集
output_file_path = 'cleaned_数据集.xlsx'
df.to_excel(output_file_path, index=False)
print(f"\n清洗后的数据已导出到：{output_file_path}")