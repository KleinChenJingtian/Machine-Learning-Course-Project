import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS','PingFang SC', 'SimHei','Songti']  # 尝试多种字体，确保至少有一种可用
plt.rcParams['axes.unicode_minus'] = False  # 确保负号也能正确显示

# 读取数据
file_path = 'cleaned_数据集2.xlsx'
df = pd.read_excel(file_path)

# 一、数据基本情况
# 1. 样本数量
sample_count = len(df)
print(f"数据集中共有 {sample_count} 条新能源二手车记录。")

# 2. 数据完整性检查（简单示例，这里只检查是否有缺失值）
missing_values = df.isnull().sum()
print("各字段缺失值数量：")
print(missing_values)


# 二、保值率的描述性统计
# 1. 中心趋势度量
print("\n保值率的中心趋势度量：")
print(f"均值：{df['保值率'].mean()}")
print(f"中位数：{df['保值率'].median()}")
mode = df['保值率'].mode()
if not mode.empty:
    print(f"众数：{mode[0]}")
else:
    print("无众数（可能保值率值较分散）")

# 2. 离散程度度量
print("\n保值率的离散程度度量：")
print(f"极差：{df['保值率'].max() - df['保值率'].min()}")
print(f"标准差：{df['保值率'].std()}")
print(f"变异系数：{df['保值率'].std() / df['保值率'].mean()}")


# 描述性统计分析
def descriptive_statistics(df):
    # 基本信息
    print("数据集形状：", df.shape)
    print("数据类型：\n", df.dtypes)

    # 空值检查
    print("\n空值统计：\n", df.isnull().sum())

    # 删除包含NaN值的行
    df.dropna(inplace=True)

    # 数值型字段描述性统计:
    print("\n数值型字段描述性统计：\n")
    print("\n里程数：\n", df[['mileAge']].describe(include=[float, int]))
    print("\n售价：\n", df[['price']].describe(include=[float, int]))
    print("\n新车价格：\n", df[['newCarPrice']].describe(include=[float, int]))
    print("\n排量：\n", df[['emission']].describe(include=[float, int]))
    print("\n油耗：\n", df[['fuelConsumption']].describe(include=[float, int]))
    print("\n折价额：\n", df[['valueLoss']].describe(include=[float, int]))
    print("\n保值率：\n", df[['resaleValueRate']].describe(include=[float, int]))
    print("\n折旧比：\n", df[['折旧比']].describe(include=[float, int]))
    print("\n使用时长(h)：\n", df[['使用时长']].describe(include=[float, int]))
    print("\n每年平均折旧额：\n", df[['每年平均折旧额']].describe(include=[float, int]))
    print("\n里程使用强度：\n", df[['里程使用强度']].describe(include=[float, int]))
    print("\n保值率变化率：\n", df[['保值率变化率']].describe(include=[float, int]))

    # 保值率计算
    df['Resale_Value_Rate'] = df['price'] / df['newCarPrice']
    print("\n保值率描述性统计：\n", df[['price', 'newCarPrice', 'resaleValueRate']].describe(include=[float, int]))

# 图表展示
def plot_graphs(df):
    # 价格分布直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=20, kde=True, color='#7fcdbb', alpha=0.6)
    plt.title('价格分布直方图', fontsize=16)
    plt.xlabel('价格（万元）', fontsize=14)
    plt.ylabel('密度', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
#
    # 里程与价格的散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mileAge', y='price', color='#b3e2cd', alpha=0.7)
    plt.title('里程与价格的散点图', fontsize=16)
    plt.xlabel('里程 (万公里)', fontsize=14)
    plt.ylabel('价格 (万元)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
#
    # 各品牌车辆价格的条形图
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='brand', y='price', palette='pastel')
    plt.title('各品牌车辆价格的条形图', fontsize=16)
    plt.xlabel('品牌', fontsize=14)
    plt.ylabel('价格 (万元)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
#
    # 各品牌车辆数量的条形图
    plt.figure(figsize=(12, 6))
    # 对DataFrame进行分组并计数，得到每个品牌的车辆数量
    brand_counts = df['brand'].value_counts()
    # 绘制条形图，x轴为品牌，y轴为数量
    sns.barplot(x=brand_counts.index, y=brand_counts.values, palette='pastel')
    plt.title('各品牌车辆数量的条形图', fontsize=16)
    plt.xlabel('品牌', fontsize=14)
    plt.ylabel('数量', fontsize=14)  # 这里应该是数量，而不是价格
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
#
    # 车辆类型的饼图
    plt.figure(figsize=(8, 8))
    plt.pie(df['bodyType'].value_counts(), labels=df['bodyType'].value_counts().index, autopct='%1.1f%%',
            startangle=140, colors=sns.color_palette("pastel"))
    plt.title('车辆类型的饼图', fontsize=16)
    plt.tight_layout()
    plt.show()
#
    # 保值率直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Resale_Value_Rate'], bins=20, kde=True, color='#fde74c', alpha=0.6)
    plt.title('保值率直方图', fontsize=16)
    plt.xlabel('保值率', fontsize=14)
    plt.ylabel('密度', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # 保值率箱型图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y='Resale_Value_Rate', color='#e41a1c')
    plt.title('保值率箱型图', fontsize=16)
    plt.ylabel('保值率', fontsize=14)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()
#
#     # 里程与保值率的线图
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, x='mileAge', y='Resale_Value_Rate', color='#6baed6', alpha=0.8)
#     plt.title('里程与保值率的线图', fontsize=16)
#     plt.xlabel('里程 (万公里)', fontsize=14)
#     plt.ylabel('保值率', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.show()
#
#     # 价格与保值率的面积图
#     plt.figure(figsize=(10, 6))
#     plt.fill_between(df['price'], df['Resale_Value_Rate'], alpha=0.5, color='#3182bd')
#     plt.title('价格与保值率的面积图', fontsize=16)
#     plt.xlabel('价格', fontsize=14)
#     plt.ylabel('保值率', fontsize=14)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.show()
#
#     # 价格与保值率的散点图
#     plt.figure(figsize=(10, 6))
#     plt.scatter(df['price'], df['Resale_Value_Rate'], alpha=0.5, label='数据点')
#     # 计算回归线
#     slope, intercept, r_value, p_value, std_err = linregress(df['price'], df['Resale_Value_Rate'])
#     line = slope * df['price'] + intercept
#     plt.plot(df['price'], line, 'r', label=f'回归线 $y={slope:.2f}x+{intercept:.2f}$')
#     # 添加标题和坐标轴标签
#     plt.title('价格与保值率的散点图及回归线', fontsize=18)
#     plt.xlabel('价格（万元）', fontsize=16)
#     plt.ylabel('保值率', fontsize=16)
#     # 添加图例
#     plt.legend()
#     # 显示网格
#     plt.grid(True)
#     # 调整布局
#     plt.tight_layout()
#     # 显示图表
#     plt.show()
#
# # 运行描述性统计分析和图表展示
# descriptive_statistics(df)
# plot_graphs(df)
#
# def export_to_excel(df, filename):
#     df.to_excel(filename, index=False)
#
# # 导出描述性统计结果到Excel文件
# descriptive_statistics(df)
# export_to_excel(df[['price', 'newCarPrice', 'Resale_Value_Rate']].describe(include=[float, int]), '描述性统计结果.xlsx')

# ——————————————————————————————————————————————————————————————————
# # 折线图
# plt.figure(figsize=(10, 6))
# df['brand'].value_counts().plot(kind='line', marker='o')
# plt.title('品牌折线图', fontsize=16)
# plt.xlabel('品牌', fontsize=14)
# plt.ylabel('数量', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.tight_layout()
# plt.show()

# # 时间分组分析
# def time_group_analysis(df):
#     df['regDate'] = pd.to_datetime(df['regDate'])
#     df['year'] = df['regDate'].dt.year
#     average_resale_value_rate_by_year = df.groupby('year')['resaleValueRate'].mean()
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=average_resale_value_rate_by_year.reset_index(), x='year', y='resaleValueRate', marker='o', color='#FF9688')
#     plt.title('平均保值率随年份变化', fontsize=16, fontweight='bold')
#     plt.xlabel('年份', fontsize=14, fontweight='bold')
#     plt.ylabel('平均保值率', fontsize=14, fontweight='bold')
#     plt.xticks(fontsize=12, fontweight='bold')
#     plt.yticks(fontsize=12, fontweight='bold')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     # 计算上牌时间与保值率之间的相关系数
#     correlation = average_resale_value_rate_by_year.corr(df['resaleValueRate'])
#     print(f"上牌时间与保值率之间的相关系数：{correlation:.2f}")
#
# time_group_analysis(df)


#  （一）能源类型与保值率
def energy_type_resale_value_rate(df):
    fuel_type_resale_value_rate = df.groupby('fuelType')['resaleValueRate'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=fuel_type_resale_value_rate.reset_index(), x='fuelType', y='resaleValueRate', palette='viridis')
    plt.title('能源类型与平均保值率', fontsize=16, fontweight='bold')
    plt.xlabel('能源类型', fontsize=14, fontweight='bold')
    plt.ylabel('平均保值率', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_energy_type_resale_value(df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='fuelType', y='resaleValueRate', palette='viridis')
    plt.title('不同能源类型的平均保值率')
    plt.xlabel('能源类型')
    plt.ylabel('平均保值率')
    plt.xticks(rotation=45)
    plt.show()
plot_energy_type_resale_value(df)

def plot_energy_type_boxplot(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='fuelType', y='resaleValueRate', palette='coolwarm')
    plt.title('不同能源类型的保值率分布')
    plt.xlabel('能源类型')
    plt.ylabel('保值率')
    plt.xticks(rotation=45)
    plt.show()
plot_energy_type_boxplot(df)

def plot_energy_type_violinplot(df):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='fuelType', y='resaleValueRate', palette='muted')
    plt.title('不同能源类型的保值率密度分布')
    plt.xlabel('能源类型')
    plt.ylabel('保值率')
    plt.xticks(rotation=45)
    plt.show()
plot_energy_type_violinplot(df)

def plot_energy_type_scatterplot(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='mileAge', y='resaleValueRate', hue='fuelType', palette='Dark2')
    plt.title('里程与保值率的关系（按能源类型）')
    plt.xlabel('车辆里程（万公里）')
    plt.ylabel('保值率')
    plt.legend(title='能源类型')
    plt.show()
plot_energy_type_scatterplot(df)

def plot_energy_type_radarplot(df):
    categories = list(df['fuelType'].unique())
    values = df.groupby('fuelType')['resaleValueRate'].mean().values
    values = np.concatenate((values,[values[0]]))  # 闭合雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    plt.title('不同能源类型的保值率雷达图')
    plt.show()
plot_energy_type_radarplot(df)

def plot_energy_type_heatmap(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('能源类型与保值率的相关性热力图')
    plt.show()
plot_energy_type_heatmap(df)

