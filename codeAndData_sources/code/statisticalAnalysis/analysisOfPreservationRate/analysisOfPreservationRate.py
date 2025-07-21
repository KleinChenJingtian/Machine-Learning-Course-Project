import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS','PingFang SC', 'SimHei','Songti']  # 尝试多种字体，确保至少有一种可用
plt.rcParams['axes.unicode_minus'] = False  # 确保负号也能正确显示

# 读取数据
file_path = '../cleaned_数据集2.xlsx'
df = pd.read_excel(file_path)

# 描述性统计分析
def descriptive_statistics(df):
    # 基本信息
    print("数据集形状：", df.shape)
    print("数据类型：\n", df.dtypes)

    # 空值检查
    print("\n空值统计：\n", df.isnull().sum())

    # 删除包含NaN值的行
    df.dropna(inplace=True)

# （一）能源类型与保值率
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

def plot_brand_resale_bubble(df):
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['mileAge'], df['Resale_Value_Rate'], s=df['mileAge']*10, c='Resale_Value_Rate', cmap='viridis', alpha=0.6, edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter, label='保值率')
    plt.title('品牌里程与保值率的气泡图')
    plt.xlabel('车辆里程（万公里）')
    plt.ylabel('保值率')
    plt.show()


# (二)品牌与保值率
def plot_top_brands_bar_chart(df, top_n=10):
    brand_resale_mean = df.groupby('brand')['resaleValueRate'].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=brand_resale_mean.index, y=brand_resale_mean.values, palette='viridis')
    plt.title(f'前{top_n}品牌平均保值率')
    plt.xlabel('品牌')
    plt.ylabel('平均保值率')
    plt.xticks(rotation=45)
    plt.show()
plot_top_brands_bar_chart(df)

def plot_brands_pie_chart(df, top_n=10):
    brand_resale_mean = df.groupby('brand')['resaleValueRate'].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(12, 8))
    plt.pie(brand_resale_mean, labels=brand_resale_mean.index, autopct='%1.1f%%%', startangle=140)
    plt.title(f'前{top_n}品牌保值率分布')
    plt.show()
plot_brands_pie_chart(df)


def plot_brands_stacked_bar_chart(df, top_n=10):
    brand_resale_mean = df.groupby('brand')['resaleValueRate'].mean().sort_values(ascending=False).head(top_n)
    brand_resale_mean_df = brand_resale_mean.reset_index()
    brand_resale_mean_df.columns = ['Brand', 'Average Resale Rate']
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Average Resale Rate', y='Brand', data=brand_resale_mean_df, palette='viridis')
    plt.title(f'前{top_n}品牌平均保值率（堆叠）')
    plt.xlabel('平均保值率')
    plt.ylabel('品牌')
    plt.yticks(rotation=0)
    plt.show()
plot_brands_stacked_bar_chart(df)

def plot_brands_dot_chart(df, top_n=10):
    brand_resale_mean = df.groupby('brand')['resaleValueRate'].mean().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(12, 8))
    sns.stripplot(x=brand_resale_mean.index, y=brand_resale_mean.values, jitter=True, palette='viridis')
    plt.title(f'前{top_n}品牌平均保值率')
    plt.xlabel('品牌')
    plt.ylabel('平均保值率')
    plt.xticks(rotation=45)
    plt.show()
plot_brands_dot_chart(df)

def print_brand_resale_table(df, top_n=10):
    brand_resale_mean = df.groupby('brand')['resaleValueRate'].mean().sort_values(ascending=False).head(top_n)
    print(brand_resale_mean.to_string())
print_brand_resale_table(df)


def plot_brand_resale_stacked_bar(df):
    # 计算每个品牌的平均保值率并排序
    brand_resale_mean = df.groupby('brand')['resaleValueRate'].mean().sort_values(ascending=False)
    # 创建一个新的DataFrame，用于绘制条形图
    brand_resale_mean_df = brand_resale_mean.reset_index()
    brand_resale_mean_df.columns = ['Brand', 'Average Resale Rate']
    # 绘制水平条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Average Resale Rate', y='Brand', data=brand_resale_mean_df, palette='Paired')
    # 设置标题和坐标轴标签
    plt.title('各品牌平均保值率', pad=20)  # 增加标题与图表的距离
    plt.xlabel('平均保值率', labelpad=15)  # 增加x轴标签与轴的距离
    plt.ylabel('品牌', labelpad=15)  # 增加y轴标签与轴的距离
    # 设置刻度标签的字体大小
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    # 旋转y轴标签，避免重叠
    plt.yticks(rotation=0)
    # 显示网格
    plt.grid(axis='x', linestyle='--', linewidth=0.7)
    # 调整布局
    plt.tight_layout()
    plt.show()
plot_brand_resale_stacked_bar(df)

def plot_brand_resale_scatterplot(df):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='mileAge', y='resaleValueRate', hue='brand', palette='tab10', alpha=0.6)
    plt.title('里程与保值率的关系（按品牌）')
    plt.xlabel('车辆里程（万公里）')
    plt.ylabel('保值率')
    plt.legend(title='品牌')
    plt.show()
plot_brand_resale_scatterplot(df)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'Songti']  # 尝试多种字体，确保至少有一种可用
plt.rcParams['axes.unicode_minus'] = False  # 确保负号也能正确显示

# 读取数据
file_path = '品牌与保值率.xlsx'
xls = pd.ExcelFile(file_path)

# 存储所有品牌的数据
all_brands_data = {}

# 遍历所有工作表
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    all_brands_data[sheet_name] = df

    # 选取分析字段
    df_selected = df[['brand', 'valueLoss', 'resaleValueRate', '折旧比', '保值率变化率']]

    # 品牌与保值率的关系分析
    print(f"{sheet_name} 品牌与保值率的关系分析：")
    brand_resale = df_selected.groupby('brand')['resaleValueRate'].mean().sort_values()
    print(brand_resale)

    # 品牌间的保值率差异
    print(f"{sheet_name} 品牌间的保值率差异：")
    brand_depreciation = df_selected.groupby('brand')['折旧比'].mean().sort_values()
    print(brand_depreciation)

    # 品牌与保值率的关联关系分析
    print(f"\n{sheet_name} 品牌与保值率的关联关系分析：")
    print("保值率与折旧比的相关性：", pearsonr(brand_resale.index, brand_resale.values)[0])
    print("保值率与保值率变化率的相关性：", pearsonr(brand_resale.index, df_selected.groupby('brand')['保值率变化率'].mean().values)[0])

    # 可视化品牌间的保值率差异
    plt.figure(figsize=(10, 6))
    sns.barplot(x=brand_resale.index, y=brand_resale.values)
    plt.title(f'{sheet_name} 品牌间的保值率差异')
    plt.xlabel('品牌')
    plt.ylabel('平均保值率')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=brand_depreciation.index, y=brand_depreciation.values)
    plt.title(f'{sheet_name} 品牌间的折旧比差异')
    plt.xlabel('品牌')
    plt.ylabel('平均折旧比')
    plt.xticks(rotation=45)
    plt.show()
