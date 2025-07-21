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

# 删除包含 NaN 值的行
df = df.dropna()
# 替换 Inf 值为 NaN 并删除
df = df.replace([np.inf, -np.inf], np.nan).dropna()

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
    print("\n保值率描述性统计：\n", df[['price', 'newCarPrice', 'Resale_Value_Rate']].describe(include=[float, int]))

# 图表展示
def plot_graphs(df):
    # 价格分布直方图
    plt.figure(figsize=(12, 8))  # 增加图表尺寸
    sns.histplot(df['price'], bins=20, kde=True, color='#7fcdbb', alpha=0.6)
    plt.title('价格分布直方图', fontsize=18)  # 增大标题字体
    plt.xlabel('价格（万元）', fontsize=16)  # 增大坐标轴标题字体
    plt.ylabel('密度', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # 里程与价格的散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='mileAge', y='price', color='#b3e2cd', alpha=0.7)
    plt.title('里程与价格的散点图', fontsize=18)
    plt.xlabel('里程 (万公里)', fontsize=16)
    plt.ylabel('价格 (万元)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # 各品牌车辆价格的条形图
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df, x='brand', y='price', hue='brand', palette='pastel', legend=False)
    plt.title('各品牌车辆价格的条形图', fontsize=18)
    plt.xlabel('品牌', fontsize=16)
    plt.ylabel('价格 (万元)', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # 各品牌车辆数量的条形图
    plt.figure(figsize=(14, 8))
    brand_counts = df['brand'].value_counts()
    sns.barplot(x=brand_counts.index, y=brand_counts.values, hue=brand_counts.index, palette='pastel', legend=False)
    plt.title('各品牌车辆数量的条形图', fontsize=18)
    plt.xlabel('品牌', fontsize=16)
    plt.ylabel('数量', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # 车辆类型的饼图
    plt.figure(figsize=(8, 8))
    body_type_counts = df['bodyType'].value_counts()
    plt.pie(body_type_counts, labels=body_type_counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("pastel"))
    plt.title('车辆类型的饼图', fontsize=18)
    plt.tight_layout()
    plt.show()

    # 保值率直方图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Resale_Value_Rate'], bins=20, kde=True, color='#fde74c', alpha=0.6)
    plt.title('保值率直方图', fontsize=18)
    plt.xlabel('保值率', fontsize=16)
    plt.ylabel('密度', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # 保值率箱型图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y='Resale_Value_Rate', color='#e41a1c')
    plt.title('保值率箱型图', fontsize=18)
    plt.ylabel('保值率', fontsize=16)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # 里程与保值率的线图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='mileAge', y='Resale_Value_Rate', color='#6baed6', alpha=0.8)
    plt.title('里程与保值率的线图', fontsize=18)
    plt.xlabel('里程 (万公里)', fontsize=16)
    plt.ylabel('保值率', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

    # 价格与保值率的面积图
    plt.figure(figsize=(10, 6))
    plt.fill_between(df['price'], df['Resale_Value_Rate'], alpha=0.5, color='#3182bd')
    plt.title('价格与保值率的面积图', fontsize=18)
    plt.xlabel('价格', fontsize=16)
    plt.ylabel('保值率', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

# 运行描述性统计分析和图表展示
descriptive_statistics(df)
plot_graphs(df)

def export_to_excel(df, filename):
    df.to_excel(filename, index=False)


# 导出描述性统计结果到Excel文件
export_to_excel(df[['price', 'newCarPrice', 'Resale_Value_Rate']].describe(include=[float, int]), '描述性统计结果.xlsx')


# 价格与保值率的散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['price'], df['Resale_Value_Rate'], alpha=0.5, label='数据点')
# 计算回归线
slope, intercept, r_value, p_value, std_err = linregress(df['price'], df['Resale_Value_Rate'])
line = slope * df['price'] + intercept
plt.plot(df['price'], line, 'r', label=f'回归线 $y={slope:.2f}x+{intercept:.2f}$')
# 添加标题和坐标轴标签
plt.title('价格与保值率的散点图及回归线', fontsize=18)
plt.xlabel('价格（万元）', fontsize=16)
plt.ylabel('保值率', fontsize=16)
# 添加图例
plt.legend()
# 显示网格
plt.grid(True)
# 调整布局
plt.tight_layout()
# 显示图表
plt.show()
