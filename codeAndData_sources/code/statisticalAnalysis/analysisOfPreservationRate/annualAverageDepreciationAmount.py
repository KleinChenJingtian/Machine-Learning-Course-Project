import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, f_oneway

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'SimHei', 'Songti']  # 尝试多种字体，确保至少有一种可用
plt.rcParams['axes.unicode_minus'] = False  # 确保负号也能正确显示

# 读取数据
file_path = '../cleaned_数据集2.xlsx'
df = pd.read_excel(file_path)

colors = sns.color_palette('husl', 15)  # 生成多种颜色

# 数据预处理
# 确保所有列的数据类型正确
df['regDate'] = pd.to_datetime(df['regDate'])
df['mileAge'] = df['mileAge'].astype(float)
df['newCarPrice'] = df['newCarPrice'].astype(float)
df['valueLoss'] = df['valueLoss'].astype(float)
df['resaleValueRate'] = df['resaleValueRate'].astype(float)
df['折旧比'] = df['折旧比'].astype(float)
df['使用时长'] = df['使用时长'].astype(float)
df['每年平均折旧额'] = df['每年平均折旧额'].astype(float)



# 品牌与年平均折旧额的关系
# 仅取排名前10的品牌
top_brands = df['brand'].value_counts().head(10).index  # 取前10名品牌
brand_depreciation = df[df['brand'].isin(top_brands)][['brand', '每年平均折旧额']]
# 设置图表大小
plt.figure(figsize=(12, 8))
# 绘制箱线图
ax = sns.boxplot(x='brand', y='每年平均折旧额', data=brand_depreciation, palette=colors)
# 设置标题和坐标轴标签
plt.title('主要品牌的平均年折旧额对比')
plt.xlabel('品牌')
plt.ylabel('平均年折旧额')
# 自动调整y轴范围
plt.ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.1)  # 增加10%的空间
# 设置网格线
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', axis='y')
# 增加数据标签
medians = brand_depreciation['每年平均折旧额'].median()
for i, brand in enumerate(top_brands):
    plt.text(i, medians, f'{medians:.2f}', ha='center', va='center', color='white', fontsize=9, weight='bold')
# 设置图例
plt.legend(title='品牌', loc='center left', bbox_to_anchor=(1, 0.5), title_fontsize='large', fontsize='large')
# 调整图例位置，使其不遮挡图表内容
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.9)
# 显示图表
plt.show()





# 数据预处理
# 确保所有列的数据类型正确
df['fuelType'] = df['fuelType'].astype(str)  # 确保能源类型为字符串类型
df['每年平均折旧额'] = df['每年平均折旧额'].astype(float)

# 数据预处理
# 确保所有列的数据类型正确
df['brand'] = df['brand'].astype(str)  # 确保品牌为字符串类型
df['series'] = df['series'].astype(str)  # 确保系列为字符串类型
df['resaleValueRate'] = df['resaleValueRate'].astype(float)  # 确保保值率为浮点数类型

# 构建适合旭日图的层次化数据结构
# 这里我们使用groupby和sum来聚合保值率数据
hierarchical_data = df.groupby(['brand', 'series'])['resaleValueRate'].sum().reset_index()

# 绘制旭日图
fig = px.sunburst(hierarchical_data,
                  path=['brand', 'series'],
                  values='resaleValueRate',
                  title='品牌及其子类别（系列）与保值率的关系分析')
fig.show()


# 数据预处理
# 确保所有列的数据类型正确，并排除品牌名或系列为空值的数据
df.dropna(subset=['brand', 'series'], inplace=True)  # 排除品牌名或系列为空值的数据
df['brand'] = df['brand'].astype(str)  # 确保品牌为字符串类型
df['series'] = df['series'].astype(str)  # 确保系列为字符串类型
df['valueLoss'] = df['valueLoss'].astype(float)  # 确保年平均折价额为浮点数类型

# 构建适合旭日图的层次化数据结构
# 这里我们使用groupby和sum来聚合年平均折价额数据
# 并且只考虑非空的brand和series
hierarchical_data = df.groupby(['brand', 'series'])['valueLoss'].sum().reset_index()

# 绘制旭日图
fig = px.sunburst(hierarchical_data,
                  path=['brand', 'series'],
                  values='valueLoss',
                  title='品牌及其系列')
fig.show()

# 数据预处理
df.dropna(subset=['brand', 'series'], inplace=True)  # 排除品牌名或系列为空值的数据
df['brand'] = df['brand'].astype(str)  # 确保品牌为字符串类型
df['series'] = df['series'].astype(str)  # 确保系列为字符串类型

# 构建适合树状图的层次化数据结构
hierarchical_data = df.groupby(['brand', 'series'])['每年平均折旧额'].sum().reset_index()

# 绘制树状图，并在每个分支上显示年平均折旧额的具体数值
fig = px.treemap(hierarchical_data,
                 path=['brand', 'series'],
                 values='每年平均折旧额',
                 title='品牌及其子类别（系列）与年平均折旧额的关系分析',
                 color_discrete_sequence=px.colors.qualitative.Vivid)  # 使用生动的颜色序列

# 添加文本标签以显示数值
for i in range(len(fig.data)):
    fig.data[i].texttemplate = '%{label}<br>%{value}'

fig.show()