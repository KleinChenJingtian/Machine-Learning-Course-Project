# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from scipy.stats import linregress
#
# # 设置支持中文的字体
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS','PingFang SC', 'SimHei','Songti']  # 尝试多种字体，确保至少有一种可用
# plt.rcParams['axes.unicode_minus'] = False  # 确保负号也能正确显示
#
# # 读取数据
# file_path = 'cleaned_数据集1.xlsx'
# data = pd.read_excel(file_path)
#
# # 二、保值率的描述性统计
# # 1. 中心趋势度量
# print("\n保值率的中心趋势度量：")
# mean_holding_rate = data['resaleValueRate'].mean()
# print(f"均值：{mean_holding_rate}")
# median_holding_rate = data['resaleValueRate'].median()
# print(f"中位数：{median_holding_rate}")
# mode_holding_rate = data['resaleValueRate'].mode()
# if not mode_holding_rate.empty:
#     print(f"众数：{mode_holding_rate}")
# else:
#     print("无众数（可能保值率值较分散）")
#
# # 2. 离散程度度量
# print("\n保值率的离散程度度量：")
# range_holding_rate = data['resaleValueRate'].max() - data['resaleValueRate'].min()
# print(f"极差：{range_holding_rate}")
# std_holding_rate = data['resaleValueRate'].std()
# print(f"标准差：{std_holding_rate}")
# cv_holding_rate = std_holding_rate / mean_holding_rate
# print(f"变异系数：{cv_holding_rate}")
#
#
# # 三、与保值率相关的其他变量关系分析
#
# # （一）上牌时间与保值率
# # 1. 时间分组分析
# data['regDate'] = pd.to_datetime(data['regDate'])
# years = data['regDate'].dt.year
# unique_years = years.unique()
# average_holding_rate_per_year = []
# for year in unique_years:
#     subset = data[years == year]
#     average_holding_rate_per_year.append(subset['resaleValueRate'].mean())
#     # 绘制柱状图展示每年平均保值率（修正了此处的错误）
#     plt.bar(year, subset['resaleValueRate'].mean(), color='skyblue')
# plt.xlabel('上牌年份')
# plt.ylabel('平均保值率')
# plt.title('上牌时间与平均保值率关系')
# plt.xticks(unique_years)  # 设置x轴刻度为年份的具体值
# plt.show()
#
#
# # 2. 相关系数计算
# # 将年份转换为数值型数据后计算相关系数
# years = pd.to_numeric(years)
# correlation_time_holding_rate = data['resaleValueRate'].corr(years)
# print("\n上牌时间与保值率的关系：")
# print(f"相关系数：{correlation_time_holding_rate}")
#
#
# # （二）里程与保值率
# # 1. 里程分段分析
# mileage_bins = [0, 20000, 40000, 60000, np.inf]
# mileage_labels = ['0 - 20000', '20000 - 40000', '40000 - 60000', '60000+']
# data['里程分段'] = pd.cut(data['mileAge'], bins=mileage_bins, labels=mileage_labels)
# average_holding_rate_per_mileage = data.groupby('里程分段', observed = False)['resaleValueRate'].mean()
#
#  # 绘制饼图展示不同里程分段的平均保值率占比
# plt.pie(average_holding_rate_per_mileage, labels=average_holding_rate_per_mileage.index, autopct='%1.1f%%', colors=sns.color_palette('Set1'))
# plt.title('不同里程分段平均保值率占比')
# plt.show()
#
#
# # （三）品牌与保值率
# # 1. 品牌间保值率比较
# average_holding_rate_per_brand = data.groupby('brand')['resaleValueRate'].mean().sort_values()
#
#  # 绘制水平条形图展示各品牌平均保值率
# plt.barh(average_holding_rate_per_brand.index, average_holding_rate_per_brand, color='orange')
# plt.xlabel('平均保值率')
# plt.ylabel('品牌')
# plt.title('各品牌平均保值率')
# plt.show()
#
#
# # （四）新车价格与保值率
# # 1. 比例关系分析
# data['保值率与新车价格比值'] = data['resaleValueRate'] / data['newCarPrice']
# ratio_stats = data['保值率与新车价格比值'].describe()
# print("\n保值率与新车价格比值的统计信息：")
# print(ratio_stats)
#
#  # 绘制箱线图展示保值率与新车价格比值的分布
# sns.boxplot(data['保值率与新车价格比值'], color='lightgreen')
# plt.xlabel('保值率与新车价格比值')
# plt.title('保值率与新车价格比值分布')
# plt.show()
#
#
# # （五）折价额与保值率
# # 1. 简单关系分析
# scatter_plot = data.plot.scatter(x='valueLoss', y='resaleValueRate', c='purple')
# scatter_plot.set_title('折价额与保值率的关系')
#
# # 2. 对比分析
# data['保值率与折价额差值'] = data['resaleValueRate'] - data['valueLoss']
# difference_stats = data['保值率与折价额差值'].describe()
# print("\n保值率与折价额差值的统计信息：")
# print(difference_stats)
#
#  # 绘制直方图展示保值率与折价额差值的分布
# plt.hist(data['保值率与折价额差值'], bins=10, color='salmon')
# plt.xlabel('保值率与折价额差值')
# plt.ylabel('频数')
# plt.title('保值率与折价额差值分布')
# plt.show()





# ———————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS','PingFang SC', 'SimHei','Songti']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
file_path = '../cleaned_数据集2.xlsx'
data = pd.read_excel(file_path)

# 二、保值率的描述性统计
# 1. 中心趋势度量
print("\n保值率的中心趋势度量：")
mean_holding_rate = data['resaleValueRate'].mean()
print(f"均值：{mean_holding_rate}")
median_holding_rate = data['resaleValueRate'].median()
print(f"中位数：{median_holding_rate}")
mode_holding_rate = data['resaleValueRate'].mode()
if not mode_holding_rate.empty:
    print(f"众数：{mode_holding_rate}")  # 修改此处，只打印众数的第一个值（如果有多个众数）
else:
    print("无众数（可能保值率值较分散）")

# 2. 离散程度度量
print("\n保值率的离散程度度量：")
range_holding_rate = data['resaleValueRate'].max() - data['resaleValueRate'].min()
print(f"极差：{range_holding_rate}")
std_holding_rate = data['resaleValueRate'].std()
print(f"标准差：{std_holding_rate}")
cv_holding_rate = std_holding_rate / mean_holding_rate
print(f"变异系数：{cv_holding_rate}")


# 三、与保值率相关的其他变量关系分析

# （一）上牌时间与保值率
# 1. 时间分组分析
data['regDate'] = pd.to_datetime(data['regDate'])
years = data['regDate'].dt.year
unique_years = years.unique()
average_holding_rate_per_year = []
for year in unique_years:
    subset = data[years == year]
    average_holding_rate_per_year.append(subset['resaleValueRate'].mean())
    # 绘制柱状图展示每年平均保值率（修正了此处的错误）
    plt.bar(year, subset['resaleValueRate'].mean(), color='skyblue')
plt.xlabel('上牌年份')
plt.ylabel('平均保值率')
plt.title('上牌时间与平均保值率关系')
plt.xticks(unique_years)
plt.yticks(np.arange(0, max(average_holding_rate_per_year) + 0.1, 0.1))  # 添加y轴刻度，可根据实际情况调整步长
plt.show()


# 2. 相关系数计算
# 将年份转换为数值型数据后计算相关系数
years = pd.to_numeric(years)
correlation_time_holding_rate = data['resaleValueRate'].corr(years)
print("\n上牌时间与保值率的关系：")
print(f"相关系数：{correlation_time_holding_rate}")


# （二）里程与保值率
# 1. 里程分段分析
mileage_bins = [0, 20000, 40000, 60000, np.inf]
mileage_labels = ['0 - 20000', '20000 - 40000', '40000 - 60000', '60000+']
data['里程分段'] = pd.cut(data['mileAge'], bins=mileage_bins, labels=mileage_labels)
average_holding_rate_per_mileage = data.groupby('里程分段', observed = False)['resaleValueRate'].mean()

 # 绘制饼图展示不同里程分段的平均保值率占比
plt.pie(average_holding_rate_per_mileage, labels=average_holding_rate_per_mileage.index, autopct='%1.1f%%', colors=sns.color_palette('Set1'), startangle = 90)  # 添加startangle参数，使饼图从垂直方向开始
plt.title('不同里程分段平均保值率占比')
plt.show()


# （三）品牌与保值率
# 1. 品牌间保值率比较
average_holding_rate_per_brand = data.groupby('brand')['resaleValueRate'].mean().sort_values()

 # 绘制水平条形图展示各品牌平均保值率
plt.barh(average_holding_rate_per_brand.index, average_holding_rate_per_brand, color='orange')
plt.xlabel('平均保值率')
plt.ylabel('品牌')
plt.title('各品牌平均保值率')
plt.xlim(0, max(average_holding_rate_per_brand) * 1.1)  # 设置x轴范围，可根据实际情况调整
plt.show()


# （四）新车价格与保值率
# 1. 比例关系分析
data['保值率与新车价格比值'] = data['resaleValueRate'] / data['newCarPrice']
ratio_stats = data['保值率与新车价格比值'].describe()
print("\n保值率与新车价格比值的统计信息：")
print(ratio_stats)

 # 绘制箱线图展示保值率与新车价格比值的分布
sns.boxplot(data['保值率与新车价格比值'], color='lightgreen')
plt.xlabel('保值率与新车价格比值')
plt.title('保值率与新车价格比值分布')
plt.show()


# （五）折价额与保值率
# 1. 简单关系分析
scatter_plot = data.plot.scatter(x='valueLoss', y='resaleValueRate', c='purple')
scatter_plot.set_title('折价额与保值率的关系')
# 添加坐标轴标签字体大小设置，可根据实际情况调整
scatter_plot.set_xlabel('折价额', fontsize = 12)
scatter_plot.set_ylabel('保值率', fontsize = 12)


# 2. 对比分析
data['保值率与折价额差值'] = data['resaleValueRate'] - data['valueLoss']
difference_stats = data['保值率与折价额差值'].describe()
print("\n保值率与折价额差值的统计信息：")
print(difference_stats)

 # 绘制直方图展示保值率与折价额差值的分布
plt.hist(data['保值率与折价额差值'], bins=10, color='salmon')
plt.xlabel('保值率与折价额差值')
plt.ylabel('频数')
plt.title('保值率与折价额差值分布')
plt.show()
