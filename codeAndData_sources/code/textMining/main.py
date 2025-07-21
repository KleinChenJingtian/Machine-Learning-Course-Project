# 1. 引入需要的库并读取数据
import pandas as pd
from charset_normalizer import models
from snownlp import SnowNLP
import numpy as np
import re
import jieba
import stylecloud
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from wordcloud import WordCloud
from PIL import Image

data=pd.read_excel('/Users/boyi/desktop/文本数据1.xlsx',sheet_name='Sheet1')
data = data[['回答内容']]

# 2.  设置 Pandas 显示格式
pd.options.display.float_format = '{:.6f}'.format

# 3. 数据预处理
# 定义数据处理函数
def process_data(data):
    # 初始化存储处理后数据的列表
    processed_data = []
    # 遍历每一行处理数据
    for index, row in data.iterrows():
        raw_content = row['回答内容']
        # 使用正则表达式提取答主
        author_match = re.search(r'^(.*?)\n', raw_content)
        author = author_match.group(1) if author_match else ''
        # 提取发布时间
        publish_match = re.search(r'发布于\s([\d-]+\s[\d:]+)', raw_content)
        publish_time = publish_match.group(1) if publish_match else ''
        # 提取赞同数
        upvote_match = re.search(r'(\d+)\s人赞同了该回答', raw_content)
        upvote_count = int(upvote_match.group(1)) if upvote_match else 0
        # 提取评论数
        comment_match = re.search(r'(\d+)\s条评论', raw_content)
        comment_count = int(comment_match.group(1)) if comment_match else 0
        # 提取回答内容（去除无关内容）
        content = re.split(r'发布于|赞同\s\d+|\d+\s条评论', raw_content)[0].strip()
        # 将处理后的数据存储到列表
        processed_data.append({
            '答主': author,
            '内容': content,
            '发布时间': publish_time,
            '赞同数': upvote_count,
            '评论数': comment_count
        })
    # 转换为DataFrame
    processed_df = pd.DataFrame(processed_data)
    return processed_df
# 调用数据处理函数
processed_data = process_data(data)

# 保存处理结果
#with pd.ExcelWriter('/Users/boyi/desktop/文本数据1.xlsx', mode='a', engine='openpyxl') as writer:
    #processed_data.to_excel(writer, sheet_name='处理结果', index=False)

# 读取处理后数据
data=pd.read_excel('/Users/boyi/desktop/文本数据1.xlsx',sheet_name='处理结果')
data = data[['内容']]

# 检查数据类型和是否有缺失值
print(data.info())
print(data.describe(include=['object','float']).T)
# 去除空值与重复值
data = data.dropna(subset=['内容'])  # 删除空值行
data = data.drop_duplicates()  # 去除重复值
# 确保所有内容都是字符串
data['内容'] = data['内容'].astype(str)  # 转换为字符串

# 4. 情感分析
data['情感得分'] = data['内容'].apply(lambda x: SnowNLP(x).sentiments)
# 查看结果
print(data.head())
print(data.sample(10))
# 可视化整体情感倾向
## 手动设置字体
font_path = '/System/Library/Fonts/Supplemental/Songti.ttc'  # 替换为你的中文字体路径
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
# 确保负号正常显示
plt.rcParams['axes.unicode_minus'] = False
# 绘制图表
plt.figure(figsize=(12, 6))
rate = data['情感得分']
ax=sns.distplot(rate,
                hist_kws={'color':'blue','label':'直方图'},
                kde_kws={'color':'red','label':'密度曲线'},
                bins=20)
ax.set_title("大众对新能源汽车整体情感倾向",fontproperties=font_prop)
plt.legend()
plt.show()

# 5. 绘制词云图
# 重新读取Excel数据
data = pd.read_excel('/Users/boyi/desktop/文本数据1.xlsx', sheet_name='处理结果')
# 文本分词处理
data['cut'] = data['内容'].apply(lambda x: " ".join(jieba.lcut(x)))  # 分词并以空格连接
# 拼接所有回答内容的分词结果
text = " ".join(data['cut'])
# 加载停用词
stopword_path = '/Users/boyi/Desktop/hit_stopwords copy.txt'
with open(stopword_path, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())
# 加载词云背景图
mask_path = '/Users/boyi/Desktop/the-new-york-public-library-1-fp5zUitxE-unsplash.jpg'
mask = np.array(Image.open(mask_path))
# 生成词云
wordcloud = WordCloud(
    font_path='/System/Library/Fonts/Supplemental/Songti.ttc',
    background_color='white',
    max_words=200,
    mask=mask,
    stopwords=stopwords
).generate(text)
# 绘制词云
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("大众就新能源汽车回答内容词云图", fontsize=20)
plt.show()



# 6. LDA主题分析
from gensim import corpora, models
# 停用词路径
stop_words_path = '/Users/boyi/Desktop/hit_stopwords copy.txt'
 # 加载停用词
with open(stop_words_path, encoding='utf-8-sig') as f:
     stop_words = [line.strip() for line in f.readlines()]

# 查看数据内容
data['内容'] = data['内容'].dropna().fillna('')  # 确保数据内容是字符串类型
data['分词内容'] = data['内容'].apply(lambda x: " ".join(jieba.cut(x)))  # 使用jieba进行中文分词
data['情感得分'] = data['内容'].apply(lambda x: SnowNLP(x).sentiments)
# 筛选情感倾向
positive = data[data['情感得分'] >= 0.85]['分词内容']
negative = data[data['情感得分'] <= 0.2]['分词内容']

# 分词处理函数并过滤停用词
def preprocess_text(text, stop_words):
    word_list = text.split()
    filtered_words = [word for word in word_list if word not in stop_words]
    return filtered_words
# 过滤停用词
positive['preprocessed'] = positive.apply(lambda x: preprocess_text(x, stop_words), )
negative['preprocessed'] = negative.apply(lambda x: preprocess_text(x, stop_words), )

# 创建字典和语料库
# LDA需要将文本数据转换成向量表示
positive_dict = corpora.Dictionary(positive['preprocessed'])
positive_corpus = [positive_dict.doc2bow(text) for text in positive['preprocessed']]
negative_dict=corpora.Dictionary(negative['preprocessed'])
negative_corpus=[negative_dict.doc2bow(text) for text in negative['preprocessed']]

# LDA建模
lda_model = models.LdaModel(positive_corpus, num_topics=5, id2word=positive_dict, passes=10)
lda_model_n=models.LdaModel(negative_corpus,num_topics=5,id2word=negative_dict,passes=10)

# 查看主题结果
print("积极主题分析结果：")
for i in range(5):
    print(f"Topic {i+1}:")
    print(lda_model.print_topic(i))
    print('-'*30)
print("消极主题分析结果：")
for i in range(5):
    print(f"Topic {i+1}:")
    print(lda_model_n.print_topic(i))
    print('-'*30)

# 7. 评论相关数据可视化
# 重新读取数据
file_path='/Users/boyi/desktop/文本数据1.xlsx'
sheet_name='处理结果'
df=pd.read_excel(file_path,sheet_name=sheet_name)
# 定义并统计各条回答热度值
df['热度值']=df['赞同数']+df['评论数']
# 清洗掉可能重复的数据
df = df.drop_duplicates(subset=['答主'])
# 对上述数值型数据进行描述性统计
columns_to_analyze=['赞同数','评论数','热度值']
stats= df[columns_to_analyze].describe()
print("描述性统计：")
print(stats)

# 引入必要的库
from pyecharts.charts import Bar
from pyecharts import options as opts
#（1）绘制互动热度Top30的热度柱状图
top30_df= df.nlargest(30,'热度值')[['答主','热度值']]
bar=(
    Bar()
    .add_xaxis(top30_df['答主'].tolist())  # 横坐标：答主
    .add_yaxis("热度值", top30_df['热度值'].tolist(), label_opts=opts.LabelOpts(is_show=True,position='top'))  # 显示热度值
    .set_global_opts(
        title_opts=opts.TitleOpts(title="互动热度 Top30 柱状图"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),  # 旋转横坐标标签防止重叠
        yaxis_opts=opts.AxisOpts(name="热度值"),
        legend_opts=opts.LegendOpts(is_show=True),
        )
    )
bar.render('热度值Top30 答主柱状图.html')

#（2）绘制回答发布年份分布的玫瑰图
from pyecharts.charts import Pie
# 将不一致日期格式转换为日期时间格式
df['发布时间']=pd.to_datetime(df['发布时间'],errors='coerce')
# 提取年份信息
df['年份']=df['发布时间'].dt.year

# 按年份统计回答数量并降序排序
year_distribution=df['年份'].value_counts().sort_values(ascending=False)
print('回答发布年份分布（降序）：')
print(year_distribution)
data_pair=[(str(int(year)), count) for year, count in year_distribution.items()]
print(data_pair)
# 创建玫瑰图
pie=(
    Pie(init_opts=opts.InitOpts(width="800px", height="600px"))
    .add(
        series_name="回答发布年份分布",
        data_pair=data_pair,
        rosetype="radius",  # 设置为玫瑰图
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="回答发布年份分布玫瑰图"),
        legend_opts=opts.LegendOpts(is_show=True, pos_right="10%"),
    )
    .set_series_opts(
        label_opts=opts.LabelOpts(is_show=True, position='outside',formatter="{b}: {c} ({d}%)")  # 显示百分比
    )
)

# 渲染玫瑰图为HTML文件
pie.render('回答发布年份分布玫瑰图_v2.html')
