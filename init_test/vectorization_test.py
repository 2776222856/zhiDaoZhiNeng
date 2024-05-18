import pandas as pd
import matplotlib.pyplot as plt

csv_url = '../dataBase/ONET/vectorized_ONET/New Occupation Data vectorized.csv'
# 读取CSV文件
df = pd.read_csv(csv_url)

# 选择前20条数据
df_top20 = df.head(10)

# 提取vectorization列的值并转换为二维列表
vectors = df_top20['vectorization'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# 绘制折线图
plt.figure(figsize=(15, 10))

for i, vector in enumerate(vectors):
    plt.plot(vector, label=f'Row {i+1}')

plt.title('Vectorization Line Plot for First 20 Rows')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
