import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# 加载 embeddings.csv 和 renamed_files.csv
embeddings_file = 'embeddings.csv'
renamed_files_file = 'renamed_files.csv'

embeddings_df = pd.read_csv(embeddings_file)
renamed_files_df = pd.read_csv(renamed_files_file)

# 合并两个数据框
merged_df = pd.merge(embeddings_df, renamed_files_df, left_on='filename', right_on='UUID')

# 检查合并后的数据框
print(merged_df.head())

# 进行聚类
X = merged_df[['x', 'y']]

# 使用 KMeans 进行聚类
kmeans = KMeans(n_clusters=300, random_state=42)  # 假设分成5类，可以根据需要调整
merged_df['cluster'] = kmeans.fit_predict(X)

# 使用 Plotly 进行可视化并保存为 HTML
fig = px.scatter(merged_df, x='x', y='y', color='cluster', hover_data=['New Filename'])
fig.update_layout(title='t-SNE Clustering Visualization', xaxis_title='x', yaxis_title='y')

# 保存为 HTML 文件
html_output_file = 'visualization.html'
fig.write_html(html_output_file)

# 打印保存路径
print(f"Visualization saved to {html_output_file}")

# 保存带有聚类结果的数据框
csv_output_file = 'merged_and_clustered.csv'
merged_df.to_csv(csv_output_file, index=False)
