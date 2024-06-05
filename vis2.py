import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px


source_folder = "audio"
csv_output_file = "embeddings.csv"
html_output_file = "vis2.html"


embeddings_df = pd.read_csv(csv_output_file)

# 计算 t-SNE 降维（如果没有保存 t-SNE 结果，则取消注释以下代码进行计算）
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(np.vstack(embeddings_df['embedding'].apply(eval).tolist()))
embeddings_df['x'] = embeddings_2d[:, 0]
embeddings_df['y'] = embeddings_2d[:, 1]
embeddings_df = pd.read_csv(csv_output_file)

# 使用 Plotly 生成 HTML 可视化
fig = px.scatter(
    embeddings_df, x='x', y='y', 
    hover_data=['filename'], 
    title='t-SNE Visualization of Audio Embeddings'
)

# 添加播放音频文件的功能
fig.update_traces(marker=dict(size=12),
                  selector=dict(mode='markers+text'))

# 添加自定义 JavaScript 以在点击时播放音频并下载选中区域的数据
fig.add_layout_image(
    dict(
        source="data:image/png;base64, [base64 data of a play button image]",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        sizex=0.2, sizey=0.2,
        xanchor="center", yanchor="middle"
    )
)

fig.update_layout(
    hovermode='closest',
    clickmode='event+select',
    dragmode='lasso',
)

# 在生成的 HTML 文件中添加 JavaScript 以播放音频和下载选中区域的数据
html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="plotly-div">{fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    <button id="download-button">Download Selected Data</button>
    <script>
        var selectedData = [];
        
        document.getElementById('plotly-div').on('plotly_selected', function(eventData) {{
            selectedData = eventData.points.map(pt => pt.customdata[0]);
        }});
        
        document.getElementById('plotly-div').on('plotly_click', function(data) {{
            var fileName = data.points[0].customdata[0];
            var audio = new Audio('{source_folder}/' + fileName);
            audio.play();
        }});

        document.getElementById('download-button').onclick = function() {{
            if (selectedData.length > 0) {{
                var csvContent = "data:text/csv;charset=utf-8,";
                csvContent += "filename\\n";
                selectedData.forEach(function(rowArray) {{
                    csvContent += rowArray + "\\n";
                }});
                
                var encodedUri = encodeURI(csvContent);
                var link = document.createElement("a");
                link.setAttribute("href", encodedUri);
                link.setAttribute("download", "selected_data.csv");
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }} else {{
                alert("No data selected");
            }}
        }};
    </script>
</body>
</html>
"""

with open(html_output_file, 'w') as f:
    f.write(html_template)

print(f"HTML visualization with audio playback and data download saved to {html_output_file}")
