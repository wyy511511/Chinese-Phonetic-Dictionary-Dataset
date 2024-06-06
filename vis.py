# pip install pandas plotly sklearn

import os
import shutil
import torch
import librosa
import pandas as pd
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.manifold import TSNE
import plotly.express as px


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


source_folder = "audio"
output_folder = "vec1"
csv_output_file = "embeddings.csv"
html_output_file = "vis.html"
os.makedirs(output_folder, exist_ok=True)

# init
embeddings_list = []


for root, dirs, files in os.walk(source_folder):
    for filename in files:

        if filename.endswith(".mp3"):

            src_file = os.path.join(root, filename)
            dst_file = os.path.join(output_folder, filename)


            shutil.copy2(src_file, dst_file)
            print(f"Copied: {filename}")


            audio_input, sample_rate = librosa.load(src_file, sr=16000)


            inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state


            print(f"Audio Embedding for {filename}:")
            print(hidden_states.shape)  #  (batch_size, sequence_length, hidden_size)
            print(hidden_states)


            embeddings_list.append({
                'filename': filename,
                'embedding': hidden_states.squeeze().tolist()  
            })



embeddings_df = pd.DataFrame(embeddings_list)

# embeddings_df.to_csv(csv_output_file, index=False)
# print(f"Embeddings saved to {csv_output_file}") 

# Cal t-SNE 
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(np.vstack(embeddings_df['embedding']))

embeddings_df['x'] = embeddings_2d[:, 0]
embeddings_df['y'] = embeddings_2d[:, 1]


embeddings_df.to_csv(csv_output_file, index=False)
print(f"Embeddings saved to {csv_output_file}")



fig = px.scatter(embeddings_df, x='x', y='y', hover_data=['filename'], title='t-SNE Visualization of Audio Embeddings')
fig.write_html(html_output_file)
print(f"HTML visualization saved to {html_output_file}")
