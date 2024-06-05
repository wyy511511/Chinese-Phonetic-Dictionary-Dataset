



import os
import shutil
import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Modify
source_folder = "audio"
output_folder = "vec"
csv_output_file = "embeddings.csv"


# check if exist
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
embeddings_expanded_df = embeddings_df['embedding'].apply(pd.Series)
embeddings_expanded_df['filename'] = embeddings_df['filename']
embeddings_expanded_df = embeddings_expanded_df[['filename'] + list(embeddings_expanded_df.columns[:-1])]


embeddings_expanded_df.to_csv(csv_output_file, index=False)
print(f"Embeddings saved to {csv_output_file}")


