import os
import re
import csv
import shutil


txt_file = "Chinese__Most Common 3000 Hanzi.txt"  
audio_folder = "audio"  
csv_output_file = "renamed_files.csv"

pattern = re.compile(
    r'\[sound:(sapi5-\w+\.mp3)\].*?<span class=pinyin>(.*?)</span>.*?href=.*?>(.*?)</a>',
    re.DOTALL
)


data = []

with open(txt_file, 'r', encoding='utf-8') as file:
    content = file.read()
    entries = content.split('\n\n\n\n')

    for entry in entries:
        match = re.search(
            r'\[sound:(sapi5-[\w-]+\.mp3)\].*?<span class=pinyin>(.*?)</span>.*?href=.*?zi=(.*?)\'',
            entry, re.DOTALL
        )
        if match:
            uuid, pinyin, character = match.groups()
            print(uuid, pinyin, character)
            new_filename = f"{pinyin}_{character}.mp3"
            data.append([uuid, pinyin, character, new_filename])




with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['UUID', 'Pinyin', 'Character', 'New Filename'])

    for uuid, pinyin, character, new_filename in data:
        old_filepath = os.path.join(audio_folder, uuid)
        new_filepath = os.path.join(audio_folder, new_filename)

        if os.path.exists(old_filepath):
            shutil.move(old_filepath, new_filepath)
            writer.writerow([uuid, pinyin, character, new_filename])
            print(f"Renamed {uuid} to {new_filename}")

print(f"Renaming completed and saved to {csv_output_file}")
