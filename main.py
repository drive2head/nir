'''
ШАГ 1
СЧИТЫВАЕМ ДАННЫЕ
'''

import os

entity_files_dict = {}

resources_folder = './resources'
records_folders = [f.path for f in os.scandir(resources_folder) if f.is_dir()]
for record_folder in records_folders:
    entity_folders = [f for f in os.scandir(record_folder) if f.is_dir()]
    for entity_folder in entity_folders:
        entity_records_folder = entity_folder.path
        entity_name = entity_folder.name
        entity_records_paths = [f.path for f in os.scandir(entity_records_folder)]

        if entity_files_dict.get(entity_name) is None:
            entity_files_dict[entity_name] = entity_records_paths
        else:
            entity_files_dict[entity_name] += entity_records_paths

'''
ШАГ 2
ПРИВОДИМ К ВЕКТОРНОМУ ВИДУ
'''

import librosa
import pandas as pd

# mfcc
# for entity, files in entity_files_dict.items():
#     for file in files:
#         print('entity:', entity)
#         print('file:', file)
#         signal, sample_rate = librosa.load(file)
#         mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate)
#         print(mfcc.shape)
#         print(mfcc)

data = []
# lpc
for entity, files in entity_files_dict.items():
    for file in files:
        y, sr = librosa.load(file)
        lpc = librosa.lpc(y, order=16)
        coeffs = lpc.tolist()[1::]
        data.append(coeffs + [entity])

df = pd.DataFrame(data, columns=[str(i) for i in range(16)] + ['entity'])

'''
ШАГ 3
РАЗБИВАЕМ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
'''

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.33, random_state=42)

print(X_train)
print(X_test)

'''
ШАГ 4
ОБУЧАЕМ МОДЕЛЬ
'''

'''
ШАГ 5
ТЕСТИРУЕМ МОДЕЛЬ
'''
