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
        entity_coeffs = lpc.tolist()[1::]
        data.append(entity_coeffs + [entity])

coeff_names = [str(i) for i in range(16)]
df = pd.DataFrame(data, columns=coeff_names + ['entity'])

unique_entities = df['entity'].unique()
unique_entities_size = unique_entities.size
entity_label_map = {}
entity_id_map = {}

for i in range(len(unique_entities)):
    entity = unique_entities[i]
    entity_label_map[entity] = i
    entity_id_map[str(i)] = entity

def mapEntityLabelToInt(label):
    return entity_label_map[label]

df['entity'] = df['entity'].map(lambda x: mapEntityLabelToInt(x))

def mapIntToEntityLabel(label):
    return entity_id_map[str(label)]

df['entity_label'] = df['entity'].map(lambda x: mapIntToEntityLabel(x))

'''
ШАГ 3
РАЗБИВАЕМ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
'''

from sklearn.model_selection import train_test_split

X = df[coeff_names]
y = df['entity']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(X_train)
# print(X_test)
#
# print(Y_train)
# print(Y_test)

'''
ШАГ 4
ОБУЧАЕМ МОДЕЛЬ
'''

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(unique_entities_size)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)

'''
ШАГ 5
ТЕСТИРУЕМ МОДЕЛЬ
'''

test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)

print(df.groupby('entity_label').size())

print('Train data size:', X.size)

print('\nTest accuracy:', test_acc)