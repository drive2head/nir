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

for key, value in entity_files_dict.items():
    print('entity:', key)
    print('audio files:', value)


'''
ШАГ 2
ПРИВОДИМ К ВЕКТОРНОМУ ВИДУ
'''

'''
ШАГ 3
РАЗБИВАЕМ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
'''

'''
ШАГ 4
ОБУЧАЕМ МОДЕЛЬ
'''

'''
ШАГ 5
ТЕСТИРУЕМ МОДЕЛЬ
'''
