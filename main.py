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

import scipy.signal as signal

audio_file = "audio_file.wav"
sample_rate, signal_data = wavfile.read(audio_file)

import librosa

# mfcc
for key, value in entity_files_dict.items():
    signal, sample_rate = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(signal, sr=sample_rate, n_mfcc=13)
    print(mfcc.shape)

import numpy as np
# fft
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sample_rate, len(magnitude))

print(magnitude)
print(frequency)

# lpc
frame_length = int(sample_rate * 0.025)
frame_step = int(sample_rate * 0.010)
frames = signal.frame(signal_data, frame_length, frame_step)

n_fft = frame_length  # размер окна для STFT
stft = np.abs(signal.stft(signal_data, nperseg=n_fft)[2])[:, :frames.shape[1]]

order = 12  # порядок LPC
lpc_coeffs = np.zeros((frames.shape[1], order + 1))
for i in range(frames.shape[1]):
    lpc_coeffs[i] = signal.lpc(frames[:, i], order)

print(lpc_coeffs)

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
