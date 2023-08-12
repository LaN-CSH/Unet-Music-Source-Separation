import numpy as np
import librosa
import os
from config import *


def get_data():
    X = []
    Y = []
    file_list = os.listdir(r'E:\Stem_Np\Test')
    for i in range(len(file_list)):
        S = np.load(r'E:\Stem_Np\Test\%s' % (file_list[i]))
        for j in range((S.shape[1])//sample_size):
            print(j, '/', (S.shape[1])//sample_size)
            np.save(r'E:\Stem_Np\Test_frame\%s_%d' % (file_list[i][:-4], j),
                    S[:, j*sample_size:(j+1)*sample_size, 0])
            if j == (((S.shape[1])//sample_size) - 1):
                print('last')
                np.save(r'E:\Stem_Np\Test_frame\%s_%d' % (file_list[i][:-4], j+1),
                        S[:, -sample_size:, 0])
                break
        # X.append(S)
        print(S[:1, :, 0].shape)
        # print((S[:1, :, 1].shape[1])//sample_size)
        # print(S[1:, :, 0].shape)
        # print(S[1:, :, 1].shape)


def MakeSpec():   # Making Spectrogram from data set, Not a common Spectrogram making function
    train_list = os.listdir('/workspace/seungho/Data/Train_frame')    # shape = (?, 1024, 256, 1), (?, 1024, 256, 4)
    for batch_num in range(BATCH):
        for train_file in range(len(train_list)//BATCH):
            frame1 = np.load('/workspace/seungho/Data/Train_frame/%s' % (train_list[batch_num*(len(train_list)//BATCH) + train_file]))
            stft_mix = librosa.stft(frame1[0, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_mix = stft_mix[:-1, :]
            # print(stft_mix.shape)
            mag, phase = librosa.magphase(stft_mix)

            stft_drum = librosa.stft(frame1[1, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_drum = stft_drum[:-1, :]
            stft_drum = np.expand_dims(stft_drum, axis=-1)

            stft_bass = librosa.stft(frame1[2, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_bass = stft_bass[:-1, :]
            stft_bass = np.expand_dims(stft_bass, axis=-1)

            stft_other = librosa.stft(frame1[3, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_other = stft_other[:-1, :]
            stft_other = np.expand_dims(stft_other, axis=-1)

            stft_vocal = librosa.stft(frame1[4, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_vocal = stft_vocal[:-1, :]
            stft_vocal = np.expand_dims(stft_vocal, axis=-1)

            # print(type(mag[0][0]))
            x_mix = np.expand_dims(stft_mix, axis=-1)
            y_label = np.concatenate((stft_drum, stft_bass, stft_other, stft_vocal), axis=-1)

            x_mix = np.expand_dims(x_mix, axis=0)
            y_label = np.expand_dims(y_label, axis=0)

            x_mix = np.abs(x_mix)
            y_label = np.abs(y_label)

            # norm = x_mix.max()

            if x_mix.max() == 0:
                norm = 1
                print()
                print('MAX = 0 !')
                print()
                print()
            else:
                norm = x_mix.max()
                print()
                print('norm is not ZERO')
                print()
            print('N:', norm, train_file,'B:', batch_num, '/workspace/seungho/Data/Train_frame/%s' % (train_list[batch_num*(len(train_list)//BATCH) + train_file]))
            x_mix /= norm
            y_label /= norm

            # print(x_mix.shape, y_label.shape)
            if train_file == 0:
                X_train = x_mix
                Y_train = y_label
            else:
                X_train = np.concatenate((X_train, x_mix), axis=0)
                Y_train = np.concatenate((Y_train, y_label), axis=0)
        print(X_train.shape, Y_train.shape)
        np.save('/workspace/seungho/Data/Train_patch/X/patch_%d' % (batch_num), X_train)
        np.save('/workspace/seungho/Data/Train_patch/Y/patch_%d' % (batch_num), Y_train)
    return X_train, Y_train

def MakeSpec_test():
    train_list = os.listdir(r'E:\Stem_Np\Test_frame')

    print(len(train_list)//test_BATCH)
    for batch_num in range(test_BATCH):
        for train_file in range(len(train_list)//test_BATCH):
            frame1 = np.load(r'E:\Stem_Np\Test_frame\%s' % (train_list[batch_num*(len(train_list)//test_BATCH) + train_file]))
            stft_mix = librosa.stft(frame1[0, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_mix = stft_mix[:-1, :]
            # print(stft_mix.shape)
            mag, phase = librosa.magphase(stft_mix)

            stft_drum = librosa.stft(frame1[1, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_drum = stft_drum[:-1, :]
            stft_drum = np.expand_dims(stft_drum, axis=-1)

            stft_bass = librosa.stft(frame1[2, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_bass = stft_bass[:-1, :]
            stft_bass = np.expand_dims(stft_bass, axis=-1)

            stft_other = librosa.stft(frame1[3, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_other = stft_other[:-1, :]
            stft_other = np.expand_dims(stft_other, axis=-1)

            stft_vocal = librosa.stft(frame1[4, :], n_fft=window_size, hop_length=hop_length, center=False)
            stft_vocal = stft_vocal[:-1, :]
            stft_vocal = np.expand_dims(stft_vocal, axis=-1)

            # print(type(mag[0][0]))
            x_mix = np.expand_dims(stft_mix, axis=-1)
            y_label = np.concatenate((stft_drum, stft_bass, stft_other, stft_vocal), axis=-1)

            x_mix = np.expand_dims(x_mix, axis=0)
            y_label = np.expand_dims(y_label, axis=0)

            x_mix = np.abs(x_mix)
            y_label = np.abs(y_label)

            if x_mix.max() == 0:
                norm = 1
                print()
                print('MAX = 0 !')
                print()
                print()
            else:
                norm = x_mix.max()
                print()
                print('norm is not ZERO')
                print()
            print('N:', norm, train_file, 'B:', batch_num, r'E:\Stem_Np\Test_frame\%s' % (train_list[batch_num*(len(train_list)//BATCH) + train_file]))
            x_mix /= norm
            y_label /= norm

            # print(x_mix.shape, y_label.shape)
            if train_file == 0:
                X_train = x_mix
                Y_train = y_label
            else:
                X_train = np.concatenate((X_train, x_mix), axis=0)
                Y_train = np.concatenate((Y_train, y_label), axis=0)
        print(X_train.shape, Y_train.shape)
        np.save(r'E:\Stem_Np\Test_patch\X\patch_%d' % (batch_num), X_train)
        np.save(r'E:\Stem_Np\Test_patch\Y\patch_%d' % (batch_num), Y_train)
    return X_train, Y_train

def check_np():
    nu = np.load('/workspace/seungho/Data/Train_patch/X/patch_10.npy')
    for len_np in range(nu.shape[0]):
        print(nu[len_np])
    nu = np.load('/workspace/seungho/Data/Train_patch/Y/patch_10.npy')
    for len_np in range(nu.shape[0]):
        print(nu[len_np])


# get_data()
# MakeSpec()
MakeSpec_test()
# check_np()
