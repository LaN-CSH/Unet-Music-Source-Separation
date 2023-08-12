import numpy as np
import librosa
import tensorflow as tf
import os

from UNET_model import UNET

from config import *


def readAud(S):   # audio = np.ndarray of loaded data, 9초 이상의 wav, 44100 파일
    # S = np.load(r'wav file path')
    print(S.shape)
    if len(S.shape) == 1:
        S = np.expand_dims(S, axis=-1)
        S = np.concatenate((S, S), axis=-1)
    print(S.shape)
    print(((S.shape[0] // sample_size) + 1) * sample_size - S.shape[0])
    zero_2ch = np.concatenate((np.expand_dims(np.zeros(((S.shape[0] // sample_size) + 1) * sample_size - S.shape[0]), axis=-1)
                               , np.expand_dims(np.zeros(((S.shape[0] // sample_size) + 1) * sample_size - S.shape[0]), axis=-1)), axis=-1)
    S = np.concatenate((S, zero_2ch), axis=0)
    S = np.expand_dims(S, axis=0)
    print(S.shape)
    print((S.shape[1]) // sample_size)
    for j in range((S.shape[1]) // sample_size):
        print(j+1, '/', (S.shape[1]) // sample_size)
        if j == 0:
            wave_batch_0 = S[:, j * sample_size:(j + 1) * sample_size, 0]
            wave_batch_1 = S[:, j * sample_size:(j + 1) * sample_size, 1]
        else:
            wave_batch_0 = np.concatenate((wave_batch_0, S[:, j * sample_size:(j + 1) * sample_size, 0]), axis=0)
            wave_batch_1 = np.concatenate((wave_batch_1, S[:, j * sample_size:(j + 1) * sample_size, 1]), axis=0)
        # if j == ((S.shape[1] // sample_size) - 1):
        #     print('last')
        #     wave_batch_0 = np.concatenate((wave_batch_0, S[:, (j + 1) * sample_size:, 0]), axis=0)
        #     wave_batch_1 = np.concatenate((wave_batch_1, S[:, (j + 1) * sample_size:(j + 2) * sample_size, 0]), axis=0)
    print(wave_batch_0.shape)
    print(wave_batch_1.shape)
    return wave_batch_0, wave_batch_1

def stft_read_batch(w0, w1):
    for k in range(w0.shape[0]):
        stft_0 = librosa.stft(w0[k], n_fft=window_size, hop_length=hop_length, center=False)
        mag_0, phase_0 = librosa.magphase(stft_0)
        stft_0 = stft_0[:-1, :]
        stft_0 = np.expand_dims(stft_0, axis=-1)
        stft_0 = np.abs(np.expand_dims(stft_0, axis=0))
        mag_0 = np.expand_dims(mag_0, axis=0)  # (1, 1024, 256)
        stft_0 /= stft_0.max()

        stft_1 = librosa.stft(w1[k], n_fft=window_size, hop_length=hop_length, center=False)
        mag_1, phase_1 = librosa.magphase(stft_1)
        stft_1 = stft_1[:-1, :]
        stft_1 = np.expand_dims(stft_1, axis=-1)  # (1024, 256, 1)
        stft_1 = np.abs(np.expand_dims(stft_1, axis=0))  # (1, 1024, 256, 1)
        mag_1 = np.expand_dims(mag_1, axis=0)  # (1, 1024, 256)
        stft_1 /= stft_1.max()

        if k == 0:
            channel_0 = stft_0
            channel_1 = stft_1
            magni_0 = mag_0
            magni_1 = mag_1
        else:
            channel_0 = np.concatenate((channel_0, stft_0), axis=0)
            channel_1 = np.concatenate((channel_1, stft_1), axis=0)
            magni_0 = np.concatenate((magni_0, mag_0), axis=0)
            magni_1 = np.concatenate((magni_1, mag_1), axis=0)
    return channel_0, channel_1, magni_0, magni_1  # (?, 1024, 256, 1), (?, 1024, 256, 1), (?, 1024, 256), (?, 1024, 256)


s, sr = librosa.load('./y2mate.com - 349_EVGVJQtwxCY_320kbps.wav')
print(s)
wf0, wf1 = readAud(s)
wb0, wb1, m_ch0, m_ch1 = stft_read_batch(wf0, wf1)

X = tf.placeholder(tf.float32, [None, 1024, 256, 1])
train_mode = tf.placeholder(tf.bool)  # Feed True or False

# TRAIN
#########BRING SAVED WEIGHTS


unet = UNET(X, train=False)

X_multi_channel = tf.concat((X, X, X, X), axis=-1)
output = tf.multiply(unet, X_multi_channel)  # output shape = (?, 1024, 256, 4)

with tf.Session() as sess:
    output0 = sess.run(output, feed_dict={X: wb0})  # (?, 1024, 256, 4)
    output1 = sess.run(output, feed_dict={X: wb1})  # (?, 1024, 256, 4)
zero_out = np.zeros((output0.shape[0], 1, 256, 4))
output0 = np.concatenate((output0, zero_out), axis=1)   # (?, 1025, 256, 4)
output1 = np.concatenate((output1, zero_out), axis=1)   # (?, 1025, 256, 4)
zero_out = np.zeros((output0.shape[0], 1, 256))
m_ch0 = np.concatenate((m_ch0, zero_out), axis=1)       # (?, 1025, 256)
m_ch1 = np.concatenate((m_ch1, zero_out), axis=1)       # (?, 1025, 256)

out0_drum = librosa.istft(output0[0, :, :, 0]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)
out0_bass = librosa.istft(output0[0, :, :, 1]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)
out0_other = librosa.istft(output0[0, :, :, 2]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)
out0_vocal = librosa.istft(output0[0, :, :, 3]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)

out1_drum = librosa.istft(output1[0, :, :, 0]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)
out1_bass = librosa.istft(output1[0, :, :, 1]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)
out1_other = librosa.istft(output1[0, :, :, 2]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)
out1_vocal = librosa.istft(output1[0, :, :, 3]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)

for out_num in range(1, output0.shape[0]):
    out0_drum = np.append(
        (out0_drum, librosa.istft(output0[out_num, :, :, 0]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
    out0_bass = np.append(
        (out0_bass, librosa.istft(output0[out_num, :, :, 1]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
    out0_other = np.append(
        (out0_other, librosa.istft(output0[out_num, :, :, 2]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
    out0_vocal = np.append(
        (out0_vocal, librosa.istft(output0[out_num, :, :, 3]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))

    out1_drum = np.append(
        (out1_drum, librosa.istft(output0[out_num, :, :, 0]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
    out1_bass = np.append(
        (out1_bass, librosa.istft(output0[out_num, :, :, 1]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
    out1_other = np.append(
        (out1_other, librosa.istft(output0[out_num, :, :, 2]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
    out1_vocal = np.append(
        (out1_vocal, librosa.istft(output0[out_num, :, :, 3]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))


out0_drum = np.expand_dims(out0_drum, axis=-1)
out0_bass = np.expand_dims(out0_bass, axis=-1)
out0_other = np.expand_dims(out0_other, axis=-1)
out0_other = np.expand_dims(out0_other, axis=-1)

out1_drum = np.expand_dims(out1_drum, axis=-1)
out1_bass = np.expand_dims(out1_bass, axis=-1)
out1_other = np.expand_dims(out1_other, axis=-1)
out1_vocal = np.expand_dims(out1_vocal, axis=-1)

drum = np.transpose(np.concatenate((out0_drum, out1_drum), axis=-1))
bass = np.transpose(np.concatenate((out0_bass, out1_bass), axis=-1))
other = np.transpose(np.concatenate((out0_other, out1_other), axis=-1))
vocal = np.transpose(np.concatenate((out0_vocal, out1_vocal), axis=-1))

librosa.output.write_wav('./wav_sample/path drum.wav', drum, sr, norm=True)
librosa.output.write_wav('./wav_sample/path bass.wav', bass, sr, norm=True)
librosa.output.write_wav('./wav_sample/path other.wav', other, sr, norm=True)
librosa.output.write_wav('./wav_sample/vocal.wav', vocal, sr, norm=True)
