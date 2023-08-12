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
    print('00000000000000000000')
    np.save('./sample/patch/_ch0.npy', wave_batch_0)
    np.save('./sample/patch/_ch1.npy', wave_batch_1)
    print(wave_batch_0.shape,'_ch0')
    print(wave_batch_1.shape,'_ch1')
    return wave_batch_0, wave_batch_1

def stft_read_batch(w0, w1):
    for k in range(w0.shape[0]):
        stft_0 = librosa.stft(w0[k], n_fft=window_size, hop_length=hop_length, center=False)
        mag_0, phase_0 = librosa.magphase(stft_0)
        if np.abs(stft_0).all() == mag_0.all():
            print("SAMESAME%d" % (k+1))
        stft_0 = stft_0[:-1, :]
        stft_0 = np.expand_dims(stft_0, axis=-1)
        stft_0 = np.abs(np.expand_dims(stft_0, axis=0))
        mag_0 = np.expand_dims(mag_0, axis=0)  # (1, 1024, 256)
        mag_0 = np.expand_dims(mag_0, axis=-1)  # (1, 1024, 256, 1)
        phase_0 = np.expand_dims(phase_0, axis=0)  # (1, 1024, 256)
        phase_0 = np.expand_dims(phase_0, axis=-1)  # (1, 1024, 256, 1)
        stft_0 /= stft_0.max()

        stft_1 = librosa.stft(w1[k], n_fft=window_size, hop_length=hop_length, center=False)
        mag_1, phase_1 = librosa.magphase(stft_1)
        stft_1 = stft_1[:-1, :]
        stft_1 = np.expand_dims(stft_1, axis=-1)  # (1024, 256, 1)
        stft_1 = np.abs(np.expand_dims(stft_1, axis=0))  # (1, 1024, 256, 1)
        mag_1 = np.expand_dims(mag_1, axis=0)  # (1, 1024, 256)
        mag_1 = np.expand_dims(mag_1, axis=-1)  # (1, 1024, 256, 1)
        phase_1 = np.expand_dims(phase_1, axis=0)  # (1, 1024, 256)
        phase_1 = np.expand_dims(phase_1, axis=-1)  # (1, 1024, 256, 1)
        stft_1 /= stft_1.max()

        if k == 0:
            channel_0 = stft_0
            channel_1 = stft_1
            magni_0 = mag_0
            magni_1 = mag_1
            p_0 = phase_0
            p_1 = phase_1
        else:
            channel_0 = np.concatenate((channel_0, stft_0), axis=0)
            channel_1 = np.concatenate((channel_1, stft_1), axis=0)
            magni_0 = np.concatenate((magni_0, mag_0), axis=0)
            magni_1 = np.concatenate((magni_1, mag_1), axis=0)
            p_0 = np.concatenate((p_0, phase_0), axis=0)
            p_1 = np.concatenate((p_1, phase_1), axis=0)
    print('magni_shape: ', magni_0.shape)
    print('p_shape: ', p_0.shape)
    np.save('./sample/patch/stft_pt_ch0.npy', channel_0)
    # np.save('./sample/patch/stft_pt_ch0_part3.npy', channel_0[2:3, :, :, :])
    # np.save('./sample/patch/stft_pt_ch0_part4.npy', channel_0[3:4, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part6.npy', channel_0[5:6, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part9.npy', channel_0[8:9, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part10.npy', channel_0[9:10, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part14.npy', channel_0[13:14, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part15.npy', channel_0[14:15, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part16.npy', channel_0[15:16, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part17.npy', channel_0[16:17, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part18.npy', channel_0[17:18, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part19.npy', channel_0[18:19, :, :, :])
    np.save('./sample/patch/stft_pt_ch0_part20.npy', channel_0[19:20, :, :, :])


    np.save('./sample/patch/stft_pt_ch1.npy', channel_1)
    # np.save('./sample/patch/stft_pt_ch1_part3.npy', channel_1[2:3, :, :, :])
    # np.save('./sample/patch/stft_pt_ch1_part4.npy', channel_1[3:4, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part6.npy', channel_1[5:6, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part9.npy', channel_1[8:9, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part10.npy', channel_1[9:10, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part14.npy', channel_1[13:14, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part15.npy', channel_1[14:15, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part16.npy', channel_1[15:16, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part17.npy', channel_1[16:17, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part18.npy', channel_1[17:18, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part19.npy', channel_1[18:19, :, :, :])
    np.save('./sample/patch/stft_pt_ch1_part20.npy', channel_1[19:20, :, :, :])

    np.save('./sample/patch/stft_pt_mag0.npy', magni_0)
    # np.save('./sample/patch/stft_pt_mag0_part3.npy', magni_0[2:3, :, :, :])
    # np.save('./sample/patch/stft_pt_mag0_part4.npy', magni_0[3:4, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part6.npy', magni_0[5:6, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part9.npy', magni_0[8:9, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part10.npy', magni_0[9:10, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part14.npy', magni_0[13:14, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part15.npy', magni_0[14:15, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part16.npy', magni_0[15:16, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part17.npy', magni_0[16:17, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part18.npy', magni_0[17:18, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part19.npy', magni_0[18:19, :, :, :])
    np.save('./sample/patch/stft_pt_mag0_part20.npy', magni_0[19:20, :, :, :])

    np.save('./sample/patch/stft_pt_mag1.npy', magni_1)
    # np.save('./sample/patch/stft_pt_mag1_part3.npy', magni_1[2:3, :, :, :])
    # np.save('./sample/patch/stft_pt_mag1_part4.npy', magni_1[3:4, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part6.npy', magni_1[5:6, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part9.npy', magni_1[8:9, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part10.npy', magni_1[9:10, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part14.npy', magni_1[13:14, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part15.npy', magni_1[14:15, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part16.npy', magni_1[15:16, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part17.npy', magni_1[16:17, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part18.npy', magni_1[17:18, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part19.npy', magni_1[18:19, :, :, :])
    np.save('./sample/patch/stft_pt_mag1_part20.npy', magni_1[19:20, :, :, :])

    np.save('./sample/patch/stft_pt_phase0.npy', p_0)
    # np.save('./sample/patch/stft_pt_phase0_part3.npy', p_0[2:3, :, :])
    # np.save('./sample/patch/stft_pt_phase0_part4.npy', p_0[3:4, :, :])
    np.save('./sample/patch/stft_pt_phase0_part6.npy', p_0[5:6, :, :])
    np.save('./sample/patch/stft_pt_phase0_part9.npy', p_0[8:9, :, :])
    np.save('./sample/patch/stft_pt_phase0_part10.npy', p_0[9:10, :, :])
    np.save('./sample/patch/stft_pt_phase0_part14.npy', p_0[13:14, :, :])
    np.save('./sample/patch/stft_pt_phase0_part15.npy', p_0[14:15, :, :])
    np.save('./sample/patch/stft_pt_phase0_part16.npy', p_0[15:16, :, :])
    np.save('./sample/patch/stft_pt_phase0_part17.npy', p_0[16:17, :, :])
    np.save('./sample/patch/stft_pt_phase0_part18.npy', p_0[17:18, :, :])
    np.save('./sample/patch/stft_pt_phase0_part19.npy', p_0[18:19, :, :])
    np.save('./sample/patch/stft_pt_phase0_part20.npy', p_0[19:20, :, :])

    np.save('./sample/patch/stft_pt_phase1.npy', p_1)
    # np.save('./sample/patch/stft_pt_phase1_part3.npy', p_1[2:3, :, :])
    # np.save('./sample/patch/stft_pt_phase1_part4.npy', p_1[3:4, :, :])
    np.save('./sample/patch/stft_pt_phase1_part6.npy', p_1[5:6, :, :])
    np.save('./sample/patch/stft_pt_phase1_part9.npy', p_1[8:9, :, :])
    np.save('./sample/patch/stft_pt_phase1_part10.npy', p_1[9:10, :, :])
    np.save('./sample/patch/stft_pt_phase1_part14.npy', p_1[13:14, :, :])
    np.save('./sample/patch/stft_pt_phase1_part15.npy', p_1[14:15, :, :])
    np.save('./sample/patch/stft_pt_phase1_part16.npy', p_1[15:16, :, :])
    np.save('./sample/patch/stft_pt_phase1_part17.npy', p_1[16:17, :, :])
    np.save('./sample/patch/stft_pt_phase1_part18.npy', p_1[17:18, :, :])
    np.save('./sample/patch/stft_pt_phase1_part19.npy', p_1[18:19, :, :])
    np.save('./sample/patch/stft_pt_phase1_part20.npy', p_1[19:20, :, :])
    print(channel_0.shape, channel_1.shape, mag_0.shape, mag_1.shape)

    return channel_0, channel_1, magni_0, magni_1  # (?, 1024, 256, 1), (?, 1024, 256, 1), (?, 1024, 256), (?, 1024, 256)
# 3, 4번째 (배열에서 2,3 번째 요소)


def run():
    s, sr = librosa.load('./sample/wav_sample/agami.wav', sr=44100)
    print(sr)
    wf0, wf1 = readAud(s)
    wb0, wb1, m_ch0, m_ch1 = stft_read_batch(wf0, wf1)

    X = tf.placeholder(tf.float32, [None, 1024, 256, 1])
    train_mode = tf.placeholder(tf.bool)  # Feed True or False
    #
    # TRAIN
    unet = UNET(X, train_mode)
    X_multi_channel = tf.concat((X, X, X, X), axis=-1)
    out = tf.multiply(unet, X_multi_channel)
    # BRING SAVED WEIGHTS
    with tf.Session() as sess:
        # part3_0 = np.load('./sample/patch/stft_pt_ch0_part3.npy')
        # part3_1 = np.load('./sample/patch/stft_pt_ch1_part3.npy')
        # part4_0 = np.load('./sample/patch/stft_pt_ch0_part4.npy')
        # part4_1 = np.load('./sample/patch/stft_pt_ch1_part4.npy')
        part6_0 = np.load('./sample/patch/stft_pt_ch0_part6.npy')
        part6_1 = np.load('./sample/patch/stft_pt_ch1_part6.npy')
        part9_0 = np.load('./sample/patch/stft_pt_ch0_part9.npy')
        part9_1 = np.load('./sample/patch/stft_pt_ch1_part9.npy')
        part10_0 = np.load('./sample/patch/stft_pt_ch0_part10.npy')
        part10_1 = np.load('./sample/patch/stft_pt_ch1_part10.npy')
        part14_0 = np.load('./sample/patch/stft_pt_ch0_part14.npy')
        part14_1 = np.load('./sample/patch/stft_pt_ch1_part14.npy')
        part15_0 = np.load('./sample/patch/stft_pt_ch0_part15.npy')
        part15_1 = np.load('./sample/patch/stft_pt_ch1_part15.npy')
        part16_0 = np.load('./sample/patch/stft_pt_ch0_part16.npy')
        part16_1 = np.load('./sample/patch/stft_pt_ch1_part16.npy')
        part17_0 = np.load('./sample/patch/stft_pt_ch0_part17.npy')
        part17_1 = np.load('./sample/patch/stft_pt_ch1_part17.npy')
        part18_0 = np.load('./sample/patch/stft_pt_ch0_part18.npy')
        part18_1 = np.load('./sample/patch/stft_pt_ch1_part18.npy')
        part19_0 = np.load('./sample/patch/stft_pt_ch0_part19.npy')
        part19_1 = np.load('./sample/patch/stft_pt_ch1_part19.npy')
        part20_0 = np.load('./sample/patch/stft_pt_ch0_part20.npy')
        part20_1 = np.load('./sample/patch/stft_pt_ch1_part20.npy')
        print(part6_0.shape, part6_1.shape)
        print(part9_0.shape, part9_1.shape)
        print(part10_0.shape, part10_1.shape)
        print(part19_0.shape, part19_1.shape)

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print()
            print()
            print('CHECKPOINT EXISTS')
            print()
            print()
        else:
            sess.run(tf.global_variables_initializer())
            print()
            print()
            print('NO CHECKPOINT??')
            print()
            print()
        out0 = sess.run(out, feed_dict={X: wb0, train_mode: False})
        out1 = sess.run(out, feed_dict={X: wb1, train_mode: False})
        # partial3_0 = sess.run(unet, feed_dict={X: part3_0, train_mode: False})
        # partial3_1 = sess.run(unet, feed_dict={X: part3_1, train_mode: False})
        # partial4_0 = sess.run(unet, feed_dict={X: part4_0, train_mode: False})
        # partial4_1 = sess.run(unet, feed_dict={X: part4_1, train_mode: False})
        partial6_0 = sess.run(unet, feed_dict={X: part6_0, train_mode: False})
        partial6_1 = sess.run(unet, feed_dict={X: part6_1, train_mode: False})
        partial9_0 = sess.run(unet, feed_dict={X: part9_0, train_mode: False})
        partial9_1 = sess.run(unet, feed_dict={X: part9_1, train_mode: False})
        partial10_0 = sess.run(unet, feed_dict={X: part10_0, train_mode: False})
        partial10_1 = sess.run(unet, feed_dict={X: part10_1, train_mode: False})
        partial14_0 = sess.run(unet, feed_dict={X: part14_0, train_mode: False})
        partial14_1 = sess.run(unet, feed_dict={X: part14_1, train_mode: False})
        partial15_0 = sess.run(unet, feed_dict={X: part15_0, train_mode: False})
        partial15_1 = sess.run(unet, feed_dict={X: part15_1, train_mode: False})
        partial16_0 = sess.run(unet, feed_dict={X: part16_0, train_mode: False})
        partial16_1 = sess.run(unet, feed_dict={X: part16_1, train_mode: False})
        partial17_0 = sess.run(unet, feed_dict={X: part17_0, train_mode: False})
        partial17_1 = sess.run(unet, feed_dict={X: part17_1, train_mode: False})
        partial18_0 = sess.run(unet, feed_dict={X: part18_0, train_mode: False})
        partial18_1 = sess.run(unet, feed_dict={X: part18_1, train_mode: False})
        partial19_0 = sess.run(unet, feed_dict={X: part19_0, train_mode: False})
        partial19_1 = sess.run(unet, feed_dict={X: part19_1, train_mode: False})
        partial20_0 = sess.run(unet, feed_dict={X: part20_0, train_mode: False})
        partial20_1 = sess.run(unet, feed_dict={X: part20_1, train_mode: False})
        print(out0.shape, out1.shape)
        print(partial6_0.shape, partial6_1.shape)
        np.save('./sample/patch/learned_pt_ch0.npy', out0)
        np.save('./sample/patch/learned_pt_ch1.npy', out1)
        # np.save('./sample/patch/learned_part3_ch0.npy', partial3_0)
        # np.save('./sample/patch/learned_part3_ch1.npy', partial3_1)
        # np.save('./sample/patch/learned_part4_ch0.npy', partial4_0)
        # np.save('./sample/patch/learned_part4_ch1.npy', partial4_1)
        np.save('./sample/patch/learned_part6_ch0.npy', partial6_0)
        np.save('./sample/patch/learned_part6_ch1.npy', partial6_1)
        np.save('./sample/patch/learned_part9_ch0.npy', partial9_0)
        np.save('./sample/patch/learned_part9_ch1.npy', partial9_1)
        np.save('./sample/patch/learned_part10_ch0.npy', partial10_0)
        np.save('./sample/patch/learned_part10_ch1.npy', partial10_1)
        np.save('./sample/patch/learned_part14_ch0.npy', partial14_0)
        np.save('./sample/patch/learned_part14_ch1.npy', partial14_1)
        np.save('./sample/patch/learned_part15_ch0.npy', partial15_0)
        np.save('./sample/patch/learned_part15_ch1.npy', partial15_1)
        np.save('./sample/patch/learned_part16_ch0.npy', partial16_0)
        np.save('./sample/patch/learned_part16_ch1.npy', partial16_1)
        np.save('./sample/patch/learned_part17_ch0.npy', partial17_0)
        np.save('./sample/patch/learned_part17_ch1.npy', partial17_1)
        np.save('./sample/patch/learned_part18_ch0.npy', partial18_0)
        np.save('./sample/patch/learned_part18_ch1.npy', partial18_1)
        np.save('./sample/patch/learned_part19_ch0.npy', partial19_0)
        np.save('./sample/patch/learned_part19_ch1.npy', partial19_1)
        np.save('./sample/patch/learned_part20_ch0.npy', partial20_0)
        np.save('./sample/patch/learned_part20_ch1.npy', partial20_1)


def batch_istft(patch0, patch1, m_ch0, m_ch1, ph0, ph1, n):  # patch0,1 channel 0,1의 np.ndarray 파일
    s, sr = librosa.load('./sample/wav_sample/if_there_was_practice_in_love.wav', sr=44100)
    print(sr, len(s))
    part6_0 = np.load('./sample/patch/stft_pt_ch0_part6.npy')
    part6_1 = np.load('./sample/patch/stft_pt_ch1_part6.npy')
    zero_out = np.zeros((patch0.shape[0], 1, 256, 4))
    output0 = np.concatenate((patch0, zero_out), axis=1)   # (1, 1025, 256, 4)
    output1 = np.concatenate((patch1, zero_out), axis=1)   # (1, 1025, 256, 4)
    zero_out = np.zeros((patch0.shape[0], 1, 256, 1))
    part6_0 = np.concatenate((part6_0, zero_out), axis=1)
    part6_1 = np.concatenate((part6_1, zero_out), axis=1)
    zero_out = np.zeros((output0.shape[0], 1, 256))
    print(m_ch0.shape)
    # m_ch0 = np.concatenate((m_ch0, zero_out), axis=1)       # (1, 1025, 256)
    # m_ch1 = np.concatenate((m_ch1, zero_out), axis=1)       # (1, 1025, 256)

    out0_mix = librosa.istft(m_ch0[0, :, :, 0]*ph0[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out0_drum = librosa.istft(output0[0, :, :, 0]*m_ch0[0, :, :, 0]*ph0[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out0_bass = librosa.istft(output0[0, :, :, 1]*m_ch0[0, :, :, 0]*ph0[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out0_other = librosa.istft(output0[0, :, :, 2]*m_ch0[0, :, :, 0]*ph0[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out0_vocal = librosa.istft(output0[0, :, :, 3]*m_ch1[0, :, :, 0]*ph0[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)

    out1_mix = librosa.istft(m_ch1[0, :, :, 0]*ph1[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out1_drum = librosa.istft(output1[0, :, :, 0]*m_ch1[0, :, :, 0]*ph1[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out1_bass = librosa.istft(output1[0, :, :, 1]*m_ch1[0, :, :, 0]*ph1[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out1_other = librosa.istft(output1[0, :, :, 2]*m_ch1[0, :, :, 0]*ph1[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    out1_vocal = librosa.istft(output1[0, :, :, 3]*m_ch1[0, :, :, 0]*ph1[0, :, :, 0], win_length=window_size, hop_length=hop_length, center=False)
    print(out0_other.shape)
    print(out1_other.shape)
    out0_mix = np.expand_dims(out0_mix, axis=-1)
    out0_drum = np.expand_dims(out0_drum, axis=-1)
    out0_bass = np.expand_dims(out0_bass, axis=-1)
    out0_other = np.expand_dims(out0_other, axis=-1)
    out0_vocal = np.expand_dims(out0_vocal, axis=-1)
    print(out0_drum.shape)
    print(out0_bass.shape)
    print(out0_other.shape)
    print(out0_vocal.shape)

    out1_mix = np.expand_dims(out1_mix, axis=-1)
    out1_drum = np.expand_dims(out1_drum, axis=-1)
    out1_bass = np.expand_dims(out1_bass, axis=-1)
    out1_other = np.expand_dims(out1_other, axis=-1)
    out1_vocal = np.expand_dims(out1_vocal, axis=-1)

    print(out0_other.shape)
    print(out1_other.shape)
    mix = np.transpose(np.concatenate((out0_mix, out1_mix), axis=-1))
    drum = np.transpose(np.concatenate((out0_drum, out1_drum), axis=-1))
    bass = np.transpose(np.concatenate((out0_bass, out1_bass), axis=-1))
    other = np.transpose(np.concatenate((out0_other, out1_other), axis=-1))
    vocal = np.transpose(np.concatenate((out0_vocal, out1_vocal), axis=-1))

    print('Final Shape:', vocal.shape)
    librosa.output.write_wav('./sample/mix_%d.wav' %n, 2*mix, sr)
    librosa.output.write_wav('./sample/drum_%d.wav' %n, 2*drum, sr)
    librosa.output.write_wav('./sample/bass_%d.wav' %n, 2*bass, sr)
    librosa.output.write_wav('./sample/other_%d.wav' %n, 2*other, sr)
    librosa.output.write_wav('./sample/vocal_%d.wav' %n, 2*vocal, sr)
    print("SAVED!")


# X_multi_channel = tf.concat((X, X, X, X), axis=-1)
# output = tf.multiply(unet, X_multi_channel)  # output shape = (?, 1024, 256, 4)
#
# with tf.Session() as sess:
#     output0 = sess.run(output, feed_dict={X: wb0})  # (?, 1024, 256, 4)
#     output1 = sess.run(output, feed_dict={X: wb1})  # (?, 1024, 256, 4)
# zero_out = np.zeros((output0.shape[0], 1, 256, 4))
# output0 = np.concatenate((output0, zero_out), axis=1)   # (?, 1025, 256, 4)
# output1 = np.concatenate((output1, zero_out), axis=1)   # (?, 1025, 256, 4)
# zero_out = np.zeros((output0.shape[0], 1, 256))
# m_ch0 = np.concatenate((m_ch0, zero_out), axis=1)       # (?, 1025, 256)
# m_ch1 = np.concatenate((m_ch1, zero_out), axis=1)       # (?, 1025, 256)
#
# out0_drum = librosa.istft(output0[0, :, :, 0]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)
# out0_bass = librosa.istft(output0[0, :, :, 1]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)
# out0_other = librosa.istft(output0[0, :, :, 2]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)
# out0_vocal = librosa.istft(output0[0, :, :, 3]*m_ch0[0], n_fft=window_size, hop_length=hop_length, center=False)
#
# out1_drum = librosa.istft(output1[0, :, :, 0]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)
# out1_bass = librosa.istft(output1[0, :, :, 1]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)
# out1_other = librosa.istft(output1[0, :, :, 2]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)
# out1_vocal = librosa.istft(output1[0, :, :, 3]*m_ch1[0], n_fft=window_size, hop_length=hop_length, center=False)
#
# for out_num in range(1, output0.shape[0]):
#     out0_drum = np.append(
#         (out0_drum, librosa.istft(output0[out_num, :, :, 0]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#     out0_bass = np.append(
#         (out0_bass, librosa.istft(output0[out_num, :, :, 1]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#     out0_other = np.append(
#         (out0_other, librosa.istft(output0[out_num, :, :, 2]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#     out0_vocal = np.append(
#         (out0_vocal, librosa.istft(output0[out_num, :, :, 3]*m_ch0[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#
#     out1_drum = np.append(
#         (out1_drum, librosa.istft(output0[out_num, :, :, 0]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#     out1_bass = np.append(
#         (out1_bass, librosa.istft(output0[out_num, :, :, 1]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#     out1_other = np.append(
#         (out1_other, librosa.istft(output0[out_num, :, :, 2]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#     out1_vocal = np.append(
#         (out1_vocal, librosa.istft(output0[out_num, :, :, 3]*m_ch1[out_num], n_fft=window_size, hop_length=hop_length, center=False)))
#
#
# out0_drum = np.expand_dims(out0_drum, axis=-1)
# out0_bass = np.expand_dims(out0_bass, axis=-1)
# out0_other = np.expand_dims(out0_other, axis=-1)
# out0_other = np.expand_dims(out0_other, axis=-1)
#
# out1_drum = np.expand_dims(out1_drum, axis=-1)
# out1_bass = np.expand_dims(out1_bass, axis=-1)
# out1_other = np.expand_dims(out1_other, axis=-1)
# out1_vocal = np.expand_dims(out1_vocal, axis=-1)
#
# drum = np.transpose(np.concatenate((out0_drum, out1_drum), axis=-1))
# bass = np.transpose(np.concatenate((out0_bass, out1_bass), axis=-1))
# other = np.transpose(np.concatenate((out0_other, out1_other), axis=-1))
# vocal = np.transpose(np.concatenate((out0_vocal, out1_vocal), axis=-1))
#
# librosa.output.write_wav('./wav_sample/path drum.wav', drum, sr)
# librosa.output.write_wav('./wav_sample/path bass.wav', bass, sr)
# librosa.output.write_wav('./wav_sample/path other.wav', other, sr)
# librosa.output.write_wav('./wav_sample/vocal.wav', vocal, sr)

run()
# p1 = np.load('./sample/patch/learned_part6_ch0.npy')
# p2 = np.load('./sample/patch/learned_part6_ch1.npy')
# m1 = np.load('./sample/patch/stft_pt_mag0_part6.npy')
# m2 = np.load('./sample/patch/stft_pt_mag1_part6.npy')
# phase0 = np.load('./sample/patch/stft_pt_phase0_part6.npy')
# phase1 = np.load('./sample/patch/stft_pt_phase1_part6.npy')
for asfds in range(20, 21):
    p1 = np.load('./sample/patch/learned_part%d_ch0.npy' %asfds)
    p2 = np.load('./sample/patch/learned_part%d_ch1.npy' %asfds)
    m1 = np.load('./sample/patch/stft_pt_mag0_part%d.npy' %asfds)
    m2 = np.load('./sample/patch/stft_pt_mag1_part%d.npy' %asfds)
    phase0 = np.load('./sample/patch/stft_pt_phase0_part%d.npy' %asfds)
    phase1 = np.load('./sample/patch/stft_pt_phase1_part%d.npy' %asfds)
    batch_istft(p1, p2, m1, m2, phase0, phase1, asfds)
