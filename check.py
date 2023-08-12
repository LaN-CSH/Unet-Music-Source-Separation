import numpy as np
import librosa

d, srd = librosa.load('./sample/drum.wav', sr=44100)
b, srb = librosa.load('./sample/bass.wav', sr=44100)
o, sro = librosa.load('./sample/other.wav', sr=44100)
v, srv = librosa.load('./sample/vocal.wav', sr=44100)
print(d)
print(b)
print(o)
print(v)
print(d.shape)
print(b.shape)
print(o.shape)
print(v.shape)
