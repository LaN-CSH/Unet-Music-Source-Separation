import numpy as np
import librosa


inst = ['drum', 'bass', 'other', 'vocal', 'mix']
# for inst_name in inst:
#     # a, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 9), sr=44100)
#     # b, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 10), sr=44100)
#     c, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 14), sr=44100)
#     d, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 15), sr=44100)
#     e, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 16), sr=44100)
#     f, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 17), sr=44100)
#     g, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 18), sr=44100)
#     h, sr = librosa.load('./sample/%s_%d.wav' %(inst_name, 19), sr=44100)
#     print(type(c), type(d))
#     z = np.concatenate((c, d, e, f, g, h))
#     librosa.output.write_wav('./sample/agami_%s_14-19.wav'%(inst_name), z, sr=44100)

# demo
c, sr = librosa.load('./sample/%s_%d.wav' %(inst[4], 14), sr=44100)
d, sr = librosa.load('./sample/%s_%d.wav' %(inst[0], 15), sr=44100)
e, sr = librosa.load('./sample/%s_%d.wav' %(inst[1], 16), sr=44100)
f, sr = librosa.load('./sample/%s_%d.wav' %(inst[2], 17), sr=44100)
g, sr = librosa.load('./sample/%s_%d.wav' %(inst[3], 18), sr=44100)
h, sr = librosa.load('./sample/%s_%d.wav' %(inst[4], 19), sr=44100)
print(type(c), type(d))
z = np.concatenate((c, d, e, f, g, h))
librosa.output.write_wav('./sample/agami_demo_14-19.wav', z, sr=44100)
