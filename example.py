from keras.models import Input, Model
from keras.layers import Lambda
from keras_signal import Spectrogram,MelSpectrogram,MFCC,STFT,InverseSTFT

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

frame_length = 512
frame_step = frame_length//4
window = "hann"

rate,src = wavfile.read("test.wav")
src = src.reshape((1,-1))

# Power spectrogram
x = Input(shape=(None,))
y = Spectrogram(frame_length,frame_step)(x)
model_spec = Model(inputs=x,outputs=y)
spec = model_spec.predict(src)

print("Powerspec shape: ",spec.shape)
plt.imshow(np.log(spec[0].T))
plt.show()

# MelSpectrogram
x = Input(shape=(None,))
y = MelSpectrogram(frame_length,frame_step,num_mel_bins=40,num_spectrogram_bins=frame_length//2+1,sample_rate=rate)(x)
model_melspec = Model(inputs=x,outputs=y)
melspec = model_melspec.predict(src)
print("Melspec shape: ",melspec.shape)
plt.imshow(melspec[0].T)
plt.show()

# MelSpectrogram
x = Input(shape=(None,))
y = MFCC(frame_length,frame_step,num_mel_bins=40,num_spectrogram_bins=frame_length//2+1,sample_rate=rate)(x)
model_mfcc = Model(inputs=x,outputs=y)
mfcc = model_mfcc.predict(src)
print("MFCC shape: ",mfcc.shape)
plt.imshow(mfcc[0].T)
plt.show()

# STFT -ISTFT
x = Input(shape=(None,))
y = STFT(frame_length,frame_step,window_fn_type=window)(x)
model_stft = Model(inputs=x,outputs=y)
f = model_stft.predict(src)
print("STFT shape: ",f.shape)
x = Input(shape=(None,frame_length//2+1,2))
y = InverseSTFT(frame_length,frame_step,window_fn_type=window)(x)
model_istft = Model(inputs=x,outputs=y)
dst = model_istft.predict(f)
print("ISTFT shape: ",dst.shape)

# compare original and reconstructed signals.
plt.plot(src.flatten())
plt.plot(dst.flatten())
plt.show()


