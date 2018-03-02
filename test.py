import unittest
import keras
import librosa
import numpy as np
from matplotlib import pyplot as plt
from keras_signal import Frame,Unframe,STFT,InverseSTFT,MelSpectrogram,Spectrogram

class TestSTFT(unittest.TestCase):
    def setUp(self):
        self.length = 512*2000
        self.frame_length = 512
        self.frame_step = self.frame_length//2
        self.src,self.rate = librosa.load(librosa.util.example_audio_file())
        self.src = self.src[:self.length].reshape((1,-1))

        self.eps = 1e-6

    def testFrame(self):
        # keras
        x = keras.layers.Input(shape=(None,))
        y = Frame(self.frame_length,self.frame_step)(x)
        model = keras.Model(inputs=x,outputs=y)
        dst_keras = model.predict(self.src)
        dst_keras = dst_keras[0]

        # librosa
        dst_librosa = librosa.util.frame(self.src.flatten(),frame_length=self.frame_length,hop_length=self.frame_step)
        dst_librosa = np.transpose(dst_librosa,(1,0))

        # compare
        self.assertEqual(dst_keras.shape,dst_librosa.shape)
        m = min( np.mean(np.abs(dst_keras)),np.mean(np.abs(dst_librosa)) )
        self.assertLess(np.mean(np.abs(dst_keras-dst_librosa)),m*self.eps)

    def testUnframeReconstruction(self):
        # Frame model
        x = keras.layers.Input(shape=(None,))
        y = Frame(self.frame_length,self.frame_step,pad_end=True)(x)
        model_frame = keras.Model(inputs=x,outputs=y)

        # Unframe model
        x = keras.layers.Input(shape=(None,self.frame_length))
        y = Unframe(self.frame_step)(x)
        model_unframe = keras.Model(inputs=x,outputs=y)

        # reconstruct
        f = model_frame.predict(self.src)
        dst = model_unframe.predict(f)

        self.assertLess(np.mean(np.abs(self.src-dst[:,:self.length])),self.eps)

    def testSTFT(self):
        # keras
        x = keras.layers.Input(shape=(None,))
        y = STFT(self.frame_length,self.frame_step)(x)
        model = keras.Model(inputs=x,outputs=y)
        dst_keras = model.predict(self.src)
        dst_keras = dst_keras[0]

        # librosa
        dst_librosa = librosa.stft(self.src.flatten(),n_fft=self.frame_length,hop_length=self.frame_step,center=False)
        dst_librosa = np.transpose(dst_librosa,(1,0))
        dst_librosa = np.stack([np.real(dst_librosa),-np.imag(dst_librosa)],axis=-1)
        
        # compare
        self.assertEqual(dst_keras.shape,dst_librosa.shape)
        m = min( np.mean(np.abs(dst_keras)),np.mean(np.abs(dst_librosa)) )
        self.assertLess(np.mean(np.abs(dst_keras-dst_librosa)),m*self.eps)

    def testSpectrogram(self):
        # keras
        x = keras.layers.Input(shape=(None,))
        y = Spectrogram(self.frame_length,self.frame_step)(x)
        model = keras.Model(inputs=x,outputs=y)
        dst_keras = model.predict(self.src)
        dst_keras = dst_keras[0]

        # librosa
        dst_librosa = librosa.stft(self.src.flatten(),n_fft=self.frame_length,hop_length=self.frame_step,center=False)
        dst_librosa = np.absolute(dst_librosa)**2
        dst_librosa = np.transpose(dst_librosa,(1,0))

        # compare
        self.assertEqual(dst_keras.shape,dst_librosa.shape)
        m = min( np.mean(np.abs(dst_keras)),np.mean(np.abs(dst_librosa)) )
        self.assertLess(np.mean(np.abs(dst_keras-dst_librosa)),m*self.eps)

    ### NOTE:implementation of mel filterbank is different between librosa and tensorflow. 
    ### librosa/tensorflow uses interpolation on frequency/mel domain.
    # def testMelSpectrogram(self):
    #     nmels = 20
    #     fmin = 100
    #     fmax = 4000

    #     # keras
    #     x = keras.layers.Input(shape=(None,))
    #     y = MelSpectrogram(\
    #         self.frame_length,self.frame_step,\
    #         num_mel_bins=nmels,num_spectrogram_bins=self.frame_length//2+1,sample_rate=self.rate,\
    #         lower_edge_hertz=fmin,upper_edge_hertz=fmax)(x)
    #     model = keras.Model(inputs=x,outputs=y)
    #     dst_keras = model.predict(self.src)
    #     dst_keras = dst_keras[0]

    #     # librosa
    #     s = librosa.stft(self.src.flatten(),n_fft=self.frame_length,hop_length=self.frame_step,center=False)
    #     s = np.absolute(s)**2
    #     dst_librosa = librosa.feature.melspectrogram(\
    #         None,S=s,sr=self.rate,n_fft=self.frame_length,hop_length=self.frame_step,\
    #         n_mels=nmels,fmin=fmin,fmax=fmax,htk=True,norm=None)
    #     dst_librosa = np.transpose(dst_librosa,(1,0))

    #     # compare
    #     self.assertEqual(dst_keras.shape,dst_librosa.shape)
    #     m = min( np.mean(np.abs(dst_keras)),np.mean(np.abs(dst_librosa)) )
    #     self.assertLess(np.mean(np.abs(dst_keras-dst_librosa)),m*self.eps)

    def testInverseSTFTReconstruction(self):
        for window in ["hann","hamming",None]:
            # STFT model
            x = keras.layers.Input(shape=(None,))
            y = STFT(self.frame_length,self.frame_step,window_fn_type=window,pad_end=True)(x)
            model_stft = keras.Model(inputs=x,outputs=y)

            # ISTFT model
            x = keras.layers.Input(shape=(None,self.frame_length//2+1,2))
            y = InverseSTFT(self.frame_length,self.frame_step,window_fn_type=window)(x)
            model_istft = keras.Model(inputs=x,outputs=y)

            # reconstruct
            f = model_stft.predict(self.src)
            dst = model_istft.predict(f)

            self.assertLess(np.mean(np.abs(self.src-dst[:,:self.length])),self.eps)

if __name__=="__main__":
    print("Not testing for MelSpectrogram(),MFCC().")
    unittest.main()