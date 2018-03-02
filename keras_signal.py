import functools
import keras
from keras.models import Model
from keras.layers import Input, Lambda
import tensorflow as tf 
from tensorflow.contrib.signal import frame,overlap_and_add,stft,inverse_stft
from tensorflow.contrib.signal import linear_to_mel_weight_matrix,hann_window,hamming_window,inverse_stft_window_fn,mfccs_from_log_mel_spectrograms

def choose_window_fn(window_fn_type):
    if window_fn_type == "hamming":
        return functools.partial(hamming_window, periodic=True)
    elif window_fn_type == "hann":
        return functools.partial(hann_window, periodic=True)
    elif window_fn_type is None:
        return None
    else:
        raise ValueError("window_fn must be \"hamming\" or \"hann\" or None")

def Frame(
    frame_length,
    frame_step,
    pad_end=False,
    pad_value=0):
    """
    Wrapper of tensorflow.contrib.signal.frame()

    # Arguments
        Same as tensorflow.contrib.signal.frame().
    # Input shape
        2D tensor with shape: `(batch_size, steps)`
    # Output shape
        3D tensor with shape: `(batch_size, frames, frame_length)`
    """
    def tf_frame(x):
        f = tf.contrib.signal.frame(x,frame_length,frame_step,pad_end,pad_value)
        return f
    return Lambda(tf_frame)

def Unframe(frame_step):
    """
    Wrapper of tensorflow.contrib.signal.overlap_and_add()
    NOTE: this layer automatically scale outputs for accurate recostruction.

    # Arguments
        Same as tensorflow.contrib.signal.overlap_and_add().
    # Input shape
        3D tensor with shape: `(batch_size, frames, frame_length)`
    # Output shape
        2D tensor with shape: `(batch_size, steps)`
    """
    def tf_unframe(x):
        f = tf.contrib.signal.overlap_and_add(x,frame_step)
        f *= frame_step/x.shape[-1].value
        return f
    return Lambda(tf_unframe)

def STFT(
    frame_length,
    frame_step,
    fft_length=None,
    window_fn_type="hann",
    pad_end=False):
    """
    Wrapper of tensorflow.contrib.signal.stft()

    # Arguments
        Same as tensorflow.contrib.signal.stft().
    # Input shape
        2D tensor with shape: `(batch_size, steps)`
    # Output shape
        4D tensor with shape: `(batch_size, frames, fft_length//2+1,2)`
        Last dimension is concatenation of real/imaginery part.
    """
    def tf_stft(x):
        window_fn = choose_window_fn(window_fn_type)
        fc = stft(x,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length,window_fn=window_fn,pad_end=pad_end)
        f_real = tf.real(fc)
        f_imag = tf.imag(fc)
        f = tf.stack([f_real,f_imag],-1)
        return f
    return Lambda(tf_stft)

def InverseSTFT(
    frame_length,
    frame_step,
    fft_length=None,
    window_fn_type="hann"):
    """
    Wrapper of tensorflow.contrib.signal.inverse_stft()
    NOTE: this layer automatically applies inverse_stft_window_fn() for accurate recostruction.

    # Arguments
        Same as tensorflow.contrib.signal.inverse_stft().
    # Input shape
        4D tensor with shape: `(batch_size, frames, fft_length//2+1,2)`
        Last dimension is concatenation of real/imaginery part.
    # Output shape
        2D tensor with shape: `(batch_size, steps)`
    """
    def tf_inverse_stft(f):
        fc = tf.complex(*tf.unstack(f,axis=-1))
        if window_fn_type is None:
            window_fn = None
            x = inverse_stft(fc,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length,window_fn=window_fn)
            x *= frame_step/frame_length
        else:
            window_fn = inverse_stft_window_fn(frame_step,choose_window_fn(window_fn_type))
            x = inverse_stft(fc,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length,window_fn=window_fn)
        return x
    return Lambda(tf_inverse_stft)

def Spectrogram(
    frame_length,
    frame_step,
    fft_length=None,
    window_fn_type="hann",
    pad_end=False,
    power=2.0):
    """
    Calculates spectrogram.

    # Arguments
        power: Exponent for magnitude calculation.
        Other arguments are same as tensorflow.contrib.signal.stft().
    # Input shape
        2D tensor with shape: `(batch_size, steps)`
    # Output shape
        3D tensor with shape: `(batch_size, frames, fft_length//2+1)`
    """
    def tf_spectrogram(x):
        window_fn = choose_window_fn(window_fn_type)
        fc = stft(x,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length,window_fn=window_fn,pad_end=pad_end)
        f = tf.abs(fc)**power
        return f
    return Lambda(tf_spectrogram)

def MelSpectrogram(
    frame_length,
    frame_step,
    fft_length=None,
    window_fn_type="hann",
    pad_end=False,
    num_mel_bins=20,
    num_spectrogram_bins=129,
    sample_rate=8000,
    lower_edge_hertz=125.0,
    upper_edge_hertz=3800.0,
    power=2.0):
    """
    Calculates Mel spectrogram.
    Mel filterbank is defined by tf.contrib.signal.linear_to_mel_weight_matrix().
    NOTE:Implementation of mel filterbank may differ from other libraries ( e.g. librosa/tensorflow uses interpolation on linear/mel domain ).

    # Arguments
        power: Exponent for magnitude calculation.
        Other arguments are same as tensorflow.contrib.signal.stft().
    # Input shape
        2D tensor with shape: `(batch_size, steps)`
    # Output shape
        3D tensor with shape: `(batch_size, frames, num_mel_bins)`
    """
    def tf_melspectrogram(x):
        window_fn = choose_window_fn(window_fn_type)
        fc = stft(x,frame_length=frame_length,frame_step=frame_step,fft_length=fft_length,window_fn=window_fn,pad_end=pad_end)
        f = tf.abs(fc)**power
        w = linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=upper_edge_hertz)
        r = tf.tensordot(f,w,1)
        r.set_shape(f.shape[:-1].concatenate(num_mel_bins))
        return r
    return Lambda(tf_melspectrogram)

def MFCC(frame_length, frame_step, **kwargs):
    """
    Calculates MFCC.
    Mel filterbank is defined by tf.contrib.signal.linear_to_mel_weight_matrix().
    NOTE:Implementation of mel filterbank may differ from other libraries ( e.g. librosa/tensorflow uses interpolation on linear/mel domain ).

    # Arguments
        Same as MelSpectrogram().
    # Input shape
        2D tensor with shape: `(batch_size, steps)`
    # Output shape
        3D tensor with shape: `(batch_size, frames, mfcc_bins)`
    """
    def tf_mfcc(x):
        mel_spectrograms = MelSpectrogram(frame_length, frame_step, **kwargs)(x)
        log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)
        mfccs = mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
        return mfccs
    return Lambda(tf_mfcc)
