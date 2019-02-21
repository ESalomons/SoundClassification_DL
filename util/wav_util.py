import scipy.io.wavfile as wv
import signal as sig
import numpy as np

def createDownsampledFile(srcFile, targetFileName, fsTarget):
    fsOrg, signal = wv.read(filename)
    dsSignal = getDownsampledSignal(signal, fsOrg, fsTarget)
    

def getDownsampledSignal(signal, fsOrg, fsTarget):
    lpSignal = applyLowpassFilter(signal, int(fsTarget / 2), fsOrg)
    print('lowpassed')
    numSamples = int(fsTarget * 1.0 * len(lpSignal) / fsOrg)
    print('numsamples: {}'.format(numSamples))
    dsSignal = np.array(sig.resample(lpSignal, numSamples))
    print('resampled')
    return dsSignal

def applyLowpassFilter(signal, hiFrq, fs):
    b, a = sig.butter(2, 1.0 * hiFrq / fs)
    return sig.lfilter(b, a, signal)