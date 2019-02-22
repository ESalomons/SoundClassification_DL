import wave
import os
import numpy as np
import itertools
import operator
from util import WavFileParts

def lowpass(signal, alpha):
    lowpassSig = [0] * len(signal)
    lowpassSig[0] = abs(signal[0])
    abssigTimesAlpha = alpha * np.array(abs(signal))

    for i in range(1, len(abssigTimesAlpha)):
        lowpassSig[i] = (abssigTimesAlpha[i] + (1 - alpha) * lowpassSig[i - 1])
    return lowpassSig

def getSoundThresholdFromSignal(signal, alpha, thresholdFactor):
    lpsig = lowpass(signal, alpha)
    return thresholdFactor * np.max(lpsig)

def getSoundThresholdFromFile(filename, start, end, alpha, thresholdfactor, fsTarget=None):
    if not fsTarget:
        signal, fsFile = librosa.load(filename, offset=start, duration=(end-start))
    else:
        signal, fsFile = librosa.load(filename, sr=fsTarget, offset=start, duration=(end-start))
    return getSoundThresholdFromSignal(signal, alpha, thresholdfactor)

def determineThresholds(microphones,
                        datasetDir='/Volumes/SAA_DATA/datasets/localizationRecordings/20160919',
                        fileDate=160919,
                        fileNum=218,
                        silenceStart=1,
                        silenceEnd=5
                        ):
    thresholds = {}

    filename = datasetDir + '/{:d}_{:d}'.format(fileDate, fileNum) + '_mono{:d}.wav'

    for microphone in microphones:
        wavFileName = filename.format(microphone)
        soundThreshold = getSoundThresholdFromFile(wavFileName, silenceStart, silenceEnd)
        thresholds[microphone] = soundThreshold

    return thresholds


def getSoundChunks(filename, start, end, soundThreshold):
    timeChunks = []
    wavFile = wave.open(filename, 'rb')
    wavFile.rewind()
    nFrames = wavFile.getnframes()
    framerate = wavFile.getframerate()
    signal = np.fromstring(
        wavFile.readframes(int(end * framerate))[int(start * 2 * framerate):int(end * 2 * framerate)],
        np.int16)  # 2* framerate, because conversion int16
    wavFile.close()

    soundChunks = getSoundChunkIndices(signal, soundThreshold)
    nr = 1

    for chunk in soundChunks:
        startTime = chunk[0] * 1.0 / framerate + start
        endTime = chunk[1] * 1.0 / framerate + start

        if endTime - startTime > minimalSoundTime:
            timeChunks.append([startTime, endTime])

    return timeChunks


def findAndSaveSoundChunks(fileparts,
                           dirName='/Users/etto/Dropbox/git/wavLocalization/results',
                           datasetDir='/Volumes/SAA_DATA/datasets/localizationRecordings/20160919',
                           fileDate=160919,
                           thresholdFileNum=218,
                           thrSilenceStart=1,
                           thrSilenceEnd=5):
    """
    print soundchunks in json format
    store the printout in results/soundchunks.py
    :return:
    """
    microphones = [1, 2, 3, 4]

    # determine thresholds
    thresholds = determineThresholds(microphones, datasetDir=datasetDir,
                                     fileDate=fileDate, fileNum=thresholdFileNum,
                                     silenceStart=thrSilenceStart, silenceEnd=thrSilenceEnd
                                     )  # dictionary: int -> float

    
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    filename = dirName + '/soundChunks.py'
    # filename = 'dummy'  # to prevent accidental overwriting soundChunks
    with open(filename, 'w') as out:
        out.write('soundChunks = [\n')
        for filepart in fileparts:  # type: WavFilePart
            for microphone in microphones:
                filename = datasetDir + '/{:d}_{:d}_mono{:d}.wav'.format(fileDate, filepart.fileNr, microphone)
                chunks = getSoundChunks(filename, filepart.getStartSecs(), filepart.getEndSecs(),
                                                            thresholds[microphone])
                filepart.setSoundChunks(microphone, chunks)
            out.write(filepart.toJSON())
            print('written ' + str(filepart))
            out.write(',\n')
        out.write(']\n')

    open(dirName + '/__init__.py', 'w').close()


def getSoundChunkIndices(signal, alpha, threshold):
    lpSig = lowpass(signal, alpha)
    indices = np.array(range(len(signal)))
    indices = indices[lpSig > threshold]

    soundIndexList = []
    for k, g in itertools.groupby(enumerate(indices), lambda i_x: i_x[0] - i_x[1]):
        soundIndexList.append(list(map(operator.itemgetter(1), g)))

    chunkIndices = []

    for chunk in soundIndexList:
        if len(chunk) > 1:
            chunkIndices.append([chunk[0], chunk[-1]])

    return chunkIndices



