import os
import numpy as np
import itertools
import operator
import librosa

def lowpass(signal, fs, alpha):
    # alpha correction; originally determined for 44100 Hz
    alpha = (44100 * alpha) / fs
    lowpassSig = [0] * len(signal)
    lowpassSig[0] = abs(signal[0])
    abssigTimesAlpha = alpha * np.array(abs(signal))

    for i in range(1, len(abssigTimesAlpha)):
        lowpassSig[i] = (abssigTimesAlpha[i] + (1 - alpha) * lowpassSig[i - 1])
    return lowpassSig

def getSoundThresholdFromSignal(signal, fs, alpha, thresholdFactor):
    lpsig = lowpass(signal, fs, alpha)
    return thresholdFactor * np.max(lpsig)

def getSoundThresholdFromFile(filename, start, end, alpha, thresholdfactor, fsTarget=None):
    if not fsTarget:
        print('Using original Fs')
        signal, fsFile = librosa.load(filename, offset=start, duration=(end-start))
    else:
        signal, fsFile = librosa.load(filename, sr=fsTarget, offset=start, duration=(end-start))
    return getSoundThresholdFromSignal(signal, fsTarget, alpha, thresholdfactor)

def determineThresholds(microphones,
                        datasetDir,
                        fileDate,
                        fileNum,
                        silenceStart,
                        silenceEnd,
                        alpha,
                        thresholdfactor,
                        targetFs
                        ):
    thresholds = {}

    filename = datasetDir + '/{:d}_{:d}'.format(fileDate, fileNum) + '_mono{:d}.wav'

    for microphone in microphones:
        wavFileName = filename.format(microphone)
        soundThreshold = getSoundThresholdFromFile(filename=wavFileName,
                                                   start=silenceStart,
                                                   end=silenceEnd,
                                                   alpha=alpha,
                                                   thresholdfactor=thresholdfactor,
                                                   fsTarget=targetFs)
        thresholds[microphone] = soundThreshold

    return thresholds


def getSoundChunks(filename, start, end, soundThreshold, alpha, minimalSoundTime, targetFs):
    timeChunks = []

    signal, framerate = librosa.load(filename, sr=targetFs, offset=start, duration=(end - start))
    soundChunks = getSoundChunkIndices(signal=signal,
                                       fs=targetFs,
                                       threshold=soundThreshold,
                                       alpha=alpha)
    nr = 1

    for chunk in soundChunks:
        startTime = chunk[0] * 1.0 / framerate + start
        endTime = chunk[1] * 1.0 / framerate + start

        if endTime - startTime > minimalSoundTime:
            timeChunks.append([startTime, endTime])

    return timeChunks


def findAndSaveSoundChunks(fileparts,
                           dirName,
                           datasetDir,
                           fileDate,
                           thresholdFileNum,
                           thrSilenceStart,
                           thrSilenceEnd,
                           alpha,
                           thresholdfactor,
                           minimalSoundTime,
                           targetFs):
    """
    print soundchunks in json format
    store the printout in results/soundchunks.py
    :return:
    """
    microphones = [1, 2, 3, 4]

    # determine thresholds
    thresholds = determineThresholds(microphones, datasetDir=datasetDir,
                                     fileDate=fileDate, fileNum=thresholdFileNum,
                                     silenceStart=thrSilenceStart, silenceEnd=thrSilenceEnd,
                                     alpha=alpha,
                                     thresholdfactor=thresholdfactor,
                                     targetFs=targetFs)  # dictionary: int -> float

    dirName = dirName + '_Fs{}'.format(targetFs)
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    filename = dirName + '/soundChunks.py'
    # filename = 'dummy'  # to prevent accidental overwriting soundChunks
    with open(filename, 'w') as out:
        out.write('soundChunks = [\n')
        for filepart in fileparts:  # type: WavFilePart
            for microphone in microphones:
                filename = datasetDir + '/{:d}_{:d}_mono{:d}.wav'.format(fileDate, filepart.fileNr, microphone)
                chunks = getSoundChunks( filename=filename,
                                         start=filepart.getStartSecs(),
                                         end=filepart.getEndSecs(),
                                         soundThreshold=thresholds[microphone],
                                         alpha=alpha,
                                         minimalSoundTime=minimalSoundTime,
                                         targetFs=targetFs)
                filepart.setSoundChunks(microphone, chunks)
            out.write(filepart.toJSON())
            print('written ' + str(filepart))
            out.write(',\n')
        out.write(']\n')

    open(dirName + '/__init__.py', 'w').close()


def getSoundChunkIndices(signal, fs, alpha, threshold):
    lpSig = lowpass(signal, fs, alpha)
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



