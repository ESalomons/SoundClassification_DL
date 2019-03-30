from util.soundchunk_util_20190222 import findAndSaveSoundChunks
import os
import util.WavFileParts as WavFileParts

alpha = 0.005
thresholdFactor = 1
minimalSoundTime = 0.15
targetFs = 44100

osRecordingsDir = '/Volumes/SAA_DATA/datasets/localizationRecordings/'
osRecordingsDir = 'E:/SAA_DATA/localizationRecordings/'

scTargetBase = 'chunks'
if not os.path.isdir(scTargetBase):
    os.mkdir(scTargetBase)

dirName = scTargetBase + '/G527'
datasetDir = osRecordingsDir + '20170221'
fileDate = 170221
thresholdFileNum = 551
thrSilenceStart = 10
thrSilenceEnd = 15

wavFileParts = WavFileParts.getWavFileParts20170221_G5_27()
findAndSaveSoundChunks(wavFileParts,
                       dirName,
                       datasetDir,
                       fileDate,
                       thresholdFileNum,
                       thrSilenceStart,
                       thrSilenceEnd,
                       alpha=alpha,
                       thresholdfactor=thresholdFactor,
                       minimalSoundTime=minimalSoundTime,
                       targetFs=targetFs)