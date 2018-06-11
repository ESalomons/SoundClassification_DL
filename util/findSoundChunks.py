from util import WavFileParts
from util import classifySoundChunks

def findSoundChunks20180609(osRecordingsDir='/Volumes/SAA_DATA/datasets/localizationRecordings/'):
    defsWavLocalization.alpha = 0.005
    defsWavLocalization.thresholdFactor = 1
    defsWavLocalization.minimalSoundTime = 0.15

    defsWavLocalization.correlatedTimeDiffThreshold = 0.025  # ~8.5 m
    defsWavLocalization.onlySkipOldestPointForCorrelation = False

    scTargetBase = 'results'
    if not os.path.isdir(scTargetBase):
        os.mkdir(scTargetBase)

    dirName = scTargetBase + '/Studio'
    datasetDir = osRecordingsDir + '20171011'
    fileDate = 170816
    thresholdFileNum = 745
    thrSilenceStart = 1
    thrSilenceEnd = 2.5

    wavFileParts = WavFileParts.getWavFileParts20171014_Studio()
    classifySoundChunks.findAndSaveSoundChunks(wavFileParts,
                                               dirName,
                                               datasetDir,
                                               fileDate,
                                               thresholdFileNum,
                                               thrSilenceStart,
                                               thrSilenceEnd)


    dirName = scTargetBase + '/G428'
    datasetDir = '/Volumes/SAA_DATA/datasets/localizationRecordings/20170221'
    fileDate = 170221
    thresholdFileNum = 542
    thrSilenceStart = 14
    thrSilenceEnd = 17
    wavFileParts = WavFileParts.getWavFileParts20170221_G4_28()

    classifySoundChunks.findAndSaveSoundChunks(wavFileParts,
                                               dirName,
                                               datasetDir,
                                               fileDate,
                                               thresholdFileNum,
                                               thrSilenceStart,
                                               thrSilenceEnd)

    dirName = scTargetBase + '/G527'
    datasetDir = '/Volumes/SAA_DATA/datasets/localizationRecordings/20170221'
    fileDate = 170221
    thresholdFileNum = 551
    thrSilenceStart = 10
    thrSilenceEnd = 15

    wavFileParts = WavFileParts.getWavFileParts20170221_G5_27()
    classifySoundChunks.findAndSaveSoundChunks(wavFileParts,
                                               dirName,
                                               datasetDir,
                                               fileDate,
                                               thresholdFileNum,
                                               thrSilenceStart,
                                               thrSilenceEnd)
    
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
                chunks = identifySoundChunks.getSoundChunks(filename, filepart.getStartSecs(), filepart.getEndSecs(),
                                                            thresholds[microphone])
                filepart.setSoundChunks(microphone, chunks)
            out.write(filepart.toJSON())
            print('written ' + str(filepart))
            out.write(',\n')
        out.write(']\n')

    open(dirName + '/__init__.py', 'w').close()