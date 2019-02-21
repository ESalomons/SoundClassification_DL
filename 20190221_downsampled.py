
import util.util_20190221 as nb_util



classes = ['music', 'voice', 'environment']

macDir = '/Volumes/SAA_DATA/datasets/'
winDir = 'E:/SAA_DATA/'
osDir = winDir
recordingDir = osDir + '/localizationRecordings'

if osDir == winDir:
    storageFolder = 'E:/SAA_DATA/storedData/'
else:
    storageFolder = '/Users/etto/Desktop/storedData/'

baseSrcDir = osDir + 'localizationFiles/20171025AllExtractionsMic4'
orgWavDirs1 = ['G428_0.0_1.4',
              'G527_0.5_1.4',
              'Studio_2.0_4.2'
              ]

orgWavDirs2 = ['G428_2.1_2.4',
              'G527_1.2_5.8',
              'Studio_3.0_2.0'
              ]

orgsG428 = ['G428_0.0_1.4','G428_2.1_2.4']
orgsG527 = ['G527_0.5_1.4','G527_1.2_5.8']
orgsStudio = ['Studio_2.0_4.2','Studio_3.0_2.0']

chunksBaseDir = 'chunks'
rooms = ['Studio', 'G428', 'G527']


# # Centrale parameters: targetFreq, windowLength / NFFT

# In[3]:


targetFreq = 8000
windowLength = 0.032 
NFFT = 256


# ### trainen van model obv: alles G428, G527 en Studio

# In[4]:


# Train voor alle lagen
baseModelFilename = '20190221_fs_{}_NFFT{}'.format(targetFreq, NFFT)
modelFilePath = storageFolder
logPrefix = 'Alle orgs'
orgWavDirs = orgsG428 + orgsG527 + orgsStudio
orgWavDirs = ['G428_2.1_2.4']


# In[6]:


X_train, Y_train, realTrainClasses = nb_util.getTrainDataFromFolders(orgWavDirs, targetFreq,
                                                                     NFFT, baseSrcDir, classes)


# In[ ]:


# np.shape(X_train)
#
# # Maak test data
# LOG_HEADER('maak spectra voor testdata G428, NFFT={}'.format(NFFT), True)
# wvPts = readSoundChunksDynamic('chunks.G428.soundChunks')
# baseDir = recordingDir + '/20170221'
# fileDate = 170221
# filename = 'testData_G428'
# for micNr in [1,2,3,4]:
#     keyname = 'mic{}'.format(micNr)
#     createAndStoreTestData(wvPts, baseDir, fileDate, micNr, filename, NFFT, keyname)
#
# LOG('Klaar: spectra voor testdata G428', True)# Maak test data
# LOG_HEADER('maak spectra voor testdata G527, NFFT={}'.format(NFFT), True)
# wvPts = readSoundChunksDynamic('chunks.G527.soundChunks')
# baseDir = recordingDir + '/20170221'
# fileDate = 170221
# filename = 'testData_G527'
# for micNr in [1,2,3,4]:
#     keyname = 'mic{}'.format(micNr)
#     createAndStoreTestData(wvPts, baseDir, fileDate, micNr, filename, NFFT, keyname)
#
# LOG('Klaar: spectra voor testdata G527', True)LOG_HEADER('maak spectra voor testdata Studio, NFFT={}'.format(NFFT), True)
# wvPts = readSoundChunksDynamic('chunks.Studio.soundChunks')
# baseDir = recordingDir + '/20171011'
# fileDate = 170816
# filename = 'testData_Studio'
# for micNr in [1,2,3,4]:
#     keyname = 'mic{}'.format(micNr)
#     createAndStoreTestData(wvPts, baseDir, fileDate, micNr, filename, NFFT, keyname)
#
# LOG('Klaar: spectra voor testdata Studio', True)layersList = [[100, 20], [400, 250, 100, 20], [400, 300, 200, 100, 50, 20, 10],
#               [450, 400, 350, 300, 250, 200, 150, 100, 50, 21]]
# nrEpochs = 5
# nrsEpochs = range(1, nrEpochs + 1)
# #nrsEpochs = [nrEpochs]
# bSize = 128for layers in layersList:
#     LOG_HEADER(logPrefix + ', lagen: {}'.format(str(layers)), True)
#     train_and_evaluate_per_epoch(X_train, Y_train, realTrainClasses,
#                                  layers, nrEpochs, modelFilePath, baseModelFilename, NFFT,
#                                 batch_size=bSize)# Test alle modellen
# fileName='plots/20190103_NFFT{}.txt'.format(NFFT)
# with open(fileName, 'w') as f:
#     with redirect_stdout(f):
#         testModellen(modelFilePath=storageFolder,
#                      layerss=layersList,
#                      NFFT=NFFT,
#                      micNr=4,
#                      testFileNames=['testData_G428', 'testData_G527', 'testData_Studio'],
#                      showOverall=True)                from parseExperimentResults import parseRedoExperimentexperimentId='20190103_NFFT{}'.format(NFFT)
# filePattern='20190103_NFFT{}'.format(NFFT)
# parseRedoExperiment(experimentId, filePattern)