import matplotlib.pyplot as plt
import os
from util import createPlotHtml


def doMain():
#     doMainRegular()
    parseRedoExperiment()
        
def doMainRegular():
    # fileName = 'plots/20180817_train_Studio.txt'
    # pattern = '# 20180817_orgStudio'
    experimentId = '20180822_StudioModel_norm'

    fileName = 'plots/{}.txt'.format(experimentId)
    pattern = '# ' + experimentId

    targetPlotFolder = 'plots/' + pattern[2:] + '/'
    if not os.path.isdir(targetPlotFolder):
        os.mkdir(targetPlotFolder)

    print('creating plots in ' + targetPlotFolder)

    rooms = []
    features = ['F1 overall', 'F1 music', 'F1 voice', 'F1 environment']

    for feature in features:
        roomExperiments = {
            'G428': {},
            'G527': {},
            'Studio': {},
            'Overall': {}
        }
        room = ''

        with open(fileName) as resFile:
            expName = ''
            epochNr = 0
            for line in resFile:
                if 'testData_' in line:
                    room = line.split('_')[1][:-1]
                elif pattern in line:
                    splt = line.split('ep')
                    expName = splt[0][2:]
                    epochNr = int(splt[1])
                elif feature in line:
                    if not expName in roomExperiments[room]:
                        roomExperiments[room][expName] = []
                    FValue = float(line.split(': ')[1])
                    roomExperiments[room][expName].append((epochNr, FValue))

        for room in roomExperiments:
            experiments = roomExperiments[room]
            plt.figure()
            for exp in experiments:
                values = sorted(experiments[exp], key=lambda tup: tup[0])
                unzipped = list(zip(*values))

                plt.plot(unzipped[0], unzipped[1], label=exp)

            plt.title('Room: ' + room)
            plt.ylabel(feature)
            plt.xlabel('epochs')
            plt.ylim([0.8, 1])
            plt.legend()

            figname = targetPlotFolder + pattern[2:] + '_' + room + '_' + feature + '.png'
            plt.savefig(figname, bbox_inches='tight', dpi=200)  # 600

    createPlotHtml.createPlotsInDir('plots/' + experimentId)
        
def parseRedoExperiment(experimentId='20180720_allOrgs',filePattern='20180823_redoAllOrgs'):
    fileName = 'plots/{}.txt'.format(filePattern)
    pattern = '# ' + experimentId

    targetPlotFolder = 'plots/{}/'.format(filePattern)
    if not os.path.isdir(targetPlotFolder):
        os.mkdir(targetPlotFolder)

    print('creating plots in ' + targetPlotFolder)

    rooms = []
    features = ['F1 overall', 'F1 music', 'F1 voice', 'F1 environment']

    for feature in features:
        roomExperiments = {
            'G428': {},
            'G527': {},
            'Studio': {},
            'Overall': {}
        }
        room = ''

        with open(fileName) as resFile:
            expName = ''
            epochNr = 0
            for line in resFile:
                if 'testData_' in line:
                    room = line.split('_')[1][:-1]
                elif pattern in line:
                    splt = line.split('ep')
                    expName = splt[0][2:]
                    epochNr = int(splt[1])
                elif feature in line:
                    if not expName in roomExperiments[room]:
                        roomExperiments[room][expName] = []
                    FValue = float(line.split(': ')[1])
                    roomExperiments[room][expName].append((epochNr, FValue))

        for room in roomExperiments:
            experiments = roomExperiments[room]
            plt.figure()
            for exp in experiments:
                values = sorted(experiments[exp], key=lambda tup: tup[0])
                unzipped = list(zip(*values))

                plt.plot(unzipped[0], unzipped[1], label=exp)

            plt.title('Room: ' + room)
            plt.ylabel(feature)
            plt.xlabel('epochs')
            plt.ylim([0.8, 1])
            plt.legend()

            figname = targetPlotFolder + pattern[2:] + '_' + room + '_' + feature + '.png'
            plt.savefig(figname, bbox_inches='tight', dpi=200)  # 600

    createPlotHtml.createPlotsInDir(targetPlotFolder[:-1])
    
def version1():
    fileFolder = '/Users/etto/Desktop/trainResults/'
    pattern = '# 20180720_allOrgs'
    targetPlotFolder = 'plots/' + pattern[2:] + '/'
    if not os.path.isdir(targetPlotFolder):
        os.mkdir(targetPlotFolder)

    rooms = ['G428', 'G527', 'Studio']
    features = ['F1 overall', 'F1 music', 'F1 voice', 'F1 environment']

    for room in rooms:
        for feature in features:
            fileNameShort = 'allOrgs{}.txt'.format(room)
            fileName = fileFolder + fileNameShort

            experiments = {}
            with open(fileName) as resFile:
                expName = ''
                epochNr = 0
                for line in resFile:
                    if pattern in line:
                        splt = line.split('ep')
                        expName = splt[0][2:]
                        epochNr = int(splt[1])
                        if not expName in experiments:
                            experiments[expName] = []
                    if feature in line:
                        FValue = float(line.split(': ')[1])
                        experiments[expName].append((epochNr, FValue))

            plt.figure()
            for exp in experiments:
                values = sorted(experiments[exp], key=lambda tup:tup[0])
                unzipped = list(zip(*values))

                plt.plot(unzipped[0], unzipped[1], label=exp)

            plt.title('Room: ' + room)
            plt.ylabel(feature)
            plt.xlabel('epochs')
            plt.ylim([0.85, 1])
            plt.legend()

            figname = targetPlotFolder + pattern[2:] + '_' + room + '_' + feature + '.png'
            plt.savefig(figname, bbox_inches='tight', dpi=200)  # 600
# plt.show()

if __name__ == '__main__':
    doMain()
