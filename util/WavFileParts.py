import json

class WavFilePart(object):
    def __init__(self, fileNr, soundType, locX, locY, startMin, startSec, endMin, endSec):
        self.fileNr = fileNr
        self.soundType = soundType
        self.locX = locX
        self.locY = locY
        self.soundBegin = (startMin, startSec)
        self.soundEnd = (endMin, endSec)
        self.sound_chunks = {}
        self.sound_classifications = {1: [], 2: [], 3: [], 4: []}

    def __str__(self):
        return 'File: {:d} - {:s} : ({:.2f}, {:.2f})'.format(self.fileNr, self.soundType, self.locX, self.locY)

    def setSoundChunks(self, micNr, chunks):
        self.sound_chunks[micNr] = chunks

    def setClassifications(self, micNr, classifications):
        self.sound_classifications[micNr] = classifications

    def getSoundChunks(self, micNr):
        return self.sound_chunks[micNr]

    def getClassification(self, micNr):
        return self.sound_classifications[micNr]

    def getSoundType(self):
        return self.soundType

    def getStartSecs(self):
        return 60 * self.soundBegin[0] + self.soundBegin[1]

    def getEndSecs(self):
        return 60 * self.soundEnd[0] + self.soundEnd[1]

    def toJSON(self):
        # return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        return json.dumps(self.__dict__, sort_keys=True)

    def addClassification(self, micNr, prediction):
        self.sound_classifications[micNr].append(prediction)




def doMain():
    parts = getWavFileParts()
    # for part in parts:
    #     print(part.toJSON())
    jsonString = {"end": [5, 1], "soundChunks": {}, "start": [2, 56], "soundType": "Voice", "locX": 2, "locY": 0,
                  "fileNr": 218}
    print(WavFilePartFromJson(jsonString).toJSON())


def WavFilePartFromJson(jsonString):
    wfp = WavFilePart(jsonString["fileNr"], jsonString["soundType"], jsonString["locX"], jsonString["locY"],
                      jsonString["soundBegin"][0], jsonString["soundBegin"][1], jsonString["soundEnd"][0],
                      jsonString["soundEnd"][1])
    soundChunks = jsonString["sound_chunks"]
    classifications = jsonString["sound_classifications"]
    for micNr in soundChunks:
        wfp.setSoundChunks(int(micNr), soundChunks[micNr])
        wfp.setClassifications(int(micNr), classifications[micNr])
    return wfp


def getWavFilePartsShort():
    parts = []
    filenr = 219
    locX = 3
    locY = 2
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 0, 12, 2, 8))

    return parts


def getWavFilePartsWithGunshots():
    return getWavFilePartsGunshotsOnly() + getWavFileParts()


def getWavFilePartsGunshotsOnly():
    parts = []
    filenr = 218
    parts.append(WavFilePart(filenr, 'Gunshot', 1, 2.5, 0, 24, 0, 51))
    parts.append(WavFilePart(filenr, 'Gunshot', 3, 2, 1, 4, 1, 20))
    parts.append(WavFilePart(filenr, 'Gunshot', 3.5, 4, 1, 31, 1, 48))
    parts.append(WavFilePart(filenr, 'Gunshot', 2, 0, 2, 1, 2, 25))
    return parts


def getWavFileParts():
    parts = []
    filenr = 218
    locX = 2
    locY = 0
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 56, 5, 1))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 5, 6, 7, 21))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 8, 29, 10, 44))

    filenr = 219
    locX = 3
    locY = 2
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 0, 12, 2, 8))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 2, 25, 5, 23))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 5, 30, 7, 36))

    filenr = 220
    locX = 1
    locY = 2.5
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 1, 0, 3, 0))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 3, 10, 4, 52))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 5, 1, 6, 55))

    filenr = 221
    locX = 3.5
    locY = 5
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 0, 10, 2, 13))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 2, 33, 4, 27))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 4, 41, 6, 25))

    filenr = 222
    locX = 0
    locY = 0
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 0, 5, 2, 26))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 2, 34, 4, 18))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 4, 26, 7, 11))

    return parts

    #### parts with noise


def getWavFilePartsWithNoise():
    parts = []
    # filenr = 223
    # soundType = 'Gunshots'
    # locX = 2
    # locY = 0
    # start = (0, 1)
    # end = (0, 16)
    # parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))
    #
    # filenr = 223
    # soundType = 'Gunshots'
    # locX = 3
    # locY = 2
    # start = (0, 26)
    # end = (0, 42)
    # parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))
    #
    # filenr = 223
    # soundType = 'Gunshots'
    # locX = 3.5
    # locY = 5
    # start = (0, 49)
    # end = (1, 4)
    # parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))

    filenr = 224
    soundType = 'Voice'
    locX = 3.5
    locY = 5
    start = (0, 4)
    end = (0, 48)
    parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))

    filenr = 224
    soundType = 'Voice'
    locX = 3
    locY = 2
    start = (1, 12)
    end = (1, 44)
    parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))

    filenr = 224
    soundType = 'Voice'
    locX = 2
    locY = 0
    start = (1, 53)
    end = (2, 38)
    parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))

    filenr = 225
    soundType = 'Environment'
    locX = 2
    locY = 0
    start = (0, 2)
    end = (0, 48)
    parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))

    filenr = 225
    soundType = 'Environment'
    locX = 3
    locY = 2
    start = (0, 54)
    end = (1, 33)
    parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))

    filenr = 225
    soundType = 'Environment'
    locX = 3.5
    locY = 5
    start = (1, 40)
    end = (2, 21)
    parts.append(WavFilePart(filenr, soundType, locX, locY, start[0], start[1], end[0], end[1]))

    return parts


def getWavFileParts20170221_G4_28():
    parts = []
    filenr = 545
    locX = 0
    locY = 1.4
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 6.3, 0, 19.5))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 28.6, 1, 45.5))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 46, 3, 8))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 3, 9, 4, 43.5))

    filenr = 547
    locX = 2.1
    locY = 2.4
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 13.7, 0, 30.4))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 37.1, 1, 38))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 40, 2, 48.7))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 49, 4, 6))

    # filenr = 548
    # locX = 0
    # locY = 4
    # parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 14, 0, 30))
    # parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 32, 1, 40))
    # parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 40, 2, 44.5))
    # parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 46, 4, 16))

    # filenr = 549
    # locX = 2.1
    # locY = 3
    # parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 10, 0, 25))
    # parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 28, 1, 34.5))
    # parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 34.5, 2, 44.3))
    # parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 44.3, 3, 56))

    return parts


def getWavFileParts20170221_G5_27():
    parts = []
    # filenr = 551
    # locX = 0.45
    # locY = 1.4
    # parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 20, 0, 35))
    # parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 39, 1, 23.5))
    #
    # filenr = 552
    # locX = 0.45
    # locY = 1.4
    # parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 4, 0, 41))
    # parts.append(WavFilePart(filenr, 'Music', locX, locY, 0, 41, 2, 3.5))
    # parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 3.5, 3, 24))

    filenr = 553
    locX = 1.9
    locY = 2.8
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 7.5, 0, 19))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 25, 1, 31))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 31, 2, 37))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 37, 3, 58.5))

    # filenr = 554
    # locX = 3.1
    # locY = 6.3
    # parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 9, 0, 27))
    # parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 38, 1, 52.5))
    # parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 54, 3, 16))
    # parts.append(WavFilePart(filenr, 'Voice', locX, locY, 3, 16, 4, 46))
    #
    # filenr = 555
    # locX = 1.2
    # locY = 6.3 - 0.5
    # parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 13, 0, 26))
    # parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 33.5, 2, 8))
    # parts.append(WavFilePart(filenr, 'Music', locX, locY, 2, 8, 3, 18.3))
    # parts.append(WavFilePart(filenr, 'Voice', locX, locY, 3, 18.3, 4, 58))

    return parts


def getWavFileParts20171014_Studio():
    parts = []

    filenr = 740
    locX = 2
    locY = 4.2
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 6.9, 0, 28.6))

    filenr = 741
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 13, 1, 34))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 37, 3, 3))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 3, 4, 4, 46))

    filenr = 742
    locX = 3
    locY = 2
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 8, 0, 24.5))

    filenr = 744
    locX = 3
    locY = 2
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 8, 1, 32))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 35, 3, 5.5))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 3, 8, 4, 44))

    filenr = 745
    locX = 1.8
    locY = 2.8
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 2.5, 0, 21.3))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 38, 1, 53))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 55, 3, 20))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 3, 22, 5, 5))

    filenr = 746
    locX = 1.5
    locY = 1.2
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 1.6, 0, 21.1))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 22, 1, 31))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 33, 2, 39))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 43, 4, 6.2))

    filenr = 747
    locX = 0.8
    locY = 0.8
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 3, 0, 20))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 25, 1, 33))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 36, 2, 58))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 3, 2, 4, 48))

    filenr = 748
    locX = 0
    locY = 1
    parts.append(WavFilePart(filenr, 'Gunshot', locX, locY, 0, 2, 0, 15))
    parts.append(WavFilePart(filenr, 'Environment', locX, locY, 0, 16.5, 1, 22))
    parts.append(WavFilePart(filenr, 'Music', locX, locY, 1, 25, 2, 44))
    parts.append(WavFilePart(filenr, 'Voice', locX, locY, 2, 48, 4, 23))

    return parts


if __name__ == "__main__":
    doMain()
    print('Ready')
