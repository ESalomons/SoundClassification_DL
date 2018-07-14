import copy

def classNameShort(name):
    return name[0]

class ConfusionMatrix:

    def __init__(self, classes):
        self.matrix = []
        self.nrClasses = len(classes)
        self.classes = classes
        for i in range(0,self.nrClasses):
            row = []
            for j in range(0,self.nrClasses):
                row.append(0)
            self.matrix.append(row)

    def add(self, row, col, num):
        self.matrix[row][col] += num

    def get(self,row,col):
        return self.matrix[row][col]


    def addTextPrediction(self, actual, predicted):
        actualNr = self.classes.index(actual.lower())
        predictedNr = self.classes.index(predicted.lower())
        self.add(actualNr, predictedNr, 1)

    def correctPredictions(self):
        correctPreds = []
        for classNr in range(0,self.nrClasses):
            correctPreds.append(self.get(classNr,classNr))
        return correctPreds

    def sensitivities(self):
        corr = self.correctPredictions()
        sumRows = [sum(x) for x in self.matrix]
        return [1.0*x/y if (y!=0) else 0 for x,y in zip(corr,sumRows)]

    def getSensitivityMap(self):
        senses = self.sensitivities()
        mp = {}
        index = 0
        for clz in self.classes:
            mp[clz] = senses[index]
            index += 1
        return mp

    def precisions(self):
        corr = self.correctPredictions()
        sumCols = [sum(x) for x in zip(*self.matrix)]
        return [1.0*x/y if y != 0 else 0 for x,y in zip(corr, sumCols) ]

    def getClassAccuracy(self, classNr):
        row = self.matrix[classNr]
        return (1.0 * row[classNr]) / sum(row)

    def f1Scores(self):
        f1s = []
        sensitivites = self.sensitivities()
        precisions = self.precisions()
        for i in range(self.nrClasses):
            f1s.append(self.calcF1(sensitivites[i], precisions[i]))
        return f1s

    def f1Avg(self):
        f1s = self.f1Scores()
        return 1.0 * sum(f1s) / len(f1s)

    def f1Norm(self):
        matrixCopy = copy.deepcopy(self)
        matrixCopy.normalizeRows()
        return matrixCopy.f1Avg()

    def normalizedCopy(self):
        matrixCopy = copy.deepcopy(self)
        matrixCopy.normalizeRows()
        return matrixCopy

    def clone(self):
        return copy.deepcopy(self)

    # def f1M(self):
    #     precisions = self.precisions()
    #     sensitivities = self.sensitivities()
    #     avgPrecision = sum(precisions) / len(precisions)
    #     avgSens = sum(sensitivities) / len(sensitivities)
    #     return self.calcF1(avgSens, avgPrecision)

    def calcF1(self, sensitivity, precision):
        if sensitivity + precision > 0:
            return 2 * sensitivity * precision / (sensitivity + precision)
        else:
            return 0

    def totalObservations(self):
        total = 0
        for row in range(0, self.nrClasses):
            for col in range(0, self.nrClasses):
                total += self.get(row,col)
        return total

    def accuracies(self):
        accs = []
        totalObs = self.totalObservations() * 1.0
        for classNr in range(0,self.nrClasses):
            truePos = self.get(classNr, classNr)
            trueNeg = 0
            for rowNr in range(0, self.nrClasses):
                for colNr in range(0,self.nrClasses):
                    if (rowNr != classNr) and (colNr != classNr):
                        trueNeg += self.get(rowNr,colNr)
            if totalObs > 0:
                accs.append((truePos + trueNeg) / totalObs)
            else:
                accs.append(0)
        return accs

    def addMatrix(self, other):
        for row in range(0,self.nrClasses):
            for col in range(0, self.nrClasses):
                self.add(row,col,other.get(row,col))

    def normalizeRows(self):
        for row in self.matrix:
            sm = 1.0 * sum(row)
            if sm:
                row[:] = [val / sm for val in row]

    def f1ClassScore(self,clz):
        return self.f1Scores()[self.classes.index(clz)]

    def toF1String(self):
        f1Scores = self.f1Scores()
        res = 'F1 overall: {:.2f}\n'.format(self.f1Avg())
        for clz in self.classes:
            res += 'F1 {:s}: {:.2f}\n'.format(clz, f1Scores[self.classes.index(clz)])
        return res
    
    def toString(self):
        numFormatStr = "{:>8d}"
        if type(self.matrix[0][0]) is float:
            numFormatStr = "{:>8.2f}"
        outLine = "{:>12}".format(' ')
        sens = self.sensitivities()
        precs = self.precisions()
        accs = self.accuracies()
        for classNr in range(0,self.nrClasses):
            outLine += '{:>8s}'.format(classNameShort(self.classes[classNr]))
        outLine += " |{:>6s}{:>6s}\n".format("sens","acc")
        classNr = -1
        for row in self.matrix:
            classNr += 1
            outLine += '{:12s}'.format(self.classes[classNr])
            for col in row:
                outLine += numFormatStr.format(col)
            outLine += ' |{:6.2f}{:6.2f}\n'.format(sens[classNr],accs[classNr])

        for i in range(0,self.nrClasses):
            outLine += '--------'
        outLine += '--------------\n'
        outLine += '{:12s}'.format('prec')
        for i in range(0,self.nrClasses):
            outLine += '{:8.2f}'.format(precs[i])
        return outLine
