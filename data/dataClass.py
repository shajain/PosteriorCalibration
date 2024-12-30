import numpy as np

class Data():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getPUData(self):
        return self.x[self.y == 1], self.x[self.y == 0]


    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getXY(self):
        return self.x, self.y



class TrainingTestingData(Data):
    def __init__(self, x, y, ixTrain, ixTest):
        super(Data, self).__init__(x, y)
        self.ixTrain = ixTrain
        self.ixTest = ixTest
        self.trainData = Data(x[ixTrain], y[ixTrain])
        self.testData = Data(x[ixTest], y[ixTest])

    def testingData(self):
        return self.testData

    def trainingData(self):
        return self.trainData

    def swappedData(self):
        return TrainingTestingData(self.x, self.y, self.ixTest, self.ixTrain)

    def divideData(self):
        train = self.trainingData()
        test = self.testingData( )
        return train.getX( ), train.getY( ), test.getX( ), test.getY( )

    def getPUTestX(self):
        return self.testData.getPUData()

    def getPUTrainX(self):
        return self.trainData.getPUData()


    @classmethod
    def randomTrainingTestingData(cls, x, y):
        n = x.shape[0]
        ixTrain = np.random.choice(np.arange(n), int(n/2))
        ixTest = np.setdiff1d(np.arange(n), ixTrain)
        return TrainingTestingData(x, y, ixTrain, ixTest)

