from misc import sortedplot as sp
from IPython.display import display
import numpy as np


class Debug:

    def __init__(self):
        self.lossTrain = []
        self.SELoss = []
        fig, axs = sp.subplots(2, 2, figsize=(8,8))
        self.fig = fig
        self.axs = axs

    def attachTarget(self, x, posterior):
        self.x = x
        self.posterior = posterior
        self.posteriorFit = [ ]


    def attachNets(self, nets):
        self.nets = nets


    def plotTrainningLossHistory(self, loss):
        self.lossTrain.append(loss)
        #pdb.set_trace()
        sp.sortedplot(self.lossTrain, label='Training Loss', ax=self.axs[0,0])

    def plotSELossHistory(self):
        net = self.nets[-1]
        self.SELoss.append(net.SELossOnSavedData())
        sp.sortedplot(self.SELoss, label='SELoss', ax=self.axs[0, 1])

    def plotPosteriorFit(self):
        net = self.nets[-1]
        posteriorFit = net.posterior(self.x)
        self.posteriorFit.append(posteriorFit)
        if hasattr(self, 'posterior'):
            sp.sortedplot(self.x, self.posterior, label='Posterior True', ax=self.axs[1, 0])
        sp.sortedplot(self.x, posteriorFit, label='Posterior Est. New', ax=self.axs[1, 0])
        #pdb.set_trace()
        if len(self.posteriorFit) >=2:
           sp.sortedplot(self.x, self.posteriorFit[-2], label='Posterior Est. Old',  ax=self.axs[1, 0])
        #pdb.set_trace()




    def afterUpdate(self, loss):
        print('after Update')
        self.plotPosteriorFit()
        self.plotTrainningLossHistory(loss)
        self.plotSELossHistory()
        self.displayPlots()
        #sp.show()

    def beforeUpdate(self, iter):
        if np.remainder(iter, 10) == 0:
            print('Iteration' + str(iter))
        return

    def beforeTraining(self):
        # print('before Training')
        return


    def displayPlots(self):
        self.axs[0, 0].legend( )
        self.axs[0, 1].legend( )
        self.axs[1, 0].legend( )
        self.axs[1, 1].legend( )
        display(self.fig)
        sp.close( )
        self.axs[0, 0].clear( )
        self.axs[0, 1].clear()
        self.axs[1, 0].clear()
        self.axs[1, 1].clear()
        fig, axs = sp.subplots(2, 2, figsize=(8, 8))
        self.fig = fig
        self.axs = axs


