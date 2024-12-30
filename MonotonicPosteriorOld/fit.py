from MonotonicPosteriorOld.Net import MontonicPosteriorNetBCE as PostNet
from NN.models import Basic01 as NNPost
from NN.trainer import Trainer as Trainer
from MonotonicPosteriorOld.debug import Debug
from misc.dictUtils import safeUpdate
from misc.dictUtils import safeRemove
from misc import sortedplot as sp
from data.datagen import GaussianDG
import pdb


class PosteriorFitting:

    netDEF = {'n_units': 10, 'n_hidden': 5, 'dropout_rate': 0.0}
    trainDEF =  { 'batchSize': 500, 'maxIter': 1000, 'debug': False}

    def __init__(self, **kwargs):
        self.netDEF = safeUpdate(PosteriorFitting.netDEF, kwargs)
        self.trainDEF = safeUpdate(PosteriorFitting.trainDEF, kwargs)
        netPost = NNPost(**self.netDEF)
        netPost.build((None, 1))
        self.postNet = PostNet(netPost)

    def fit(self, x, w1, w0, **kwargs):
        self.fitArgs = {'x': x, 'w1': w1, 'w0': w0, **kwargs}
        self.postNet.save(x, w1, w0)
        trainer = Trainer(self.postNet, x, w1, w0, **safeRemove(self.trainDEF, 'debug'))
        if self.trainDEF['debug']:
            self.debug = Debug()
            if 'posterior' in kwargs:
                self.debug.attachTarget(x, kwargs['posterior'])
            trainer.attachDebugger(self.debug)
        #pdb.set_trace()
        trainer.fit( )

    def getNet(self):
        return self.postNet

    def setNet(self, postNet):
        self.postNet = postNet

    def refit(self):
        self.fit(**self.fitArgs)

    @classmethod
    def demo(cls):
        alpha = 0.5
        n_p = 50
        n_u = 50
        mu = -1
        sig = 1
        dg = GaussianDG(mu=mu, sig=sig, alpha=alpha, n_p=n_p, n_u=n_u)
        n = 2000
        [x, y] = dg.pn_data(n, alpha)
        posterior = dg.pn_posterior_cc(x)
        sp.sortedplot(x, posterior)
        sp.sortedplot(x, dg.dens_neg(x))
        sp.sortedplot(x, dg.dens_pos(x))
        sp.hist(x[(y == 0).flatten( ), :], bins=20, density=True)
        sp.hist(x[(y == 1).flatten( ), :], bins=20, density=True)
        sp.show( )
        fitting = PosteriorFitting(debug=True)
        w1 = y
        w0 = 1-y
        fitting.fit(x, w1, w0, posterior=posterior)
        return fitting