import tensorflow as tf
import numpy as np
from NN.NetWithLoss import NetWithLoss
from misc.randomSort import randomSort
from tensorflow.keras.losses import BinaryCrossentropy
from misc import sortedplot as sp
import pdb


class MontonicPosteriorNetBCE(NetWithLoss):
    DEBUG = False
    def __init__(self, net):
        #Assume that x is sorted
        super(MontonicPosteriorNetBCE, self).__init__(net)
        self.BCE =BinaryCrossentropy()

    def SELossOnSavedData(self):
        return self.SELoss(self.x, self.w1, self.w0)

    def SELoss(self, x, w1, w0):
        posterior = self.net.predict(x)
        n1 = np.sum(w1)
        n0 = np.sum(w0)
        n = (n0 + n1)/2
        # Update w1 and w0 to ensure that both w1 and w2 capture the same number of weighted points in x
        w1 = w1*n/n1
        w0 = w0*n/n0
        y = w1/(w1+w0)
        lossvec = (posterior-y)**2
        loss = np.sum(lossvec, axis=0)
        return loss

    def loss(self, x, w1, w0):
       return self.SELoss(x, w1, w0)

    # def lossFast(self, x, w1, w0, batchSize):
    #     p1 = w1/np.sum(w1)
    #     p0 = w1/np.sum(w1)
    #
    #     return loss

    def lossTF(self, x, y, nn):
        posterior = self.net(x)
        loss_bce = tf.keras.losses.binary_crossentropy(y, posterior)

        #diff = tf.keras.activations.relu(posterior[0:(nn-1)] - posterior[1:nn])
        #loss_monotonicity = tf.reduce_sum(diff)
        #loss = loss_bce + 10*loss_monotonicity
        loss = loss_bce
        #loss = tf.reduce_mean((y-posterior)**2)
        pdb.set_trace()
        return loss

    def gradients(self, x, y, batchSize):
        #self.save(x, w1, w0)
        n = x.shape[0]
        x1 = x[(y == 1).flatten(), :]
        x0 = x[(y == 0).flatten(), :]
        ix1 = np.random.choice(x1.shape[1], batchSize, replace=True)
        ix0 = np.random.choice(x0.shape[1], batchSize, replace=True)
        xx1 = x1[ix1, :]
        xx0 = x0[ix0, :]
        xx = np.vstack((xx1,xx0))
        #self.alpha = alpha
        if MontonicPosteriorNetBCE.DEBUG:
            sp.hist(xx1)
            sp.hist(xx0)
            sp.show()
            #pdb.set_trace()
        yy = np.vstack((np.ones((batchSize,1)), np.zeros((batchSize,1))))
        #xx, ix = randomSort(xx)[0:2]
        #pp = self.net.predict(xx)

        #yy = yy[ix,:]
        nn = xx.shape[0]
        #pdb.set_trace()
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            #pdb.set_trace()
            tape.watch(self.net.trainable_variables)
            loss = self.lossTF(xx, yy, nn)
            gradient = tape.gradient(loss, self.net.trainable_variables)
            #pdb.set_trace()
        return loss, gradient

    # def gradients(self, x, w1, w0, batchSize):
    #     # Assumes x to be sorted. If x is unsorted the training would be still fine since the batch is sorted anyways.
    #     # But the saved x should also be sorted. The posterior estimate relies on it being sorted
    #     #self.save(x, w1, w0)
    #     n = x.shape[0]
    #     n1 = np.sum(w1, axis=0)
    #     n0 = np.sum(w0, axis=0)
    #     #pdb.set_trace()
    #     p1 = w1 / n1
    #     p0 = w0 / n0
    #     ix1 = np.random.choice(n, batchSize, replace=True, p=p1.flatten())
    #     ix0 = np.random.choice(n, batchSize, replace=True, p=p0.flatten())
    #     xx1 = x[ix1, :]
    #     xx0 = x[ix0, :]
    #     xx = np.concatenate((xx1,xx0), axis=0)
    #     #self.alpha = alpha
    #     if MontonicPosteriorNetBCE.DEBUG:
    #         sp.hist(xx1)
    #         sp.hist(xx0)
    #         sp.show()
    #         #pdb.set_trace()
    #     yy = np.concatenate((np.ones((batchSize,1)), np.zeros((batchSize,1))), axis=0)
    #     xx, ix = randomSort(xx)[0:2]
    #     #pp = self.net.predict(xx)
    #
    #     yy = yy[ix,:]
    #     nn = xx.shape[0]
    #     #pdb.set_trace()
    #     with tf.GradientTape(watch_accessed_variables=False) as tape:
    #         #pdb.set_trace()
    #         tape.watch(self.net.trainable_variables)
    #         loss = self.lossTF(xx, yy, nn)
    #         gradient = tape.gradient(loss, self.net.trainable_variables)
    #         #pdb.set_trace()
    #     return loss, gradient

    def posterior(self, x):
        posterior = self.net.predict(x)
        #pdb.set_trace()
        return posterior


    def save(self, x, w1, w0):
        self.x, ix = randomSort(x)[0:2]
        self.w1 = w1[ix]
        self.w0 = w0[ix]

    def getWeightedSample(self):
        return self.x, self.w1, self.w0


    def copy(self):
        copy = MontonicPosteriorNetBCE(self.copyNet())
        copy.save(self.x, self.w1, self.w0)
        return copy


# class MontonicPosteriorNetBCE(NetWithLoss):
#     DEBUG = False
#     def __init__(self, net):
#         #Assume that x is sorted
#         super(MontonicPosteriorNetBCE, self).__init__(net)
#         self.BCE =BinaryCrossentropy()
#         self.dirtyPost = True
#
#     def SELossOnSavedData(self):
#         return self.SELoss(self.x, self.w1, self.w0)
#
#     def SELoss(self, x, w1, w0):
#         der = self.net.predict(x)
#         n1 = np.sum(w1)
#         n0 = np.sum(w0)
#         n = (n0 + n1)/2
#         # Update w1 and w0 to ensure that both w1 and w2 capture the same number of weighted points in x
#         w1 = w1*n/n1
#         w0 = w0*n/n0
#         phi = w1 + w0
#         y = w1/(w1+w0)
#         phiN = phi/(2*n)
#         # Reweighing der by phi is important here. It ensures that the summation is over a mixture
#         # with half positives and half negatives. This is assumed while the gradients are computed,
#         # although in an unweighted manner
#         posterior = np.cumsum(phi*der, axis=0)/(2*n)
#         posterior = tf.keras.activations.sigmoid(posterior).numpy()
#         posterior = 2 * posterior - 1
#         lossvec = phiN*((posterior-y)**2)
#         loss = np.sum(lossvec, axis=0)
#         return loss
#
#     def loss(self, x, w1, w0):
#        return self.SELoss(x, w1, w0)
#
#     # def lossFast(self, x, w1, w0, batchSize):
#     #     p1 = w1/np.sum(w1)
#     #     p0 = w1/np.sum(w1)
#     #
#     #     return loss
#
#     def lossTF(self, x, y, nn):
#         der = self.net(x)
#         posterior = tf.math.cumsum(der, axis=0)/nn
#         posterior = tf.keras.activations.sigmoid(posterior)
#         posterior = 2*posterior - 1
#         loss = self.BCE(y, posterior)
#         #pdb.set_trace()
#         return loss
#
#     def gradients(self, x, w1, w0, batchSize):
#         # Assumes x to be sorted. If x is unsorted the training would be still fine since the batch is sorted anyways.
#         # But the saved x should also be sorted. The posterior estimate relies on it being sorted
#         #self.save(x, w1, w0)
#         n = x.shape[0]
#         n1 = np.sum(w1, axis=0)
#         n0 = np.sum(w0, axis=0)
#         #pdb.set_trace()
#         p1 = w1 / n1
#         p0 = w0 / n0
#         ix1 = np.random.choice(n, batchSize, replace=True, p=p1.flatten())
#         ix0 = np.random.choice(n, batchSize, replace=True, p=p0.flatten())
#         xx1 = x[ix1, :]
#         xx0 = x[ix0, :]
#         xx = np.concatenate((xx1,xx0), axis=0)
#         if MontonicPosteriorNetBCE.DEBUG:
#             sp.hist(xx1)
#             sp.hist(xx0)
#             sp.show()
#             pdb.set_trace()
#         yy = np.concatenate((np.ones((batchSize,1)), np.zeros((batchSize,1))), axis=0)
#         xx, ix = randomSort(xx)[0:2]
#         yy = yy[ix,:]
#         nn = xx.shape[0]
#         #pdb.set_trace()
#         with tf.GradientTape(watch_accessed_variables=False) as tape:
#             #pdb.set_trace()
#             tape.watch(self.net.trainable_variables)
#             loss = self.lossTF(xx, yy, nn)
#             gradient = tape.gradient(loss, self.net.trainable_variables)
#             #pdb.set_trace()
#         return loss, gradient
#
#     def posterior(self, x):
#         # Assumes self.x to be sorted and available
#         # ix contains the indices of self.x closest to elements of x
#         #pdb.set_trace()
#         ix = np.searchsorted(self.x.flatten(), x.flatten(), side='right')
#         if self.dirtyPost:
#             self.recomputePosterior()
#         #nSaved is the size of the saved x
#         nSaved=self.post.shape[0]
#         low_ix = np.where((ix <= 0).flatten())[0]
#         up_ix = np.where((ix >= nSaved-1).flatten())[0]
#         posterior = np.zeros(x.shape)
#         posterior[low_ix] = self.post[0]
#         posterior[up_ix] = self.post[nSaved-1]
#         n = x.shape[0]
#         # ixx are the indices of x that are in the interior of self.x. These can be interpolated
#         ixx = np.setdiff1d(np.arange(n), np.concatenate((low_ix, up_ix), axis=0))
#         ix = ix[ixx]
#         p_up = self.post[ix]
#         p_low = self.post[ix-1]
#         x_up = self.x[ix]
#         x_low = self.x[ix-1]
#         slope = (p_up -p_low)/(x_up -x_low)
#         posterior[ixx] = p_low + (x[ixx]-x_low)*np.where(np.isfinite(slope), slope, 0)
#         #pdb.set_trace()
#         return posterior
#
#     def recomputePosterior(self):
#         x = self.x
#         w1 = self.w1
#         w0 = self.w0
#         n1 = np.sum(w1)
#         n0 = np.sum(w0)
#         n = (n1 + n0)/2
#         w1 = w1*n/n1
#         w0 = w0*n/n0
#         phi = w1 + w0
#         der = self.net.predict(x)
#         self.post = np.cumsum(phi * der, axis=0)/(2*n)
#         self.post = tf.keras.activations.sigmoid(self.post).numpy()
#         self.post = 2*self.post - 1
#         self.dirtyPost = False
#
#     def save(self, x, w1, w0):
#         self.x, ix = randomSort(x)[0:2]
#         self.w1 = w1[ix]
#         self.w0 = w0[ix]
#         self.dirtyPost = True
#
#     def getWeightedSample(self):
#         return self.x, self.w1, self.w0
#
#
#     def copy(self):
#         copy = MontonicPosteriorNetBCE(self.copyNet())
#         copy.save(self.x, self.w1, self.w0)
#         return copy



# class MontonicPosteriorNetBCE(NetWithLoss):
#     def __init__(self, net):
#         #Assume that x is sorted
#         super(MontonicPosteriorNetBCE, self).__init__(net)
#         self.BCE =BinaryCrossentropy()
#         self.dirtyPost = True
#
#     def SELossOnSavedData(self):
#         return self.SELoss(self.x, self.w1, self.w0)
#
#     def SELoss(self, x, w1, w0):
#         der = self.net.predict(x)
#         n1 = np.sum(w1)
#         n0 = np.sum(w0)
#         n = (n0 + n1)/2
#         # Update w1 and w0 to ensure that both w1 and w2 capture the same number of weighted points in x
#         w1 = w1*n/n1
#         w0 = w0*n/n1
#         phi = w1 + w0
#         y = w1/(w1+w0)
#         phiN = phi/(2*n)
#         preActivation = np.cumsum(der, axis=0)
#         preActivation = self.net.subtractC(preActivation).numpy()
#         posterior = tf.keras.activations.sigmoid(preActivation).numpy()
#         lossvec = phiN*((posterior-y)**2)
#         loss = np.sum(lossvec, axis=0)
#         return loss
#
#     def loss(self, x, w1, w0):
#        return self.SELoss(x, w1, w0)
#
#     # def lossFast(self, x, w1, w0, batchSize):
#     #     p1 = w1/np.sum(w1)
#     #     p0 = w1/np.sum(w1)
#     #
#     #     return loss
#
#     def lossTF(self, x, y, nn):
#         der = self.net(x)
#         preActivation = tf.math.cumsum(der, axis=0)/nn
#         preActivation = self.net.subtractC(preActivation)
#         posterior = tf.keras.activations.sigmoid(preActivation)
#         loss = self.BCE(y, posterior)
#         #pdb.set_trace()
#         return loss
#
#     def gradients(self, x, w1, w0, batchSize):
#         self.save(x, w1, w0)
#         n = x.shape[0]
#         n1 = np.sum(w1, axis=0)
#         n0 = np.sum(w0, axis=0)
#         #pdb.set_trace()
#         p1 = w1 / n1
#         p0 = w0 / n0
#         ix1 = np.random.choice(n, batchSize, replace=True, p=p1.flatten())
#         ix0 = np.random.choice(n, batchSize, replace=True, p=p0.flatten())
#         xx1 = x[ix1, :]
#         xx0 = x[ix0, :]
#         xx = np.concatenate((xx1,xx0), axis=0)
#         yy = np.concatenate((np.ones((batchSize,1)), np.zeros((batchSize,1))), axis=0)
#         xx, ix = randomSort(xx)[0:2]
#         yy = yy[ix,:]
#         nn = xx.shape[0]
#         #pdb.set_trace()
#         with tf.GradientTape(watch_accessed_variables=False) as tape:
#             # pdb.set_trace()
#             tape.watch(self.net.trainable_variables)
#             loss = self.lossTF(xx, yy, nn)
#         return loss, tape.gradient(loss, self.net.trainable_variables)
#
#     def posterior(self, x):
#         # Assumes self.x to be sorted and available
#         # ix contains the indices of self.x closest to elements of x
#         #pdb.set_trace()
#         ix = np.searchsorted(self.x.flatten(), x.flatten(), side='right')
#         if self.dirtyPost:
#             self.recomputePosterior()
#         #nSaved is the size of the saved x
#         nSaved=self.post.shape[0]
#         low_ix = np.where((ix<=0).flatten())[0]
#         up_ix = np.where((ix>=nSaved-1).flatten())[0]
#         posterior = np.zeros(x.shape)
#         posterior[low_ix] = self.post[0]
#         posterior[up_ix] = self.post[nSaved-1]
#         n = x.shape[0]
#         # ixx are the indices of x that are in the interior of self.x. These can be interpolated
#         ixx = np.setdiff1d(np.arange(n), np.concatenate((low_ix, up_ix), axis=0))
#         ix = ix[ixx]
#         p_up = self.post[ix]
#         p_low = self.post[ix-1]
#         x_up = self.x[ix]
#         x_low = self.x[ix-1]
#         slope = (p_up -p_low)/(x_up -x_low)
#         posterior[ixx] = p_low + (x[ixx]-x_low)*np.where(np.isfinite(slope), slope, 0)
#         #pdb.set_trace()
#         return posterior
#
#
#
#     def recomputePosterior(self):
#         der = self.net.predict(self.x)
#         n = self.x.shape[0]
#         preActivation = np.cumsum(der, axis=0)/n
#         preActivation = self.net.subtractC(preActivation).numpy()
#         self.post = tf.keras.activations.sigmoid(preActivation).numpy()
#         self.dirtyPost = False
#
#     def save(self, x, w1, w0):
#         self.x = x
#         self.w1 = w1
#         self.w0 = w0
#         self.dirtyPos = True
#
#
#     def copy(self):
#         copy = MontonicPosteriorNetBCE(self.copyNet())
#         copy.save(self.x, self.w1, self.w0)
#         return copy
#
#
