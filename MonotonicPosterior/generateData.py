from misc import sortedplot as sp
from data.datagen import GaussianDG
from misc import sortedplot as sp

def generateData(n=5000, mu0=-1, sig0=2, alpha=0.5):
    n_p = 50
    n_u = 50
    dg = GaussianDG(mu=mu0, sig=sig0, alpha=alpha, n_p=n_p, n_u=n_u)
    [X, y] = dg.pn_data(n, alpha)
    posterior = dg.pn_posterior_cc(X)
    return X, y, posterior, dg