from scipy.stats import gaussian_kde
import numpy as np

def get_kde_bootstrap(data,evalpoints,nsamples,bw):
    estimates=np.zeros([len(evalpoints),nsamples])
    for i in range(nsamples):
        xstar=np.random.choice(data,len(data))
        dstar=gaussian_kde(xstar,bw_method=bw)
        estimates[:,i]=dstar(evalpoints)
    return estimates
