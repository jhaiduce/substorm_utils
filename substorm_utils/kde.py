from scipy.stats import gaussian_kde
import numpy as np

class transformed_kde(object):

    def __init__(self,dataset,bw_method=None,transform=lambda x: x, Dtrans=lambda x: 1):
    
        self.kde=gaussian_kde(transform(dataset),bw_method)
        self.transform=transform
        self.Dtrans=Dtrans

    def __call__(self,x):
        return self.kde(self.transform(x))*np.abs(self.Dtrans(x))

def get_kde_bootstrap(data,evalpoints,nsamples,bw,transform=lambda x: x, Dtrans = lambda x: 1):
    estimates=np.zeros([len(evalpoints),nsamples])
    for i in range(nsamples):
        xstar=np.random.choice(data,len(data))
        dstar=transformed_kde(xstar,bw_method=bw,transform=transform,Dtrans=Dtrans)
        estimates[:,i]=dstar(evalpoints)
    return estimates

