import numpy as np

def get_counts(model_counts,obs_counts,axis=None):

    model_counts=np.array(model_counts,dtype=bool)
    obs_counts=np.array(obs_counts,dtype=bool)

    true_positive=np.sum((model_counts) & (obs_counts),axis=axis)
    false_positive=np.sum((model_counts) & np.logical_not(obs_counts),axis=axis)
    true_negative=np.sum(np.logical_not(model_counts) & np.logical_not(obs_counts),axis=axis)
    false_negative=np.sum(np.logical_not(model_counts) & (obs_counts),axis=axis)
    
    return true_positive,false_positive,false_negative,true_negative

def peirces_skill(true_positive,false_positive,false_negative,true_negative):
    return float(true_positive)/(true_positive+false_negative)-float(false_positive)/(false_positive+true_negative)

def rate_ci(p,n,alpha=0.05):

    from scipy.stats import norm

    # z_alpha/2
    za22=norm.ppf(1-alpha/2)**2

    ci=(p+za22/2/n-za22*np.sqrt((p*(1-p)+za22/4/n)/n))/(1+za22/n)

    return ci

def heidke_ci(model_substorms,obs_substorms,nsamples=4000,ci=97.5,axis=None):

    samples=[]

    if axis is None:
        nbins=len(model_substorms)
        nbins_obs=len(obs_substorms)
    else:
        nbins=model_substorms.shape[axis]
        nbins_obs=obs_substorms.shape[axis]

    if nbins!=nbins_obs:
        raise ValueError('Model substorms and obs substorms must have same dimensions')
        

    for i in range(nsamples):

        inds=np.random.randint(0,nbins,nbins)

        model_counts_sample=np.take(model_substorms,inds,axis)
        obs_counts_sample=np.take(obs_substorms,inds,axis)
        true_positive,false_positive,false_negative,true_negative=get_counts(model_counts_sample,obs_counts_sample,axis)
        samples.append(heidke_skill(true_positive,false_positive,false_negative,true_negative))

    return np.percentile(samples,100-ci,axis=0),np.percentile(samples,ci,axis=0)

def heidke_skill(true_positive,false_positive,false_negative,true_negative):
    N=true_positive+false_positive+true_negative+false_negative
    try:
        expected_correct=((true_positive+false_negative)*(true_positive+false_positive) + (true_negative + false_negative)*(true_negative + false_positive))/np.array(N,dtype=float)
        skill=(true_positive+true_negative-expected_correct)/np.array(N-expected_correct,dtype=float)
    except ZeroDivisionError:
        return None

    return skill

def hit_rate(true_positive,false_positive,false_negative,true_negative):
    try:
        return np.array(true_positive,dtype=float)/(true_positive+false_negative)
    except ZeroDivisionError:
        return None

def false_alarm_rate(true_positive,false_positive,false_negative,true_negative):
    try:
        return np.array(false_positive,dtype=float)/(true_negative+false_positive)
    except ZeroDivisionError:
        return None

def dump_stats(forecast_substorms,obs_substorms):
    true_positive,false_positive,false_negative,true_negative=get_counts(forecast_substorms,obs_substorms)
    print true_positive,false_positive
    print false_negative,true_negative

    print 'Heidke skill score:',heidke_skill(float(true_positive),float(false_positive),float(false_negative),float(true_negative))
    print 'Peirce skill score:',peirces_skill(float(true_positive),float(false_positive),float(false_negative),float(true_negative))
    print 'Hit rate:',hit_rate(true_positive,false_positive,false_negative,true_negative)
    print 'False alarm rate:',false_alarm_rate(true_positive,false_positive,false_negative,true_negative)
