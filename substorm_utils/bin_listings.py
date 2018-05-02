from datetime import datetime, timedelta
import numpy as np
from pytz import UTC

def make_grid(signatures,tstart=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),tstep=timedelta(0,1800),signature_filters=None,return_times=False):

    tmax_tnum=(tmax-tstart).total_seconds()

    tstep=tstep.total_seconds()

    nsteps=int(tmax_tnum/tstep)

    if signature_filters is not None:
        nsigs=len(signature_filters)
    else:
        nsigs=len(signatures)

    grid=np.zeros((nsigs,nsteps))

    grid_tnums=np.arange(0,(tmax-tstart).total_seconds(),tstep)

    if return_times:
        times=np.ma.array(np.zeros((nsigs,nsteps)),mask=1)

    names=[]

    i=0

    for name,signature_tnums in signatures.iteritems():

        if signature_filters and name not in signature_filters:
            continue

        grid_inds,unique_idx=np.unique(np.searchsorted(grid_tnums,signature_tnums,side='right'),return_index=True)
        mask=(grid_inds>0) & (grid_inds<grid.shape[1])
        grid_inds=grid_inds[mask]
        unique_idx=unique_idx[mask]

        grid[i,grid_inds-1]=1

        if return_times:
            times[i,grid_inds]=signature_tnums[unique_idx]
        
        names.append(name)
        i+=1

    if return_times:
        return grid,times,names
    else:
        return grid,names

def substorm_occurrences(grid,threshold,mandatory_signature_inds=()):

    substorms=((grid).sum(axis=0)>=threshold)

    for ind in mandatory_signature_inds:
        substorms=np.logical_and(substorms,grid[ind])

    return substorms

def interval_counts(times,tmin=datetime(2005,1,1),tmax=datetime(2005,2,1),tstep=timedelta(0,1800)):

    nsteps=(tmax-tmin).total_seconds()/tstep.total_seconds()

    grid=np.zeros((1,nsteps))

    for time in times:
        if time<tmin or time>tmax: continue
        ti=int((time-tmin).total_seconds()/tstep.total_seconds())
        grid[0,ti]+=1

    return grid

from cache_decorator import cache_result

@cache_result(clear=False)
def convolve_onsets(onset_tnums,tmin=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),resolution=1./(24*60),bandwidth=15./(24*60)):

    out_tnums=np.arange(0,(tmax-tmin).total_seconds(),resolution.total_seconds())

    out=np.zeros(out_tnums.shape)

    bw_sec=bandwidth.total_seconds()

    for tnum in onset_tnums:
        out+=np.exp(-(out_tnums-tnum)**2/2/bw_sec**2)

    return out,out_tnums

def convolved_substorm_scores(signatures,signature_weights={},bandwidth=timedelta(0,60*15),resolution=timedelta(0,60),tmin=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC)):
    signature_scores=[]
    tnums=None
    for key in signatures.keys():
        weight=signature_weights.get(key,1)
        if weight>0:
            scores,tnums=convolve_onsets(tuple(signatures[key]),resolution=resolution,
                                         bandwidth=bandwidth,tmin=tmin,tmax=tmax)
            signature_scores.append(scores*weight)
    return np.sum(signature_scores,axis=0),tnums

def find_substorms_convolution(signatures,threshold,signature_weights={},tstep=timedelta(0,1800),bandwidth=timedelta(0,60*15),tmin=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),convolution_resolution=timedelta(0,60)):

    scores,score_tnums=convolved_substorm_scores(signatures,signature_weights,bandwidth,convolution_resolution,tmin=tmin,tmax=tmax)

    bin_tnums=np.arange(0,(tmax-tmin).total_seconds(),tstep.total_seconds())
    split_inds=np.searchsorted(score_tnums,bin_tnums)

    bin_maxes=np.zeros([len(bin_tnums)])
    bin_maxtimes=np.zeros([len(bin_tnums)])

    for ibin,(split_scores,split_tnums) in enumerate(zip(
            np.split(scores,split_inds[1:]),
            np.split(score_tnums,split_inds[1:]))):
        max_ind=np.argmax(split_scores)
        bin_maxes[ibin]=split_scores[max_ind]
        bin_maxtimes[ibin]=split_tnums[max_ind]

    substorm_bins=(bin_maxes>threshold)
    #substorm_times=[tmin+timedelta(seconds=maxtime) for maxtime in bin_maxtimes[substorm_bins]]
    substorm_times=[tmin+timedelta(seconds=maxtime) for maxtime in bin_maxtimes[substorm_bins]]
    return substorm_bins,substorm_times

def find_substorms(signatures,threshold,signature_filters=None,mandatory_signatures=[],tstep=timedelta(0,1800),return_times=False):

    if signature_filters is None:
        signature_filters=signatures.keys()

    retvals=make_grid(signatures,signature_filters=signature_filters,tstep=tstep,return_times=return_times)
    if return_times:
        grid,times,keys=retvals
    else:
        grid,keys=retvals

    mandatory_signature_inds=[keys.index(key) for key in mandatory_signatures]

    substorms=substorm_occurrences(grid,threshold,mandatory_signature_inds)

    if return_times:
        if len(mandatory_signature_inds)>0:
            times=np.ma.min(times[mandatory_signature_inds,:],axis=0)
        else:
            times=np.ma.min(times,axis=0)

        return substorms,times
    else:
        return substorms

def filter_substorms(substorms,data_values,threshold):
    above_threshold=(data_values>threshold)
    substorms=substorms.copy()
    substorms[substorms]=above_threshold
    return substorms
