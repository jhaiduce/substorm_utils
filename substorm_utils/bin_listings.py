from datetime import datetime, timedelta
import numpy as np
from pytz import UTC
from scipy.special import erf

def filter_onsets(substorms,onset_list,tstart=datetime(2005,1,1,tzinfo=UTC),tstep=timedelta(0,1800)):
    onset_bins=[int((onset-tstart).total_seconds()/tstep.total_seconds()) for onset in onset_list]

    onset_list_filtered=np.array(onset_list)[substorms[onset_bins]]
                
    return onset_list_filtered

def make_grid(signatures,tstart=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),tstep=timedelta(0,1800),signature_filters=None,return_times=False,epoch=datetime(2005,1,1,tzinfo=UTC)):

    tmax_tnum=(tmax-epoch).total_seconds()

    tstep=tstep.total_seconds()

    nsteps=int(tmax_tnum/tstep)

    if signature_filters is not None:
        nsigs=len(signature_filters)
    else:
        nsigs=len(signatures)

    grid=np.zeros((nsigs,nsteps))

    grid_tnums=np.arange((tstart-epoch).total_seconds(),(tmax-epoch).total_seconds(),tstep)

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

def convolve_onsets(onset_tnums,tmin=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),resolution=timedelta(seconds=60),bandwidth=timedelta(seconds=60*15),epoch=datetime(2005,1,1,tzinfo=UTC)):

    out_tnums=np.arange((tmin-epoch).total_seconds(),(tmax-epoch).total_seconds(),resolution.total_seconds())

    bw_sec=bandwidth.total_seconds()

    pulses=np.zeros(out_tnums.shape)
    pulses[:-1]=np.histogram(onset_tnums,out_tnums)[0]

    m=int(bw_sec*6/resolution.total_seconds())
    x=np.arange(-m,m)*resolution.total_seconds()
    g=np.exp(-x**2/2/bw_sec**2)

    out=np.convolve(pulses,g,mode='same')

    out=erf(out)

    return out,out_tnums

def convolved_substorm_scores(signatures,signature_weights={},bandwidth=timedelta(0,60*15),resolution=timedelta(0,60),tmin=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),epoch=datetime(2005,1,1,tzinfo=UTC)):
    signature_scores=[]
    tnums=None
    for key in signatures.keys():
        weight=signature_weights.get(key,1)
        if weight>0:
            scores,tnums=convolve_onsets(tuple(signatures[key]),resolution=resolution,
                                         bandwidth=bandwidth,tmin=tmin,tmax=tmax,
                                         epoch=epoch)
            signature_scores.append(scores*weight)
    return np.sum(signature_scores,axis=0),tnums

def search_convolution_scores(scores,threshold,require_continuous=True):
    local_max_inds,=np.where((scores[1:-1]>scores[:-2]) & (scores[1:-1]>scores[2:]))

    try:
        len(threshold)
    except TypeError:
        pass
    else:
        if len(threshold)!=len(scores):
            raise ValueError('threshold and scores must have same length')

    local_max_inds+=1

    event_inds=[]

    for i,local_max_ind in enumerate(local_max_inds):
        
        try:
            above_threshold=(scores[local_max_ind]>threshold[local_max_ind])
            threshold_is_scalar=False
        except (TypeError,IndexError):
            above_threshold=(scores[local_max_ind]>threshold)
            threshold_is_scalar=True
            
        if above_threshold:

            # Get threshold for this segment (scalar or sequence)
            if threshold_is_scalar:
                segment_threshold=threshold
            else:
                if len(event_inds)>0:
                    segment_threshold=threshold[event_inds[-1]:local_max_ind]
                
            if require_continuous==True \
               and len(event_inds)>0 \
               and np.all(scores[event_inds[-1]:local_max_ind]>segment_threshold):
                
                # This max and the previous one are part of a continuous period of above-threshold scores
                
                if scores[local_max_ind]>scores[event_inds[-1]]:

                    # This max is higher than the last one so we keep it and discard the last one.
                    event_inds[-1]=local_max_ind

            else:
                event_inds.append(local_max_ind)
    return event_inds

def find_convolution_onsets(signatures,threshold,signature_weights={},bandwidth=timedelta(0,60*10),tmin=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),convolution_resolution=timedelta(0,60),require_continuous=True,epoch=datetime(2005,1,1,tzinfo=UTC)):
    scores,score_tnums=convolved_substorm_scores(signatures,signature_weights,bandwidth,convolution_resolution,tmin=tmin,tmax=tmax,epoch=epoch)

    onset_inds=search_convolution_scores(scores,threshold,require_continuous)

    return score_tnums[onset_inds]

def find_substorms_convolution(signatures,threshold,signature_weights={},tstep=timedelta(0,1800),bandwidth=timedelta(0,60*10),tmin=datetime(2005,1,1,tzinfo=UTC),tmax=datetime(2005,2,1,tzinfo=UTC),convolution_resolution=timedelta(0,60),return_times=False,epoch=datetime(2005,1,1,tzinfo=UTC)):

    score_tnums=find_convolution_onsets(signatures,threshold,signature_weights=signature_weights,bandwidth=bandwidth,convolution_resolution=convolution_resolution,tmin=tmin,tmax=tmax,epoch=epoch)

    bin_tnums=np.arange((tmin-epoch).total_seconds(),(tmax-epoch).total_seconds(),tstep.total_seconds())
    split_inds=np.searchsorted(score_tnums,bin_tnums)

    substorm_bin_inds=np.searchsorted(bin_tnums,score_tnums)
    substorm_bin_inds=substorm_bin_inds[(substorm_bin_inds>0) & (substorm_bin_inds<len(bin_tnums))]

    substorm_bins=np.zeros(len(bin_tnums),dtype=bool)
    substorm_bins[substorm_bin_inds]=True

    #substorm_times=[tmin+timedelta(seconds=maxtime) for maxtime in bin_maxtimes[substorm_bins]]
    if return_times:
        substorm_times=[epoch+timedelta(seconds=tnum) for tnum in score_tnums]
        return substorm_bins,substorm_times
    else:
        return substorm_bins

def find_substorms(signatures,threshold,signature_filters=None,mandatory_signatures=[],tstep=timedelta(0,1800),epoch=datetime(2005,1,1,tzinfo=UTC),return_times=False):

    if signature_filters is None:
        signature_filters=signatures.keys()

    retvals=make_grid(signatures,signature_filters=signature_filters,tstep=tstep,return_times=return_times,epoch=epoch)
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
