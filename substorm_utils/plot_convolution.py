from pytz import UTC
from datetime import datetime, timedelta
from substorm_utils.bin_listings import find_convolution_onsets, convolved_substorm_scores
from matplotlib import pyplot as plt
import numpy as np

def plot_convolution_score(signatures,ax,tmin,tmax,convolution_resolution=timedelta(0,60),bandwidth=timedelta(minutes=10),epoch=datetime(2005,1,1,tzinfo=UTC),**kwargs):
    scores,tnums=convolved_substorm_scores(signatures,resolution=convolution_resolution,bandwidth=bandwidth,tmin=tmin,tmax=tmax,epoch=epoch)
    times=np.array([epoch+timedelta(0,s) for s in tnums])
    in_range=(times>=tmin) & (times<=tmax)
    times=times[in_range]
    scores=scores[in_range]
    ax.plot(times,scores)
    return scores,times

def make_convolution_figure(signatures,threshold,tstart,tend,bandwidth=timedelta(minutes=10),epoch=datetime(2005,1,1,tzinfo=UTC),signature_type_labels={}):
    onsets_all=find_convolution_onsets(signatures,threshold,bandwidth=bandwidth,epoch=epoch,tmin=tstart,tmax=tend)
    onsets_all=[epoch+timedelta(0,s) for s in onsets_all]
    fig=plt.figure(figsize=[5.5,5.5])
    from matplotlib.gridspec import GridSpec

    gs=GridSpec(len(signatures)+1,1,hspace=0,right=0.95,top=0.98,left=0.08,bottom=0.08)
    axes=[]

    for i,key in enumerate(signatures.keys()):
        if i>0:
            subplot_kwargs={'sharex':axes[0]}
        else:
            subplot_kwargs={}
        ax=fig.add_subplot(gs[i,0],**subplot_kwargs)
        axes.append(ax)
        plot_convolution_score({key:signatures[key]},ax,tstart,tend,bandwidth=bandwidth,epoch=epoch)
        ax.set_ylabel(signature_type_labels.get(key,key))
        ax.set_xlim(tstart,tend)

    ax=fig.add_subplot(gs[-1,0],sharex=axes[0])
    axes.append(ax)
    ax.set_ylabel('All')
    scores,times=plot_convolution_score(signatures,ax,tstart,tend,bandwidth=bandwidth,epoch=epoch)
    try:
        len(threshold)
    except:
        ax.axhline(threshold,color='r',alpha=0.5,linewidth=1)
    else:
        from matplotlib.dates import date2num
        tstart_num=date2num(tstart)
        mpl_dates=np.arange(len(threshold))/1440.+tstart_num
        ax.plot(mpl_dates,threshold,color='r',alpha=0.5,linewidth=1)
        
    ax.set_xlim(tstart,tend)
    
    labelpos=(0.98,0.94)
    from string import ascii_lowercase
    subplot_labels=[ascii_lowercase[i] for i in range(len(signatures)+1)]

    from matplotlib.dates import DateFormatter
    for i,ax in enumerate(axes):
        if i!=len(signatures):
            plt.setp(ax.get_xticklabels(),visible=False)

        # Add a label to the axis
        label=subplot_labels[i]
        text=ax.text(labelpos[0],labelpos[1],label,transform=ax.transAxes,weight='bold',fontsize=11,verticalalignment='top',color='k',horizontalalignment='right')

        ax.tick_params('x',which='both',direction='inout',top=True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3,integer=True))

        for onset in onsets_all:
            ax.axvline(onset,linewidth=0.5,color='k',linestyle='--')

        ymin,ymax=ax.get_ylim()
        min_ymax=1.2
        if ymax<min_ymax:
            ax.set_ylim(ymin,min_ymax)

    return axes,fig
