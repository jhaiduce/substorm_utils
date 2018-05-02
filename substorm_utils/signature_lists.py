from datetime import datetime, timedelta
import numpy as np
from borovsky_id_algorithm import borovsky_id_algorithm
import spacepy.datamodel as dm
from datetime import datetime
from find_dipolarizations import find_dipolarizations_br_bz_theta
from cache_decorator import cache_result
import itertools
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache
from mpb_parsers import parse_onset_tmax, parse_index, parse_onsets
from matplotlib.dates import num2date, date2num
from pytz import UTC
from timeseries import interp_timeseries
import os

@lru_cache(maxsize=10)
@cache_result()
def get_dipolarizations(run_name,satname,datadir='.'):

    namestr=run_name.replace('/','').replace(' ','_')
    satdata=dm.fromHDF5(os.path.join(datadir,namestr+'_'+satname+'.h5'))
    times=np.array(num2date(satdata['numtime']))
    mlt=(np.arctan2(-satdata['Y'],-satdata['X'])*12/np.pi)%24
    newtimes=np.array([datetime(2005,1,1,tzinfo=UTC)+timedelta(seconds=s*60) for s in range(1440*31)])
    bx=interp_timeseries(satdata['bx'],times,newtimes)
    by=interp_timeseries(satdata['by'],times,newtimes)
    bz=interp_timeseries(satdata['bz'],times,newtimes)
    x=interp_timeseries(satdata['X'],times,newtimes)
    y=interp_timeseries(satdata['Y'],times,newtimes)
    mlt=interp_timeseries(mlt,times,newtimes)
    times=newtimes

    br=(bx*x+by*y)/np.sqrt(x**2+y**2)
    theta=np.arctan2(bz,np.sqrt(bx**2+by**2))*180/np.pi

    dip_inds=find_dipolarizations_br_bz_theta(newtimes,br,bz,theta)
    night_inds=np.where((mlt<6)|(mlt>18))[0]
    dip_inds=np.intersect1d(dip_inds,night_inds)
    dip_times=times[dip_inds]

    satdata['time']=newtimes
    satdata['bx']=bx
    satdata['by']=by
    satdata['bz']=bz
    satdata['br']=br
    satdata['theta']=theta

    return dip_times

def get_tnums(times,epoch=datetime(2005,1,1,tzinfo=UTC)):
    return np.array([(t-epoch).total_seconds() for t in times])

def get_model_signature_lists(runprops,epoch=datetime(2005,1,1,tzinfo=UTC)):

    onset_lists={}

    auroral_inds=dm.fromHDF5(os.path.join(datadir,runprops['name'].replace('/','').replace(' ','_')+'_auroral_inds.h5'))

    onsets_borovsky=borovsky_id_algorithm(auroral_inds['AL'])
    onsets_borovsky=[datetime(2005,1,1,tzinfo=UTC)+timedelta(minutes=m) for m in onsets_borovsky]

    onset_lists['AL']=get_tnums(onsets_borovsky,epoch)

    plasmoid_data=dm.fromHDF5(os.path.join(datadir,'plasmoids_'+runprops['name'].replace('/','').replace(' ','_')+'.h5'))
    plasmoid_times=np.array([datetime.strptime(s,'%Y-%m-%dT%H:%M:%S').replace(tzinfo=UTC) for s in plasmoid_data['time']])
    
    onset_lists['plasmoids']=get_tnums(plasmoid_times,epoch)

    midn_distances=(3,5,7,10,)#15,20,30,40,50)

    dipolarizations=list(itertools.chain.from_iterable([
        get_dipolarizations(runprops['name'],satname)
        for satname in ['goes10','goes12']+
        ['midn_{0:02d}'.format(distance) for distance in midn_distances]]))
    dipolarizations.sort()

    onset_lists['dipolarizations']=get_tnums(dipolarizations,epoch)

    namestr=runprops['name'].replace('/','').replace(' ','_')
    onset,tmax=parse_onset_tmax(os.path.join(datadir,namestr+'_onset_tmax.txt'))
    onset_lists['MPB']=get_tnums(onset,epoch)

    return onset_lists

def get_obs_signature_lists(epoch=datetime(2005,1,1,tzinfo=UTC)):
    
    onset_lists={}

    supermag_data=np.loadtxt(os.path.join(datadir,'20160728-19-38-supermag.txt'),skiprows=88)
    obs_al=supermag_data[:,6]

    onsets_borovsky=borovsky_id_algorithm(obs_al)
    onsets_borovsky=[datetime(2005,1,1,tzinfo=UTC)+timedelta(minutes=m) for m in onsets_borovsky]

    onset_lists['AL']=get_tnums(onsets_borovsky,epoch)

    dipolarizations=list(itertools.chain.from_iterable([
        get_dipolarizations('obs',satname)
        for satname in ['goes10','goes12']]))
    dipolarizations.sort()

    onset_lists['dipolarizations']=get_tnums(dipolarizations,epoch)

    onsets=parse_onsets(os.path.join(datadir,'obs_mpb_onsets.txt'))
    onset_lists['MPB']=get_tnums(onsets,epoch)

    borovsky_epdata_substorms=np.loadtxt(os.path.join(datadir,'borovsky_epdata_substorms.txt'),skiprows=1)
    in_month=(borovsky_epdata_substorms[:,2]==2005) & (borovsky_epdata_substorms[:,3]<32)
    borovsky_epdata_substorms=borovsky_epdata_substorms[in_month]
    borovsky_epdata_substorms=((borovsky_epdata_substorms[:,3]-1)*1440).astype(int)
    borovsky_epdata_substorms=[datetime(2005,1,1,tzinfo=UTC)+timedelta(minutes=m) for m in borovsky_epdata_substorms]

    onset_lists['epdata']=get_tnums(borovsky_epdata_substorms,epoch)

    image_fuv_substorms=np.loadtxt('substorms_2000_2005.log',skiprows=2,dtype=[('datestr','|S22'),('x',float),('y',float),('dist',float),('counts',float),('latgeo',float),('longeo',float),('latmag',float),('lonmag',float),('MLT',float)])
    image_fuv_times=[datetime.strptime(datestr[4:],'%Y_%m%d_%H:%M:%S').replace(tzinfo=UTC) for datestr in image_fuv_substorms['datestr']]
    onset_lists['image']=get_tnums(image_fuv_times,epoch)

    return onset_lists

