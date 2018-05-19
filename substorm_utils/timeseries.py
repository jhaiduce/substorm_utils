import numpy as np

def interp_timeseries(data,oldtimes,newtimes):
    from scipy.interpolate import interp1d

    epoch=oldtimes[0]

    oldtimes_f=[(t-epoch).total_seconds() for t in oldtimes]
    newtimes_f=[(t-epoch).total_seconds() for t in newtimes]

    interpolator=interp1d(oldtimes_f,data,fill_value=np.nan,bounds_error=False)

    return interpolator(newtimes_f)

