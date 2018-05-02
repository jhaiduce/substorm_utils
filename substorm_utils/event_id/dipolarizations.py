import numpy as np
from matplotlib.dates import date2num,num2date
from datetime import datetime, timedelta
from pytz import UTC

def highpass(data,period=1440):
    import scipy.signal as signal

    N,Wn=signal.buttord(3./(period),3./(period*2),3,20,False)
    b,a=signal.butter(N,Wn,btype='highpass')

    filtered=signal.filtfilt(b,a,data)
    return filtered

def check_ranges(i,arr,ranges,op=np.max):
    satisfied=True

    for offs_min,offs_max,threshold_min,threshold_max in ranges:
        imin=max(i+offs_min,0)
        imax=min(i+offs_max,len(arr))

        if imin>imax-1:
            # Not enough points to evaluate
            satisfied=False
            break

        range_lim=op(arr[imin:imax])

        if threshold_min is not None and range_lim-arr[i]<threshold_min:
            # Threshold was not satisfied
            satisfied=False
            break

        if threshold_max is not None and range_lim-arr[i]>threshold_max:
            # Threshold was not satisfied
            satisfied=False
            break

    return satisfied

def find_dipolarizations_fu2012(numtimes,bx,by,bz):
    """
    Dipolarization ID algorithm described in Fu (2012). Only the magnetic field criteria are used, not the pressure/velocity criteria.

    Arguments:
    numtimes: Numeric time in the format returned by matplotlib's date2num
    bx,by,bz: Magnetic field components in GSM coordinates
    """

    theta=np.arctan2(bz,np.sqrt(bx**2+by**2))
    window_pos=numtimes[0]
    window_width=3.0/1440
    window_step=1.5/1440
    dip_times=[]

    while window_pos<=numtimes[-1]-window_width:
        in_window=(numtimes>window_pos)&(numtimes<=window_pos+window_width)
        theta_in_window=theta[in_window]
        theta_max=np.max(theta_in_window)
        
        # Theta must reach 45 deg or more and increase by 10 deg in the interval
        if theta_max>=np.pi/4 and \
           theta_max-np.min(theta_in_window)>10*np.pi/180 and\
           np.argmax(theta_in_window)>np.argmin(theta_in_window):
            
            # Candidate dipolarization time occurs at minimum bz in window
            dip_time=numtimes[in_window][np.argmin(bz[in_window])]

            # Only record if dipolarization follows the last one by at least 30 sec
            if len(dip_times)==0 or dip_time-dip_times[-1]>0.5/1440:
                dip_times.append(dip_time)
                
        window_pos+=window_step

    return dip_times 

def find_dipolarizations(times,bz_values,threshold_m5min=-1,threshold_m10min=-2,threshold_1min=0.5,threshold_2min=1,threshold_3min=2,threshold_30min=8,threshold_60min=15,min_time_between_events=20):
    minutes_from_start=(date2num(times)-date2num(times[0]))*1440
    deltas=minutes_from_start[1:]-minutes_from_start[:-1]
    if np.max(np.abs(deltas-1))>1e-2:
        raise ValueError('Times must be in 1-minute intervals')

    bz_filtered=highpass(bz_values)
    
    event_times=[]
    event_inds=[]
    i=20
    while i<len(bz_filtered)-60:
        al0=bz_filtered[i]
        if bz_filtered[i]-np.mean(bz_filtered[np.max(i-10,0):i])>threshold_m10min:
            i+=1
            continue
        if bz_filtered[i]-np.mean(bz_filtered[np.max(i-10,0):i])>threshold_m5min:
            i+=1
            continue
        #if bz_filtered[i+2]-al0 < threshold_1min: 
        #    i+=1
        #    continue
        #if bz_filtered[i+4]-al0 < threshold_2min:
        #    i+=1
        #    continue
        if np.max(bz_filtered[i:i+6])-al0 < threshold_3min:
            i+=1
            continue

        #if np.min(bz_filtered[i:i+5])-al0 < 0:
        #    i+=1
        #    continue

        if np.max(bz_filtered[i+4:i+31]) - al0 < threshold_30min:
            i+=1
            continue
        
        if np.max(bz_filtered[i:i+61]) - al0 < threshold_60min:
            i+=1
            continue

        event_inds.append(i)
        i+=1
        #i+=min_time_between_events

    return event_inds

def find_dipolarizations_theta(times,theta_values):

    
    minutes_from_start=(date2num(times)-date2num(times[0]))*1440
    deltas=minutes_from_start[1:]-minutes_from_start[:-1]
    if np.max(np.abs(deltas-1))>1e-2:
        raise ValueError('Times must be in 1-minute intervals')

    theta_filtered=highpass(theta_values)

    event_times=[]
    event_inds=[]
    i=20
    while i<len(theta_filtered)-60:

        # Check that angle decreases prior to onset
        #if theta_filtered[i]-np.mean(theta_filtered[np.max(i-60,0):i])<1:
        #    i+=1
        #    continue

        # Check that decrease is not too steep
        #if theta_filtered[i]-np.mean(theta_filtered[np.max(i-10,0):i])>-5:
        #    i+=1
        #    continue

        # Make sure increase is reasonably quick
        if np.max(theta_filtered[i:i+10])-theta_filtered[i]<5:
            i+=1
            continue

        # Make sure peak is high enough:
        if np.max(theta_filtered[i:i+60])-theta_filtered[i]<15:
            i+=1
            continue

        event_inds.append(i)
        i+=1
        #i+=min_time_between_events

    return event_inds

def find_dipolarizations_br_bz_theta(times,br,bz,theta):
    minutes_from_start=(date2num(times)-date2num(times[0]))*1440
    deltas=minutes_from_start[1:]-minutes_from_start[:-1]
    if np.max(np.abs(deltas-1))>1e-2:
        raise ValueError('Times must be in 1-minute intervals')

    theta_local_mins=np.where((theta[2:]>theta[1:-1]) &
                              (theta[:-2]>theta[1:-1]))[0]+1

    br_abs=np.abs(br)
    
    event_times=[]
    event_inds=[]
    i=20

    bz_max_ranges=(
        #(-10,0,0,None),
        #(-5,0,0,None),
        (0,10,2,None),
        (0,30,10,None),
        (0,60,16,None)
    )

    bz_min_ranges=(
        #(-10,0,-1,None),
        #(-5,2,0,None),
        #(1,10,-1,None),
        #(10,30,1,None),
        #(30,60,5,None),
        #(0,60,-3,None)
    )

    theta_max_ranges=(
        (-60,-10,5,None),
        (-10,-1,1,None),
        (0,15,5,None),
    )
    
    theta_min_ranges=(
        #(-60,-10,None,5),
        #(-10,5,None,2),
        #(2,5,None,3),
        #(0,15,None,8)
    )

    br_min_ranges=(
        #(-40,-10,None,-3),
        (-10,-2,None,-0.25),
        (2,10,None,-1),
        (10,40,None,-3),
    )

    br_max_ranges=(
        #(-30,-10,3,None),
        #(-10,0,1,None),
        #(0,10,1,None),
        #(10,30,3,None),
    )

    for i in theta_local_mins:
    #for i in range(len(times)):

        bz_min_satisfied=check_ranges(i,bz,bz_min_ranges,np.min)
        bz_min_satisfied=True
        bz_max_satisfied=check_ranges(i,bz,bz_max_ranges,np.max)
        #bz_max_satisfied=True
        br_min_satisfied=check_ranges(i,br_abs,br_min_ranges,np.min)
        #br_min_satisfied=True
        br_max_satisfied=check_ranges(i,br_abs,br_max_ranges,np.max)
        theta_min_satisfied=check_ranges(i,theta,theta_min_ranges,np.min)
        theta_min_satisfied=True
        theta_max_satisfied=check_ranges(i,theta,theta_max_ranges,np.max)
        theta_max_satisfied=True

        thresholds_satisfied=(bz_max_satisfied and bz_min_satisfied
            and br_min_satisfied and br_max_satisfied
            and theta_max_satisfied
        )
        
        if thresholds_satisfied:
            # All thresholds were satisfied, add to list
            event_inds.append(i)

    return event_inds
