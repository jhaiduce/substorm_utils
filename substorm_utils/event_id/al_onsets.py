import numpy as np
from matplotlib.dates import date2num,num2date

def borovsky_id_algorithm(al_values):
    """
    Identifies candidate substorm onsets from the AL index using the procedure described in Borovsky (2017).

    al_values: A sequence of AL index values in nT, with a time cadence of 1 minute
    """

    event_inds=[]
    i=0

    # Find indices where al_values decreases by 150 or more in 15 minutes
    descent_intervals=np.where((al_values[:-15]-al_values[15:])>=150)[0]

    for i,idescent in enumerate(descent_intervals):

        # Ignore this interval if it is within 15 minutes of a previous interval
        if i>0:
            if idescent-descent_intervals[i-1]<30: continue

        # Find decreases greater than 10 nT in 2 minutes
        candidate_inds=np.where((al_values[idescent:idescent+13]-al_values[idescent+2:idescent+15])>10)[0]

        for candidate_ind in candidate_inds:

            # 45-minute time integral prior to candidate onset
            before_integral=np.sum(
                al_values[
                    max(candidate_ind+idescent-45,0):candidate_ind+idescent])

            # 45-minute time integral after candidate onset
            after_integral=np.sum(
                al_values[
                    candidate_ind+idescent:min(candidate_ind+idescent+45,
                                               len(al_values))])

            # Ratio after_integral/before_integral must be at least 1.5
            if after_integral < before_integral*1.5:

                # Candidate satisfies all selection criteria
                event_inds.append(candidate_ind+idescent)

                # Ignore remaining candidates in this descent interval
                break

    return event_inds

def supermag_id_algorithm(al_values,threshold_1min=-15,threshold_2min=-30,threshold_3min=-45,threshold_30min=-100,min_time_between_events=20):
    """
    Identifies candidate substorm onsets from the AL index using the procedure described in Newell (2011).

    al_values: A sequence of AL index values in nT, with a time cadence of 1 minute
    """

    event_inds=[]
    i=0
    while i<len(al_values)-30:
        al0=al_values[i]
        if al_values[i+1]-al0 >= threshold_1min: 
            i+=1
            continue
        if al_values[i+2]-al0 >= threshold_2min:
            i+=1
            continue
        if al_values[i+3]-al0 >= threshold_3min:
            i+=1
            continue
        if np.average(np.array(al_values[i+4:i+31])) - al0 >= threshold_30min:
            i+=1
            continue
        
        event_inds.append(i)
        i+=min_time_between_events

    return event_inds
