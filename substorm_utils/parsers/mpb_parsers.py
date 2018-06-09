from datetime import datetime
from pytz import UTC

def parse_mpbdate(string):
    yearstr,timestr=string.split('/')
    year,month,day=[int(s) for s in yearstr.split('-')]
    hourstr,minutestr,secondstr=timestr.split(':')

    hour=int(hourstr)
    minute=int(minutestr)
    second_f=float(secondstr)

    second=int(second_f)

    microsecond=int((second_f-second)*1000000)

    if year==0:
        return None

    return datetime(year,month,day,hour,minute,second,microsecond,tzinfo=UTC)

def parse_onsets(filename):

    onsets=[]

    with open(filename) as fh:
        for line in fh.readlines():
            
            onsets.append(parse_mpbdate(line))

    return onsets
    
def parse_onset_tmax(filename):

    onsets=[]
    tmax=[]

    with open(filename) as fh:
        for line in fh.readlines():
            onset_str,tmax_str=line.split()
            onsets.append(parse_mpbdate(onset_str))
            tmax.append(parse_mpbdate(tmax_str))

    return onsets,tmax

def parse_index(filename):
    times=[]
    index=[]
    with open(filename) as fh:
        for line in fh.readlines():
            timestr,indstr=line.split()
            times.append(parse_mpbdate(timestr))
            index.append(float(indstr))

    return times,index
