import numpy as np
import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord
from collections import OrderedDict
from astropy.time import Time
import h5py
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
import time
import ephem

## Gets RA and Dec from an azimuth and elevation using Ephem
## +7 comes from time zone correction

def get_ra_dec_ha(az, el, year, month, day, hour, minute, second):
    observer = ephem.Observer()
    observer.lon = str(-111.600562)
    observer.lat = str(31.958092)
    observer.elevation = 2091
    observer.date = ephem.Date((year, month, day, hour + 7, minute, second))
    ra,dec = observer.radec_of(str(az), str(el))
    ha = observer.sidereal_time() - ra
    return np.degrees(ra), np.degrees(dec), np.degrees(ha)
    
# Path leading to regular WIYN data
wiynfile = "/xdisk/kkratter/kkratter/LBTO_data/WIYN/vimba.log" ## Change this to location of vimba log
wiyndat=[]

## Separating features from vimba log
with open(wiynfile) as f:
    gar=f.readline()    
    head=f.readline().split('>')
    gar = f.readline()
    head=head[0:-1]

    wiyndict=OrderedDict((key,None) for key in head)
    lineind=0
    for line in f:
        templist=(f.readline().split())
        index=0
        for key in wiyndict:
            if lineind!=0:
                wiyndict[key].append(templist[index])
            else:
                wiyndict[key]=[(templist[index])]
                
            index=index+1
        lineind=lineind+1


## Vectorizing dictionary data
dates = wiyndict['<date']
times = wiyndict[' <time']
wiyn_rot = np.asarray(wiyndict[' <wiyn rot (rad)'])
mos_rot = np.asarray(wiyndict[' <mos rot (rad)'])
parallactic = np.asarray(wiyndict[' <parallactic (rad)'])
az = np.asarray(wiyndict[' <az (deg)'])
el = np.asarray(wiyndict[' <el (deg)'])
TCS_Ra = np.asarray(wiyndict[' <TCS Ra'])
TCS_Dec = np.asarray(wiyndict[' <TCS Dec'])
AST_Ra = np.asarray(wiyndict[' <AST Ra (mid)'])
AST_Dec = np.asarray(wiyndict[' <AST Dec (mid)'])
X_pixel = np.asarray(wiyndict[' <X pixel'])
Y_pixel = np.asarray(wiyndict[' <Y pixel'])
XY_Ra = np.asarray(wiyndict[' <XY Ra (deg)'])
XY_Dec = np.asarray(wiyndict[' <XY Dec (deg)'])

## Splitting up dates into features, also recovering pure horizontal to equatorial conversion
dts = []
years, months, days, hours, minutes, seconds = [], [], [], [], [], []
ra, dec, hour_angle = [], [], []

for index in range(len(dates)):
    d,t = dates[index],times[index]
    itemsd = d.split('.')
    itemst = t.split(':')
    
    dt = datetime(int(itemsd[0]), int(itemsd[1]), int(itemsd[2]), int(itemst[0]), int(itemst[1]), int(itemst[2]))
    dt_time = Time(dt)
    years.append(int(itemsd[0]))
    months.append(int(itemsd[1]))
    days.append(int(itemsd[2]))
    hours.append(int(itemst[0]))
    minutes.append(int(itemst[1]))
    seconds.append(int(itemst[2]))
    unixtime = time.mktime(dt.timetuple())
    dts.append(unixtime)
    
    currRa, currDec, currHa = get_ra_dec_ha(az[index], el[index], int(itemsd[0]), int(itemsd[1]), int(itemsd[2]), int(itemst[0]), int(itemst[1]), int(itemst[2]))
    ra.append(currRa)
    dec.append(currDec)
    hour_angle.append(currHa)


dts = np.asarray(dts)
ra = np.asarray(ra)
dec = np.asarray(dec)
years = np.asarray(years)
monts = np.asarray(months)
days = np.asarray(days)
hours = np.asarray(hours)
minutes = np.asarray(minutes)
seconds = np.asarray(seconds)
hour_angle = np.asarray(hour_angle)

## Getting true outputs AST and current solution TCS
wiyndiffs = np.zeros([len(wiyndict['<date']),4])
for i in range (len(wiyndict['<date'])):
    c=SkyCoord(wiyndict[' <TCS Ra'][i],wiyndict[' <TCS Dec'][i],unit=(u.hourangle, u.deg))
    d=SkyCoord(wiyndict[' <AST Ra (mid)'][i],wiyndict[' <AST Dec (mid)'][i],unit=(u.hourangle, u.deg))
    wiyndiffs[i,0]=float(c.ra.deg)
    wiyndiffs[i,1]=float(c.dec.deg)
    wiyndiffs[i,2]=float(d.ra.deg)
    wiyndiffs[i,3]=float(d.dec.deg)

TCS_Ra = wiyndiffs[:,0]
TCS_Dec = wiyndiffs[:,1]
AST_Ra = wiyndiffs[:,2]
AST_Dec = wiyndiffs[:,3]

## Assembling features
features = np.zeros((len(el), 33))
features[:,0] = wiyn_rot
features[:,1] = mos_rot
features[:,2] = parallactic
features[:,3] = az
features[:,4] = el
features[:,5] = TCS_Ra
features[:,6] = TCS_Dec
features[:,7] = X_pixel
features[:,8] = Y_pixel
features[:,9] = XY_Ra
features[:,10] = XY_Dec
features[:,11] = dts
features[:,12] = years
features[:,13] = months
features[:,14] = days
features[:,15] = hours
features[:,16] = minutes
features[:,17] = seconds
features[:,18] = ra
features[:,19] = dec
features[:,20] = hour_angle

## Begins assembling weather features
h5f = h5py.File('compiled_weather.hdf5', 'r')
weather_data = np.asarray(h5f.get('dataset1'))
h5f.close()


weather_times = weather_data[:,0]
weather_vals = []
weather_index, old_index = 0, 0

## Creates features array for weather data
## Matches time indices in weather log to those in regular features
for dt_index in range(len(features[:,11])):
    x = features[:,11][dt_index]
    difference_array = np.absolute(weather_times-x)[old_index:]
    weather_index = difference_array.argmin()
    weather_vals.append(weather_data[weather_index,:])

weather_vals = np.asarray(weather_vals)

features[:,21] = weather_vals[:,0]
features[:,22] = weather_vals[:,1]
features[:,23] = weather_vals[:,2]
features[:,24] = weather_vals[:,3]
features[:,25] = weather_vals[:,4]
features[:,26] = weather_vals[:,5]
features[:,27] = weather_vals[:,6]
features[:,28] = weather_vals[:,7]
features[:,29] = weather_vals[:,8]
features[:,30] = weather_vals[:,9]

features[:,31] = AST_Ra
features[:,32] = AST_Dec


## Some garbage data in this range, could come from telescope calibration
features = np.delete(features, list(range(20900, 21700)), axis=0)


f = h5py.File('TotalDataWIYN_with_weather.hdf5', 'w')
f.create_dataset('dataset1', data=features)

f.close()











