import matplotlib.pyplot as plt
import numpy as np
import astropy
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from collections import OrderedDict
from astropy.time import Time
import h5py
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
import time
import astropy.coordinates as coord
from astropy.coordinates import AltAz
import ephem

## Gets RA and Dec from an azimuth and elevation
## Doesn't involve WHAM frame transformation--still figuring that out

def get_ra_dec_ha(az, el, year, month, day, hour, minute, second):
    observer = ephem.Observer()
    observer.lon = str(-111.600562)
    observer.lat = str(31.958092)
    observer.elevation = 2091
    observer.date = ephem.Date((year, month, day, hour + 7, minute, second))
    ra,dec = observer.radec_of(str(az), str(el))
    ha = observer.sidereal_time() - ra
    return np.degrees(ra), np.degrees(dec), np.degrees(ha)

def radec2azel(ra, dec, dtime, az_val):
    observing_location = EarthLocation(lat='31.958092', lon='-111.600562')  
    observing_time = Time(dtime)  
    aa = AltAz(location=observing_location, obstime=observing_time)
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    aa_coos = coord.transform_to(aa)
    new_az, new_alt = aa_coos.az.deg, aa_coos.alt.deg
    if az_val < 0 and abs(az_val - new_az) > 350:
        new_az = new_az - 360
    return new_az, new_alt
    
    
# Path leading to regular WIYN data
wiynfile = "/xdisk/kkratter/kkratter/LBTO_data/WIYN/vimba.log"
wiyndat=[]

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

dts = []
years, months, days, hours, minutes, seconds = [], [], [], [], [], []
ra, dec, hour_angle = [], [], []
ast_az, ast_el, tcs_az, tcs_el = [], [], [], []

##convert from annoying hour formatting
wiyndiffs = np.zeros([len(wiyndict['<date']),4])
for i in range (len(wiyndict['<date'])):
    print('wiyndiffs', i)
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

for index in range(len(dates)):
    print('big loop', index)
    d,t = dates[index],times[index]
    itemsd = d.split('.')
    itemst = t.split(':')
    
    ## Fix az
    hour_val = int(itemst[0]) + 7
    if hour_val > 23:
        hour_val -= 24
    dt = datetime(int(itemsd[0]), int(itemsd[1]), int(itemsd[2]), int(itemst[0]), int(itemst[1]), int(itemst[2]))
    other_dt = datetime(int(itemsd[0]), int(itemsd[1]), int(itemsd[2]), hour_val, int(itemst[1]), int(itemst[2]))
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
    curr_ast_az, curr_ast_el = radec2azel(AST_Ra[index], AST_Dec[index], other_dt, float(az[index]))
    curr_tcs_az, curr_tcs_el = radec2azel(TCS_Ra[index], TCS_Dec[index], other_dt, float(az[index]))
    

    ra.append(currRa)
    dec.append(currDec)
    hour_angle.append(currHa)
    ast_az.append(curr_ast_az)
    ast_el.append(curr_ast_el)
    tcs_az.append(curr_tcs_az)
    tcs_el.append(curr_tcs_el)


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
ast_az = np.asarray(ast_az)
ast_el = np.asarray(ast_el)
tcs_az = np.asarray(tcs_az)
tcs_el = np.asarray(tcs_el)

features = np.zeros((len(el), 33))
features[:,0] = wiyn_rot
features[:,1] = mos_rot
features[:,2] = parallactic
features[:,3] = ast_az
features[:,4] = ast_el
features[:,5] = tcs_az
features[:,6] = tcs_el
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
features[:,18] = AST_Ra
features[:,19] = AST_Dec
features[:,20] = hour_angle

h5f = h5py.File('compiled_weather.hdf5', 'r')
weather_data = np.asarray(h5f.get('dataset1'))
h5f.close()

weather_times = weather_data[:,0]

weather_vals = []

weather_index, old_index = 0, 0


## Creates features array for weather data
for dt_index in range(len(features[:,11])):
    x = features[:,11][dt_index]
    difference_array = np.absolute(weather_times-x)[old_index:]
    weather_index = difference_array.argmin()
    weather_vals.append(weather_data[weather_index,:])

weather_vals = np.asarray(weather_vals)
print(weather_vals.shape)

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

#outputs = np.zeros((len(el),2))
features[:,31] = az
features[:,32] = el

features = np.delete(features, list(range(20900, 21700)), axis=0)


index = 0
cutsize = 10
new_features = []

while features.shape[0] - index > cutsize:
    series = features[index:index+cutsize,:]
    new_features.append(series)
    index += cutsize
new_features = np.asarray(new_features)

f = h5py.File('TotalDataWIYN_recurrent_with_weather_reverse.hdf5', 'w')
f.create_dataset('dataset1', data=new_features)

f.close()











