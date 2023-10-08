#import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from collections import OrderedDict
import h5py
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
import time
import ephem
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy import units as u
import numpy as np
import astropy.coordinates as coord

## Compiles weather data from environment logs
path = '/xdisk/kkratter/kkratter/LBTO_data/WIYN'

files = [
    'wiyn_environment_CustomWindow_14.txt',
    'wiyn_environment_CustomWindow_15.txt',
    'wiyn_environment_CustomWindow_16.txt',
    'wiyn_environment_CustomWindow_17.txt',
    'wiyn_environment_CustomWindow_18.txt',
    'wiyn_environment_CustomWindow_19.txt',
    'wiyn_environment_CustomWindow_20.txt',
    'wiyn_environment_CustomWindow_21.txt',
    'wiyn_environment_CustomWindow_22.txt',
]
final = []
j = 0
for file in files:
    currFile = open('/xdisk/kkratter/kkratter/LBTO_data/WIYN/' + file, 'r')
    for line in currFile:
        j += 1
        items = line.split()
        date = items[0].split('-')
        if date[0] != 'Date':
            times = items[1].split(':')
            dt = datetime(int(date[0]), int(date[1]), int(date[2]), int(times[0]), int(times[1]), int(times[2]))
            unixtime = float(time.mktime(dt.timetuple()))
            l = [unixtime]
            for i in range(2, len(items)):
                l.append(float(items[i]))
            final.append(l)
            if j % 100000 == 0:
final = np.asarray(final)

f = h5py.File('compiled_weather.hdf5', 'w')
f.create_dataset('dataset1', data=final)

f.close()






