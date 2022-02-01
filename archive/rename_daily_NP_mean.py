#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
import os

#%%
freq = 'D'
time_stamp = pd.date_range(start='2001-01', end='2018', freq=freq)

with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
    rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(time=Time(rough_orbit.mjd, format='mjd').datetime64)
    rough_orbit = rough_orbit.interp(time=time_stamp, kwargs=dict(fill_value='extrapolate')).round()
    rough_orbit = rough_orbit.where(rough_orbit.orbit>0, drop=True).astype(int)

#%%
path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/'
days_ready = [int(f[14:].replace('.nc', '')) for f in os.listdir(path) if 'nc' in f]

for day in list(range(len(rough_orbit.time))):
    if day in days_ready:
        # print(day, 'is ready')
        
        date_str = rough_orbit.time[day].pipe(lambda x: (str(x.dt.year.item())[-2:], str(x.dt.month.item()).zfill(2), str(x.dt.day.item()).zfill(2)))
        # print('{}{}{}'.format(*date_str))
        os.rename(path+'Daily_NP_mean_{}.nc'.format(day), 
            path+'daily_mean_NP_{}{}{}.nc'.format(*date_str))
    else:
        # print('no day', day)
        pass

# %%
