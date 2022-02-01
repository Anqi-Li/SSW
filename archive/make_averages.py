#%%
import numpy as np
from numpy.lib.function_base import _msort_dispatcher
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from os import listdir
from dask.diagnostics import ProgressBar
from multiprocessing import Pool

#%%
freq = 'D'
time_stamp = pd.date_range(start='2001-01', end='2018', freq=freq)

with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
    rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(time=Time(rough_orbit.mjd, format='mjd').datetime64)

    rough_orbit = rough_orbit.interp(time=time_stamp, kwargs=dict(fill_value='extrapolate')).round()
    rough_orbit = rough_orbit.where(rough_orbit.orbit>0, drop=True).astype(int)

#%%
path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/'
filelist_full = [f for f in listdir(path+'orbits/') if 'nc' in f]
#%%
def pre(ds):
    return ds.drop_vars('orbit').sel(time=~ds.indexes['time'].duplicated())
#%%
# for day in range(len(rough_orbit.time)):
def process_one_day(day):
    '''
    input day is the day number count from 2001-02-21
    '''
    day_ready = [int(f[14:].replace('.nc', '')) for f in listdir(path+'averages/') if 'nc' in f]
    if day not in day_ready:
        orbit_range = range(*rough_orbit.orbit.isel(time=slice(day, day+2)).values)
        filelist_day = [f for f in filelist_full if int(f[-9:-3]) in orbit_range]
        
        if len(filelist_day) > 0:
            print(day, orbit_range, filelist_day)
            try:
                with xr.open_mfdataset([path+'orbits/'+f for f in filelist_day], preprocess=pre) as ds:
                    ds = ds.sel(time=~ds.indexes['time'].duplicated())
                    cond_mr = (ds.mr>0.8) * (ds.A_diag>0.8) * (ds.A_peak>0.8)
                    ds_ver = ds[['ver']].where(cond_mr)
                    ds_other = ds['latitude longitude'.split()]
                    ds_day = xr.merge([ds_ver, ds_other])
                    cond_latitude = ds_day.latitude>70
                    ds_day_mean = ds_day.where(cond_latitude).mean('time').assign_coords(
                                    time=rough_orbit.time[day]
                                ).rename(
                                    {var:'mean_{}'.format(var) for var in ds_day.data_vars}
                                )
                    ds_ver_count = ds_ver.where(cond_latitude).count('time').rename(ver='count_ver')
                    ds_ver_std = ds_ver.where(cond_latitude).std('time').rename(ver='std_ver')
                    final = xr.merge([ds_day_mean, ds_ver_std, ds_ver_count])
                    final = final.assign_attrs(orbits=[int(f[-9:-3]) for f in filelist_day])
                    
                    final.to_netcdf(path+'averages/Daily_NP_mean_{}.nc'.format(day))
            except OSError as e:
                log = open(path+'averages/log.txt', 'a')
                log.write(str(e))
                log.close()
                pass
            except ValueError as e:
                print(e)
                print('day', day)
            except:
                return
    else:
        # print('day {} already done'.format(day))
        pass
        
for day in list(range(len(rough_orbit.time))):
    process_one_day(day)
# with Pool(processes=5) as p:
#        p.map(process_one_day, list(range(len(rough_orbit.time))))
#%% test on one day
# day=2116#2791
# orbit_range = range(*rough_orbit.orbit.isel(time=slice(day, day+2)).values)
# filelist_day = [f for f in filelist_full if int(f[-9:-3]) in orbit_range]
# try:
#     with xr.open_mfdataset([path+'orbits/'+f for f in filelist_day], preprocess=pre) as ds:
#         cond_mr = (ds.mr>0.8) * (ds.A_diag>0.8) * (ds.A_peak>0.8)
#         ds_ver = ds[['ver']].where(cond_mr)
#         ds_other = ds['latitude longitude'.split()]
#         ds_day = xr.merge([ds_ver, ds_other])
#         cond_latitude = ds_day.latitude>70
#         ds_day_mean = ds_day.where(cond_latitude).mean('time').assign_coords(
#                         time=rough_orbit.time[day]
#                     ).rename(
#                         {var:'mean_{}'.format(var) for var in ds_day.data_vars}
#                     )
#         ds_ver_count = ds_ver.where(cond_latitude).count('time').rename(ver='count_ver')
#         ds_ver_std = ds_ver.where(cond_latitude).std('time').rename(ver='std_ver')
#         final = xr.merge([ds_day_mean, ds_ver_std, ds_ver_count])
#         final = final.assign_attrs(orbits=[int(f[-9:-3]) for f in filelist_day])
# except OSError as e:
#     print(e)

#%% O3 dataset from Marcus
# file = '/home/anqil/Documents/osiris_database/ex_data/o2delta_lifetime.nc'
# with xr.open_dataset(file) as ds:
#     # print(ds)
#     lifetime = ds.lifetime 
    
#%%
# def average_year(year, am_pm):
#     print(year)
#     path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/marcus_results/{year}/'
#     filename = 'iri_ver_o3_orbit*_v2p0.nc'

#     with xr.open_mfdataset(path.format(year=year)+filename) as ds:
#         if am_pm == 'PM':
#             cond_lst = ds.lst>12 
#         elif am_pm == 'AM':
#             cond_lst = ds.lst<=12
#         elif am_pm == 'ALL':
#             cond_lst = ds.lst>0

#         cond_z = ds.z>50
#         cond_ver_mr = ds.ver_mr_rel>0.8
#         cond_ver_chisq = (ds.ver_cost_x+ds.ver_cost_y)<10
#         ds_ver = ds[['ver']].where(cond_ver_mr * cond_ver_chisq * cond_z * cond_lst)

#         cond_o3_mr = ds.ver_mr_rel>0.8
#         cond_o3_equi = (1-np.exp(-ds.hour_after_sunrise/lifetime))>0.95
#         cond_o3_lm = np.logical_and((ds.o3_status != 0), ((ds.o3_cost_x+ds.o3_cost_y)<10))
#         ds_o3 = ds[['o3']].where(cond_o3_mr * cond_o3_lm * cond_o3_equi * cond_lst)
        
#         vars = ['latitude', 'longitude', 'sza', 'hour_after_sunrise', 'lst']
#         ds_other = ds[vars].where(cond_lst)
#         ds_year = xr.merge([ds_ver, ds_o3, ds_other])

#         #zonal - daily mean
#         dlat = 20
#         latitude_bins = np.arange(-90, 90+dlat, dlat)
#         latitude_labels = latitude_bins[1:]-dlat/2
#         groups_lat = ds_year.groupby_bins(
#             ds_year.latitude, bins=latitude_bins, 
#             labels=latitude_labels)
#         lat_coord = []
#         mean_daily, std_daily, count_daily = [], [], []
#         for label, data in groups_lat:
#             mean_daily.append(data.resample(time='D').mean('time', keep_attrs=True))
#             std_daily.append(data.resample(time='D').std('time', keep_attrs=True))
#             count_daily.append(data.resample(time='D').count('time', keep_attrs=True))
#             lat_coord.append(label)
#             print(label)
#         mean_daily = xr.concat(mean_daily, dim='latitude_bin').assign_coords(latitude_bin=lat_coord).sortby('latitude_bin')
#         std_daily = xr.concat(std_daily, dim='latitude_bin').assign_coords(latitude_bin=lat_coord).sortby('latitude_bin')
#         count_daily = xr.concat(count_daily, dim='latitude_bin').assign_coords(latitude_bin=lat_coord).sortby('latitude_bin')
#         final =  xr.merge(
#             [mean_daily.rename({k: 'mean_{}'.format(k) for k in mean_daily.keys()}), 
#             std_daily.rename({k: 'std_{}'.format(k) for k in std_daily.keys()}), 
#             count_daily.rename({k: 'count_{}'.format(k) for k in count_daily.keys()})]
#             ).assign_attrs(am_pm=am_pm)

#         return final


#%%
# am_pm = input('AM or PM? \n')
# for year in range(2001, 2018):
#     ds = average_year(year, am_pm)
#     print('saving year {}'.format(year))
#     path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/dayglow_o3/averages/'
#     filename = '{}_daily_zonal_mean_{}.nc'.format(am_pm, year)
#     delayed_obj = ds.to_netcdf(path+filename, mode='w', compute=False)
#     with ProgressBar():
#         delayed_obj.compute()


#%%
# def save_nc(data, filename_save):
#     data = mds
#     encoding = {
#         'z': {'dtype': 'float32'},
#         'clima_z': {'dtype': 'int32'},
#     }
#     for var in data.data_vars:
#         if data[var].dtype == 'float64':
#             encoding[var] = {'dtype': 'float32', 'zlib': True, 'shuffle': True}
#         elif data[var].dtype == 'int64':
#             encoding[var] = {'dtype': 'int32', 'zlib': True, 'shuffle': True}

#     data.to_netcdf(filename_save, encoding=encoding)