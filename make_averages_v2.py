#%%
import numpy as np
import xarray as xr
from glob import glob
from dask.diagnostics import ProgressBar
from multiprocessing import Pool
from astropy.time import Time
from astroplan import Observer
import astropy.units as u
import warnings
warnings.filterwarnings("ignore")

# %%
def int2str_2digit(int):
    return str(int).zfill(2)

def get_years_months(year, month):
    if (month>1) & (month<12):
        years = [str(year)[-2:]] *3
        months = list(map(int2str_2digit, [month-1, month, month+1]))
    elif month == 1:
        years = [str(year-1)[-2:]] + [str(year)[-2:]] *2
        months = '12 01 02'.split()
    elif month == 12:
        years = [str(year)[-2:]]*2 + [str(year+1)[-2:]]
        months = '11 12 01'.split()
    else:
        raise ValueError('month {} does not exist'.format(month))
    return years, months

def rename_sufix(ds, sufix_str):
    return ds.rename({var: sufix_str+var for var in ds.keys()})

def stat(resampled):
    return xr.merge([
        resampled.mean('time', keep_attrs=True).pipe(rename_sufix, 'mean_'), 
        resampled.std('time', keep_attrs=True).pipe(rename_sufix, 'std_'), 
        resampled.count('time', keep_attrs=True).pipe(rename_sufix, 'count_')
    ])

def cal_sunset_h(lat, lon, datetime):
    points = Observer(longitude=lon*u.deg, latitude=lat*u.deg, elevation=89*u.km)
    times = Time(datetime, format='datetime64')
    sunset = points.sun_set_time(times, which="previous")
    sunset_h = (times-sunset).sec/3600 # unit: hour
    return sunset_h

#%% test cal_set_h
# import matplotlib.pyplot as plt

# path_pattern_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/orbits_v2/0112/'
# filename_ver = 'iri_ch3_ver_004664.nc'
# with xr.open_dataset(path_pattern_ver+filename_ver) as ds:
#     lon = ds.longitude
#     lat = ds.latitude
#     datetime = ds.time.astype('datetime64[m]')
#     points = Observer(longitude=lon*u.deg, latitude=lat*u.deg, elevation=80*u.km)
#     times = Time(datetime, format='datetime64', precision=0)
#     # sunset = points.sun_set_time(times, which="previous")
#     # sunset_h = (times-sunset).sec/3600 # unit: hour

#     # sunset_h = cal_sunset_h(ds.latitude, ds.longitude, ds.time)

#     # fig, ax = plt.subplots(3, 1, sharex=True)
#     # ds.ver.where(ds.mr>0.8).plot(ax=ax[0], add_colorbar=False, y='z', ylim=(60e3, 95e3), vmin=0, vmax=.5e6)
#     # ds.apparent_solar_time.plot(ax=ax[1], x='time')
#     # ax[1].plot(ds.time, sunset_h)
#     # ds.latitude.plot(ax=ax[2], x='time')

#%%

file = '/home/anqil/Documents/osiris_database/ex_data/o2delta_lifetime.nc'
with xr.open_dataset(file) as lifetime:
    lifetime=lifetime.lifetime

def process_equi_index(ds):
    path_to_save_equi = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/orbits_v2/equi_index/'
    orbit_num_ready = [f[-9:-3] for f in glob(path_to_save_equi+'*.nc')]
    
    orbit_num = str(ds.orbit.values).zfill(6)
    if orbit_num in orbit_num_ready:
        with xr.open_dataset(path_to_save_equi+'equi_index_{}.nc'.format(orbit_num)) as ds_equi:
            ds = ds.update(ds_equi)
    else:
        hour_after_sunset = cal_sunset_h(ds.latitude, ds.longitude, ds.time)
        ds = ds.assign({'hour_after_sunset': ('time', hour_after_sunset)})
        equi_index = (1 - np.exp(-ds.hour_after_sunset / lifetime.interp(z=ds.z*1e-3)))
        ds = ds.assign({'equilibrium_index': equi_index})

        ds['hour_after_sunset equilibrium_index'.split()].to_netcdf(
            path_to_save_equi + 'equi_index_{}.nc'.format(orbit_num), mode='w')
    return ds

# %%
# year, month = 2001, 12 #test
# year = 2001

# for month in range(1,13):
def mk_stat_month(month):
    print('process', year, month)
    path_pattern_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/orbits_v2/{}{}/'
    filename_pattern_ver = 'iri_ch3_ver_*.nc'
    years, months = get_years_months(year, month) #get list of string with two digits both
    temp = [glob(path_pattern_ver.format(yy, mm)+filename_pattern_ver) for yy, mm in zip(years, months)]
    # print(path_pattern_ver.format(str(year)[-2:], str(month).zfill(2))+filename_pattern_ver)
    filelist = [item for sublist in temp for item in sublist]

    path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/Daily_NP_stats/no_equi/'
    filename_save_stat = 'Daily_NP_mean_{}{}.nc'.format(years[1], months[1])

    try:
        with xr.open_mfdataset(filelist) as mds:
            # print(mds)
            mds = mds.sel(time='{}-{}'.format(year, str(month).zfill(2)))
            cond_lat = mds.latitude>70
            cond_mr = (mds.mr>0.8) * (mds.A_peak>0.8) * (mds.A_diag>0.8)
            # cond_equi = mds.equilibrium_index>0.95
            # cond_cost = (mds.ver_cost_x + mds.ver_cost_y) > 10
            ver_NP_daily_resampled = mds[['ver']].where(
                    cond_lat * cond_mr #* cond_equi
                ).resample(time='1D')
            misc_NP_daily_resampled = mds['latitude sza apparent_solar_time'.split()].where(
                    cond_lat
                ).resample(time='1D')
            
            ver_daily_stat = stat(ver_NP_daily_resampled)
            misc_daily_stat = stat(misc_NP_daily_resampled)
            daily_stat = xr.merge([ver_daily_stat, misc_daily_stat])

            def str_unique(x):
                return str(np.unique(x))
            orbits_in_the_day = mds.orbit.to_series().resample('1D').apply(str_unique).to_xarray().rename('orbits_in_the_day')
            daily_stat = daily_stat.update({orbits_in_the_day.name: orbits_in_the_day})
            daily_stat.to_netcdf(path_save_stat+filename_save_stat, mode='w')
        return daily_stat

    except OSError as e:
        print(year, month, e)
        pass

    except KeyError as e:
        print(year, month, e.__class__)
        pass

#%%
if __name__ == '__main__':
    # year = int(input('Enter year: \n'))
    for year in range(2008, 2009):
        with Pool(processes=6) as p:
            result = p.map(mk_stat_month, range(1,13))


# %%
