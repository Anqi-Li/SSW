#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from os import listdir


#%%
def plot_one_year(polar_ver, year, ax=None):
    date_range = ('{}-10'.format(year-1), '{}-03-10'.format(year))
    one_year = polar_ver.sel(
            time=slice(*date_range), 
            z=slice(70e3,95e3),
        ).chunk(dict(time=-1)
        ).interpolate_na(
            dim='time'
        )
    one_year.where(one_year>0).plot(
        x='time', y='z',
        cmap='viridis',
        norm=LogNorm(vmin=1e3, vmax=1e5),
        ax=ax,
        add_colorbar=False,
        )
    # ax.set_title("{}-{}".format(year-1, year))
    ax.text(0,1, "{}-{}".format(year-1, year),
        backgroundcolor='w',
        horizontalalignment='left',
        verticalalignment='top', 
        transform=ax.transAxes)

    ax.set_xlim(np.array(date_range).astype(np.datetime64))

#%% O2 Delta
def change_description_o2delta(ds):
    ds.mean_ver['description'] = 'IRI O2(Delta) volume emission rate'
    ds.std_ver['description'] = 'IRI O2(Delta) volume emission rate'
    ds.count_ver['description'] = 'IRI O2(Delta) volume emission rate'
    return ds

path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/Daily_NP_stats/no_equi/'
with xr.open_mfdataset(
    path_save_stat+'Daily_NP_mean_*.nc', 
    preprocess=change_description_o2delta
    ) as mds:
    NP_ver = mds.mean_ver

    fig, ax = plt.subplots(len(range(2002, 2016)), 1, sharey=True, figsize=(10,20))
    for i, year in enumerate(range(2002, 2016)):
        plot_one_year(NP_ver, year, ax[i])
    [ax[i].set_xticklabels([]) for i in range(len(ax)-1)]

#%% OH
def reindex_lat(ds):
    return ds.reindex(latitude_bin=[-80., -60., -40., -20.,   0.,  20.,  40.,  60., 80.])

path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/zenith/'
with xr.open_mfdataset(
        # [path_save_stat+'PM_daily_zonal_mean_{}.nc'.format(y) for y in [2001, 2002]],
        path_save_stat+'PM_daily_zonal_mean_*.nc',
        preprocess=reindex_lat,    
    ) as mds:
    NP_ver = mds.mean_ver.sel(latitude_bin=80).drop('latitude_bin')

    fig, ax = plt.subplots(len(range(2002, 2016)), 1, sharey=True, figsize=(10,20))
    for i, year in enumerate(range(2002, 2016)):
        plot_one_year(NP_ver, year, ax[i])
    [ax[i].set_xticklabels([]) for i in range(len(ax)-1)]

#%% OH specific dates
def reindex_lat(ds):
    return ds.reindex(latitude_bin=[-80., -60., -40., -20.,   0.,  20.,  40.,  60., 80.])

path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/zenith/'
with xr.open_mfdataset(
        # [path_save_stat+'PM_daily_zonal_mean_{}.nc'.format(y) for y in [2001, 2002]],
        path_save_stat+'PM_daily_zonal_mean_*.nc',
        preprocess=reindex_lat,    
    ) as mds:

    NP_ver = mds.mean_ver.sel(latitude_bin=80).drop('latitude_bin')
    NP_ver.sel(time=slice('2009-01-01', '2009-02-01')).plot(
        x='time', y='z',
        cmap='viridis',
        norm=LogNorm(vmin=1e3, vmax=1e5),
        ylim=(70e3, 95e3),
        # add_colorbar=False,
        # add_legend=False,
        )













#%% O2delta
path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/Daily_NP_stats/no_equi/'
with xr.open_mfdataset(
    path_save_stat+'Daily_NP_mean_*.nc', 
    ) as mds:
    # print(mds)
    # mds.mean_ver.plot(y='z')
    NP_ver = mds.mean_ver.sel(
        z=slice(70e3,95e3),
    ).chunk(dict(time=-1)
    ).interpolate_na(
        dim='time',
    ).roll(
        time=0#int(365/2),
    ).assign_coords(
        dict(year=mds.time.dt.year, doy=mds.time.dt.dayofyear)
    ).set_index(
        time=['year', 'doy']
    ).unstack()
        
    fc = NP_ver.where(
            NP_ver>0
        ).plot(
            x='doy', y='z', row='year',
            figsize=(10,15),
            cmap='viridis',
            norm=LogNorm(),
            robust=True,
            # xlim=(100, 250)
        )
    # fc.axes[-1,0].set_xticklabels((fc.axes[-1,0].get_xticks()-int(365/2)).astype(int))
    # [fc.axes[i, 0].axvline(x=int(365/2), c='r') for i in range(len(fc.axes))]

#%%
path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/zenith/'
filename = 'PM_daily_zonal_mean_{}.nc'
years = list(range(2001,2016))
with xr.open_mfdataset([path+filename.format(y) for y in years]) as mds:
    mds = mds.rename({'latitude_bin': 'latitude_bins'})
    mds = mds.reindex(latitude_bins=mds.latitude_bins[::-1]).load()
    mds = mds.assign_coords(latitude_bins=mds.latitude_bins.astype(int),
                            z = mds.z*1e-3, #m -> km
                            )

#%%
# am_pm = 'PM'
# year = '{}'
path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/'
# filename = '{}_daily_zonal_mean_{}.nc'.format(am_pm, year)
with xr.open_mfdataset(path+'*nc', compat='override', coords=['time']) as mds:
#     # mds = mds.sel(time=~mds.indexes['time'].duplicated()) # think about other alternative?
    print(mds)

#%%
with xr.open_dataset(path+'Daily_NP_mean_3357.nc') as ds:
    # z_sample = ds.z
    dims_sample = ds.dims

orbit_ready = [int(f[14:].replace('.nc', '')) for f in listdir(path) if 'nc' in f]
for orbit in orbit_ready:
    try:
        with xr.open_dataset(path+'Daily_NP_mean_{}.nc'.format(orbit)) as ds:
            # if (ds.z == z_sample).all():
            if (ds.dims == dims_sample):
                pass
                # print(orbit, 'is ok')
            else:
                print(orbit, 'is not ok')

    except ValueError as e:
        print(orbit, 'is having ValueError')
        print(e)
# %%
# path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/dayglow_o3/averages/'
# with xr.open_mfdataset(path+'*nc') as mds:
    # mds = mds.sel(time=~mds.indexes['time'].duplicated()) # think about other alternative?

path = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/'
filename_pattern = 'daily_mean_NP_{year}{month}{day}.nc'
with xr.open_mfdataset(path+filename_pattern.format(year='01', month='*', day='*')) as mds:
    polar_ver = mds.mean_ver.sel(
            latitude_bin=80
        ).assign_coords(
            dict(year=mds.time.dt.year, doy=mds.time.dt.dayofyear)
        ).set_index(
            time=['year', 'doy']
        ).unstack()
        
    polar_ver.where(
            polar_ver>0
        ).sel(
            z=slice(70,105),
            doy=slice(50,300),
        ).plot(
            x='doy', y='z', row='year',
            figsize=(10,15),
            cmap='viridis',
            norm=LogNorm(),
            robust=True,
        )

# .plot(x='time', y='z', ylim=[50, 105])