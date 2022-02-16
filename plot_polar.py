#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, NoNorm

def reindex_lat(ds):
    return ds.reindex(latitude_bin=np.arange(-80,90,20))

#%% specific one winter at NP to see SSW
# OH
path_save_stat = './data_IRI/OH/PM/'
with xr.open_mfdataset(
        path_save_stat+'PM_daily_zonal_mean_*.nc',
        preprocess=reindex_lat,    
    ) as mds:
    NP_ver = mds.mean_ver.sel(latitude_bin=80).drop('latitude_bin').rename('OH_mean')

    oh_NP = NP_ver.copy()

# O2delta
path_save_stat = './data_IRI/O2del/AM/'
with xr.open_mfdataset(
    path_save_stat+'AM_Daily_NP_mean_*.nc', 
    ) as mds:
    NP_ver = mds.mean_ver.rename('O2Delta_mean')
    o2delta_NP = NP_ver.copy()

#%%
# SMR T, o3, ho2...
# path_smr = './data_SMR/'
def read_smr_daily(filename_pattern, profile):
    path_smr = './data_SMR/'
    with xr.open_mfdataset(
        path_smr+filename_pattern,
        preprocess=lambda x: x.reindex(latitude=np.arange(-80,90,20)),
        ) as mds:
        ds_NP = mds[profile+'_mean'].sel(
                latitude=80
            ).drop(
                'latitude'
            ).rename(altitude='z').compute()
    return ds_NP


filename = 'Daily_'+'Odin-SMR_L2_ALL19lowTunc_O3-557-GHz-45-to-90-km_{}-*.nc'.format('*')
o3_NP_19 = read_smr_daily(filename, 'O3')

filename = 'Daily_' + 'Odin-SMR_L2_ALL-Meso-v3.0.0_O3-557-GHz-45-to-90-km_{}-*.nc'.format('*')
o3_NP_13 = read_smr_daily(filename, 'O3')

# filename = 'Daily_'+'Odin-SMR_L2_ALL-Strat-v3.0.0_O3-545-GHz-20-to-85-km_2009-*.nc'
# o3_NP_strat = read_smr_daily(filename, 'O3')

filename = 'Daily_'+'Odin-SMR_L2_ALL19lowTunc_H2O-557-GHz-45-to-100-km_{}-*.nc'.format('*')
h2o_NP_19 = read_smr_daily(filename, 'H2O')

filename = 'Daily_'+'Odin-SMR_L2_ALL-Meso-v3.0.0_H2O-557-GHz-45-to-100-km_{}-*.nc'.format('*')
h2o_NP_13 = read_smr_daily(filename, 'H2O')

filename = 'Daily_'+'Odin-SMR_L2_ALL19lowTunc_Temperature-557-(Fmode-19)-45-to-90-km_{}-*.nc'.format('*')
T_NP_19 = read_smr_daily(filename, 'Temperature')

filename = 'Daily_'+'Odin-SMR_L2_ALL-Meso-v3.0.0_Temperature-557-(Fmode-13)-45-to-90-km_{}-*.nc'.format('*')
T_NP_13 = read_smr_daily(filename, 'Temperature')


#%% combine FM19 and FM13
o3_NP = xr.concat([o3_NP_13, o3_NP_19], dim='time').sortby('time')
T_NP = xr.concat([T_NP_13, T_NP_19], dim='time').sortby('time')
h2o_NP = xr.concat([h2o_NP_13, h2o_NP_19], dim='time').sortby('time')

#%% plot OH O2del O3 H2O T
year = 2013
sel_arg = dict(
    time=slice('{}-11-01'.format(year-1), '{}-02-25'.format(year)),
    z=slice(50e3,100e3),
    )
plot_arg = dict(
    x='time', y='z', cmap='viridis', robust=True,
    )
fig, ax = plt.subplots(5,1, figsize=(8,10), facecolor='w', sharex=True, sharey=True)

oh_NP.sel(**sel_arg).plot(
    ax=ax[0],
    **plot_arg,
    norm=LogNorm(vmin=1e3, vmax=2e5),
    )

o2delta_NP.sel(**sel_arg).plot(
    ax=ax[1],
    **plot_arg,
    norm=LogNorm(vmin=1e3, vmax=2e5),
    )

o3_NP.sel(**sel_arg).dropna('time','all').plot.contourf(
    ax=ax[2],
    **plot_arg,
    )

h2o_NP.sel(**sel_arg).dropna('time','all').plot.contourf(
    ax=ax[3],
    **plot_arg,
    )

T_NP.sel(**sel_arg).dropna('time','all').plot.contourf(
    ax=ax[4],
    **plot_arg,
    )

smr_data_list = [o3_NP_19, h2o_NP_19, T_NP_19]
[[ax[i+2].axvline(x=t,c='k',ls=':') for t in ds.dropna('time','all').time.values] for i,ds in enumerate(smr_data_list) ]
smr_data_list = [o3_NP_13, h2o_NP_13, T_NP_13]
[[ax[i+2].axvline(x=t,c='r',ls=':') for t in ds.dropna('time','all').time.values] for i,ds in enumerate(smr_data_list) ]

[ax[i].set(title=title, xlabel='') for i,title in enumerate('OH O2Del O3 H2O T'.split())]
plt.show()





# %%
# am_pm = 'PM'
def fetch_data(am_pm):
    # path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/Daily_NP_stats/no_equi/{}/'.format(am_pm)
    path_save_stat = f'./data_IRI/O2del/{am_pm}/'
    with xr.open_mfdataset(
        path_save_stat+'{}_Daily_NP_mean_*.nc'.format(am_pm), 
        # preprocess=change_description_o2delta
        ) as mds:
        NP_ver = mds.mean_ver.rename('O2Delta_mean')
        o2delta_NP = NP_ver.copy()
    return o2delta_NP

ds_am = fetch_data('AM')
ds_pm = fetch_data('PM')
ds = xr.concat([ds_am, ds_pm], dim='am_pm').assign_coords(am_pm=['AM', 'PM'])
# ds_all = fetch_data('ALL')
#%%
year = 2013
sel_arg = dict(
    time=slice('{}-11-01'.format(year-1), '{}-02-25'.format(year)),
    z=slice(60e3,95e3),
    )
plot_arg = dict(
    x='time', y='z', cmap='viridis', 
    robust=True,
    norm=LogNorm(vmin=1e3, vmax=2e5),
    figsize=(5, 3), facecolor='w',
    )
ds.sel(**sel_arg).plot(**plot_arg, row='am_pm' )


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

#%% SMR: compare FM 19 and FM 13
fig, ax = plt.subplots(6,1, figsize=(10,15), sharex=True, sharey=True)
fig.patch.set_facecolor('w')
plt_args = dict(x='time', cmap='viridis')
T_NP_13.dropna('time','all').plot.contourf(ax=ax[0], **plt_args)
T_NP_19.dropna('time','all').plot.contourf(ax=ax[1], **plt_args)
T_NP.dropna('time','all').plot.contourf(ax=ax[2], **plt_args)

h2o_NP_13.dropna('time','all').plot.contourf(ax=ax[3], robust=True, **plt_args)
h2o_NP_19.dropna('time','all').plot.contourf(ax=ax[4], robust=True, **plt_args)
h2o_NP.dropna('time','all').plot.contourf(ax=ax[5], robust=True, **plt_args)

[ax[i].set(title='FM 13', xlabel='') for i in [0,3]]
[ax[i].set(title='FM 19', xlabel='') for i in [1,4]]
[ax[i].set(title='FM 13&19', xlabel='') for i in [2,5]]
[[ax[i].axvline(x=t,c='k',ls=':') for t in ds.dropna('time','all').time.values] for i,ds in enumerate([T_NP_13,T_NP_19,T_NP,h2o_NP_13,h2o_NP_19,h2o_NP]) ]
plt.show()



#%%
fig, ax = plt.subplots(3,1, figsize=(8,8), facecolor='w', sharex=True, sharey=True)

T_NP.dropna('time','all').plot.contourf(ax=ax[0], robust=True, x='time')
h2o_NP.dropna('time','all').plot.contourf(ax=ax[1], robust=True, x='time')
o3_NP.dropna('time','all').plot.contourf(ax=ax[2], robust=True, x='time')

[ax[0].axvline(x=t,c='k',ls=':') for t in T_NP_13.dropna('time','all').time.values]
[ax[1].axvline(x=t,c='k',ls=':') for t in h2o_NP_13.dropna('time','all').time.values]
[ax[2].axvline(x=t,c='k',ls=':') for t in o3_NP_13.dropna('time','all').time.values]

[ax[0].axvline(x=t,c='r',ls=':') for t in T_NP_19.dropna('time','all').time.values]
[ax[1].axvline(x=t,c='r',ls=':') for t in h2o_NP_19.dropna('time','all').time.values]
[ax[2].axvline(x=t,c='r',ls=':') for t in o3_NP_19.dropna('time','all').time.values]

plt.show()

#%% OH and O2del all winters
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


# O2 Delta
# path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/averages/Daily_NP_stats/'
path_save_stat = './data_IRI/O2del/ALL/'
with xr.open_mfdataset(
    path_save_stat+'Daily_NP_mean_*.nc', 
    preprocess=change_description_o2delta
    ) as mds:
    NP_ver = mds.mean_ver

    fig, ax = plt.subplots(len(range(2002, 2016)), 1, sharey=True, figsize=(10,20))
    for i, year in enumerate(range(2002, 2016)):
        plot_one_year(NP_ver, year, ax[i])
    [ax[i].set_xticklabels([]) for i in range(len(ax)-1)]

# OH
# path_save_stat = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel1/nightglow/averages/zenith/'
path_save_stat = './data_IRI/OH/PM/'
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

# %%