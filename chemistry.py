#%%
import numpy as np
from sympy import nroots
import xarray as xr
import matplotlib.pyplot as plt

def reindex_lat(ds):
    return ds.reindex(latitude_bin=np.arange(-80,90,20))

# %% fetch data
# IRI OH and O2del
def fetch_iri_data(am_pm, species):
    path_save_stat = f'./data_IRI/{species}/{am_pm}/'
    if species == 'O2del':
        with xr.open_mfdataset(
            path_save_stat+f'{am_pm}_Daily_NP_mean_*.nc', 
            ) as mds:
            ds_NP = mds.mean_ver.rename('O2Delta_mean')
    elif species == 'OH':
        with xr.open_mfdataset(
                path_save_stat+f'{am_pm}_daily_zonal_mean_*.nc',
                preprocess=reindex_lat,    
            ) as mds:
            ds_NP = mds.mean_ver.sel(latitude_bin=80).drop('latitude_bin').rename('OH_mean')
    
    return ds_NP

am_pm = 'AM'
oh_NP = fetch_iri_data(am_pm, 'OH')
o2delta_NP = fetch_iri_data(am_pm, 'O2del')

# SMR T, o3, ho2...
def fetch_smr_data(filename_pattern, profile):
    path_smr = './data_SMR/'
    with xr.open_mfdataset(
        path_smr+filename_pattern,
        preprocess=lambda x: x.reindex(latitude=np.arange(-80,90,20)),
        ) as mds:
        ds_NP = mds[profile+'_mean'].sel(
                latitude=80
            ).drop(
                'latitude'
            ).dropna(
                'time', how='all',
            ).rename(
                altitude='z'
            )
    return ds_NP.compute()

filename = 'Daily_pressure_fm19_*.nc'
p_NP_19 = fetch_smr_data(filename, 'Pressure')

filename = 'Daily_pressure_fm13_*.nc'
p_NP_13 = fetch_smr_data(filename, 'Pressure')

filename = 'Daily_'+'Odin-SMR_L2_ALL19lowTunc_O3-557-GHz-45-to-90-km_*.nc'
o3_NP_19 = fetch_smr_data(filename, 'O3')

filename = 'Daily_' + 'Odin-SMR_L2_ALL-Meso-v3.0.0_O3-557-GHz-45-to-90-km_*.nc'
o3_NP_13 = fetch_smr_data(filename, 'O3')

filename = 'Daily_'+'Odin-SMR_L2_ALL19lowTunc_H2O-557-GHz-45-to-100-km_*.nc'
h2o_NP_19 = fetch_smr_data(filename, 'H2O')

filename = 'Daily_'+'Odin-SMR_L2_ALL-Meso-v3.0.0_H2O-557-GHz-45-to-100-km_*.nc'
h2o_NP_13 = fetch_smr_data(filename, 'H2O')

filename = 'Daily_'+'Odin-SMR_L2_ALL19lowTunc_Temperature-557-(Fmode-19)-45-to-90-km_*.nc'
T_NP_19 = fetch_smr_data(filename, 'Temperature')

filename = 'Daily_'+'Odin-SMR_L2_ALL-Meso-v3.0.0_Temperature-557-(Fmode-13)-45-to-90-km_*.nc'
T_NP_13 = fetch_smr_data(filename, 'Temperature')


#%% combine FM19 and FM13
o3_NP = xr.concat([o3_NP_13, o3_NP_19], dim='time').sortby('time')
T_NP = xr.concat([T_NP_13, T_NP_19], dim='time').sortby('time')
h2o_NP = xr.concat([h2o_NP_13, h2o_NP_19], dim='time').sortby('time')
p_NP = xr.concat([p_NP_13, p_NP_19], dim='time').sortby('time')

#%% find duplicated data in smr
t_dp = T_NP.time.where(T_NP.indexes['time'].duplicated()).dropna('time')
for i in range(len(t_dp)):
    T_NP_13.sel(time=t_dp[i].values).plot.line(y='z', label='13')
    T_NP_19.sel(time=t_dp[i].values).plot.line(y='z', label='19')
    plt.legend()
    plt.show()

#%% fetch bg data
with xr.open_dataset('./data_bg/msis_cmam_climatology_z200_lat8576.nc') as ds:
    ds_bg = ds.drop('date').sel(
        # month = 1,
        lat = slice(70, 90),
    ).mean(
        'lat'
    ).interp(
        z=o2delta_NP.z*1e-3
    # ).assign(
    #     date=ds.date.astype(np.datetime64)
    # ).swap_dims(
    #     month='date'
    )
    for s in ['n2', 'o2', 'o']:
        ds_bg[s] = ds_bg[s]*1e-6 #cm-3
# ds_bg
# 
#%% attempt to calculate O  (relative)
def q_o2delta(T, o2, n2):
    k_o2 = 3.6e-18*np.exp(-220/T)
    k_n2 = 1.4e-19#1e-20
    # k_o = 1.3e-16
    return k_o2*o2 + k_n2*n2# + k_o*o

# A_o2delta = 2.23e-4 # 2.58e-4
# Q_o2delta = q_o2delta(T_NP, ds_bg.o2, ds_bg.n2)

# # o2(sig v=0) reactions ======================
# k_oom = 4.7e-33*(300/T_NP)
# # c_o2 = 6.6 #empirical quenchin coefficient
# # c_o = 19 #empirical quenchin coefficient
# # prod_o2sig_from_barth = k_oom * o**2 * m * o2 / (c_o2*o2 + c_o*o)
    
# sel_arg = dict(time=slice('2009-01-01', '2009-01-30'))
# o = np.sqrt(o2delta_NP.sel(**sel_arg) * (Q_o2delta.sel(**sel_arg) + A_o2delta)/(k_oom.sel(**sel_arg) * (ds_bg.o2 + ds_bg.n2)))

def cal_o_from_o2delta(o2delta, T, bg):
    ds_bg = bg.interp(month=T.time.dt.month)
    A_o2delta = 2.23e-4 # 2.58e-4
    Q_o2delta = q_o2delta(T, ds_bg.o2, ds_bg.n2)
    k_oom = 4.7e-33*(300/T)
    o = np.sqrt(o2delta * (Q_o2delta + A_o2delta)/(k_oom * (ds_bg.o2 + ds_bg.n2)))
    return o.rename('[O]_rel')

# sel_arg = dict(time=slice('2009-01-01', '2009-03-01'))
year = 2009
sel_arg = dict(
    # time=slice('{}-11-01'.format(year-1), '{}-02-25'.format(year)),
    time=slice(f'{year}-01-01', f'{year}-03-01'),
    z=slice(50e3,100e3),
    )
o = cal_o_from_o2delta(
    o2delta_NP.sel(**sel_arg), 
    T_NP.sel(**sel_arg).interp_like(o2delta_NP.sel(**sel_arg)), 
    ds_bg)

#%% o line plot
plt.figure(facecolor='w')
o.plot.line(y='z', add_legend=False)
ds_bg.sel(month=[1]).o.pipe(lambda x: x*1e-2).plot.line(y='z', c='k', label='msis/100', xscale='linear')
# plt.title('2009-01')
plt.legend()
plt.show()
#%% o conoutrf plot
plt.figure(facecolor='w')
o.plot.contourf(y='z', x='time', vmin=0, vmax=9e9)
plt.show()
#%% derive OH (reletive)
def cal_oh_from_o(p,T,o):
    return (p * T**(-3.4) * o).rename('[OH*]_rel')
oh = cal_oh_from_o(
    p_NP.sel(**sel_arg).interp_like(o2delta_NP.sel(**sel_arg)), 
    T_NP.sel(**sel_arg).interp_like(o2delta_NP.sel(**sel_arg)),
    o)

#%% oh contourf plot
plt.figure(facecolor='w')
oh.plot(y='z', x='time', vmin=0, vmax=30)

# show smr sampling
# for i in T_NP.sel(**sel_arg).time.values:
#     plt.axvline(x=i, c='k', ls=':')
plt.show()

#%% all contour plots
from matplotlib.colors import LogNorm
plot_arg = dict(
    x='time', y='z', cmap='viridis', robust=True,
    )
fig, ax = plt.subplots(7,1, figsize=(8,12), facecolor='w', sharex=True, sharey=True)

oh_NP.sel(**sel_arg).dropna('time','all').plot(
    ax=ax[0],
    **plot_arg,
    norm=LogNorm(),
    vmin=1e3, vmax=2e5,
    )

o2delta_NP.sel(**sel_arg).dropna('time', 'all').plot(
    ax=ax[1],
    **plot_arg,
    norm=LogNorm(),
    vmin=1e3, vmax=2e5,
    )

p_NP.sel(**sel_arg).dropna('time','all').plot.contourf(
    ax=ax[2],
    **plot_arg,
    levels=np.logspace(-2,2,10),
    )

h2o_NP.sel(**sel_arg).dropna('time','all').plot.contourf(
    ax=ax[3],
    **plot_arg,
    )

T_NP.sel(**sel_arg).dropna('time','all').plot.contourf(
    ax=ax[4],
    **plot_arg,
    )

o.plot.contourf(
    ax=ax[5],
    **plot_arg,
    )

oh.plot.contourf(
    ax=ax[6],
    **plot_arg,
    )
smr_data_list = [p_NP_19, h2o_NP_19, T_NP_19]
[[ax[i+2].axvline(x=t,c='k',ls=':') for t in ds.dropna('time','all').time.values] \
    for i,ds in enumerate(smr_data_list) ]
smr_data_list = [p_NP_13, h2o_NP_13, T_NP_13]
[[ax[i+2].axvline(x=t,c='r',ls=':') for t in ds.dropna('time','all').time.values] \
    for i,ds in enumerate(smr_data_list) ]

[ax[i].set(title=title, xlabel='') for i,title in enumerate('OH O2Del P H2O T O OH'.split())]
plt.show()


# %%
PV = nRT
m = P/(RT)