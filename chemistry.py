#%%
import numpy as np
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
o3_NP = xr.concat([o3_NP_13, o3_NP_19], dim='time').sortby('time')#.reindex_like(o2delta_NP)
T_NP = xr.concat([T_NP_13, T_NP_19], dim='time').sortby('time')
h2o_NP = xr.concat([h2o_NP_13, h2o_NP_19], dim='time').sortby('time')

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
        month = 1,
        lat = slice(70, 90),
    ).mean(
        'lat'
    ).interp(z=o2delta_NP.z*1e-3)
    for s in ['n2', 'o2', 'o']:
        ds_bg[s] = ds_bg[s]*1e-6
# ds_bg

#%% attempt to calculate O 
def q_o2delta(T, o2, n2):
    k_o2 = 3.6e-18*np.exp(-220/T)
    k_n2 = 1.4e-19#1e-20
    # k_o = 1.3e-16
    return k_o2*o2 + k_n2*n2# + k_o*o

A_o2delta = 2.23e-4 # 2.58e-4
Q_o2delta = q_o2delta(T_NP, ds_bg.o2, ds_bg.n2)

# o2(sig v=0) reactions ======================
k_oom = 4.7e-33*(300/T_NP)
# c_o2 = 6.6 #empirical quenchin coefficient
# c_o = 19 #empirical quenchin coefficient
# prod_o2sig_from_barth = k_oom * o**2 * m * o2 / (c_o2*o2 + c_o*o)

sel_arg = dict(time=slice('2009-01-15', '2009-02-28'))
o = np.sqrt(o2delta_NP.sel(**sel_arg) * (Q_o2delta.sel(**sel_arg) + A_o2delta)/(k_oom.sel(**sel_arg) * (ds_bg.o2 + ds_bg.n2)))
#%%
plt.figure()
o.plot.line(y='z', add_legend=False)
ds_bg.o.pipe(lambda x: x*1e-2).plot(y='z', c='k', label='msis/100')
plt.legend()
plt.show()
#%%
plt.figure()
o.plot.contourf(y='z', x='time', vmin=0, vmax=9e9)
plt.show()
#%%
oh = p * T_NP**-3.4 * o