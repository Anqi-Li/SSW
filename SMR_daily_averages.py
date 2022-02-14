# %%
import xarray as xr
import numpy as np
import requests
from tqdm import tqdm
from os import listdir
from matplotlib import pyplot as plt
path_save = "./data_SMR/"
baseurl = 'http://odin.rss.chalmers.se/level2_download/ALL-Meso-v3.0.0/'

def download_from_url(url):
    filename = url.split('/')[-1]
    #download the nc file if not done already
    if filename not in listdir(path_save):
        r = requests.get(url)
        if r.status_code == 404:
            pass
        else:
            with open(path_save + filename, "wb") as handle:
                for data in tqdm(r.iter_content()):
                    handle.write(data)

# %%
#interpolate in altitude grid
def alt_interp(ds, profile, alt_range):
  profile_alt = []
  # Apriori_alt=[]
  for i in range(len(ds.time)):
      profile_alt.append(np.interp(alt_range,ds.Altitude.isel(time=i),ds[profile].where(ds.MeasResponse>0.8).isel(time=i)))
      # Apriori_alt.append(np.interp(alt_range, ds.Altitude.isel(time=i),ds.Apriori.isel(time=i)))
  ds_alt = xr.Dataset({profile: (["time","altitude"],profile_alt), 
                      # "Apriori": (["time","altitude"],Apriori_alt),
                      "latitude":(["time"], ds.Lat1D.data),
                      "longitude":(["time"], ds.Lon1D.data),
                      }, 
                    coords={
                      "altitude": (["altitude"], alt_range),
                      "time":(["time"], ds.Time.data)
                      })
  return ds_alt

# %%
# make daily mean, std, count for all latitude bins
def mk_daily(ds_alt, data_vars):
    lat_bins = np.linspace(-90, 90, 10)
    lat_labels = []
    daily_mean = []
    daily_count = []
    daily_std = []
    for label, group in ds_alt[data_vars].groupby_bins(ds_alt.latitude, bins=lat_bins):
        lat_labels.append(label.mid) #make the labels to string so that we can save in nc file
        resampled = group.resample(time='1D')
        daily_mean.append(resampled.mean('time').rename({name:name+'_mean' for name in group}))
        daily_count.append(resampled.count('time').rename({name:name+'_count' for name in group}))
        daily_std.append(resampled.std('time').rename({name:name+'_std' for name in group}))

    ds_daily = xr.merge([
        xr.concat(daily_mean, dim='latitude').assign_coords(latitude=lat_labels),
        xr.concat(daily_count, dim='latitude').assign_coords(latitude=lat_labels),
        xr.concat(daily_std, dim='latitude').assign_coords(latitude=lat_labels)
    ])
    return ds_daily

#%%
def process_file(url, profile):
    download_from_url(url)
    filename = url.split('/')[-1]
    print(profile, filename)

    if filename not in listdir(path_save):
        pass
    else:
        if 'Daily_'+filename not in listdir(path_save):
            #load data
            with xr.open_dataset(path_save+filename) as ds:
                # ds = ds.assign_coords(pressure=ds.Pressure[0]).swap_dims({"level":"pressure"})
                if profile == 'Temperature':
                    pass
                else:
                    ds = ds.rename(dict(Profile=profile))

                #altitude interpolation
                alt_range = np.arange(50,101,1)*1000 #meter
                ds_alt = alt_interp(ds, profile, alt_range)

                #calculate daily mean at all latitudes
                ds_daily = mk_daily(ds_alt, [profile])
                ds_daily.to_netcdf(path_save+'Daily_'+filename)

        else:
            ds_daily = xr.open_dataset(path_save+'Daily_'+filename)
            ds_daily.close()
        
        return ds_daily

#%% General profile
# month = '11'
# year = '2008'

def fun(year):
    for month in '11 12 01 02'.split():
        print(year, month)

        profile ='O3'
        url = baseurl+'Odin-SMR_L2_ALL-Meso-v3.0.0_O3-557-GHz-45-to-90-km_{}-{}.nc'.format(year, month)
        process_file(url, profile)
        url = baseurl+'Odin-SMR_L2_ALL19lowTunc_O3-557-GHz-45-to-90-km_{}-{}.nc'.format(year, month)
        process_file(url, profile)

        profile = 'Temperature'
        url = baseurl+'Odin-SMR_L2_ALL-Meso-v3.0.0_Temperature-557-(Fmode-13)-45-to-90-km_{}-{}.nc'.format(year, month)
        process_file(url, profile)
        url = baseurl+'Odin-SMR_L2_ALL19lowTunc_Temperature-557-(Fmode-19)-45-to-90-km_{}-{}.nc'.format(year, month)
        process_file(url, profile)

        profile = 'H2O'
        url = baseurl+'Odin-SMR_L2_ALL-Meso-v3.0.0_H2O-557-GHz-45-to-100-km_{}-{}.nc'.format(year, month)
        process_file(url, profile)
        url = baseurl+'Odin-SMR_L2_ALL19lowTunc_H2O-557-GHz-45-to-100-km_{}-{}.nc'.format(year, month)
        process_file(url, profile)

for year in range(2001, 2016):
    fun(year)
    
# ds_daily = process_file(url, profile)
# ds_daily.sel(latitude=80)[profile+'_mean'].dropna('time','all').plot.contourf(x='time')




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

# %%

# %%

# %%


