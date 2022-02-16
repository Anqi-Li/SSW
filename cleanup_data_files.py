#%%
import numpy as np
import xarray as xr
import os
# %% rename files 
path = "./data_IRI/O2del/ALL/"
for file in os.listdir(path):
	# print(path + file)
	# print(path + f"ALL_{file}")
	os.rename(path+file, path+f"ALL_{file}")

#%% rename attributes in daily average files
def change_description_o2delta(ds):
    ds.mean_ver.attrs['description'] = 'IRI O2(Delta) volume emission rate'
    ds.std_ver.attrs['description'] = 'IRI O2(Delta) volume emission rate'
    ds.count_ver.attrs['description'] = 'IRI O2(Delta) volume emission rate'
    return ds

am_pm = 'PM'
# year = '02'
for year in range(2001, 2017):
	year = str(year)[-2:]
	try:
		# month = '10'
		for month in range(1,13):
			month = str(month).zfill(2)
			path_org = f'./data_IRI/O2del/{am_pm}/'
			with xr.open_dataset(
				path_org+f'{am_pm}_Daily_NP_mean_{year}{month}.nc', 
				) as ds:
				ds_new = change_description_o2delta(ds)
			ds_new.to_netcdf(
				f"./data_IRI/O2del/attr_corrected/{am_pm}/"\
					f"{am_pm}_Daily_NP_mean_{year}{month}.nc",
				mode = "w",
				)
	except FileNotFoundError:
		pass

# with xr.open_mfdataset(
#     path_org+f'{am_pm}_Daily_NP_mean_{year}*.nc', 
#     preprocess=change_description_o2delta
#     ) as mds:
# 	# print(mds.mean_ver.description)
# 	months, datasets = zip(*mds.groupby("time.month"))
# 	paths = [f"./data_IRI/O2del/attr_corrected/{am_pm}/"\
# 		f"{am_pm}_Daily_NP_mean_{year}{m}.nc" for m in months]
# 	xr.save_mfdataset(datasets, paths, mode='w')

#%% test O2del daily average attrs
am_pm = 'AM'
with xr.open_mfdataset(f"./data_IRI/O2del/{am_pm}/*.nc") as ds:
	print(ds.mean_ver.description)
# %% test data files
year = 2001
path = "~/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/orbits_v2/0303/"
with xr.open_dataset(path+f"iri_ch3_ver_011424.nc") as ds:
	print(ds)

#%% rename attributes in orbit files
def change_ver_description(ds):
	ds.ver.attrs['description'] = "IRI O2(Delta) volume emission rate"
	return ds

# year = 2001
for year in range(2002, 2017):
	print(year)
	year = str(year)[-2:]
	# month = 10
	for month in range(1,13):
		month = str(month).zfill(2)
		path = "/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/orbits_v2/"
		path_save = path+'attr_corrected/'
		try:
			with xr.open_mfdataset(
				path+f"{year}{month}/iri_ch3_ver_*.nc",
				preprocess=change_ver_description,
				) as mds:
				if f"{year}{month}" not in os.listdir(path_save):
					os.makedirs(path_save+f"{year}{month}")
				# print(mds)
				orbits, datasets = zip(*mds.groupby(mds.orbit))
				paths = [path_save+f"{year}{month}/iri_ch3_ver_{str(orb).zfill(6)}.nc" for orb in orbits]
				xr.save_mfdataset(datasets, paths, mode='w')
		except OSError:
			pass
