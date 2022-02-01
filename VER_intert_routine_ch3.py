#%%
import numpy as np
import xarray as xr
import glob
from multiprocessing import Pool
from os import listdir
import pandas as pd
from astropy.time import Time
import sys

#%% path length 1d
def pathl1d_iris(h, z=np.arange(40e3, 110e3, 1e3), z_top=150e3):
    #z: retrieval grid in meter
    #z_top: top of the atmosphere under consideration 
    #h: tangent altitude of line of sight
    if z[1]<z[0]:
        z = np.flip(z) # retrieval grid has to be ascending
        print('z has to be fliped')
    
#    if h[1]<h[0]: # measred tangent alt grid has to be ascending
#        h = np.flip(h)
#        print('h has to be fliped')
    
    Re = 6370e3 # earth's radius in m
    z = np.append(z, z_top) + Re
    h = h + Re
    pl = np.zeros((len(h), len(z)-1))
    for i in range(len(h)):
        for j in range(len(z)-1):
            if z[j+1]> h[i]:
                pl[i,j] = np.sqrt(z[j+1]**2 - h[i]**2)
                
    pathl = np.append(np.zeros((len(h),1)), pl[:,:-1], axis=1)
    pathl = pl - pathl
    pathl = 2*pathl        
    
    return pathl #in meter (same as h)

def pathlength_geometry_correction(z, h, abs_table, background_atm):
    #z, h: numpy array in km
    #abs_table, background_atm: xarray, coordinate in km
    if z[1]<z[0]:
        z = np.flip(z) # retrieval grid has to be ascending
        print('z has to be fliped')
    Re = 6375 # earth's radius in km
    z_top_Re = abs_table.z_top.values + Re # in km
    z_Re = z + Re # in km
    h_Re = h + Re # in km 
    p_sum = np.zeros((len(h), len(z)))
    for i in range(len(h)):
        for j in range(len(z)):
            if z[j] > h[i]:
                p_sum[i,j] = np.sqrt(z_Re[j]**2 - h_Re[i]**2)
    p = np.roll(p_sum, 1, axis=1)
    p[:,0] = 0
    p = p_sum - p # pahtlength of each layer (half side) in km (len(h), len(z))
    p_extend = np.append(np.fliplr(p), p, axis=1) # pathlength of each layer (2 sides) in km (len(h), len(z)*2)
    extend_layer_index = np.append(np.flip(np.arange(len(z))), np.arange(len(z))) #(len(z)*2)
    p_top = np.sqrt(z_top_Re**2-h_Re**2) - np.sqrt(z_Re[-1]**2-h_Re**2) # pathlength from z_top (200km) to the last layer of z in km (len(h)) 
    d_extend = (p_extend.cumsum(axis=1).T + p_top).T - p_extend/2 # distance from z top in km (len(h), len(z)*2)
    # p_extend_corr = np.zeros(p_extend.shape)
    p_corr = np.zeros(p.shape)
    for i in range(len(h)):
        corr = abs_table.factor.interp(tangent_pressure=background_atm.p.interp(z=h[i]), distance=d_extend[i,:])
    #     p_extend_corr[i,:] = p_extend[i,:]*corr
        p_corr[i,:] = np.bincount(extend_layer_index, p_extend[i,:]*corr)
    return p_corr, p*2

#%%
def linear_oem(y, K, Se_inv, Sa_inv, xa):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    G= np.linalg.solve(K.T.dot(Se_inv).dot(K) + Sa_inv, (K.T).dot(Se_inv))        
    x_hat = xa + G.dot(y - K.dot(xa)) 
    
    return x_hat.squeeze(), G

def mr_and_Sm(x_hat, K, Sa_inv, Se_inv, G):
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(x_hat),1)
    # A = Sa_inv + K.T.dot(Se_inv).dot(K)
    # b = K.T.dot(Se_inv)
    # G = np.linalg.solve(A, b) # gain matrix
    AVK = G.dot(K)
    MR = AVK.sum(axis=1)
    Se = np.linalg.inv(Se_inv)
#    Se = np.diag(1/np.diag(Se_inv)) #only works on diagonal matrix with no off-diagonal element
    Sm = G.dot(Se).dot(G.T) #retrieval noise covariance
    Ss = (AVK - np.eye(len(AVK))).dot(np.linalg.inv(Sa_inv)).dot((AVK - np.eye(len(AVK))).T)
#     Ss = np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv).dot(Sa_inv).dot(np.linalg.inv(K.T.dot(Se_inv).dot(K) + Sa_inv))
    return MR, AVK, Sm, Ss

def oem_cost_pro(y, y_fit, x_hat, Se_inv, Sa_inv, xa, *other_args):
    if len(y.shape) == 1:
        y = y.reshape(len(y),1)
    if len(y_fit.shape) == 1:
        y_fit = y_fit.reshape(len(y_fit),1)
    if len(xa.shape) == 1:
        xa = xa.reshape(len(xa),1)
    if len(x_hat.shape) == 1:
        x_hat = x_hat.reshape(len(xa),1)    
    cost_x = (x_hat - xa).T.dot(Sa_inv).dot(x_hat - xa) / len(y)
    cost_y = (y-y_fit).T.dot(Se_inv).dot(y-y_fit) / len(y)
    return cost_x.squeeze(), cost_y.squeeze()

def prepare():
    # load climatology (xa, x_initial, Sa, and forward_args)
    path = '/home/anqil/Documents/osiris_database/ex_data/'
    file = 'msis_cmam_climatology_z200_lat8576.nc'
    clima = xr.open_dataset(path+file)#.interp(z=z*1e-3)
    # clima = clima.update({'m':(clima.o.dims, (clima.o + clima.o2 + clima.n2)*1e-6, {'unit': 'cm-3'})}) 
    
    # construct overlap filter table
    aaa = [150,  77.4, 175,  75.8, 200,  74.4, 225, 73.2, 250, 72.0, 275, 70.8, 300,  69.8, 325, 68.8, 350, 67.8] 
    T_ref = np.array(aaa[::2])
    fr_ref = np.array(aaa[1::2])*1e-2 # percent to ratio
    fr_ref = xr.DataArray(fr_ref, dims=('T'), coords=(T_ref,), name='overlap_filter', attrs={'units': '1'})

    # load absorption correction table
    file = '/home/anqil/Documents/osiris_database/ex_data/abs_table_Voigt2_80.nc'
    abs_table = xr.open_dataset(file)
    abs_table = abs_table.set_coords('tangent_pressure').swap_dims({'tangent_altitude':'tangent_pressure'})

    return clima, fr_ref, abs_table

def invert_orbit_ch3(path_filename_limb, save_file=False, save_path_filename_ver=None, im_lst=None, return_AVK=False):
    orbit_num = path_filename_limb[-13:-7]
    print('process orbit {}'.format(orbit_num))
    ir = xr.open_dataset(path_filename_limb).sel(pixel=slice(21,128))
    ir.close()
    l1 = ir.data.where(ir.data.notnull(), drop=True).where(ir.sza>90, drop=True)
    time = l1.time
    if len(time)<1:
        print('orbit {} does not have sufficient images that satisfy criterion'.format(orbit_num))
        return
    error = ir.error.sel(time=time)
    tan_alt = ir.altitude.sel(time=time)

    if im_lst == None:
        im_lst = range(0,len(time))

    tan_low = 60e3
    tan_up = 95e3
    pixel_map = ((tan_alt>tan_low)*(tan_alt<tan_up))

    # clima, fr_ref, abs_table = prepare()
    
    #%% 1D inversion
    z = np.arange(tan_low-5e3, tan_up+20e3, 1e3) # m
    z_top = z[-1] + (z[-1]-z[-2]) #m
    xa = np.ones(len(z)) * 0 

    mr = []
    error2_retrieval = []
    error2_smoothing = []
    ver = []
    time_save = []
    A_diag = []
    A_peak = []
    A_peak_height = []
    ver_cost_x, ver_cost_y = [], []
    for i in range(len(im_lst)):
        isel_args = dict(time=im_lst[i])
        h = tan_alt.isel(**isel_args).where(pixel_map.isel(**isel_args), drop=True)
        if len(h)<1:
            # print('image not enough pixels')
            continue

        #background atmosphere (mainly for T and p)
        # background_atm = clima.sel(lat=ir.latitude.isel(**isel_args),
        #                 month=time.isel(**isel_args).dt.month,
        #                 method='nearest')

        #calculate jacobian 
        K = pathl1d_iris(h.values, z, z_top)  *1e2 #m->cm
        # pathlength, pathlength_ref  = pathlength_geometry_correction(z*1e-3, h.values*1e-3, abs_table, background_atm) #in km    
        # K = pathlength *1e5 # km-->cm    

        #calculate Sa
        # peak_apprx = 5e4 * 4*np.pi / 0.7 # approximation of the airglow peak, used in Sa
        peak_apprx = 5e4 * 4*np.pi
        sigma_a = np.ones_like(xa) * peak_apprx
        n = (z<tan_low).sum()
        sigma_a[np.where(z<tan_low)] = np.logspace(-1, np.log10(peak_apprx), n)
        n = (z>tan_up).sum()
        sigma_a[np.where(z>tan_up)] = np.logspace(np.log10(peak_apprx), -1, n)
        Sa_inv = np.diag(1 / sigma_a**2)
        
        #assign y and Se vectors
        #calculate normalise factor for y
        # overlap_filter = fr_ref.interp(
        #         T=background_atm.T.interp(z=h*1e-3), 
        #         kwargs=dict(fill_value='extrapolate')
        #     )
        # normalise = 4*np.pi / overlap_filter.values
        normalise = 4*np.pi
        y = l1.isel(**isel_args).reindex_like(h).values * normalise
        Se_inv = np.diag(1/(error.isel(**isel_args).reindex_like(h).values * normalise)**2)

        #invert
        x, G = linear_oem(y, K, Se_inv, Sa_inv, xa)
        _, A, Sm, Ss = mr_and_Sm(x, K, Sa_inv, Se_inv, G)
        cost_x, cost_y = oem_cost_pro(y, y-K.dot(x), x, Se_inv, Sa_inv, xa)

        ver.append(x)
        A_diag.append(A.diagonal())
        A_peak.append(A.max(axis=1)) #max of each row
        A_peak_height.append(z[A.argmax(axis=1)]) #z of the A_peak
        mr.append(A.sum(axis=1)) #sum of each row
        error2_retrieval.append(np.diag(Sm))
        error2_smoothing.append(np.diag(Ss))
        ver_cost_x.append(cost_x)
        ver_cost_y.append(cost_y)
        time_save.append(time[im_lst[i]].values)
    
    #save data
    if len(time_save) > 0:
        def attrs_dict(long_name, units, description):
            return dict(long_name=long_name, units=units, description=description)

        ds_ver = xr.Dataset().update({
            'time': (['time'], time_save),
            'z': (['z',], z, attrs_dict('altitude', 'm', 'Altitude grid of VER retrieval')),
            # 'pixel': (['pixel',], l1.pixel),
            'ver': (['time','z'], ver, attrs_dict('volume_emission_rate','photons cm-3 s-1', 'IRI OH(3-1) volume emission rate ')), #in correct description: shall be O2delta VER
            'mr': (['time','z'], mr, attrs_dict('measurement_response', '1', 'Measurement response')),
            'A_diag': (['time','z'], A_diag, attrs_dict('AVK_diagonal', '1','Averaging kernel matrix diagonal elements')),
            'A_peak': (['time','z'], A_peak, attrs_dict('AVK_maximum', '1', 'Averaging kernel maximum in each row')),
            'A_peak_height': (['time','z'], A_peak_height, attrs_dict('AVK_max_height', 'm', 'Corresponding altitude of the averaging kernel maximum in each row')),
            'error2_retrieval': (['time','z'], error2_retrieval, attrs_dict('variance_measurement', '(photons cm-3 s-1)^2', 'Retrieval noise S_m diagonal elements (Rodgers (2000))')),
            'error2_smoothing': (['time','z'], error2_smoothing, attrs_dict('variance_smoothing', '(photons cm-3 s-1)^2', 'Smoothing error S_s diagonal elements (Rodgers (2000))')),
            'ver_cost_x': (['time',], ver_cost_x),
            'ver_cost_y': (['time',], ver_cost_y),
            'latitude': (['time',], ir.latitude.sel(time=time_save).data, attrs_dict('latitude', 'degrees north', 'Latitude at the tangent point')),
            'longitude': (['time',], ir.longitude.sel(time=time_save).data, attrs_dict('longitude', 'degrees east', 'Longitude at the tangent point')),
            'sza': (['time',], ir.sza.sel(time=time_save).data, attrs_dict('solar_zenith_angle', 'degrees', 'Solar Zenith Angle between the satellite line-of-sight and the sun')),
            'apparent_solar_time': (['time',], ir.apparent_solar_time.sel(time=time_save).data, attrs_dict('apparent_solar_time', 'hour', 'Apparent Solar Time at the line-of-sight tangent point')),
            'orbit': ir.orbit.data,
            })
        ds_ver = ds_ver.assign_attrs(dict(channel=ir.channel.values))
        
        if save_file:
            encoding = dict(
                orbit=dict(dtype='int32'), 
                time=dict(units='days since 1858-11-17'),
                )
            encoding.update({v: {'zlib': True, 'dtype': 'float32'} for v in ds_ver.data_vars 
                            if ds_ver[v].dtype == 'float64'})

            ds_ver.to_netcdf(save_path_filename_ver, encoding=encoding)
        
        if return_AVK:
            A = xr.DataArray(A, (('row_z',z), ('col_z', z)))
            return ds_ver, A
        else:
            return ds_ver
    
def find_orbit_stamps(year):
    time_stamp = pd.date_range(start='{}-12-31 23:59:59'.format(year-1), 
                                end='{}-01-31 23:59:59'.format(year+1),
                                freq='M')
    with xr.open_dataset('~/Documents/osiris_database/odin_rough_orbit.nc') as rough_orbit:
        rough_orbit = rough_orbit.rename({'mjd':'time'}).assign(
            time=Time(rough_orbit.mjd, format='mjd').datetime64
            ).interp(time=time_stamp).astype(int).orbit
    return rough_orbit

#%%
if __name__ == '__main__':
    # limb files directory and save ver files pattern
    path_limb = '/home/anqil/Documents/sshfs/oso_extra_storage/StrayLightCorrected/Channel3/'
    path_pattern_ver = '/home/anqil/Documents/sshfs/oso_extra_storage/VER/Channel3/nightglow/orbits_v2/{}{}/'
    filename_pattern_ver = 'iri_ch3_ver_{}.nc'
    orbit_error = []

    def process(year, month):
        rough_orbit = find_orbit_stamps(year=year)
        
        range_orbit = range(*tuple(rough_orbit.isel(time=slice(month-1,month+1)).values))
        orbit_num_in_month = [f[-13:-7] for f in sorted(glob.glob(path_limb+'*.nc')) 
                            if int(f[-13:-7]) in range_orbit]
        path_ver = path_pattern_ver.format(str(year)[-2:], str(month).zfill(2))
        for orbit_num in orbit_num_in_month:
            path_filename_limb = path_limb + 'ir_slc_{}_ch3.nc'.format(orbit_num)
            ver_file_lst = glob.glob(path_ver + filename_pattern_ver.format('*'))
            ver_filename = filename_pattern_ver.format(orbit_num)
            if path_ver+ver_filename in ver_file_lst:
                print('orbit {} already exist'.format(orbit_num))
            else:
                try:
                    _ = invert_orbit_ch3(path_filename_limb, save_file=True,
                        save_path_filename_ver=path_ver+ver_filename)
                except:
                    raise
            
    def fun(month):
        process(year, month)

    year = int(input('Enter year: \n'))
    months = range(1,13)
    with Pool(processes=8) as p:
        p.map(fun, months)
    # for month in months:
    #     fun(month) 


