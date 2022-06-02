# coding =utf-8
# coding='utf-8'

import sys
import xarray as xr
version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
# import raster2array
import os
from osgeo import ogr
from osgeo import osr
from tqdm import tqdm
import datetime
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import copyreg
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import random

from netCDF4 import Dataset, numpy
import shutil
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from operator import itemgetter
from itertools import groupby
# import RegscorePy
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
import pickle
from dateutil import relativedelta
from sklearn.inspection import permutation_importance
from __init__ import *

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

def nc_to_tif():

    fdir = '/Volumes/1T/wen_prj/Data/CSIF/'
    outdir = '/Volumes/1T/wen_prj/Data/CSIF/'
    mk_dir(outdir)
    for f in os.listdir(fdir):
        if f.endswith('.nc'):
            fpath = fdir + f
            nc = Dataset(fpath)
            print(nc)
            print(nc.variables.keys())
            #t = nc['time']
        # since 1582 - 10 - 15
            lat_list = nc['lat']
            lon_list = nc['lon']
            lat_list=lat_list[::-1]  #取反
            # print(lat_list[:])
            # print(lon_list[:])
            origin_x = lon_list[0]
            origin_y = lat_list[0]
            pix_width = lon_list[1] - lon_list[0]
            pix_height = lat_list[1] - lat_list[0]
            # print (origin_x)
            # print (origin_y)
            print (pix_width)
            print (pix_height)
            # exit()
            SIF_arr_list = nc['SIF']
            print(SIF_arr_list.shape)
            # print(SIF_arr_list[0])
            # exit()
            t=nc['doy']
            print(t[:])
            date_start = datetime.datetime(int(f.split('.')[-2]),1,1)
            for i in range(len(t)):
                date_delta = datetime.timedelta(days=int(t[i])-1)
                date = date_start + date_delta
                print(date)
                val = SIF_arr_list[i]
                # array to tif
                year = date.year
                mon = date.month
                day=date.day
                print (year,mon,day)
                #fname = '%s%02d'%(year,mon)
                #fname = '%s%02d%02d'%(year,mon,day)
                fname = '{}{:02d}{:02d}.tif'.format(year,mon,day)

                # print fname
                newRasterfn = outdir + fname
                # print newRasterfn
                longitude_start = origin_x
                latitude_start = origin_y
                pixelWidth = pix_width
                pixelHeight = pix_height
                # array = val
                array=SIF_arr_list
                array=np.array(array)
                array = array[::-1]
                to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)
                # plt.imshow(array)
                # plt.colorbar()
                # plt.show()


def nc_to_tif_BESS():

    fdir = '/Volumes/1T/wen_prj/Data/BESS/PAR_nc/'
    outdir = '/Volumes/1T/wen_prj/Data/BESS/PAR_tif/'
    mk_dir(outdir)
    for f in os.listdir(fdir):
        if f.endswith('.nc'):
            fpath = fdir + f
            nc = Dataset(fpath)
            fname2=fpath.split('.')[1][1:5]
            if fname2=='2000' or fname2=='2001':
                continue
            # print(nc)
            # print(nc.variables.keys())
            #t = nc['time']
        # since 1582 - 10 - 15
            lat_list = nc['lat']
            lon_list = nc['lon']
            # lat_list=lat_list[::-1]  #取反
            # print(lat_list[:])
            # print(lon_list[:])
            origin_x = lon_list[0]  #要为负数-180
            origin_y = lat_list[0]  #要为正数90
            pix_width = lon_list[1] - lon_list[0]
            pix_height = lat_list[1] - lat_list[0]
            # print (origin_x)
            # print (origin_y)
            # print (pix_width)
            # print (pix_height)
            # exit()
            # SIF_arr_list = nc['SIF']
            GPCP_arr_list = nc['PAR']
            # print(GPCP_arr_list.shape)
            # plt.imshow(GPCP_arr_list[::])
            # plt.show()
            # print(SIF_arr_list[0])
            # exit()
            # t=nc['doy']
            # print(t[:])
            fname = '{}.tif'.format(f.split('.')[1][1:])
            print (fname)
            newRasterfn = outdir + fname
            # if os.path.isfile(newRasterfn):
            #     continue
            # print newRasterfn
            longitude_start = origin_x
            latitude_start = origin_y
            pixelWidth = pix_width
            pixelHeight = pix_height
            # array = val
            array=GPCP_arr_list[::]
            array = np.array(array)
            array = array.T
            # method 2
            # array= numpy.rot90(array, 1,(0,1))
            # array = array[::-1]
            # plt.imshow(array)
            # plt.colorbar()
            # plt.show()
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)

def nc_to_tif_PET():
    f='/Volumes/SSD_sumsang/project_greening/Data/cru_ts4.05.1901.2020.pet.dat.nc'
    outdir = results_root+'Terraclimate/PET/'
    variable='pet'
    T.mk_dir(outdir,force=True)
    ncin = Dataset(f, 'r')
    ncin_xarr = xr.open_dataset(f)
    lat = ncin['lat'][::-1]
    lon = ncin['lon']
    pixelWidth = lon[1] - lon[0]
    pixelHeight = lat[1] - lat[0]
    longitude_start = lon[0]
    latitude_start = lat[0]

    time = ncin.variables['time']

    # print(time)
    # exit()
    # time_bounds = ncin.variables['time_bounds']
    # print(time_bounds)
    start = datetime.datetime(1900, 1, 1)
    # a = start + datetime.timedelta(days=5459)
    # print(a)
    # print(len(time_bounds))
    # print(len(time))
    # for i in time:
    #     print(i)
    # exit()
    # nc_dic = {}
    flag = 0


    for i in tqdm(range(len(time))):
        flag += 1
        # print(time[i])
        date = start + datetime.timedelta(days=int(time[i]))
        year = str(date.year)
        year_int=int(year)
        month = '%02d' % date.month
        month_int=date.month
        days_number_of_one_month=T.number_of_days_in_month(year_int,month_int)
        # day = '%02d'%date.day
        date_str = year + month
        # print(date_str)
        # arr = ncin.variables[f'{variable}'][i][::-1]
        arr = ncin_xarr.variables[f'{variable}'][i][::-1]
        arr = np.array(arr)
        arr_monthly=arr*days_number_of_one_month
        grid = arr < 99999
        arr_monthly[np.logical_not(grid)] = -999999
        newRasterfn = outdir + date_str + '.tif'
        ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr_monthly)


def nc_to_tif_SPEI3():

    f = '/Volumes/SSD_sumsang/project_greening/Data/Terraclimate/CO2/sEXTocNEET_v2021_daily.nc'
    outdir = '/Volumes/SSD_sumsang/project_greening/Data/Terraclimate/CO2/CO2_tif/'
    mk_dir(outdir)

    nc = Dataset(f)

    print(nc)
    print(nc.variables.keys())
    t = nc['time']
    print(t)
    basetime=datetime.datetime(1900,1,1)  # 告诉起始时间
    lat_list = nc['lat']
    lon_list = nc['lon']
    lat_list=lat_list[::-1]  #取反
    print(lat_list[:])
    print(lon_list[:])
    origin_x = lon_list[0]  #要为负数-180
    origin_y = lat_list[0]  #要为正数90
    pix_width = lon_list[1] - lon_list[0] #经度0.5
    pix_height = lat_list[1] - lat_list[0] # 纬度-0.5
    print (origin_x)
    print (origin_y)
    print (pix_width)
    print (pix_height)
    # SIF_arr_list = nc['SIF']
    SPEI_arr_list = nc['spei']
    print(SPEI_arr_list.shape)
    print(SPEI_arr_list[0])
    # plt.imshow(SPEI_arr_list[5])
    # # plt.imshow(SPEI_arr_list[::])
    # plt.show()


    # date_list=list(range(1982,2021))
    # print(date_list)
    for i in range(len(SPEI_arr_list)):
        date_delta_i=t[i]
        print(date_delta_i)
        date_delta_i=datetime.timedelta(int(date_delta_i))
        # print(date_delta_i)
        date_i=basetime+date_delta_i
        print(date_i)
        # if date_i.split('-')[0] not in date_list :
        #     continue
        # print(date_i)
        year=date_i.year
        month=date_i.month
        fname = 'SPEI3_'+'{}{:02d}.tif'.format(year, month)
        print (fname)
        newRasterfn = outdir + fname
        print (newRasterfn)
        longitude_start = origin_x
        latitude_start = origin_y
        pixelWidth = pix_width
        pixelHeight = pix_height
    # array = val
        array=SPEI_arr_list[i]
        array = np.array(array)
    # method 2
    # array= numpy.rot90(array, 1,(0,1))
        array = array[::-1]
        array[array>10]=np.nan
        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)

def nc_to_tif_CO2():

    f = '/Volumes/SSD_sumsang/project_greening/Data/Terraclimate/CO2/sEXTocNEET_v2021_daily.nc'
    outdir = '/Volumes/SSD_sumsang/project_greening/Data/Terraclimate/CO2/CO2_tif/'
    mk_dir(outdir)

    nc = Dataset(f)

    print(nc)
    print(nc.variables.keys())

    t = nc['itime']
    print(t)
    basetime=datetime.datetime(1957,1,1)  # 告诉起始时间
    lat_list = nc['lat']
    lon_list = nc['lon']
    lat_list=lat_list[::-1]  #取反
    print(lat_list[:])
    print(lon_list[:])
    origin_x = lon_list[0]  #要为负数-180
    origin_y = lat_list[0]  #要为正数90
    pix_width = lon_list[1] - lon_list[0] #经度0.5
    pix_height = lat_list[1] - lat_list[0] # 纬度-0.5
    print (origin_x)
    print (origin_y)
    print (pix_width)
    print (pix_height)
    # exit()

    # SIF_arr_list = nc['SIF']
    CO2_arr_list = nc['co2flux_land']
    print(CO2_arr_list.shape)
    print(CO2_arr_list[0])
    # plt.imshow(SPEI_arr_list[5])
    # # plt.imshow(SPEI_arr_list[::])
    # plt.show()


    # date_list=list(range(1982,2021))
    # print(date_list)
    for i in range(len(CO2_arr_list)):
        date_delta_i=t[i]
        print(date_delta_i)
        date_delta_i=datetime.timedelta(int(date_delta_i))
        # print(date_delta_i)
        date_i=basetime+date_delta_i
        print(date_i)
        exit()
        # if date_i.split('-')[0] not in date_list :
        #     continue
        # print(date_i)
        year=date_i.year
        month=date_i.month
        fname = 'SPEI3_'+'{}{:02d}.tif'.format(year, month)
        print (fname)
        newRasterfn = outdir + fname
        print (newRasterfn)
        longitude_start = origin_x
        latitude_start = origin_y
        pixelWidth = pix_width
        pixelHeight = pix_height
    # array = val
        array=SPEI_arr_list[i]
        array = np.array(array)
    # method 2
    # array= numpy.rot90(array, 1,(0,1))
        array = array[::-1]
        array[array>10]=np.nan
        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)

def nc_to_tif_GLEAM():

    f = '/Volumes/SSD_sumsang/project_greening/Data/GLEAM/SMroot_1980-2020_GLEAM_v3.5a_MO.nc'
    outdir = '/Volumes/SSD_sumsang/project_greening/Data/GLEAM/root_soil_moisture/'
    mk_dir(outdir)

    nc = Dataset(f)

    print(nc)
    print(nc.variables.keys())
    t = nc['time']
    print(t)

    basetime=datetime.datetime(1900,1,1)  # 告诉起始时间
    lat_list = nc['lat']
    lon_list = nc['lon']
    # lat_list=lat_list[::-1]  #取反
    print(lat_list[:])
    print(lon_list[:])

    origin_x = lon_list[0]  #要为负数-180
    origin_y = lat_list[0]  #要为正数90
    pix_width = lon_list[1] - lon_list[0] #经度0.5
    pix_height = lat_list[1] - lat_list[0] # 纬度-0.5
    print (origin_x)
    print (origin_y)
    print (pix_width)
    print (pix_height)
    # SIF_arr_list = nc['SIF']
    SPEI_arr_list = nc['SMroot']
    print(SPEI_arr_list.shape)
    print(SPEI_arr_list[0])
    # plt.imshow(SPEI_arr_list[5])
    # # plt.imshow(SPEI_arr_list[::])
    # plt.show()


    # date_list=list(range(1982,2021))
    # print(date_list)
    for i in range(len(SPEI_arr_list)):
        date_delta_i=t[i]
        print(date_delta_i)
        date_delta_i=datetime.timedelta(int(date_delta_i))
        # print(date_delta_i)
        date_i=basetime+date_delta_i
        print(date_i)
        # if date_i.split('-')[0] not in date_list :
        #     continue
        # print(date_i)
        year=date_i.year
        month=date_i.month
        fname = 'GLEAM_root_'+'{}{:02d}.tif'.format(year, month)
        print (fname)
        newRasterfn = outdir + fname
        print (newRasterfn)
        longitude_start = origin_x
        latitude_start = origin_y
        pixelWidth = pix_width
        pixelHeight = pix_height
    # array = val
        array=SPEI_arr_list[i]
        array = np.array(array)
    # method 2
        array = array.T
        array[array<-10]=np.nan
        # plt.imshow(array)
        # plt.colorbar()
        # plt.show()
        to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)



def montly_composite():
    fdir = '/Volumes/SSD_sumsang/project_greening/Data/CCI_SM_2020/CCI_SM_2020_TIFF/'
    outdir = '/Volumes/SSD_sumsang/project_greening/Data/CCI_SM_2020/CSIF-TIFF_montly_composite/'
    mk_dir(outdir)
    month=[]
    for i in range(1,13):
        month.append('{:02d}'.format(i))
    print(month)

    for M in month:
        arr_sum = 0
        n = 0
        average_sif = 0
        for file in os.listdir(fdir):

            file_month_extraction=file.split('.')[0].split('_')[2][4:6]
            file_year_extraction = file.split('.')[0].split('_')[2][0:4]

            if file_month_extraction==M:
              array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir +file)
              array = np.array(array, dtype=np.float)
              arr_sum=arr_sum+array
              n=n+1
        average_sif=arr_sum/n

        newRasterfn = outdir + file_year_extraction[0:4]+M+'.tif'
        print(file_year_extraction[0:4]+M+'.tif')

        to_raster.array2raster(newRasterfn, -180, 90, 0.25, -0.25, average_sif, ndv=-999999)
        plt.imshow(average_sif,vmin=0,vmax=1)
        plt.title(M)
        plt.colorbar()
        plt.show()

def nc_to_tif_precip():  #处理GPCP数据

    fdir = '/Users/admin/Downloads/practice/test/'
    outdir = '/Users/admin/Downloads/practice/test/'
    mk_dir(outdir)
    for f in os.listdir(fdir):
        if f.endswith('.nc'):
            fpath = fdir + f
            nc = Dataset(fpath)
            print(nc)
            print(nc.variables.keys())
        # since 2005 - 1 - 5
            lat_list = nc['latitude']
            lon_list = nc['longitude']
            lat_list=lat_list[::-1]  #取反
            lon_list=lon_list[::-1] #取反
            origin_x = lon_list[0]
            print(origin_x)
            origin_y = lat_list[0]
            pix_width = lon_list[1] - lon_list[0]
            pix_height = lat_list[1] - lat_list[0]
            # print origin_x
            # print origin_y
            # print pix_width
            # print pix_height
            precip1 = nc['precip']
            print(precip1)
            # SIF_arr_list.shape
            # print(SIF_arr_list[0])
            t=nc['time']
            print(t[:])
            date_start = datetime.datetime(2005,1,1)
            fname='20050105.tif'
            # print fname
            newRasterfn = outdir + fname
            # print newRasterfn
            longitude_start = origin_x
            latitude_start = origin_y
            pixelWidth = pix_width
            pixelHeight = pix_height
            array = precip1[0]
            array = array[::-1]

            array=transformation_array(array)
            array=np.array(array)

            raster2array.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)
            plt.imshow(array)
            plt.colorbar()
            plt.show()

def nc_to_tif_SM_CCI():  #Wen 处理1982-2020nc 数据

    fdir_all = '/Users/wenzhang/Downloads/ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED_1978-2020-v06.1/'
    outdir = '/Users/wenzhang/Downloads/CCI_SM_TIFF/'
    mk_dir(outdir)
    for fdir in os.listdir(fdir_all):
        if fdir.startswith('.'):
            continue
        for f in os.listdir(fdir_all+fdir):
            if f.startswith('.'):
                continue
            fpath=fdir_all+fdir+'/'+f
            nc = Dataset(fpath)
            print(nc)
            print(nc.variables.keys())
            t = nc['time']
            print(t)

            basetime = datetime.datetime(1970, 1, 1)  # 告诉起始时间
            lat_list = nc['lat']
            lon_list = nc['lon']
            # lat_list=lat_list[::-1]  #取反
            # print(lat_list[:])
            # print(lon_list[:])

            origin_x = lon_list[0]  # 要为负数-180
            origin_y = lat_list[0]  # 要为正数90
            pix_width = lon_list[1] - lon_list[0]  # 经度0.5
            pix_height = lat_list[1] - lat_list[0]  # 纬度-0.5
            # print(origin_x)
            # print(origin_y)
            # print(pix_width)
            # print(pix_height)
            # SIF_arr_list = nc['SIF']
            SPEI_arr_list = nc['sm']
            # print(SPEI_arr_list.shape) # [1,720,1440]
            # print(SPEI_arr_list[0])
            # plt.imshow(SPEI_arr_list[0])  #
            # plt.imshow(SPEI_arr_list[::])
            # plt.show()

            # date_list=list(range(1982,2021))
            # print(date_list)
            for i in range(len(SPEI_arr_list)):
                date_delta_i = t[i]
                print(date_delta_i)
                date_delta_i = datetime.timedelta(int(date_delta_i))
                # print(date_delta_i)
                date_i = basetime + date_delta_i
                print(date_i)
                # if date_i.split('-')[0] not in date_list :
                #     continue
                # print(date_i)
                year = date_i.year
                month = date_i.month
                day=date_i.day
                fname=f'CCI_SM_{year}{month:02d}{day:02d}.tif'
                # fname = 'CCI_SM_' + '{}{:02d}.tif'.format(year, month)
                print(fname)
                newRasterfn = outdir + fname
                print(newRasterfn)
                longitude_start = origin_x
                latitude_start = origin_y
                pixelWidth = pix_width
                pixelHeight = pix_height
                # array = val
                array = SPEI_arr_list[i]
                array = np.array(array)
                # method 2
                # array = array.T
                array[array < 0] = np.nan
                # plt.imshow(array)
                # plt.colorbar()
                # plt.show()
                to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array,
                                       ndv=-999999)

def nc_to_tif_landcover():

    f = '/Volumes/SSD_sumsang/project_greening/mcd12c1_halfdegree.nc'
    outdir = '/Volumes/SSD_sumsang/project_greening/Data/landcover/landcover_tif/'
    Tools().mk_dir(outdir, force=True)

    nc = Dataset(f)
    lc_list=['water', 'grass', 'shrub', 'crop', 'EBF', 'ENF', 'DBF', 'DNF', 'savanna', 'urban', 'nonveg']
    for lc in lc_list:
        print(nc)
        print(nc.variables.keys())
        year_list = nc['year']
        print(year_list)
        for year in year_list:
            print(year)

        lat_list = nc['lat']
        lon_list = nc['lon']
        # lat_list=lat_list[::-1]  #取反
        print(lat_list[:])
        print(lon_list[:])
        origin_x = lon_list[0]  #要为负数-180
        origin_y = lat_list[0]  #要为正数90
        pix_width = lon_list[1] - lon_list[0] #经度0.5
        pix_height = lat_list[1] - lat_list[0] # 纬度-0.5
        print (origin_x)
        print (origin_y)
        print (pix_width)
        print (pix_height)

        SPEI_arr_list = nc[lc]
        print(SPEI_arr_list.shape)
        print(SPEI_arr_list[0])
        # plt.imshow(SPEI_arr_list[5])
        # # plt.imshow(SPEI_arr_list[::])
        # plt.show()


        # date_list=list(range(2001,2020))
        # print(date_list)
        for i in range(len(SPEI_arr_list)):
            year=year_list[i]

            fname = f'Landcover_{lc}_{int(year)}.tif'
            print (fname)
            newRasterfn = outdir + fname
            print (newRasterfn)
            longitude_start = origin_x
            latitude_start = origin_y
            pixelWidth = pix_width
            pixelHeight = pix_height
        # array = val
            array=SPEI_arr_list[i]
            array = np.array(array)
        # method 2
            array= numpy.rot90(array, 1,(0,1))
            array = array[::-1]

            # plt.imshow(array)
            # plt.colorbar()
            # plt.show()
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)

def nc_to_tif_Trendy():
    fdir=''

    f = f'LPX-Bern_S2_lai.nc'
    outdir = '/Users/wenzhang/Downloads/LPX-Bern_S2_lai_tif'
    Tools().mk_dir(outdir, force=True)
    yearlist =list(range(1982,2021))
    nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)



def nc_to_tif_template(fname,var_name,outdir,yearlist):

    try:
        ncin = Dataset(fname, 'r')
    except:
        return
    try:
        lat = ncin.variables['lat'][:]
        lon = ncin.variables['lon'][:]
    except:
        lat = ncin.variables['latitude'][:]
        lon = ncin.variables['longitude'][:]
    shape = np.shape(lat)

    time = ncin.variables['time'][:]
    basetime = ncin.variables['time'].units
    basetime = basetime.strip('days since ')

    try:
        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
    except:
        try:
            basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M:%S')
        except:
            basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M:%S.%f')
    data = ncin.variables[var_name]
    if len(shape) == 2:
        xx,yy = lon,lat
    else:
        xx,yy = np.meshgrid(lon, lat)
    for time_i in range(len(data)):
        date = basetime + datetime.timedelta(days=int(time[time_i]))

        mon = date.month
        year = date.year
        day=date.day
        if year not in yearlist:
            continue
        outf_name = f'{year}{mon:02d}{day:02d}.tif'
        outpath = join(outdir, outf_name)
        if isfile(outpath):
            continue
        arr = data[time_i]
        arr = np.array(arr)
        lon_list = []
        lat_list = []
        value_list = []
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                lon_i = xx[i][j]
                if lon_i > 180:
                    lon_i -= 360
                lat_i = yy[i][j]
                value_i = arr[i][j]
                lon_list.append(lon_i)
                lat_list.append(lat_i)
                value_list.append(value_i)
        DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list,outpath)



def transformation_array(array):
    newarray=[]
    for i in range(len(array)):
        #for row in array
        # temp_part1=array[i][0:1800]  对于360*720
        # temp_part2 =array[i][180:360]
        temp_part1 = array[i][0:1800]   #对于3600*7200
        temp_part2 = array[i][1800:3600]
        temp_part1=list(temp_part1)
        temp_part2 = list(temp_part2)
        # print(temp_part1)
        # print(temp_part2)
        newarray.append(temp_part2+temp_part1)
    return newarray
def main():


    # nc_to_tif_BESS()
    # nc_to_tif_SPEI3()
    # nc_to_tif_CO2()
    # nc_to_tif_PET()
    # nc_to_tif_GLEAM()
    # montly_composite()
    # nc_to_tif_SM_CCI()
    # nc_to_tif_landcover()
    nc_to_tif_Trendy()

if __name__ == '__main__':
                main()