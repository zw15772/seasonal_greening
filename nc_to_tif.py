# coding =utf-8
# coding='utf-8'

import sys
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

def nc_to_tif_burning():

    fdir = '/Volumes/SSD_sumsang/project_greening/Data/burning_areas/burning_nc/'
    outdir = '/Volumes/SSD_sumsang/project_greening/Data/burning_areas/burning_tif/'
    mk_dir(outdir)
    for f in os.listdir(fdir):
        if f.endswith('.nc'):
            fpath = fdir + f
            nc = Dataset(fpath)
            fname2=fpath.split('.')[1][1:5]
            if fname2=='2016' or fname2=='2017'or fname2=='2018':
                continue
            print(nc)
            print(nc.variables.keys())
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
            GPCP_arr_list = nc['fraction_of_burnable_area']
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
            array= numpy.rot90(array, 1,(0,1))
            array = array[::-1]
            plt.imshow(array)
            plt.colorbar()
            plt.show()
            to_raster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999)

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
    fdir = '/Users/admin/Downloads/CSIF-TIFF/'
    outdir = '/Users/admin/Downloads/CSIF-TIFF_montly_composite/'
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
            if file[4:6]==M:
              array, originX, originY, pixelWidth, pixelHeight = raster2array.raster2array(fdir +file)
              array = np.array(array, dtype=np.float)
              arr_sum=arr_sum+array
              n=n+1
        average_sif=arr_sum/n
        newRasterfn = outdir + file[0:4]+M+'.tif'
        raster2array.array2raster(newRasterfn, originX, originY, pixelWidth, pixelHeight, average_sif, ndv=-999999)
        plt.imshow(average_sif,vmin=0,vmax=0.6)
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
    nc_to_tif_burning()
    # nc_to_tif_GLEAM()
    # montly_composite()

if __name__ == '__main__':
                main()