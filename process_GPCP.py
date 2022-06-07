# coding='utf-8'
import sys
from HANTS import HANTS

version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import to_raster
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
# import h5py
from netCDF4 import Dataset
import shutil
import requests
import green_driver_trend_contribution

import os
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
# from LY_Tools import *
from lytools import *
T=Tools()

project_root='/Volumes/SSD_sumsang/project_greening/'
data_root=project_root+'Data/'
result_root=project_root+'Result/new_result'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

def BESS_preprocess(): # 转成2002年里面25个dic, 2003里面25个dic

    fdir=data_root+'BESS/PAR_tif_resample_scale/'
    # outdir = '/Users /admin/Downloads/dic_CSIF_par/'
    outdir = data_root + 'BESS/dic_annually_PAR/'
    mk_dir(outdir)
    flist = os.listdir(fdir)
    multi_year_array_dic = {}
    key_list=[]
    year_list=list(range(2002,2017))
    # print(year_list)
    # exit()

    for year in year_list:  # 构造字典的键值，并且字典的键：值初始化
        multi_year_array_dic[year] = []

    # print(dic_key_list)
    for year in year_list:
        outf = outdir + '{}'.format(year)+'/'
        # print(outf)
        # exit()
        mk_dir(outf)
        for f in tqdm(sorted(flist), desc='loading...'):
            if not f.endswith('tif'):
                continue
            if f[5:9]==str(year):
                array, originX, originY, pixelWidth, pixelHeight = raster2array.raster2array(fdir + f)
                array = np.array(array, dtype=np.float)
                multi_year_array_dic[year].append(array)

        row = len(multi_year_array_dic[year][0])
        col = len(multi_year_array_dic[year][0][0])
        key_list = []
        dic = {}

        for r in range(row):  # 构造字典的键值，并且字典的键：值初始化
            for c in range(col):
                dic[(r, c)] = []
                key_list.append((r, c))
        # print(dic_key_list)

        for r in tqdm(range(row), desc='构造time series'):  # 构造time series
            for c in range(col):
                for arr in multi_year_array_dic[year]:
                    value = arr[r][c]
                    dic[(r, c)].append(value)
        flag = 0
        temp_dic = {}
        for key in tqdm(key_list, desc='output...'):  # 存数据
            flag = flag + 1
            time_series = dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outf + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outf + 'per_pix_dic_%03d' % 0, temp_dic)

        # np.save(outf, multi_year_array_dic)
        # plt.imshow(array)
        # plt.show()
        # print(np.shape(array))
        # exit()
    # exit()


def per_pixel_all_year_PAR(): # 2002年里面25个dic, 2003里面25个dic转换成一个pix 有15*365 个像素

    fdir = result_root + 'Hants_annually_smooth/Hants_annually_smooth_LST_mean/'
    outdir = result_root + 'dic_per_pixel_all_year_variables/LST_mean/'
    Tools().mk_dir(outdir)
    yearlist = list(range(2002, 2017))
    result_dic = DIC_and_TIF().void_spatial_dic()
    for year in yearlist:
        fdir_year = fdir + str(year) + '/'
        print(fdir_year)
        # Tools().mk_dir(outdir)
        dic = {}
        dic_1 = {}
        for f in tqdm(sorted(os.listdir(fdir_year))):
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir_year + f, allow_pickle=True,encoding='latin1' ).item())
                dic.update(dic_i)
        for pix in tqdm(dic, desc='{}'.format(year)):
            # r,c = pix
            # if r>50:
            #     continue
            time_series=dic[pix]
            result_dic[pix].append(time_series)
    print(len(result_dic[123,345][0]))
    exit()
    temp_dic = {}
    flag = 0
    for key in tqdm(dic, desc='output...'):  # 存数据
        flag = flag + 1
        time_series = result_dic[key]
        time_series = np.array(time_series)
        temp_dic[key] = time_series
        if flag % 10000 == 0:
            # print(flag)
            np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
            temp_dic = {}
    np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

        # np.save(outf, multi_year_array_dic)
        # plt.imshow(array)
        # plt.show()
        # print(np.shape(array))
        # exit()
    # exit()

def per_pixel_all_year_NDVI(): # 2002npy, 2003npy转换成一个pix 有15*365 个像素

    fdir = result_root + 'Hants_annually_smooth/Hants_annually_smooth_EVI/'
    outdir = result_root + 'dic_per_pixel_all_year_variables/EVI/'
    Tools().mk_dir(outdir)
    result_dic = DIC_and_TIF().void_spatial_dic()
    # dic_complete_pix=DIC_and_TIF().void_spatial_dic()
    # dic = {}
    year_list=list(range(2002,2017))
    for f in (sorted(os.listdir(fdir))):
        if int(f[:4]) not in year_list:
            continue
        if f.endswith('.npy'):
            dic = {}
            dic = dict(np.load(fdir + f, allow_pickle=True,encoding='latin1' ).item())
        for pix in tqdm(dic, desc='{}'.format(f[:4])):
            # r,c = pix
            # if r>50:
            #     continue
            time_series=dic[pix]
            result_dic[pix].append(time_series)
    # print(len(result_dic[123,345]))
    # exit()
    temp_dic = {}
    flag = 0
    for key in tqdm(dic, desc='output...'):  # 存数据
        flag = flag + 1
        time_series = result_dic[key]
        time_series = np.array(time_series)
        temp_dic[key] = time_series
        if flag % 10000 == 0:
            # print(flag)
            np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
            temp_dic = {}
    np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

        # np.save(outf, multi_year_array_dic)
        # plt.imshow(array)
        # plt.show()
        # print(np.shape(array))
        # exit()
    # exit()


def BESS_check(): # 转成2002.npy, 2003.npy
    fdir ='/Volumes/1T/wen_prj/Result/anomaly/CSIF_par_annually_transform/'
    dic = {}
    dic_1 = {}
    result_dic = {}
    for f in tqdm(os.listdir(fdir)):
        # print(fi
        if f.endswith('.npy'):
            dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
            dic.update(dic_i)
    # ///////check 字典是否不缺值///////////////////////
    x = []
    y = []
    spatial_dic = {}
    for pix in tqdm(dic):
        vals = dic[pix]
        vals = np.array(vals)
        vals[vals<0]=np.nan
        if np.isnan(np.nanmean(vals)):
            continue
        # if np.nanmean(vals)<0:
        spatial_dic[pix] = len(vals)
        print(len(vals))
        # plt.plot(vals)
        # plt.title('{}'.format(pix))
        # plt.show()
        # x.append(pix[0])
        # y.append(pix[1])
    # plt.scatter(y,x)
    # plt.show()
    # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
    # plt.imshow(arr)
    # plt.show()


def CSIF_par_annually_transform(): # 将一个pixel 15*365转换成2002npy, 2003 npy

    fdir=result_root+'/anomaly/CSIF_par_per_pixel/'
    outdir=result_root+'anomaly/CSIF_par_annually_transform/'
    Tools().mk_dir(outdir)
    yearlist=list(range(2002,2017))
    dic = {}
    dic_1 = {}
    for f in tqdm(os.listdir(fdir)):
        if not f.startswith('p'):
            continue
        if f.endswith('.npy'):
            dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
            dic.update(dic_i)
    i = 0
    for year in yearlist:
        outf = outdir + '{}'.format(year)
        result_dic= DIC_and_TIF().void_spatial_dic()

        for pix in tqdm(dic,desc='{}'.format(year)):
            time_series = dic[pix]
            time_series = np.array(time_series)
            if len(time_series)!=15:
                continue
            time_series[time_series<-99]=np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            if np.nanmean(time_series) == 0:
                continue
            # print(time_series[i])
            result_dic[pix].append(time_series[i])
        i=i+1
        np.save(outf,result_dic)

def CSIF_par_annually_inverse_transform(): # 将一个2002npy, 2003 npy 转换成 per_pixel

    fdir=result_root+'Hants_annually_smooth/Hants_annually_smooth_CSIF/'
    outdir=result_root+'Hants_annually_smooth/CSIF_annually_transform/'
    Tools().mk_dir(outdir)
    yearlist=list(range(2002,2017))
    dic = {}

    result_dic = DIC_and_TIF().void_spatial_dic()
    for year in yearlist:
        dic= dict(np.load(fdir + str(year)+'.npy', allow_pickle=True, ).item())
        for pix in tqdm(dic,desc='{}'.format(year)):
            time_series = dic[pix]
            time_series = np.array(time_series)

            time_series[time_series<-99]=np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            if np.nanmean(time_series) == 0:
                continue
            # print(time_series[i])
            result_dic[pix].append(time_series)

    flag = 0
    temp_dic = {}
    for key in tqdm(result_dic, desc='output...'):  # 存数据
        flag = flag + 1
        time_series = result_dic[key]
        time_series = np.array(time_series)
        temp_dic[key] = time_series
        if flag % 10000 == 0:
            # print(flag)
            np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
            temp_dic = {}
    np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

class interpolate:

    def run(self):
        # self.interpolation_temp()
        self.interpolation_NIRv()


    def interpolation(self):  #函数实现CO2 的共计168个月的 的缺失值插值，最后生成一个字典
        fdir = data_root +'Terraclimate/Temp/temp_dic/'
        outdir = data_root + 'Terraclimate/Temp/temp_dic_interpolation/'
        Tools().mk_dir(outdir)
        dic = {}

        for f in tqdm(os.listdir(fdir)):
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        result_dic = {}
        for pix in tqdm(dic, desc='interpolate'):
         # r,c = pix
         # if r>50:
         #     continue
            time_series = dic[pix]
         # print(time_series)
            time_series[time_series < -999] = np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            matix = np.isnan(time_series)   #因为检查time series 发现
            matix = list(matix)
            valid_number = matix.count(False)
            # print(pix,valid_number)
            if valid_number/len(time_series)<0.85:
                continue
            ynew = np.array(time_series)
            # if ynew[0][0] < -99:
            #     continue
            ynew = Tools().interp_nan(ynew)
            result_dic[pix]=ynew
            # result_dic[pix]=ynew
            # plt.plot(ynew)
            # plt.title('{}'.format(pix))
            # plt.show()
        # arr=DIC_and_TIF().pix_dic_to_spatial_arr(result_dic)
        # # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(arr)
        # plt.show()
        flag = 0
        temp_dic = {}
        for key in tqdm(result_dic, desc='output...'):  # 存数据
            flag = flag + 1
            time_series = result_dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)
        np.save(outdir +'dic_CO2_interpolation', result_dic)

    def interpolation_NDVI(self):

        mask_tif='/Volumes/SSD_sumsang/project_greening/Data/GIMMS_NDVI/NDVI_mask.tif'

        mask_dic=DIC_and_TIF().spatial_tif_to_dic(mask_tif)

        periods = ['early', 'peak', 'late']
        time_range = '2002-2015'
        len_year = 14


        for period in periods:
            fdir = result_root + 'extraction_original_val/{}_original_extraction_all_seasons_MODIS/'.format(time_range)
            outdir = result_root + '/extraction_original_val/{}_original_extraction_all_seasons/{}_original_extraction_all_seasons_clean/'.format(
                time_range, time_range, period)
            Tools().mk_dir(outdir, force=True)

            dic_NDVI = {}

            for f in tqdm(os.listdir(fdir)):
                # if 'GIMMS' not in f:
                #     continue
                if f.endswith('.npy'):
                    # if not '005' in f:
                    #     continue
                    dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                    dic_NDVI.update(dic_i)

            dic_spatial_count = {}
            result_dic = {}
            for pix in tqdm(mask_dic):
                if pix not in dic_NDVI:
                    continue
                time_series = dic_NDVI[pix]

                time_series = time_series
                if np.isnan(mask_dic[pix]):  # 用已经mask 好的模板
                    continue

                # 1. 去除无效值  2 插值
                time_series[time_series<-1]=np.nan  #将序列中的无效值变成nan--进行下一步处理

                matix = np.isnan(time_series)  # 因为检查time series 发现
                matix = list(matix)
                valid_number = matix.count(False)
                # print(matix)
                # print(pix,valid_number)
                if valid_number / len(time_series) < 0.80:
                    continue
                ynew = np.array(time_series)
                # if ynew[0][0] < -99:
                #     continue
                ynew = Tools().interp_nan(ynew)
                if ynew[0]==None:
                    continue
                result_dic[pix] = ynew
                # plt.plot(ynew)
                # plt.title('{}'.format(pix))
                # plt.show()
                # print(len(result_dic[pix]))
                dic_spatial_count[pix] = len(result_dic[pix])
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
            # print(arr.shape)
            # # # DIC_and_TIF().plot_back_ground_arr()
            # plt.imshow(arr)
            # plt.show()
            np.save(outdir + 'during_NDVI', result_dic)


    def interpolation_MODIS_LAI(self):

        mask_tif='/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'

        mask_dic=DIC_and_TIF().spatial_tif_to_dic(mask_tif)
        variables=['MODIS_LAI']

        periods = ['early', 'peak', 'late']
        # time_range = '1988-2016'

        dic_NDVI = {}
        for variable in variables:
            for period in periods:
                outdir = result_root + f'/Main_flow/arr/DIC_Daily/{variable}_interpolate/'
                Tools().mk_dir(outdir, force=True)
                fdir = result_root + f'/Main_flow/arr/DIC_Daily/{variable}/'
                f='during_{}_{}.npy'.format(period,variable)


                dic_NDVI = dict(np.load(fdir + f, allow_pickle=True, ).item())
                # dic_NDVI.update(dic_i)

                dic_spatial_count = {}
                result_dic = {}
                for pix in tqdm(mask_dic):
                    if pix not in dic_NDVI:
                        continue
                    time_series = dic_NDVI[pix]
                    time_series_array=np.array(time_series)

                    # time_series_array = time_series_array/10000.
                    time_series_array = time_series_array
                    if np.isnan(mask_dic[pix]):  # 用已经mask 好的模板
                        continue
                    # print(time_series_array)

                    # 1. 去除无效值  2 插值

                    if np.isnan(np.nanmean(time_series)):
                        continue
                    if len(time_series_array)==0:
                        continue

                    time_series_array[time_series_array<0]=np.nan  #将序列中的无效值变成nan--进行下一步处理

                    matix = np.isnan(time_series_array)  # 因为检查time series 发现
                    matix = list(matix)
                    valid_number = matix.count(False)
                    # print(matix)
                    print(pix,valid_number)
                    if valid_number / len(time_series_array) < 0.70:
                        continue
                    ynew = np.array(time_series_array)
                    # if ynew[0][0] < -99:
                    #     continue
                    ynew = Tools().interp_nan(ynew)
                    if ynew[0]==None:
                        continue
                    result_dic[pix] = ynew
                    # plt.plot(ynew)
                    # plt.title('{}'.format(pix))
                    # plt.show()
                    # print(len(result_dic[pix]))
                    dic_spatial_count[pix] = len(result_dic[pix])
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                # print(arr.shape)
                # DIC_and_TIF().plot_back_ground_arr()
                plt.imshow(arr)
                plt.show()
                np.save(outdir + f'during_{period}_{variable}_1', result_dic)


    def interpolation_temp(self):  # 函数实现temp 的共计444个月的 的缺失值插值，最后生成一个字典
        fdir = data_root + 'Terraclimate/Temp/temp_dic/'
        outdir = data_root + 'Terraclimate/Temp/temp_dic_interpolation/'
        Tools().mk_dir(outdir)
        dic = {}

        for f in tqdm(os.listdir(fdir)):
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        result_dic = {}
        spatial_dic={}
        for pix in tqdm(dic, desc='interpolate'):
            # r,c = pix
            # if r>50:
            #     continue
            time_series = dic[pix]
            # print(time_series)
            time_series[time_series < -60] = np.nan
            time_series[time_series > 60] = np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            matix = np.isnan(time_series)  # 因为检查time series 发现
            matix = list(matix)
            valid_number = matix.count(False)
            # print(pix,valid_number)
            if valid_number / len(time_series) < 0.85:
                continue
            ynew = np.array(time_series)
            # if ynew[0][0] < -99:
            #     continue
            ynew = Tools().interp_nan(ynew)
            result_dic[pix] = ynew
            # plt.plot(ynew)
            # plt.title('{}'.format(pix))
            # plt.show()
            # print(len(result_dic[pix]))
            spatial_dic[pix] = len(result_dic[pix])
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # print(arr.shape)
        # # # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(arr)
        # plt.show()


        flag = 0
        temp_dic = {}
        for key in tqdm(result_dic, desc='output...'):  # 存数据
            flag = flag + 1
            time_series = result_dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)
        np.save(outdir + 'dic_temp_interpolation', result_dic)

    def interpolation_NIRv(self):  # 函数实现NIRv 的共计444个月的 的缺失值插值，最后生成一个字典
        fdir = data_root + '/NIRv/NIRv_dic/'
        outdir = data_root + '/NIRv/NIRv_dic_interpolation/'
        Tools().mk_dir(outdir,True)
        dic = {}

        for f in tqdm(os.listdir(fdir)):
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        result_dic = {}
        spatial_dic={}
        for pix in tqdm(dic, desc='interpolate'):
            # r,c = pix
            # if r>50:
            #     continue
            time_series = dic[pix]
            # print(time_series)
            # time_series[time_series < 0] = np.nan
            # time_series[time_series > 1] = np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            matix = np.isnan(time_series)  # 因为检查time series 发现
            matix = list(matix)
            valid_number = matix.count(False)
            # print(pix,valid_number)
            if valid_number / len(time_series) < 0.8:
                continue
            ynew = np.array(time_series)
            # if ynew[0][0] < -99:
            #     continue
            ynew = Tools().interp_nan(ynew)
            result_dic[pix] = ynew
            # plt.plot(ynew)
            # plt.title('{}'.format(pix))
            # plt.show()
            # print(len(result_dic[pix]))
            spatial_dic[pix] = len(result_dic[pix])
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # print(arr.shape)
        # # # DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(arr)
        plt.show()


        # flag = 0
        # temp_dic = {}
        # for key in tqdm(result_dic, desc='output...'):  # 存数据
        #     flag = flag + 1
        #     time_series = result_dic[key]
        #     time_series = np.array(time_series)
        #     temp_dic[key] = time_series
        #     if flag % 10000 == 0:
        #         # print(flag)
        #         np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
        #         temp_dic = {}
        # np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)
        # # np.save(outdir + 'dic_nirv_interpolation', result_dic)



    def interpolation_CSIF(self):  # CSIF/CSIF_par 的共计192个月的 的缺失值插值，最后生成一个字典
        fdir = data_root + 'CSIF_fpar/CSIF_fpar_dic/'
        outdir = data_root + '/CSIF_fpar/CSIF_fpar_dic_interpolation/'
        Tools().mk_dir(outdir, True)
        dic = {}

        for f in tqdm(os.listdir(fdir)):
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic= dict(np.load(fdir + f, allow_pickle=True, ).item())

        result_dic = {}
        spatial_dic = {}
        for pix in tqdm(dic, desc='interpolate'):
            # r,c = pix
            # if r>50:
            #     continue
            time_series = dic[pix]
            # print(time_series)
            # time_series[time_series < 0] = np.nan
            # time_series[time_series > 1] = np.nan
            if np.isnan(np.nanmean(time_series)):
                continue
            matix = np.isnan(time_series)  # 因为检查time series 发现
            matix = list(matix)
            valid_number = matix.count(False)
            # print(pix,valid_number)
            if valid_number / len(time_series) < 0.8:
                continue
            ynew = np.array(time_series)
            # if ynew[0][0] < -99:
            #     continue
            ynew = Tools().interp_nan(ynew)
            result_dic[pix] = ynew
            # plt.plot(ynew)
            # plt.title('{}'.format(pix))
            # plt.show()
            # print(len(result_dic[pix]))
            spatial_dic[pix] = len(result_dic[pix])
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # print(arr.shape)
        # # # DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(arr)
        plt.show()

        flag = 0
        temp_dic = {}
        for key in tqdm(result_dic, desc='output...'):  # 存数据
            flag = flag + 1
            time_series = result_dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)
        # np.save(outdir + 'dic_nirv_interpolation', re

    def interpolation_VOD(self):
        fdir = result_root + '/extraction_original_val/extraction_during_early_growing_season_static/'
        outdir = result_root + '/extraction_original_val/extraction_during_early_growing_season_static/'
        Tools().mk_dir(outdir, True)
        dic = {}

        for f in tqdm(os.listdir(fdir)):
            if 'CCI' not in f:
                continue
            if f.endswith('.npy'):
                dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

            result_dic = {}
            spatial_dic = {}
            for pix in tqdm(dic, desc='interpolate'):
                # r,c = pix
                # if r>50:
                #     continue
                time_series = dic[pix]
                time_series=np.array(time_series)
                time_series[time_series < 0] = np.nan
                # time_series[time_series > 1] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    continue
                matix = np.isnan(time_series)  # 因为检查time series 发现
                matix = list(matix)
                valid_number = matix.count(False)
                # print(pix,valid_number)
                if valid_number / len(time_series) < 0.7:
                    continue
                ynew = np.array(time_series)
                # if ynew[0][0] < -99:
                #     continue
                ynew = Tools().interp_nan(ynew)
                result_dic[pix] = ynew
                # plt.plot(ynew)
                # plt.title('{}'.format(pix))
                # plt.show()
                # print(len(result_dic[pix]))
                spatial_dic[pix] = len(result_dic[pix])
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # print(arr.shape)
            # # # DIC_and_TIF().plot_back_ground_arr()
            plt.imshow(arr)
            plt.show()

            np.save(outdir + f.split('.')[0]+'_interpolation', result_dic)

def plot_dic():  # LAI4g
    build_dataframe=green_driver_trend_contribution.Build_dataframe()

    # f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_original_val/extraction_during_late_growing_season_static/during_late_LAI3g_interpolation.npy'
    fdir='/Volumes/SSD_sumsang/project_greening/Data/original_dataset/MODIS_LAI_dic/'
    for f in tqdm(os.listdir(fdir)):

        if f.endswith('.npy'):
            dic = dict(np.load(fdir + f, allow_pickle=True, ).item())

    NDVI_dic = dic

    # NDVI_dic = T.load_npy(f)
    dics={'late_LAI_raw':NDVI_dic}
    df=T.spatial_dics_to_df(dics)
    df=build_dataframe.add_NDVI_mask(df)
    df = build_dataframe.add_row(df)

    P_PET_dic=build_dataframe.P_PET_ratio(build_dataframe.P_PET_dir)
    P_PET_reclass_dic=build_dataframe.P_PET_reclass(P_PET_dic)
    df=T.add_spatial_dic_to_df(df,P_PET_reclass_dic,'HI_class')

    df = df[df['row'] < 120]
    df = df[df['NDVI_MASK'] == 1]
    df = df[df['HI_class'] == 'Humid']
    # df=df.dropna(subset=['late_LAI_raw'])
    late_LAI_raw=df['late_LAI_raw'].tolist()
    # late_LAI_raw=np.array(late_LAI_raw)
    late_LAI_raw=np.array(late_LAI_raw)
    print(type(late_LAI_raw))
    T.print_head_n(df)
    print(df)
    mean_series=np.nanmean(late_LAI_raw,axis=0)
    plt.plot(mean_series)
    plt.show()


    pass

def foo():

    # f='/Volumes/SSD_sumsang/project_greening/Result/detrend/extraction_during_late_growing_season_static/during_late_CSIF_par/per_pix_dic_008.npy'
    f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Main_flow/arr/DIC_Daily/LAI3g/per_pix_dic_007.npy'
    # f='/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_window/1982-2015_during_early/during_early_CO2.npy'
    result_dic = {}
    spatial_dic={}
    # array = np.load(f)
    # dic = DIC_and_TIF().spatial_arr_to_dic(array)
    dic = dict(np.load(f, allow_pickle=True, encoding='latin1').item())
    # ///////check 字典是否不缺值////
    for pix in tqdm(dic, desc='interpolate'):
        r,c =pix
        china_r=list(range(750,150))
        china_c=list(range(550,620))

        # china_r = list(range(140, 570))
        # china_c = list(range(55, 57))
        if r not in china_r:
            if c not in china_c:
                continue
        # if not r==250:
        #     continue
        # if not c==550:
        #     continue
        print(len(dic[pix]))
        # exit()
        if len(dic[pix])==0:
            continue
        time_series = dic[pix]
        print(time_series)
        if len(time_series)==0:
            continue
        # print(time_series)
        # time_series_reshape=time_series.reshape(15,-1)
        # plt.plot(time_series_reshape[0])
        plt.plot(time_series)
        # # # # plt.imshow(time_series_reshape)
        plt.title(str(pix))
        plt.show()
        # spatial_dic[pix]=len(time_series)
    #     spatial_dic[pix] = time_series[0]
    # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
    # plt.imshow(arr)
    # plt.title(str(pix))
    # plt.show()
def foo2():

    fdir = '/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_val/extraction_during_peak_growing_season_static/'
    for f in tqdm(os.listdir(fdir)):
        print(f)

        if f.startswith ('.') :
            continue

        dic = T.load_npy(fdir + f)
        data_length = 0
        for pix in dic:
            vals = dic[pix]
            if len(vals) == 0:
                continue
            data_length = len(vals)
            break
        time_series = []
        for i in tqdm(range(data_length)):
            picked_vals = []
            for pix in dic:
                r, c = pix
                if r > 120:
                    continue
                vals = dic[pix]
                if len(vals) != data_length:
                    continue
                # print(i)
                val_i = vals[i]
                if np.isnan(val_i):
                    continue
                picked_vals.append(val_i)
            mean_val = np.mean(picked_vals)
            time_series.append(mean_val)

        plt.plot(time_series, label=f)
        # plt.title(f)
    plt.legend()
    plt.show()

def foo3(): #做平均

    fdir = '/Volumes/SSD_sumsang/project_greening/Result/new_result/detrend_extraction/during_early_growing_season_static/during_early_CCI_SM.npy'
    dic={}
    for f in tqdm(sorted(os.listdir(fdir))):
        if f.startswith('.') :
            continue
        if f.endswith('.npy'):
            # if not '005' in f:
            #     continue
            dic_i = T.load_npy(fdir+f)
            dic.update(dic_i)

    data_length = 0
    for pix in dic:
        vals = dic[pix]
        if len(vals) == 0:
            continue
        data_length = len(vals)
        break
    time_series = []
    for i in tqdm(range(data_length)):
        picked_vals = []
        for pix in dic:
            r, c = pix
            if r > 120:
                continue
            vals = dic[pix]
            if len(vals) != data_length:
                continue
            # print(i)
            val_i = vals[i]
            if np.isnan(val_i):
                continue
            picked_vals.append(val_i)
        mean_val = np.mean(picked_vals)
        time_series.append(mean_val)

    plt.plot(time_series, label=f)
    # plt.title(f)
    plt.legend()
    plt.show()




def spatial_plot():
    spatial_dic_value={}
    fdir1=data_root+'/original_dataset/CCI_SM_dic/'
    f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_original_val/2000-2018/during_peak_CCI_SM.npy'

    dic=T.load_npy(f)
    # dic=T.load_npy_dir(fdir1)
    spatial_dic={}
    spatial_length_dic={}

    for pix in tqdm(dic):

        val=dic[pix]

        val=val
        if len(val)==0:
            continue
        # length = len(val)
        isnan_list=np.isnan(val)
        length=T.count_num(isnan_list,False)
        # print(val)
        # exit()
        val_array = np.array(val)
        spatial_dic[pix]=val_array
        spatial_length_dic[pix] = length


    # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
    # arr = np.array(arr)
    #  看长度
    arr_len = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_length_dic)
    arr_len = np.array(arr_len)

    plt.figure()
    plt.imshow(arr_len, cmap='jet',)
    # plt.imshow(arr_len, cmap='jet', vmin=-0.1, vmax=0.01)
    plt.colorbar()
    plt.show()

    pass
def foo4():
    f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_val/extraction_during_late_growing_season_static/during_late_NIRv.npy'

    dic = T.load_npy(f)
    data_length = 0
    for pix in dic:
        vals = dic[pix]
        if len(vals) == 0:
            continue
        data_length = len(vals)
        break
    time_series = []
    for i in tqdm(range(data_length)):
        picked_vals = []
        for pix in dic:
            r, c = pix
            if r > 120:
                continue
            vals = dic[pix]
            if len(vals) != data_length:
                continue
            # print(i)
            val_i = vals[i]
            if np.isnan(val_i):
                continue
            picked_vals.append(val_i)
        mean_val = np.mean(picked_vals)
        time_series.append(mean_val)

    plt.plot(time_series, label=f)
        # plt.title(f)
    plt.legend()
    plt.show()

def beta_plot():  # 该功能实现所有因素的beta
    # time_range='1982-1998'
    period='early'
    # f = '/Volumes/sult/multi_linear_anomaly_NDVI/{}_multi_linear{}_anomaly.npy'.format(time_range,period)
    # f='/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_correlation_anomaly_NDVI/1982-1998_partial_correlationpeak_anomaly.npy'
    f='/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_window/1982-2000_during_early_window15/partial_correlation_early_1982-2000_window10_correlation.npy'
    # outdir='/Volumes/SSD_sumsang/project_greening/Result/new_result/multiregression_anomaly/MODIS_NDVI_{}/'.format(period)
    # T.mk_dir(outdir,force=True)
    dic = T.load_npy(f)
    var_list = []
    for pix in dic:
        # print(pix)
        vals = dic[pix]
        for var_i in vals:
            var_list.append(var_i)
    var_list = list(set(var_list))
    for var_i in var_list:
        spatial_dic = {}
        for pix in dic:
            dic_i = dic[pix]
            if not var_i in dic_i:
                continue
            val = dic_i[var_i]
            spatial_dic[pix] = val
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # DIC_and_TIF().arr_to_tif(arr,outdir+var_i+'.tif')
        std = np.nanstd(arr)
        mean = np.nanmean(arr)
        vmin = mean - std
        vmax = mean + std
        plt.figure()
        arr[arr>0.1]=1
        # plt.imshow(arr,vmin=vmin,vmax=vmax)
        plt.imshow(arr)
        plt.title(var_i)
        plt.colorbar()
    plt.show()

def beta_save_():  # 该功能实现所有因素的beta
    time_range='2002-2018'
    period='late'
    # f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/multiregression/LAI_GIMMS/detrend_1982-2001_multi_linearearly_LAI_GIMMS.npy'
    f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Sankey_plot/Data/LAI4g/2000-2018_PLS_late_LAI4g.npy'
    outdir='/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_correlation_original_detrend/TIFF_{}_{}_8/'.format(time_range,period)
    T.mk_dir(outdir,force=True)
    dic = T.load_npy(f)
    var_list = []
    for pix in dic:
        # print(pix)
        vals = dic[pix]


        for var_i in vals:
            var_list.append(var_i)
    var_list = list(set(var_list))
    for var_i in var_list:
        spatial_dic = {}
        for pix in dic:
            dic_i = dic[pix]
            if not var_i in dic_i:
                continue
            val = dic_i[var_i]
            spatial_dic[pix] = val
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        arr[arr<-99]=np.nan
        # DIC_and_TIF().arr_to_tif(arr,outdir+var_i+'.tif')
        std = np.nanstd(arr)
        mean = np.nanmean(arr)
        vmin = mean - std
        vmax = mean + std
        plt.figure()
        # plt.imshow(arr,vmin=vmin,vmax=vmax)
        plt.imshow(arr, vmin=-0.5, vmax=0.5,cmap='jet')
        plt.title(var_i)
        plt.colorbar()
    plt.show()

def check_pcorr():
    f='/Users/wenzhang/Desktop/parial_corr_window_00-15.npy'
    dic = T.load_npy(f)
    dic_pcorr = {}
    for pix in dic:
        dic_i = dic[pix]['pcorr']
        dic_pcorr[pix] = dic_i
    df = T.dic_to_df(dic_pcorr, 'pix')
    print(df)


def spatial_check():  ## 空间上查看有多少个月

    # fdir ='/Volumes/SSD_sumsang/project_greening/Result/new_result/trend_window/1982-2018_during_early_window15/trend_during_early_Aridity.npy'

    f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_window/1982-2018_during_early_window15/partial_correlation_early_1982-2018_window00_correlation.npy'
    # outdir = data_root + 'VOD/VOD_interpolation/'
    # Tools().mk_dir(outdir, True)
    dic = {}
    dic = dict(np.load(f, allow_pickle=True, ).item())
    # for f in tqdm(os.listdir(fdir)):
    #     if not f.startswith('p'):
    #         continue
    #     if f.endswith('.npy'):
    #         dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
    #         dic.update(dic_i)
    result_dic = {}
    spatial_dic = {}
    for pix in tqdm(dic, desc=''):
        # r,c = pix
        # if r>50:
        #     continue
        time_series = dic[pix]


        # time_series[time_series < -999] = np.nan
        # time_series[time_series < 0] = np.nan

        if np.isnan(np.nanmean(time_series)):
            continue

        spatial_dic[pix] = time_series

        # spatial_dic[pix] = len(dic[pix])
    arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
    # print(arr.shape)
    # # # DIC_and_TIF().plot_back_ground_arr()
    plt.imshow(arr)
    plt.show()

def foo1():
    f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_val/extraction_during_early_growing_season_static/during_early_NIRv.npy'
    dic = dict(np.load(f, allow_pickle=True, ).item())
    year_list=range(15)
    print(year_list)

    mean_val = {}
    confidence_value = {}
    val={}

    for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
        mean_val[year] = []
        confidence_value[year] = []
        val[year]=[]

    for year in year_list:
        for pix in tqdm(dic, desc=''):
            if len(dic[pix])<15:
                continue
            r, c = pix
            if r>120:
                continue
            val_i=dic[pix][year]
            val[year].append(val_i)
        val_list = np.array(val[year])
        n = len(val_list)
        mean_val_i = np.nanmean(val_list)
        se = stats.sem(val_list)
        h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
        confidence_value[year] = h
        mean_val[year] = mean_val_i

    mean_value_yearly = []
    up_list = []
    bottom_list = []

    for year in year_list:
        mean_value_yearly.append(mean_val[year])
        up_list.append(mean_val[year] + confidence_value[year])
        bottom_list.append(mean_val[year] - confidence_value[year])
    plt.plot(mean_value_yearly, label='test')
    plt.fill_between(range(len(mean_value_yearly)), up_list, bottom_list, alpha=0.3, zorder=-1)
    plt.xticks(range(len(mean_value_yearly)), year_list)
    plt.show()







def copy_CCI_to_one_folder():
    fdir='/Volumes/SSD_sumsang/project_greening/Data/CCI_monthly/'
    outdir='/Volumes/SSD_sumsang/project_greening/Data/CCI_all_year/'
    Tools().mk_dir(outdir)
    for folder in os.listdir(fdir):
        for f in os.listdir(fdir+folder):
            inpath=os.path.join(fdir, folder,f)
            fname=folder+f.split('.')[0].split('_')[-1]
            outpath=outdir+fname+'.tif'
            shutil.copy(inpath,outpath)


def main():
    # BESS_preprocess()
    # BESS_check()
    # interpolate().interpolation_CSIF()
    # interpolate().interpolation_NDVI()
    # interpolate().interpolation_VOD()
    # interpolate().interpolation_NIRv()
    # interpolate().interpolation_MODIS_LAI()

    # per_pixel_all_year_PAR()
    # spatial_check()
    # CSIF_par_annually_transform()
    # plot_dic()

    foo()
    # spatial_plot_Yang()
    # spatial_plot()
    # beta_plot()
    # beta_save_()
    # check_pcorr()
    # foo4()
    #  foo3()
    # interpolate().run()
    # foo1()

    # CSIF_par_annually_inverse_transform()
    # copy_CCI_to_one_folder()



if __name__ == '__main__':
    main()
