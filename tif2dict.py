# coding='utf-8'
import sys
from HANTS import HANTS
import pingouin
import pingouin as pg
from green_driver_trend_contribution import *

version = sys.version_info.major
assert version == 3, 'Python Version Error'
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import re
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
from lytools import *
from osgeo import gdal

from osgeo import gdal




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
T=Tools()



# project_root='/Volumes/SSD_sumsang/project_greening/'
# data_root=project_root+'Data/'
# result_root=project_root+'Result/new_result/'


this_root = 'D:/Greening/'
data_root = 'D:/Greening/Data/'
results_root = 'D:/Greening/Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

def tif2dict():
    fdir='/Volumes/SSD_sumsang/project_greening/Data/LAI_3g/LAI_3g_resample'


    NDVI_mask_f='/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'
    array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
    array_mask[array_mask<0]=np.nan

    T.mk_dir(outdir,force=True)
    flist=os.listdir(fdir)
    all_array=[]
    year_list=list(range(2001,2021))  # ??????????????????
    for f in tqdm(sorted(flist),desc='loading...'):
        if f.startswith('.'):
            continue
        if not f.endswith('.tif'):
            continue
        print(f)
        # print(f.split('.')[0].split('_')[2][0:4])
        # print(f.split('.')[0][0:4])

        # if int(f.split('.')[0].split('_')[2][0:4]) not in year_list:  #
        #     continue

        # if int(f.split('.')[0][0:4]) not in year_list:  #
        #     continue
        # if int(f.split('_')[2][0:4]) not in year_list:  #
        #     continue
        # print(f.split('.')[0][0:4])
        # if int(f.split('.')[0][0:4]) not in year_list:  #
        #     continue

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
        array = np.array(array, dtype=np.float)
        array=array[:360]  # PAR???361*720

        # array[array<-999]=np.nan
        array[array ==0] = np.nan
        # array[array < 0] = np.nan # ????????????LAI ????????????<0!!
        # plt.imshow(array)
        # plt.show()
        array_mask=np.array(array_mask,dtype=np.float)
        # plt.imshow(array_mask)
        # plt.show()
        array=array * array_mask
        # plt.imshow(array)
        # plt.show()

        # print(np.shape(array))
        # exit()
        all_array.append(array)
    # exit()

    row=len(all_array[0])
    col = len(all_array[0][0])
    key_list=[]
    dic={}

    for r in tqdm(range(row),desc='??????key'): #?????????????????????????????????????????????????????????
        for c in range(col):
            dic[(r,c)]=[]
            key_list.append((r,c))
    #print(dic_key_list)


    for r in tqdm(range(row),desc='??????time series'): # ??????time series
        for c in range(col):
            for arr in all_array:
                value=arr[r][c]
                dic[(r,c)].append(value)
            # print(dic)
    time_series=[]
    flag=0
    temp_dic={}
    for key in tqdm(key_list,desc='output...'): #?????????
        flag=flag+1
        time_series=dic[key]
        time_series=np.array(time_series)
        temp_dic[key]=time_series
        if flag %10000 == 0:
            # print(flag)
            np.save(outdir +'per_pix_dic_%03d' % (flag/10000),temp_dic)
            temp_dic={}
    np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

def tif2dict_daily():
    fdir=results_root+'Main_flow/arr/Phenology/Hants/LAI3g/'
    outdir=results_root+'Main_flow/arr/DIC/Hants/LAI3g/'
    NDVI_mask_f='/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'
    array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
    array_mask[array_mask<0]=np.nan

    T.mk_dir(outdir,force=True)
    flist=os.listdir(fdir)
    all_array=[]
    year_list=list(range(1982,2021))  # ??????????????????


    dic = {}
    key_list = []

    for r in tqdm(range(360), desc='??????key'):  # ?????????????????????????????????????????????????????????
        for c in range(720):
            dic[(r, c)] = []
            key_list.append((r, c))

    for f in tqdm(sorted(flist), desc='loading each year...'):
        if f.startswith('.'):
            continue
        if not f.endswith('.npy'):
            continue
        print(f)
        year = int(f.split('.')[0])
        if year not in year_list:
            continue
        dic_file = dict(np.load(fdir + f, allow_pickle=True, ).item())
        for key in (key_list):
            if key not in dic_file:
                continue
            valus = dic_file[key]
            values_array = np.array(valus)
            values_array[values_array < 0] = np.nan # ????????????LAI ????????????<0!!
            dic[key].append(valus)


    time_series = []
    flag = 0
    temp_dic = {}
    for key in tqdm(key_list, desc='output...'):  # ?????????
        flag = flag + 1
        time_series = dic[key]
        time_series = np.array(time_series)
        temp_dic[key] = time_series
        if flag % 10000 == 0:
            # print(flag)
            np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
            temp_dic = {}
    np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)



            # array = np.array(array, dtype=np.float)
            # array=array[:360]  # PAR???361*720
            #
            # # array[array<-999]=np.nan
            # array[array ==0] = np.nan
            # # array[array < 0] = np.nan # ????????????LAI ????????????<0!!
            # # plt.imshow(array)
            # # plt.show()
            # array_mask=np.array(array_mask,dtype=np.float)
            # # plt.imshow(array_mask)
            # # plt.show()
            # array=array * array_mask
            # # plt.imshow(array)
            # # plt.show()
            #
            # # print(np.shape(array))
            # # exit()
            # all_array.append(array)
            #

def tif2dict_trendy():
    fdir_all = 'D:/greening/Data/Trendy_TIFF_resample_unify_2/'
    outdir_all = 'D:/greening/Data/DIC2/'
    NDVI_mask_f = 'D:/greening/Data/Base_data/NDVI_mask.tif'
    array_mask, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(NDVI_mask_f)
    array_mask[array_mask < 0] = np.nan


    for fdir in tqdm(os.listdir(fdir_all)):
        print(fdir)
        if not 'ensemble' in fdir:
            continue
        outdir = join(outdir_all, fdir + '/')
        T.mk_dir(outdir, force=True)
        year_list=list(range(1982,2021))# ??????????????????
        all_array = []
        for f in tqdm(sorted(os.listdir(fdir_all+fdir+'/')),desc='loading...'):
            if f.startswith('.'):
                continue
            if not f.endswith('.tif'):
                continue
            print(f)
            # if int(f.split('.')[0][0:4]) not in year_list:  #
            #     continue

            if int(f.split('.')[0][-6:-2]) not in year_list:  #
                continue

            array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir_all+fdir+'/' + f)
            array = np.array(array, dtype=np.float)
            array=array[:360]  # PAR???361*720

            array[array<-999]=np.nan
            # array[array ==0] = np.nan
            array[array <= 0] = np.nan # ????????????LAI ????????????<0!!
            # plt.imshow(array)
            # plt.show()
            array_mask=np.array(array_mask,dtype=np.float)
            # plt.imshow(array_mask)
            # plt.show()
            array=array * array_mask
            # plt.imshow(array)
            # plt.show()

            # print(np.shape(array))
            # exit()
            all_array.append(array)
        # exit()

        row=len(all_array[0])
        col = len(all_array[0][0])
        key_list=[]
        dic={}

        for r in tqdm(range(row),desc='??????key'): #?????????????????????????????????????????????????????????
            for c in range(col):
                dic[(r,c)]=[]
                key_list.append((r,c))

        for r in tqdm(range(row),desc='??????time series'): # ??????time series
            for c in range(col):
                for arr in all_array:
                    value=arr[r][c]
                    dic[(r,c)].append(value)
                # print(dic)
        time_series=[]
        flag=0
        temp_dic={}

        for key in tqdm(key_list,desc='output...'): #?????????
            flag=flag+1
            time_series=dic[key]
            time_series=np.array(time_series)
            temp_dic[key]=time_series
            if flag %10000 == 0:
                # print(flag)
                np.save(outdir +'per_pix_dic_%03d' % (flag/10000),temp_dic)
                temp_dic={}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

def tif2dic_single_file():
    inf = '/Volumes/1T/wen_prj/Result/trend/EOS_trend_CSIF_par_threshold_20%.tif'
    outf = '/Volumes/1T/wen_prj/Result/trend/EOS_trend_CSIF_par_threshold_20%.npy'

    array, originX, originY, pixelWidth, pixelHeight = raster2array.raster2array(inf)
    array = np.array(array, dtype=np.float)

    row = len(array)
    col = len(array[0])
    key_list = []
    dic = {}

    for r in range(row):  # ?????????????????????????????????????????????????????????
        for c in range(col):
            dic[(r, c)] = []
            key_list.append((r, c))

    for r in tqdm(range(row), desc='??????time series'):  # ??????time series
        for c in range(col):
                value = array[r][c]
                dic[(r, c)].append(value)
    time_series = []
    flag = 0
    temp_dic = {}
    for key in tqdm(key_list, desc='output...'):  # ?????????
        flag = flag + 1
        time_series = dic[key]
        time_series = np.array(time_series)
        temp_dic[key] = time_series
    np.save(outf, temp_dic)

def Hants_average_smooth():  #??????????????????????????????????????????,????????????
    fdir='/Users/admin/Downloads/dic_CSIF_par/'
    dic = {}
    dic_1 = {}
    result_dic = {}
    for f in tqdm(os.listdir(fdir)):
        # print(fi
        if f.endswith('.npy'):
            dic_i = dict(np.load(fdir+f,allow_pickle=True,).item())
            dic.update(dic_i)
    #///////check ?????????????????????///////////////////////
    # x = []
    # y = []
    # spatial_dic = {}
    # for pix in tqdm(dic):
    #     vals = dic[pix]
    #     vals = np.array(vals)
    #     vals[vals<0]=np.nan
    #     # if np.isnan(np.nanmean(vals)):
    #     # if np.nanmean(vals)<0:
    #     if np.nanmean(vals) == 0:
    #         continue
    #     spatial_dic[pix] = 1
    #
    #     x.append(pix[0])
    #     y.append(pix[1])
    # # plt.scatter(y,x)
    # # plt.show()
    # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
    # plt.imshow(arr)
    # plt.show()
    # exit()

    # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    # max_val_dic={}
    # min_val_dic={}
    # median_val_dic={}
    # for pix in tqdm(dic):
    #     time_series = dic[pix]
    #     r,c=pix
    #     china_r = list(range(75,150))
    #     china_c = list(range(550,620))
    #     if not r in china_r:
    #         continue
    #     if not c in china_c:
    #         continue
    for pix in tqdm(dic):
        time_series = dic[pix]
        if np.isnan(np.nanmean(time_series)):
           continue
        if np.nanmean(time_series) == 0:
            continue
    #         ??????
    #     max_val = np.max(time_series)
    #     if max_val<0:
    #         max_val_dic[pix] = max_val
    #     min_val = np.min(time_series)
    #     min_val_dic[pix] = min_val
    #     median_val=np.median(time_series)
    #     if median_val<0:
    #         median_val_dic[pix] = median_val
    # arr_max=DIC_and_TIF().pix_dic_to_spatial_arr(max_val_dic)
    # arr_min = DIC_and_TIF().pix_dic_to_spatial_arr(min_val_dic)
    # arr_median = DIC_and_TIF().pix_dic_to_spatial_arr(median_val_dic)
    #
    # DIC_and_TIF().plot_back_ground_arr()
    # plt.imshow(arr_max,vmin=-0.2, vmax=0.2)
    # plt.title('max_result')
    #
    # plt.figure()
    # plt.imshow(arr_min,cmap='jet', vmin=-0.2, vmax=0.2)
    # DIC_and_TIF().plot_back_ground_arr()
    # plt.title('min_result')
    #
    # plt.figure()
    # plt.imshow(arr_median,vmin=-0.2, vmax=0.2)
    # DIC_and_TIF().plot_back_ground_arr()
    # plt.title('median_result')
    #
    # plt.show()
        time_series_reshape = np.reshape(time_series,(18,-1))
        time_series_reshape=time_series_reshape.T
        averageCSIF=[]
        for i in time_series_reshape:
            # print(time_series_reshape.shape)
            ii=np.mean(i)
            averageCSIF.append(ii)
        # plt.plot(averageCSIF)
        # plt.show()
        # print(averageCSIF)
        # exit()
        averageCSIF=np.array(averageCSIF)
        # averageCSIF[(averageCSIF)<0]=0
        ynew=Tools().interp_1d(averageCSIF,-999999)
        ynew=np.array([ynew])
        if ynew[0][0]==None:
            continue
        xnew, ynew365=interp(ynew[0])
        ynew365 = np.array([ynew365])
        try:
            result=HANTS(sample_count=365, inputs=ynew365,
                        frequencies_considered_count=3,
                        outliers_to_reject='Hi',
                        low=-1, high=2,
                        fit_error_tolerance=np.std(ynew365),
                        delta=0.1)
            # result_dic[pix]=1
            result_dic[pix] = result[0]
            # ###### ??????
            # plt.subplot(211)
            # plt.plot(ynew365[0])
            # plt.title('spline average_result',)
            # plt.scatter(np.linspace(0, 365, len(ynew[0])),ynew[0])
            #
            # # plt.figure()
            # plt.plot(result[0])
            # plt.title('Hants_average_result', )
            #
            # plt.subplot(212)
            # dic_temp = {}
            # c,r = pix
            # for ci in range(c-5,c+5):
            #     for ri in range(r-5,r+5):
            #         pix_new = (ci,ri)
            #         dic_temp[pix_new] = 10
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_temp)
            # plt.imshow(arr,'jet')
            # DIC_and_TIF().plot_back_ground_arr()
            # plt.title(str(pix))
            #
            # plt.show()
        except Exception as e:
            # print(e)
            #
            # print(ynew365)
            # plt.plot(ynew365[0])
            # plt.show()
            # time.sleep(2)
            continue
    # arr=DIC_and_TIF().pix_dic_to_spatial_arr(result_dic)
    # # DIC_and_TIF().plot_back_ground_arr()
    # plt.imshow(arr)
    # plt.show()

    outdir='/Volumes/1T/wen_prj/average_phenology_CSIF_par/'
    mk_dir(outdir)
    flag = 0
    temp_dic = {}
    for key in tqdm(result_dic, desc='output...'):  # ?????????
        flag = flag + 1
        hants_time_series = result_dic[key]
        hants_time_series = np.array(hants_time_series)
        temp_dic[key] = hants_time_series
        if flag % 10000 == 0:
            # print(flag)
            np.save(outdir + 'average_hants_per_pix_dic_%03d' % (flag / 10000), temp_dic)
            temp_dic = {}
    np.save(outdir + 'average_hants_per_pix_dic_%03d' % 0, temp_dic)

def Hants_annually_smooth(): # ???????????????????????????

    fdir=data_root+'BESS/dic_annually_PAR/'
    outdir=result_root+'Hants_annually_smooth/Hants_annually_smooth_LST_mean/'

    Tools().mk_dir(outdir)
    dic = {}
    dic_1 = {}
    for f in tqdm(os.listdir(fdir)):
        # print(fi
        if f.endswith('.npy'):
            dic_i = dict(np.load(fdir+f,allow_pickle=True,).item())
            dic.update(dic_i)
    # result_dic = DIC_and_TIF().void_spatial_dic()
    for y in range(2003, 2017):
        outf = outdir+'{}'.format(y)
        result_dic={}
        for pix in tqdm(dic,desc='{}'.format(y)):
            # r,c = pix
            # if r>50:
            #     continue
            time_series = dic[pix]
            if np.isnan(np.nanmean(time_series)):
                continue
            if np.nanmean(time_series) == 0:
                continue
            # if np.nanmean(time_series) <= 0:
            #     continue
            time_series_reshape = np.reshape(time_series, (14, -1))
            # print(time_series_reshape[int(y) - 2003])
            # exit()
            ynew=Tools().interp_1d(time_series_reshape[int(y)-2003],-999999)
            ynew=np.array([ynew])
            if ynew[0][0]==None:
                continue
            # print(ynew)
            # plt.plot(ynew[0])
            # plt.show()
            xnew, ynew365=interp(ynew[0])
            # plt.plot(ynew365)
            ynew365 = np.array([ynew365])
            try:
                result=HANTS(sample_count=365, inputs=ynew365,
                            frequencies_considered_count=3,
                            outliers_to_reject='Hi',
                            low=-70, high=70,
                            fit_error_tolerance=np.std(ynew365),
                            delta=0.1)
                # plt.plot(result[0])
                # plt.scatter(np.linspace(0,365,len(time_series_reshape[int(y) - 2000])),time_series_reshape[int(y) - 2000])
                # plt.title('{}'.format(pix))
                # plt.show()
                result_dic[pix]=result[0]
            except Exception as e:
                # print(e)
                # print(ynew365)
                # plt.plot(ynew365[0])
                # plt.show()
                # exit()
                pass
        np.save(outf,result_dic)
    # np.save(outdir + 'result_dic', result_dic)


def interp(vals):

    # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

    inx = range(len(vals))
    iny = vals
    x_new = np.linspace(min(inx), max(inx), 365)
    # print(x_new)
    func = interpolate.interp1d(inx, iny)
    y_new = func(x_new)

    return x_new, y_new


class Main_flow_Early_Peak_Late_Dormant:

    def __init__(self):
        self.this_class_arr='/Volumes/SSD_sumsang/project_greening/Result/new_result'
        pass


    def run(self):
        self.annual_phelogy()
        pass


    def annual_phelogy(self):
        # # 4 calculate phenology
        # self.early_peak_late_dormant_period_annual()
        # # 5 transform daily to monthly
        # self.transform_early_peak_late_dormant_period_annual()
        # # pass

        self.early_peak_late_dormant_period_multiyear()
        # 5 transform daily to monthly
        self.transform_early_peak_late_dormant_period_multiyear()
        pass


    def transform_early_peak_late_dormant_period_annual(self):
        vars_dic = {
            'early_length': '',
            'mid_length': '',
            'late_length': '',
            'dormant_length': '',
            'early_start': '',
            'early_start_mon': '',

            'early_end': '',
            'early_end_mon': '',

            'peak': 'peak',
            'peak_mon': '',

            'late_start': '',
            'late_start_mon': '',

            'late_end': '',
            'late_end_mon': '',
        }


        outdir = self.this_class_arr + 'early_peak_late_dormant_period_annually/20%_transform_early_peak_late_dormant_period_annually_CSIF_par/'
        T.mk_dir(outdir)
        fdir = self.this_class_arr + 'early_peak_late_dormant_period_annually/20%_early_peak_late_dormant_period_annually_CSIF_par/'
        sif_dic_hants_dir = self.this_class_arr + 'hants_annually_smooth/hants_annually_smooth_CSIF_par/'

        # sif_dic = Tools().load_npy_dir(sif_dic_f)

        #
        for var in vars_dic:
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            for y in tqdm(range(2002,2017),desc=var):
                f = fdir + '{}.npy'.format(y)
                dic = T.load_npy(f)
                for pix in dic:
                    var_val = dic[pix][var]
                    spatial_dic[pix].append(var_val)
            # valid_dic = {}
            # for pix in spatial_dic:
            #     vals_len = len(spatial_dic[pix])
            #     if vals_len > 0:
            #         valid_dic[pix] = 1
            # valid_arr = DIC_and_TIF().pix_dic_to_spatial_arr(valid_dic)
            # plt.imshow(valid_arr)
            # plt.show()
            np.save(outdir + var,spatial_dic)

        ############### Dormant Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(2002, 2017), desc='dormant_mons_list'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            for pix in dic:
                sos = dic[pix]['early_start_mon']
                eos = dic[pix]['late_end_mon']
                gs = range(sos,eos+1)
                winter_mons = []
                for m in range(1,13):
                    if m in gs:
                        continue
                    winter_mons.append(m)
                spatial_dic[pix].append(winter_mons)
        np.save(outdir + 'dormant_mons', spatial_dic)

        ############### GS Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(2002, 2017), desc='dormant_mons_list'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            for pix in dic:
                sos = dic[pix]['early_start_mon']
                eos = dic[pix]['late_end_mon']
                if sos >= eos:
                    continue
                gs_mons = range(sos, eos + 1)
                spatial_dic[pix].append(gs_mons)
        np.save(outdir + 'gs_mons', spatial_dic)

        ########  peak vals  #######
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(2002, 2017), desc='peak_val'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            sif_dic = T.load_npy(sif_dic_hants_dir + str(y) + '.npy')
            for pix in dic:
                var_val = dic[pix]['peak']
                peak_val = sif_dic[pix][var_val]
                spatial_dic[pix].append(peak_val)
        np.save(outdir + 'peak_vals', spatial_dic)

                ########  peak_average vals  #######

        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for y in tqdm(range(2002, 2017), desc='peak_val'):
            f = fdir + '{}.npy'.format(y)
            dic = T.load_npy(f)
            sif_dic = T.load_npy(sif_dic_hants_dir + str(y) + '.npy')
            for pix in dic:
                # var_val = dic[pix]['peak']
                early_end_index = dic[pix]['early_end']
                late_start_index = dic[pix]['late_start']
                # peak_val = sif_dic[pix][var_val]
                # spatial_dic[pix].append(peak_val)
                peak_selected_vals = sif_dic[pix][early_end_index:late_start_index]
                peak_average = np.mean(peak_selected_vals)
                spatial_dic[pix].append(peak_average)

        np.save(outdir + 'peak_average', spatial_dic)

        pass


    def early_peak_late_dormant_period_annual(self,threshold_i=0.2):
        hants_smooth_dir = self.this_class_arr + 'hants_annually_smooth/hants_annually_smooth_CSIF_par/'

        # print(hants_smooth_dir)
        # print('/Users/wenzhang/project/drought_legacy/results_root_main_flow_2002/arr/Main_flow_Early_Peak_Late_Dormant/hants_smooth_annual')
        # exit()
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_annually/20%_early_peak_late_dormant_period_annually_CSIF_par/'
        T.mk_dir(outdir)

        for f in os.listdir(hants_smooth_dir):
            outf_i = outdir + f
            year = int(f.split('.')[0])
            hants_smooth_f = hants_smooth_dir + f
            hants_dic = T.load_npy(hants_smooth_f)
            result_dic = {}

            for pix in tqdm(hants_dic,desc=str(year)):
                vals = hants_dic[pix]
                if len(vals)==0:
                    continue
                peak = np.argmax(vals)
                if peak == 0 or peak == (len(vals)-1):
                    continue
                # if np.nanmean(vals) <= 0:  # ??????????????????CSIF, ?????????NDVI??????EVI ?????????2000
                #     continue
                try:
                    early_start = self.__search_left(vals, peak, threshold_i)
                    late_end = self.__search_right(vals, peak, threshold_i)
                except:
                    early_start = 60
                    late_end = 130
                    # print vals
                    plt.plot(vals)
                    plt.show()
                # method 1
                # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
                # method 2
                early_end, late_start = self.__median_early_late(vals,early_start,late_end,peak)

                early_period = early_end - early_start
                peak_period = late_start - early_end
                late_period = late_end - late_start
                dormant_period = 365 - (late_end - early_start)

                result = {
                    'early_length':early_period,
                    'mid_length':peak_period,
                    'late_length':late_period,
                    'dormant_length':dormant_period,
                    'early_start':early_start,
                    'early_start_mon':self.__day_to_month(early_start),

                    'early_end':early_end,
                    'early_end_mon':self.__day_to_month(early_end),

                    'peak':peak,
                    'peak_mon':self.__day_to_month(peak),

                    'late_start':late_start,
                    'late_start_mon':self.__day_to_month(late_start),

                    'late_end':late_end,
                    'late_end_mon':self.__day_to_month(late_end),
                }
                # print(result)
                # exit()
                result_dic[pix] = result
            np.save(outf_i,result_dic)

    def early_peak_late_dormant_period_multiyear(self, threshold_i=0.2):
        # hants_smooth_dir = self.this_class_arr + 'hants_annually_smooth_CSIF_par/'
        hants_smooth_dir = self.this_class_arr + 'multiyear_average_phenology_CSIF_par/'

        # print(hants_smooth_dir)
        # print('/Users/wenzhang/project/drought_legacy/results_root_main_flow_2002/arr/Main_flow_Early_Peak_Late_Dormant/hants_smooth_annual')
        # exit()
        outdir = self.this_class_arr + 'early_peak_late_dormant_period_multiyear/20%_early_peak_late_dormant_period_multiyear_CSIF_par/'
        T.mk_dir(outdir)

        dic = {}
        result_dic={}
        for f in tqdm(os.listdir(hants_smooth_dir),desc='??????'):
            if f.endswith('.npy'):
                dic_i = dict(np.load(hants_smooth_dir + f, allow_pickle=True, ).item())
                dic.update(dic_i)

        for pix in tqdm(dic, desc='????????????'):
            vals = dic[pix]
            plt.plot(vals)
            plt.show()
            peak = np.argmax(vals)
            if peak == 0 or peak == (len(vals) - 1):
                continue
            if np.nanmean(vals) <= 0:  # ??????????????????CSIF, ?????????NDVI??????EVI ?????????2000
                continue
            try:
                early_start = self.__search_left(vals, peak, threshold_i)
                late_end = self.__search_right(vals, peak, threshold_i)
            except:
                early_start = 60
                late_end = 130
                # print vals
                plt.plot(vals)
                plt.show()
            # method 1
            # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak)
            # method 2
            early_end, late_start = self.__median_early_late(vals, early_start, late_end, peak)

            early_period = early_end - early_start
            peak_period = late_start - early_end
            late_period = late_end - late_start
            dormant_period = 365 - (late_end - early_start)

            result = {
                'early_length': early_period,
                'mid_length': peak_period,
                'late_length': late_period,
                'dormant_length': dormant_period,
                'early_start': early_start,
                'early_start_mon': self.__day_to_month(early_start),

                'early_end': early_end,
                'early_end_mon': self.__day_to_month(early_end),

                'peak': peak,
                'peak_mon': self.__day_to_month(peak),

                'late_start': late_start,
                'late_start_mon': self.__day_to_month(late_start),

                'late_end': late_end,
                'late_end_mon': self.__day_to_month(late_end),
            }
            # print(result)
            # exit()
            result_dic[pix] = result
        np.save(outdir+'phenology_extraction', result_dic)

    def transform_early_peak_late_dormant_period_multiyear(self):
        vars_dic = {
            'early_length': '',
            'mid_length': '',
            'late_length': '',
            'dormant_length': '',
            'early_start': '',
            'early_start_mon': '',

            'early_end': '',
            'early_end_mon': '',

            'peak': 'peak',
            'peak_mon': '',

            'late_start': '',
            'late_start_mon': '',

            'late_end': '',
            'late_end_mon': '',
        }

        outdir = self.this_class_arr + 'early_peak_late_dormant_period_multiyear/20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/'
        T.mk_dir(outdir)
        f1 = self.this_class_arr + 'early_peak_late_dormant_period_multiyear/20%_early_peak_late_dormant_period_multiyear_CSIF_par/phenology_extraction.npy'
        sif_dic_hants_dir = self.this_class_arr + 'multiyear_average_phenology_CSIF_par/'
        # sif_dic_hants_dir = self.this_class_arr + 'Hants_annually_smooth_CSIF_par/'

        # sif_dic = Tools().load_npy_dir(sif_dic_f)

        #
        for var in vars_dic:
            spatial_dic = DIC_and_TIF().void_spatial_dic()
            dic= dict(np.load(f1, allow_pickle=True, ).item())
            for pix in dic:
                var_val = dic[pix][var]
                spatial_dic[pix].append(var_val)
            # valid_dic = {}
            # for pix in spatial_dic:
            #     vals_len = len(spatial_dic[pix])
            #     if vals_len > 0:
            #         valid_dic[pix] = 1
            # valid_arr = DIC_and_TIF().pix_dic_to_spatial_arr(valid_dic)
            # plt.imshow(valid_arr)
            # plt.show()
            np.save(outdir + var, spatial_dic)

        ############### Dormant Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        dic = dict(np.load(f1, allow_pickle=True, ).item())
        for pix in dic:
            sos = dic[pix]['early_start_mon']
            eos = dic[pix]['late_end_mon']
            gs = range(sos, eos + 1)
            winter_mons = []
            for m in range(1, 13):
                if m in gs:
                    continue
                winter_mons.append(m)
            spatial_dic[pix].append(winter_mons)
        np.save(outdir + 'dormant_mons', spatial_dic)

        ############### GS Mons #############
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        dic = dict(np.load(f1, allow_pickle=True, ).item())
        for pix in dic:
            sos = dic[pix]['early_start_mon']
            eos = dic[pix]['late_end_mon']
            if sos >= eos:
                continue
            gs_mons = range(sos, eos + 1)
            spatial_dic[pix].append(gs_mons)
        np.save(outdir + 'gs_mons', spatial_dic)

        ########  peak vals  #######
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        sif_dic={}
        dic=dict(np.load(f1, allow_pickle=True, ).item())
        for f in tqdm(os.listdir(sif_dic_hants_dir), desc='??????'):
            if f.endswith('.npy'):
                dic_i = dict(np.load(sif_dic_hants_dir + f, allow_pickle=True, ).item())
                sif_dic.update(dic_i)
        for pix in dic:
            var_val = dic[pix]['peak']
            peak_val = sif_dic[pix][var_val]
            spatial_dic[pix].append(peak_val)
        np.save(outdir + 'peak_vals', spatial_dic)

        ########  peak_average vals  #######

        spatial_dic = DIC_and_TIF().void_spatial_dic()
        dic = dict(np.load(f1, allow_pickle=True, ).item())
        for pix in dic:
            # var_val = dic[pix]['peak']
            early_end_index = dic[pix]['early_end']
            late_start_index = dic[pix]['late_start']
            # peak_val = sif_dic[pix][var_val]
            # spatial_dic[pix].append(peak_val)
            peak_selected_vals = sif_dic[pix][early_end_index:late_start_index]
            peak_average = np.mean(peak_selected_vals)
            spatial_dic[pix].append(peak_average)

        np.save(outdir + 'peak_average', spatial_dic)

        pass

    def __interp__(self, vals):

        # x_new = np.arange(min(inx), max(inx), ((max(inx) - min(inx)) / float(len(inx))) / float(zoom))

        inx = range(len(vals))
        iny = vals
        # x_new = np.linspace(min(inx), max(inx), 168)
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)

        return x_new, y_new

    def __search_left(self, vals, maxind, threshold_i):
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        # if left_min < 2000:
        #     left_min = 2000
        if left_min < 0:
            left_min = 0
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind

    def __search_right(self, vals, maxind, threshold_i):
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        # if right_min < 2000:
        #     right_min = 2000
        if right_min < 0:
            right_min = 0
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind



    def __slope_early_late(self,vals,sos,eos,peak):
        # 1 slope????????????????????????????????????????????early late ??????????????????????????
        # ??????????????early late ??????????????????
        slope_left = []
        for i in range(sos,peak):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_left.append(slope_i)

        slope_right = []
        for i in range(peak,eos):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_right.append(slope_i)

        max_ind = np.argmax(slope_left) + sos
        min_ind = np.argmin(slope_right) + peak

        return max_ind, min_ind


    def __median_early_late(self,vals,sos,eos,peak):
        # 2 ??????????sos-peak peak-eos???????????????????????sos?????eos??????????????????????????

        median_left = int((peak-sos)*2./3.)
        median_right = int((eos - peak)*1./3.)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind

    def __day_to_month(self,doy):
        base = datetime.datetime(2002,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day

        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    def trend(self):

        # -----------------------------??????----------------------------------------
        # fdir = '/Volumes/1T/wen_prj/Result/anomaly/CSIF_par_annually_transform/2002.npy'
        # dic_1 = {}
        # dic = dict(np.load(fdir, allow_pickle=True, ).item())
        # val_singleyear=[]
        # spatial_dic = {}
        # for pix in tqdm(dic):
        #     val = dic[pix]
        #     if len(val) == 0:
        #         continue
        #     spatial_dic[pix]=len(val[0])
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.title('peak_value {}')
        # plt.show()
        # exit()


        # f_dir=self.this_class_arr+ 'extraction/extraction_pre_EOS_static/20%_Pre_GPCP_extraction/'
        # f_dir = self.this_class_arr + 'extraction/extraction_original_val/temperature/'
        f_dir = self.this_class_arr + 'extraction_original_val/1982-2015_original_extraction_all_seasons/'
        outdir=self.this_class_arr+'trend/original_trend/trend_during_static/'
        # outdir = self.this_class_arr + 'mean/mean_pre_EOS_statiEc/'
        Tools().mk_dir(outdir)
        # exit()
        dic={}
        spatial_dic = {}
        spatial_dic_p_value={}
        spatial_dic_count = {}

        for f in tqdm(os.listdir(f_dir)):
            # if f.endswith('.npy'):
            #     continue
            dic= dict(np.load(f_dir + f, allow_pickle=True, ).item())
            outf=outdir+f.split('.')[0]
            print(outf)
            for pix in tqdm(dic):
                val= dic[pix]
                if len(val)!= 15:
                    continue
                val = np.array(val)
                val[val < -99] = np.nan
                if np.isnan(np.nanmean(val)):
                    continue
                try:
                    xaxis = range(len(val))
                    # a, b, r = KDE_plot().linefit(xaxis, val)
                    r, p = stats.pearsonr(xaxis, val)
                    k, b = np.polyfit(xaxis, val, 1)

                    # print(k)
                    spatial_dic_count[pix]=len(val)

                except Exception as e:
                    print(val)
                    k = np.nan
                    p=np.nan
                spatial_dic[pix] = k  #???trend
                spatial_dic_p_value[pix]=p
                # spatial_dic_r_value[pix] = r
                # spatial_dic[pix] = np.nanmean(val)   #????????????
            count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
            plt.imshow(count_arr)
            plt.colorbar()
            plt.show()
            trend_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            trend_arr = np.array(trend_arr)
            p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
            p_value_arr = np.array(p_value_arr)
            # trend_
            #
            # trend_arr < -10] = np.nan
            # trend_arr[trend_arr > 10] = np.nan

            hist = []
            for i in trend_arr:
                for j in i:
                    if np.isnan(j):
                        continue
                    hist.append(j)

            plt.hist(hist, bins=80)
            plt.figure()
            plt.imshow(trend_arr, cmap='jet', vmin=1, vmax=2.5)
            plt.title(f.split('.')[0])
            plt.colorbar()
            plt.show()
            #
            #     # save arr to tif
            DIC_and_TIF().arr_to_tif(trend_arr,outf+'_trend.tif')
            DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
            np.save(outf+'_trend', trend_arr)
            np.save(outf+'_p_value', p_value_arr)

    def mean(self):

        f_dir = self.this_class_arr + 'extraction/extraction_anomaly/CSIF_par_during_extraction/'
        outdir=self.this_class_arr+'trend/original_trend/trend_during_static/trend_during_whole_static/'
        # outdir = self.this_class_arr + 'mean/mean_pre_EOS_statiEc/'
        Tools().mk_dir(outdir)
        # exit()
        dic={}
        spatial_dic = {}
        spatial_dic_p_value={}
        spatial_dic_count = {}

        for f in tqdm(os.listdir(f_dir)):
            # if f.endswith('.npy'):
            #     continue
            dic= dict(np.load(f_dir + f, allow_pickle=True, ).item())
            outf=outdir+f.split('.')[0]
            print(outf)
            for pix in tqdm(dic):
                val= dic[pix]
                if len(val)!= 15:
                    continue
                val = np.array(val)
                val[val < -99] = np.nan
                if np.isnan(np.nanmean(val)):
                    continue
                try:
                    xaxis = range(len(val))
                    # a, b, r = KDE_plot().linefit(xaxis, val)
                    r,p=stats.pearsonr(xaxis,val)
                    k,b = np.polyfit(xaxis,val,1)
                    # print(k)
                    spatial_dic_count[pix]=len(val)

                except Exception as e:
                    print(val)
                    k = np.nan
                    p=np.nan
                spatial_dic[pix] = k  #???trend
                spatial_dic_p_value[pix]=p
                # spatial_dic_r_value[pix] = r
                # spatial_dic[pix] = np.nanmean(val)   #????????????
            count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
            plt.imshow(count_arr)
            plt.colorbar()
            plt.show()
            trend_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            trend_arr = np.array(trend_arr)
            p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
            p_value_arr = np.array(p_value_arr)
            # trend_arr[trend_arr < -10] = np.nan
            # trend_arr[trend_arr > 10] = np.nan

            hist = []
            for i in trend_arr:
                for j in i:
                    if np.isnan(j):
                        continue
                    hist.append(j)

            plt.hist(hist, bins=80)
            plt.figure()
            plt.imshow(trend_arr, cmap='jet', vmin=1, vmax=2.5)
            plt.title(f.split('.')[0])
            plt.colorbar()
            plt.show()
            #
            #     # save arr to tif
            DIC_and_TIF().arr_to_tif(trend_arr,outf+'_trend.tif')
            DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
            np.save(outf+'_trend', trend_arr)
            np.save(outf+'_p_value', p_value_arr)

    def change_trend(self):  #change rate trend

        fdir = self.this_class_arr + '%CSIF_par/early/'
        # f1 = self.this_class_arr + 'trend/late_trend_CSIF_par_threshold_20%.npy'
        outdir=self.this_class_arr+'change_trend/'
        outf=self.this_class_arr+'/change_trend/early_change'
        Tools().mk_dir(outdir)
        dic_change={}
        dic_trend={}
        spatial_dic = {}
        spatial_dic_count = {}
        spatial_dic_slope={}
        spatial_dic_p_value={}

        # dic_trend = np.load(f1)  #load array

        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                # if not '005' in f:
                #     continue
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic_change.update(dic_i)

        for pix in tqdm(dic_change):
            change_rate = dic_change[pix]
            # trend=dic_trend[pix]
            if len(change_rate)!= 15:
                continue
            # if trend<0:
            #     continue
            spatial_dic_count[pix]=len(change_rate)
            try:
                xaxis = range(len(change_rate))
                # a, b, r = KDE_plot().linefit(xaxis, anomaly_mean)
                r,p=stats.pearsonr(xaxis,change_rate)
                k,b = np.polyfit(xaxis,change_rate,1)
                # print(k)

            except Exception as e:
                print(change_rate)
                k = np.nan
            spatial_dic_slope[pix] = k
            spatial_dic_p_value[pix]=p
        count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
        plt.imshow(count_arr)
        plt.colorbar()
        # plt.show()
        trend_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_slope)
        trend_arr = np.array(trend_arr)
        p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
        p_value_arr = np.array(p_value_arr)

        trend_arr[trend_arr > 0.05] = np.nan
        trend_arr[trend_arr < -0.05] = np.nan

        hist = []
        for i in trend_arr:
            for j in i:
                if np.isnan(j):
                    continue
                hist.append(j)
        plt.figure()
        plt.hist(hist, bins=100,)
        plt.figure()
        plt.imshow(trend_arr, cmap='jet', vmin=-0.05, vmax=0.05)
        plt.title('')
        plt.colorbar()
        plt.show()
        #
        #     # save arr to tif
        DIC_and_TIF().arr_to_tif(trend_arr, outf + '_trend.tif')
        DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
        np.save(outf + '_trend', trend_arr)
        np.save(outf + '_p_value', p_value_arr)


    def changes_NDVI_method1(self): # ??????1 GCB Gonsamo et al., 2021 and Pierre
        periods=['early','peak','late']
        time_range='2002-2015'
        len_year=14

        for period in periods:
            fdir = result_root + 'extraction_original_val/{}_original_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(
                time_range, time_range, period)
            outdir = result_root + '%NDVI_Pierre/'.format(time_range, period)
            Tools().mk_dir(outdir, force=True)
            dic_NDVI = {}

            for f in tqdm(os.listdir(fdir)):
                if 'CSIF'not in f:
                    continue

                if f.endswith('.npy'):
                    # if not '005' in f:
                    #     continue
                    dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                    dic_NDVI.update(dic_i)

            dic_spatial_count = {}
            delta_dic = {}
            for pix in tqdm(dic_NDVI):
                val_NDVI = dic_NDVI[pix]

                if len(val_NDVI) != len_year:
                    continue


                if np.isnan(np.nanmean(val_NDVI)):
                    print('error')
                    continue
                row = len(val_NDVI)

                NDVI_min = np.nanmin(val_NDVI)
                NDVI_max = np.nanmax(val_NDVI)
                NDVI_mean=np.nanmean(val_NDVI)


                delta_time_series = []

                for i in range(row):
                    # delta = ((val_NDVI[i] - NDVI_min) / (NDVI_max-NDVI_min))*100
                    delta = ((val_NDVI[i] - NDVI_mean) / (NDVI_mean)) * 100
                    delta_time_series.append(delta)
                delta_dic[pix]=delta_time_series
                # print(delta_time_series)
                # delta_dic[pix]=delta_time_series[0:17]
                # delta_dic[pix]=delta_time_series[17:]
                # print(delta_time_series[0:17])
                # exit()

                dic_spatial_count[pix] = len(delta_time_series)

                # ?????????pix???????????????
                # plt.plot(delta_time_series)
                # plt.show()

                # ?????????????????????????????????

            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
            # plt.imshow(arr, cmap='jet')
            # plt.colorbar()
            # plt.title('')
            # plt.show()


                # plt.plot(result_dic[pix])
                # plt.show()

            np.save(outdir + '{}_{}_CSIF.npy'.format(time_range,period), delta_dic)


    def changes_NDVI_keenan(self): # ???NDVI-????????????/???????????????100%
        periods=['early','peak','late']
        time_range = '1988-2015'
        len_year = 28
        variable = 'VOD'

        for period in periods:
            # fdir = result_root + 'extraction_original_val/{}_original_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(
            #     time_range, time_range, period)
            fdir = result_root + 'extraction_original_val/extraction_during_{}_growing_season_static/'.format(
                 period)
            outdir = result_root + 'Keenan_relative_change/{}_{}/'.format(time_range,variable)
            Tools().mk_dir(outdir, force=True)
            dic_NDVI = {}

            for f in tqdm(os.listdir(fdir)):
                if variable not in f:
                    continue
                if f.endswith('.npy'):
                    # if not '005' in f:
                    #     continue
                    dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                    dic_NDVI.update(dic_i)

            dic_spatial_count = {}
            delta_dic = {}
            for pix in tqdm(dic_NDVI):
                val_NDVI = dic_NDVI[pix]
                val_NDVI = np.array(val_NDVI)
                val_NDVI[val_NDVI < 0] = np.nan

                if len(val_NDVI) != len_year:
                    continue


                if np.isnan(np.nanmean(val_NDVI)):
                    print('error')
                    continue
                row = len(val_NDVI)



                delta_time_series = []

                for i in range(row):
                    delta = (((val_NDVI[i] - 1/3*(val_NDVI[0]+val_NDVI[1]+val_NDVI[2])))) *100
                    delta_time_series.append(delta)
                # print(delta_time_series)
                delta_dic[pix]=delta_time_series


                # dic_spatial_count[pix] = len(delta_time_series)

                # ?????????pix???????????????
                # plt.plot(delta_time_series)
                # plt.show()

                # ?????????????????????????????????

            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
            # plt.imshow(arr, cmap='jet')
            # plt.colorbar()
            # plt.title('')
            # plt.show()


                # plt.plot(result_dic[pix])
                # plt.show()

            np.save(outdir + '{}_{}_{}.npy'.format(time_range,variable, period), delta_dic)


    def Pierre_anomaly_variables(self):  # Pierre ???????????????anomaly

        periods = ['early', 'peak', 'late']



        # variables = ['MODIS_LAI','LAI4g','LAI3g',]
        # variables = [ 'CCI_SM', 'PAR', 'VPD','Temp']
        variables = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI',
                          'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                          'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',
                          'ISBA-CTRIP_S2_lai','Trendy_ensemble']
        # variables_list = ['CABLE-POP_S2_lai','LPX-Bern_S
        # variables = ['MODIS_LAI','LAI3g']

        for variable in variables:

            for period in periods:
                dic_NDVI={}  # so important!!

                fdir=results_root+f'extraction_original_val/extraction_original_val_Trendy2/extraction_during_{period}_growing_season_static/'
                outdir = results_root + 'Pierre_relative_change/monthly/1982-2020_Trendy/'
                Tools().mk_dir(outdir, force=True)

                file=fdir+f'during_{period}_{variable}.npy'

                dic_NDVI = dict(np.load(file, allow_pickle=True, ).item())


                delta_dic = {}
                for pix in tqdm(dic_NDVI):
                    val_NDVI = dic_NDVI[pix]
                    val_NDVI=np.array(val_NDVI)
                    # print(val_NDVI)
                    # if val_NDVI[0]==None:
                    #     continue
                    if np.nanmean(val_NDVI)==np.nan:
                        continue
                    # val_NDVI[val_NDVI<0]=np.nan

                    # if len(val_NDVI) != len_year:
                    #     continue

                    if np.isnan(np.nanmean(val_NDVI)):
                        print('error')
                        continue
                    row = len(val_NDVI)

                    NDVI_mean = np.nanmean(val_NDVI)


                    delta_time_series = []

                    for i in range(row):
                        delta = (((val_NDVI[i] - NDVI_mean))/NDVI_mean)*100
                        delta_time_series.append(delta)
                    delta_dic[pix] = delta_time_series
                    # print(delta_time_series)


                    # dic_spatial_count[pix] = len(delta_time_series)

                    # ?????????pix???????????????
                    # plt.plot(delta_time_series)
                    # plt.show()

                    # ?????????????????????????????????

                # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                # plt.imshow(arr, cmap='jet')
                # plt.colorbar()
                # plt.title('')
                # plt.show()

                # plt.plot(result_dic[pix])
                # plt.show()

                np.save(outdir + f'{variable}_{period}_relative_change.npy', delta_dic)
                # np.save(outdir + '{}_during_{}_CSIF.npy'.format(time_range, period), delta_dic)

    def anomaly(self):  #
        periods = ['early', 'peak', 'late']
        # variables = ['MODIS_LAI','LAI4g','LAI3g',]
        variables = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                             'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai', 'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai',
                             'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai', 'Trendy_ensemble']


        for variable in variables:
            for period in periods:
                dic_NDVI={}  # so important!!
                # fdir = result_root + f'extraction_original_val/extraction_during_{period}_growing_season_static/'
                fdir = results_root + f'extraction_original_val/2000-2018_Trendy/'
                outdir = results_root + 'anomaly/2000-2018_Trendy/'
                Tools().mk_dir(outdir, force=True)

                file=fdir+f'during_{period}_{variable}.npy'

                dic_NDVI = dict(np.load(file, allow_pickle=True, ).item())


                delta_dic = {}
                for pix in tqdm(dic_NDVI):
                    val_NDVI = dic_NDVI[pix]
                    val_NDVI=np.array(val_NDVI)
                    # print(val_NDVI)
                    # if val_NDVI[0]==None:
                    #     continue
                    if np.nanmean(val_NDVI)==np.nan:
                        continue
                    # val_NDVI[val_NDVI<0]=np.nan  #????????????????????????

                    # if len(val_NDVI) != len_year:
                    #     continue

                    if np.isnan(np.nanmean(val_NDVI)):
                        print('error')
                        continue
                    row = len(val_NDVI)

                    NDVI_mean = np.nanmean(val_NDVI)
                    NDVI_std=np.nanstd(val_NDVI)


                    delta_time_series = []

                    for i in range(row):
                        delta = (val_NDVI[i] - NDVI_mean)
                        delta_time_series.append(delta)
                    delta_dic[pix] = delta_time_series
                    # print(delta_time_series)


                    # dic_spatial_count[pix] = len(delta_time_series)

                    # ?????????pix???????????????
                    # plt.plot(delta_time_series)
                    # plt.show()

                    # ?????????????????????????????????

                # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                # plt.imshow(arr, cmap='jet')
                # plt.colorbar()
                # plt.title('')
                # plt.show()

                # plt.plot(result_dic[pix])
                # plt.show()

                np.save(outdir + f'{variable}_{period}_anomaly.npy', delta_dic)
                # np.save(outdir + '{}_during_{}_CSIF.npy'.format(time_range, period), delta_dic)

    def zscore(self):  #
        periods = ['early', 'peak', 'late']
        # variables = ['MODIS_LAI','LAI4g','LAI3g',]
        variables = ['CCI_SM', 'PAR', 'VPD','Temp']
        # variables = [ 'LAI3g','MODIS_LAI']
        # variables = ['MODIS_LAI']

        for variable in variables:
            for period in periods:
                dic_NDVI={}  # so important!!
                # fdir = result_root + f'extraction_original_val/extraction_during_{period}_growing_season_static/'
                fdir = result_root + f'extraction_original_val/2000-2018_daily/'
                outdir = result_root + 'zscore/2000-2018_daily/'
                Tools().mk_dir(outdir, force=True)

                file=fdir+f'during_{period}_{variable}.npy'

                dic_NDVI = dict(np.load(file, allow_pickle=True, ).item())


                delta_dic = {}
                for pix in tqdm(dic_NDVI):
                    val_NDVI = dic_NDVI[pix]
                    val_NDVI=np.array(val_NDVI)
                    # print(val_NDVI)
                    # if val_NDVI[0]==None:
                    #     continue
                    if np.nanmean(val_NDVI)==np.nan:
                        continue
                    # val_NDVI[val_NDVI<0]=np.nan  #????????????????????????

                    # if len(val_NDVI) != len_year:
                    #     continue

                    if np.isnan(np.nanmean(val_NDVI)):
                        print('error')
                        continue
                    row = len(val_NDVI)

                    NDVI_mean = np.nanmean(val_NDVI)
                    NDVI_std=np.nanstd(val_NDVI)


                    delta_time_series = []

                    for i in range(row):
                        delta = (val_NDVI[i] - NDVI_mean)/NDVI_std
                        delta_time_series.append(delta)
                    delta_dic[pix] = delta_time_series
                    # print(delta_time_series)


                    # dic_spatial_count[pix] = len(delta_time_series)

                    # ?????????pix???????????????
                    # plt.plot(delta_time_series)
                    # plt.show()

                    # ?????????????????????????????????

                # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                # plt.imshow(arr, cmap='jet')
                # plt.colorbar()
                # plt.title('')
                # plt.show()

                # plt.plot(result_dic[pix])
                # plt.show()

                np.save(outdir + f'{variable}_{period}_zscore.npy', delta_dic)
                # np.save(outdir + '{}_during_{}_CSIF.npy'.format(time_range, period), delta_dic)




    def anonmaly_winter(self):  # ?????? ??????????????????winter precipitation_anomaly
     f=result_root+'extraction_original_val/1982-2015_original_extraction_all_seasons/winter_temperature_extraction.npy'
     outdir=result_root+'anomaly_variables_independently/'
     time_range='1982-2015'


     dic_winter = dict(np.load(f, allow_pickle=True, ).item())

     delta_dic = {}
     for pix in tqdm(dic_winter):
        val_winter = dic_winter[pix]

        if len(val_winter) != 33:
            continue

        if np.isnan(np.nanmean(val_winter)):
            print('error')
            continue
        row = len(val_winter)

        winter_mean = np.nanmean(val_winter)


        delta_time_series = []

        for i in range(row):
            delta = ((val_winter[i] - winter_mean))
            delta_time_series.append(delta)
        delta_dic[pix] = delta_time_series
        # print(delta_time_series)
        # delta_dic[pix]=delta_time_series[0:17]
        # delta_dic[pix] = delta_time_series[17:]
        # print(delta_time_series[0:17])
        # exit()

        # dic_spatial_count[pix] = len(delta_time_series)

        # ?????????pix???????????????
        # plt.plot(delta_time_series)
        # plt.show()

        # ?????????????????????????????????

    # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
    # plt.imshow(arr, cmap='jet')
    # plt.colorbar()
    # plt.title('')
    # plt.show()

    # plt.plot(result_dic[pix])
    # plt.show()

     np.save(outdir + '1982-2015_winter_temperature.npy', delta_dic)
    def average_phenology_retrieval_Yao_method(self):
        fdir = '/Volumes/1T/wen_prj/dic_CSIF_par/'
        dic = {}
        SOS_index_dic={}
        EOS_index_dic={}

        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        for pix in tqdm(dic,):
            change_value=[]
            vals = dic[pix]
            if np.isnan(np.nanmean(vals)):
                continue
            if np.nanmean(vals) <= 0:
                continue
            time_series_reshape = np.reshape(vals, (18, -1))
            time_series_reshape = time_series_reshape.T
            # print(time_series_reshape.shape)
            average_val = []
            for i in time_series_reshape:
                ii = np.mean(i)
                average_val.append(ii)
            average_val = np.array(average_val)
            peak = np.argmax(average_val)
            if peak == 0 or peak == (len(average_val) - 1):
                 continue
            # plt.plot(average_val)
            # plt.show()
            # print(average_val)
            # exit()
            for i in range(len(average_val)):
                if i< len(vals):
                    change_value.append(vals[i+1]-vals[i])
            max_change_index=np.argmax(change_value)
            min_change_index = np.argmax(change_value)
            SOS_index_dic[pix]=max_change_index
            EOS_index_dic[pix]=min_change_index


    def average_phenology_retrieval(self,threshold_i=0.2):  #####Wen-4.26 ???????????????????????????

        fdir = '/Volumes/1T/wen_prj/average_phenology_CSIF_par/'
        dic = {}
        dic_1 = {}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        x = []
        y = []
        for pix in dic:
            vals = dic[pix]
            if np.isnan(vals[0]):
                continue
            dic_1[pix] = 1
            # x.append(pix[0])
            # y.append(pix[1])
            # # print(vals)
        # plt.scatter(y,x)
        # plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_1)
        plt.imshow(arr)
        plt.show()
        exit()

        spa_dic = {}

        for pix in tqdm(dic,):
            vals = dic[pix]
            peak = np.argmax(vals)
            if peak == 0 or peak == (len(vals) - 1):
                continue
            try:
                early_start = self.__search_left(vals, peak, threshold_i)
                late_end = self.__search_right(vals, peak, threshold_i)
            except:
                early_start = 60
                late_end = 130
                # print vals
                plt.plot(vals)
                plt.show()
            # for pix in tqdm(dic):  Wen-4.26
            #     seasonal_average = dic[pix]
            #     if np.isnan(seasonal_average[0]):
            #         continue
            #     max_seasonal_averageSIF = np.max(seasonal_average)
            #     max_index=np.argmax(seasonal_average)
            #     # plt.plot(seasonal_averageSIF)
            #     # plt.show()
            #     left=seasonal_average[:max_index]
            # if len(left) == 0:
            #     continue
            # sos_index = 999999
            # SOS_threshold = (max_seasonal_averageSIF - np.min(left)) * 0.3 + np.min(left)
            # for i in left:
            #     if i > SOS_threshold:
            #         sos_index = np.argwhere(left == i)
            #         sos_index = sos_index[0][0]
            #         # print(sos_index)
            #         break
            # SOS_dic[pix] = sos_index
            # right = seasonal_average[max_index:]
            # # print(right)
            # # print(right.shape)
            # # exit()
            # eos_index = 999999
            # EOS_threshold = (max_seasonal_averageSIF - np.min(right)) * 0.2 + np.min(right)
            # if len(right) == 0:
            #     continue
            # for ii in right:
            #     if ii < EOS_threshold:
            #         eos_index = np.argwhere(right == ii)
            #         eos_index = eos_index[0][0] + max_index
            #         # print(sos_index)
            #         break
            # EOS_dic[pix] = eos_index
            spa_dic[pix] = 1
        # arr_t = DIC_and_TIF().pix_dic_to_spatial_arr(EOS_dic)
        # # DIC_and_TIF().plot_back_ground_arr()
        # arr_t[arr_t > 9999] = np.nan
        # plt.imshow(arr_t, vmin=150, vmax=350, cmap='jet')
        # plt.show()

    def annual_phenology_retrieval_Yao_method(self):
        # //////////////???raw data
        # fdir = '/Volumes/1T/wen_prj/dic_CSIF_par/'
        # # outdir = '/Users /admin/Downloads/dic_CSIF_par/'
        # outdir = '/Volumes/1T/wen_prj/raw_annually_CSIF_par/'
        # Tools().mk_dir(outdir)
        # dic = {}
        # dic_1 = {}
        # for f in tqdm(os.listdir(fdir),desc='???dic'):
        #     # print(fi
        #     if f.endswith('.npy'):
        #         dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
        #         dic.update(dic_i)
        # SOS_index_dic = {}
        # EOS_index_dic = {}
        # for y in range(2002, 2017):
        #     outf = outdir + '{}'.format(y)
        #     result_dic = {}
        #     for pix in tqdm(dic, desc='{}'.format(y)):
        #         change_value = []
        #         time_series = dic[pix]
        #         if np.isnan(np.nanmean(time_series)):
        #             continue
        #         if np.nanmean(time_series) == 0:
        #             continue
        #         time_series_reshape = np.reshape(time_series, (18, -1))
        #         ynew = time_series_reshape[int(y) - 2000]
        #         ynew = np.array([ynew])
        #         if ynew[0][0] == None:
        #             continue
        #         peak = np.argmax(ynew[0])
        #         if peak == 0 or peak == (len(ynew[0]) - 1):
        #             continue
        #         if np.isnan(np.nanmean(ynew[0])):
        #             continue
        #         if np.nanmean(ynew[0]) <= 0:
        #             continue
        #         result_dic[pix]=ynew[0]
        #         # plt.plot(ynew[0])
        #         # plt.show()
        #     #     dic_1[pix] = 1
        #     # arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_1)
        #     # plt.imshow(arr)
        #     # plt.show()
        #     # exit()
        #     np.save(outf,result_dic)

        fdir = '/Volumes/1T/wen_prj/raw_annually_CSIF_par/'
        outdir = '/Volumes/1T/wen_prj/annual_phenology_retrieval_Yao_method/'
        outf1=outdir+'annual_phenology_retrieval_Yao_SOS'
        outf2 = outdir +'annual_phenology_retrieval_Yao_EOS'
        Tools().mk_dir(outdir)
        hants_dic={}
        multi_year_raw_dic={}
        for f in tqdm(sorted(os.listdir(fdir))):
            if f.endswith('.npy'):
                year = f.split('.')[0]
                one_year_raw_dic = dict(np.load(fdir + f, allow_pickle=True, ).item())
                multi_year_raw_dic[year] = one_year_raw_dic
        SOS_index_dic = DIC_and_TIF().void_spatial_dic()
        EOS_index_dic = DIC_and_TIF().void_spatial_dic()

        for pix in tqdm(SOS_index_dic):
            n_year = 0
            result_list = []
            for y in range(2002, 2017):
                one_year_raw_dic = multi_year_raw_dic[str(y)]
                # print(len(one_year_raw_dic))
                # continue
                if not pix in one_year_raw_dic:
                    continue
                vals = one_year_raw_dic[pix]
                # print(len(vals))
                # exit()
                peak = np.argmax(vals)
                if peak == 0 or peak == (len(vals) - 1):
                    continue
                if np.isnan(np.nanmean(vals)):
                    continue
                if np.nanmean(vals) <= 0:
                    continue
                # plt.plot(vals)
                # plt.show()
                change_value = []
                for i in range(len(vals)):
                    if i > 0:
                        change_value.append(vals[i] - vals[i - 1])
                # plt.plot(change_value)
                # plt.title('change v')
                # plt.figure()
                # plt.plot(vals)
                # plt.show()
                max_change_index = np.argmax(change_value)
                min_change_index = np.argmin(change_value)
                sos = max_change_index * 4
                eos = min_change_index * 4
                SOS_index_dic[pix].append(sos)
                EOS_index_dic[pix].append(eos)
        np.save(outf1, SOS_index_dic)
        np.save(outf2, EOS_index_dic)


def slices(args):
    pass


class statistic_anaysis:


    def __init__(self):
        self.this_class_arr='/Volumes/SSD_sumsang/project_greening/Result/'
        pass


    def run(self):
        # self.extraction_pre_window()
        # self.daily_anomaly()
        # self.extraction_variables_static_pre_month()
        # self.detrend_extraction_during_pre_variables()
        # self.univariate_correlation()
        # self.univariate_correlation_partial()
        # self.partial_correlation_test()
        # self.plot_moving_window()
        # self.trend_calculation()

        # self.detrend_during_variables()
        # self.correlation_calculation()
        # self.mean_winter_calculation()
        # self.trend_calculation()
        # self.CV_calculation()



        pass

    def extraction_pre_window(self):
        window=15
        month=3
        period='early'
        fdir = result_root + 'detrend_extraction/pre_{}_growing_season_static/pre_{}_{}month/'.format(period,period,month)
        outdir = result_root + 'extraction_anomaly_val_window_detrend/pre_{}_{}month/{}_year_window/'.format(period,month,window)
        # fdir = result_root + 'detrend_extraction/during_{}_growing_season_static/'.format(period)
        # outdir = result_root + 'extraction_anomaly_val_window_detrend/during_{}/{}_year_window/'.format(period,
        #                                                                                              window)
        Tools().mk_dir(outdir,force=True)
        dic = {}
        for f in tqdm(sorted(os.listdir(fdir))):
            if f!='pre_{}month_{}_CCI_SM.npy'.format(month,period):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
            print(f)
            filename=f.split('.')[0]+'/'
            print(filename)
            Tools().mk_dir(outdir+filename, force=True)
            result_dic = {}

            new_x_extraction_by_window={}
            for pix in tqdm(dic):

                time_series = dic[pix]
                time_series = np.array(time_series)

                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    print('error')
                    continue
                new_x_extraction_by_window[pix]=self.forward_window_extract(time_series,window)

            flag = 0
            temp_dic = {}
            for key in tqdm(new_x_extraction_by_window, desc='output...'):  # ?????????
                flag = flag + 1
                time_series = new_x_extraction_by_window[key]
                time_series = np.array(time_series)
                temp_dic[key] = time_series
                if flag % 10000 == 0:
                    # print(flag)
                    np.save(outdir +filename + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                    temp_dic = {}
            np.save(outdir +filename + 'per_pix_dic_%03d' % 0, temp_dic)

    def extraction_during_window(self):

        window_list=[15]
        period_list=['early','peak','late']
        variables=['VPD','CCI_SM','CO2','Temp','PAR']
        # variables = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai',
        #              'ISAM_S2_LAI',
        #              'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
        #              'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',
        #              'ISBA-CTRIP_S2_lai', 'Trendy_ensemble']

        # variables = ['LAI3g_monthly']


        for i in window_list:
            fdir = results_root + f'Pierre_relative_change/2000-2018_daily/2000-2018_X/'


            outdir = results_root + f'extract_relative_change_window/{i}_year_window_2000/Daily_X/'
            Tools().mk_dir(outdir, force=True)

            Tools().mk_dir(outdir, force=True)
            for period in period_list:

                for variable in variables:

                    for f in tqdm(sorted(os.listdir(fdir))):
                        dic = {}

                        if f != f'{variable}_{period}_relative_change.npy':
                            continue

                        if f.endswith('.npy'):
                            dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                            dic.update(dic_i)
                        print(f)
                        # filename=f.split('.')[0]+'/'
                        filename = f.split('.')[0]
                        print(filename)


                        result_dic = {}

                        new_x_extraction_by_window={}
                        for pix in tqdm(dic):

                            time_series = dic[pix]
                            time_series = np.array(time_series)

                            time_series[time_series < -999] = np.nan
                            if np.isnan(np.nanmean(time_series)):
                                print('error')
                                continue

                            # new_x_extraction_by_window[pix]=self.forward_window_extract_anomaly(time_series,i) # extraction ??????????????????anomaly
                            new_x_extraction_by_window[pix] = self.forward_window_extract(time_series, i)
                            # new_x_extraction_by_window[pix] = self.forward_window_extract_mean(time_series, i)


                        np.save(outdir +filename, new_x_extraction_by_window)


                    # np.save(outdir +filename, new_x_extraction_by_window)

    def plot_moving_window(self):

        dic=dict(np.load(result_root+'partial_window/plot_moving_window_partial_correlation/moving_partial_correlation_peak_Non Humid.npy',allow_pickle=True,).item())
        print(dic)

        early_key_list=[]
        peak_key_list = []
        late_key_list = []
        season_list=[]
        for key in dic:
            if 'early' in key:
                early_key_list.append(key)
            elif 'peak' in key:
                peak_key_list.append(key)
            elif 'late' in key:
                late_key_list.append(key)
            else:
                continue
        early_key_list.sort()
        peak_key_list.sort()
        peak_key_list.sort()

        # print(early_key_list)
        # print(peak_key_list)
        # print(late_key_list)
        season_list=[early_key_list,peak_key_list,late_key_list]
        plt.figure()
        for season in season_list:

            flag = 1
            for key in season:
                plt.subplot(1, 4, flag)
                val=dic[key]
                print(val)
                mean_array_list=val[0]
                CI_array_list=val[1]
                plt.plot(range(len(mean_array_list)),mean_array_list,zorder=99)
                plt.fill_between(range(len(mean_array_list)),y1=mean_array_list-CI_array_list,y2=mean_array_list+CI_array_list,alpha=0.3)
                plt.title(key)
                flag=flag+1
                plt.title('non-humid')
        plt.show()



    def average_peak_calculation(self):
        file1 = '/Users/admin/Downloads/transform_early_peak_late_dormant_period_annual/early_end.npy'
        file2 = '/Users/admin/Downloads/transform_early_peak_late_dormant_period_annual/late_start.npy'
        fdir = '/Users/admin/Downloads/hants_smooth_annual/'
        early_end_dic = {}
        late_start_dic = {}
        early_end_dic = dict(np.load(file1, allow_pickle=True, ).item())
        late_start_dic = dict(np.load(file2, allow_pickle=True, ).item())
        hants_smoothed_dic_allyear = {}
        for f in tqdm(sorted(os.listdir(fdir))):
            if f.endswith('.npy'):
                year = f.split('.')[0]
                hants_smoothed_dic = dict(np.load(fdir + f, allow_pickle=True, ).item())
                hants_smoothed_dic_allyear[year] = hants_smoothed_dic

        #### method 1 ###
        result_dic=DIC_and_TIF().void_spatial_dic()
        #### method 2 ###
        # result_dic = {}
        for pix in tqdm(early_end_dic):
            flag = 0
            result_list = []
            for y in range(2002,2017):
                hants_smoothed_dic = hants_smoothed_dic_allyear[str(y)]
                mean_dic = {}
                early_end = early_end_dic[pix]
                late_start = late_start_dic[pix]
                if len(early_end) !=15:
                    continue
                if len(late_start) != 15:
                    continue
                early_end = early_end_dic[pix][flag]
                late_start = late_start_dic[pix][flag]
                hants = hants_smoothed_dic[pix]
                picked_val = hants[early_end:late_start]
                if len(picked_val) ==0:
                    continue
                mean = np.mean(picked_val)
                #### method 1 ###
                result_dic[pix].append(mean)
                flag=flag+1

                #### method2 ###
                # result_list.append(mean)
            # result_dic[pix] = result_list
        outdir='/Users/admin/Downloads/transform_early_peak_late_dormant_period_annual/'
        np.save(outdir +'average_peak_calculation', result_dic)

    def daily_anomaly(self):
        # fdir = result_root + 'dic_per_pixel_all_year_variables/PAR/'
        # outdir = result_root + 'anomaly/PAR/'
        # fdir=data_root + 'dic_SM/'
        # outdir=data_root+ 'anomaly/SM/'
        fdir = result_root + 'Hants_annually_smooth/CSIF_annually_transform/'
        outdir = result_root + 'anomaly/CSIF/'
        Tools().mk_dir(outdir)
        dic = {}
        dic_1 = {}
        for f in tqdm(sorted(os.listdir(fdir))):
            # print(fi
            if f.endswith('.npy'):
                # if not '005' in f:
                #     continue
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)

        result_dic = {}
        for pix in tqdm(dic):
            time_series = dic[pix]

            # print(type(time_series))
            # print(np.shape(time_series))
            # for i in time_series:
            #     print(i)
            #     print(type(i))
            #     exit()
            # time_series_matrix = []

            time_series[time_series<-999]=np.nan
            if np.isnan(np.nanmean(time_series)):
                print('error')
                # plt.plot(time_series)
                # plt.show()
                continue
            time_series_T = time_series.T
            # print(time_series_reshape[int(y) - 2003])
            # exit()
            row=len(time_series_T)
            col=len(time_series_T[0])
            anomaly = np.ones_like(time_series_T) * np.nan
            for i in range(row):
                time_series_mean = np.nanmean(time_series_T[i])
                time_series_std = np.nanstd(time_series_T[i])
                for j in range(col):
                    anomaly[i][j]=(time_series_T[i][j]-time_series_mean)/time_series_std
            anomaly=anomaly.T
            anomaly_15_year = anomaly.flatten()
            # plt.plot(anomaly_15_year)
            # plt.show()
            result_dic[pix]=anomaly_15_year
            # plt.plot(result_dic[pix])
            # plt.show()

        flag = 0
        temp_dic = {}
        for key in tqdm(result_dic,desc='output...'): #?????????
            flag=flag+1
            time_series=result_dic[key]
            time_series=np.array(time_series)
            temp_dic[key]=time_series
            if flag %10000 == 0:
                # print(flag)
                np.save(outdir +'per_pix_dic_%03d' % (flag/10000),temp_dic)
                temp_dic={}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def monthly_anomaly(self): # ??????monthly anomaly
        fdir = data_root + 'VOD/VOD_dic/'
        outdir = data_root + 'VOD/VOD_zscore/'
        Tools().mk_dir(outdir)
        dic={}
        for f in tqdm(sorted(os.listdir(fdir))):
            if f.startswith('.'):
                continue
            if f.endswith('.npy'):
                # if not '005' in f:
                #     continue
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)

        result_dic = {}
        for pix in tqdm(dic):
            # r,c=pix
            # china_r = list(range(75, 150))
            # china_c = list(range(550, 620))
            # if not r in china_r:
            #     continue
            # if not c in china_c:
            #     continue
            time_series = dic[pix]
            time_series=np.array(time_series)
            if len(time_series) != 348:
                continue
                # plt.plot(time_series)
                # plt.show()

            time_series[time_series < -999] = np.nan
            if np.isnan(np.nanmean(time_series)):
                print('error')
                continue
            time_series = time_series.reshape(-1, 12)
            time_series_T = time_series.T
            row = len(time_series_T)
            col = len(time_series_T[0])
            anomaly = np.ones_like(time_series_T) * np.nan
            for i in range(row):
                time_series_mean = np.nanmean(time_series_T[i])
                time_series_std = np.nanstd(time_series_T[i])
                for j in range(col):
                    anomaly[i][j] = (time_series_T[i][j] - time_series_mean) / time_series_std
                    # anomaly[i][j] = (time_series_T[i][j] - time_series_mean)
            anomaly = anomaly.T
            anomaly_15_year = anomaly.flatten()
            # plt.plot(anomaly_15_year)
            # plt.show()
            result_dic[pix] = anomaly_15_year
            # plt.plot(result_dic[pix])
            # plt.show()

        flag = 0
        temp_dic = {}
        for key in tqdm(result_dic, desc='output...'):  # ?????????
            flag = flag + 1
            time_series = result_dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def monthly_anomaly_ly(self): # ??????monthly anomaly
        fdir = data_root + 'NIRv_tif_05/NIRv_dic_Yang/'
        outdir = data_root + 'NIRv_tif_05/NIRv_dic_Yang_zscore_test/'
        Tools().mk_dir(outdir)
        Pre_Process().cal_anomaly(fdir,outdir)



    def detrend(self): #
        # fdir_all = result_root + '%CSIF_par/'
        # outdir = result_root + 'detrend/%CSIF_par/'
        # Tools().mk_dir(outdir)
        # for fdir in tqdm(sorted(os.listdir(fdir_all))):
        #     # if not fdir.startswith('20%_Pre_SPEI_extraction'):
        #     #     continue
        #     print(fdir)
        #     outdir_val1=outdir+fdir.split('_')[1]+'_'+fdir.split('_')[2]+'/'
        #     print(outdir_val1)
        #     Tools().mk_dir(outdir_val1)
        #
        #     for f in tqdm(sorted(os.listdir(fdir_all+fdir+'/'))):
        #         dic={}
        #         print(f)
        #         dic = dict(np.load(fdir_all+fdir+'/'+f, allow_pickle=True, ).item())
        #         outdir_val2 = outdir_val1 +f.split('.')[0]+'/'
        #         print(outdir_val2)
        #         Tools().mk_dir(outdir_val2)
        #         result_dic = {}
        #         i=0
        #         for pix in tqdm(dic):
        #             # r,c=pix
        #             # china_r = list(range(75, 150))
        #             # china_c = list(range(550, 620))
        #             # if not r in china_r:
        #             #     continue
        #             # if not c in china_c:
        #             #     continue
        #             time_series = dic[pix]
        #             time_series=np.array(time_series)
        #             time_series[time_series < -999] = np.nan
        #             if np.isnan(np.nanmean(time_series)):
        #                 # print('error')
        #                 continue
        #             # plt.plot(time_series)
        #             try:
        #                 detrend_series=scipy.signal.detrend(time_series)
        #                 # plt.plot(detrend_series)
        #                 # plt.show()
        #                 result_dic[pix] = detrend_series
        #                 # # plt.plot(result_dic[pix])
        #                 # # plt.show()
        #             except Exception as e:
        #                 print(time_series)
        #                 i = i + 1
        #                 print(i)
        #                 # exit()
        #
        #
        #         flag = 0
        #         temp_dic = {}
        #         for key in tqdm(result_dic, desc='output...'):  # ?????????
        #             flag = flag + 1
        #             time_series = result_dic[key]
        #             time_series = np.array(time_series)
        #             temp_dic[key] = time_series
        #             if flag % 10000 == 0:
        #                 # print(flag)
        #                 np.save(outdir_val2 + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
        #                 temp_dic = {}
        #         np.save(outdir_val2 + 'per_pix_dic_%03d' % 0, temp_dic)
        time_range='2000-2018'
        periods=['early','peak','late']
        f = '/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'
        arr=ToRaster().raster2array(f)[0]
        dic_mask_NDVI=DIC_and_TIF().spatial_arr_to_dic(arr)
        variables=['X','Y']
        for period in periods:
            for variable in variables:

                fdir_all = result_root + f'Pierre_relative_change/monthly/{time_range}_{variable}/'
                outdir = result_root + f'detrend_Pierre_relative_change_monthly/detrend_{time_range}/detrend_{time_range}_during_{period}/'
                Tools().mk_dir(outdir,force=True)
                dic = {}

                for f in tqdm(sorted(os.listdir(fdir_all ))):
                    if f.startswith('._'):
                        continue
                    dic_i = dict(np.load(fdir_all + f, allow_pickle=True, ).item())
                    dic.update(dic_i)
                    result_dic = {}
                    i=0
                    for pix in tqdm(dic):
                        is_mask=dic_mask_NDVI[pix]
                        if is_mask !=1:
                            continue

                        # r,c=pix
                        # china_r = list(range(75, 150))
                        # china_c = list(range(550, 620))
                        # if not r in china_r:
                        #     continue
                        # if not c in china_c:
                        #     continue
                        time_series = dic[pix]
                        time_series=np.array(time_series)
                        time_series[time_series < -999] = np.nan
                        if np.isnan(np.nanmean(time_series)):
                            # print('error')
                            continue
                        # plt.plot(time_series)
                        try:
                            # detrend_series=scipy.signal.detrend(time_series)
                            detrend_series =T.detrend_vals(time_series)  # ????????????detrend, ?????????????????????nan ??? ??????????????????
                            # plt.plot(detrend_series)
                            # plt.show()
                            result_dic[pix] = detrend_series
                            # # plt.plot(result_dic[pix])
                            # # plt.show()
                        except Exception as e:
                            print(time_series)
                            i = i + 1
                            print(i)
                            # exit()

                    f_name=f.split('.')[0]
                    outf=outdir + f'detrend_{f_name}'
                    # print(outf)

                    np.save(outf, result_dic)


    def detrend_extraction_during_pre_variables(self): ############--------------------detrend extraction_during variables------------------------
        period='late'
        time_range='2002-2015'
        # month=1

        fdir_all = result_root + 'anomaly_variables_independently/{}_during_{}/'.format(time_range,period,)
        outdir = result_root + 'detrend_extraction_anomaly/{}_during_{}_detrend/'.format(time_range,period)

        Tools().mk_dir(outdir, force=True)
        for f in tqdm(sorted(os.listdir(fdir_all))):
            if f == '{}_during_{}_root_soil_moisture.npy'.format(time_range, period):
                continue
            if f == '{}_during_{}_surf_soil_moisture.npy'.format(time_range, period):
                continue
            if f == '{}_during_{}_NIRv.npy'.format(time_range, period):
                continue
            if f == '{}_during_{}_VOD.npy'.format(time_range, period):
                continue
            dic={}
            print(f)
            # exit()

            dic = dict(np.load(fdir_all+f, allow_pickle=True, ).item())
            print(f)
            outf = outdir +f.split('.')[0]+'.npy'
            print(outf)
            result_dic = {}
            i=0
            for pix in tqdm(dic):
                # r,c=pix
                # china_r = list(range(75, 150))
                # china_c = list(range(550, 620))
                # if not r in china_r:
                #     continue
                # if not c in china_c:
                #     continue
                time_series = dic[pix]
                time_series=np.array(time_series)
                time_series[time_series < -999] = np.nan
                if np.isnan(np.nanmean(time_series)):
                    # print('error')
                    continue
                # plt.plot(time_series)
                try:
                    detrend_series=scipy.signal.detrend(time_series)
                    # plt.plot(detrend_series)
                    # plt.show()
                    result_dic[pix] = detrend_series

                except Exception as e:
                    print(time_series)
                    i = i + 1
                    print(i)
                    # exit()


            np.save(outf, result_dic)  # ??????


    def extraction_variables_dynamic(self): #????????????pre
        # fdir = result_root + 'dic_per_pixel_all_year_variables/PAR/'
        # outdir = result_root + 'anomaly/PAR/'
        f1 = result_root+'early_peak_late_dormant_period_annually/20%_transform_early_peak_late_dormant_period_annually_CSIF_par/late_end.npy'
        fdir2 =result_root + 'anomaly/PAR/'
        # outdir = result_root + 'extraction_pre_EOS/'
        outdir = result_root + 'extraction_pre_EOS/20%_Pre_GPCP_extraction/'
        Tools().mk_dir(outdir)
        SOS_index_series = {}
        dic_variables = {}
        dic_pre_variables = DIC_and_TIF().void_spatial_dic()
        dic_pre={}
        dic_1 = {}

        dic_SOS = dict(np.load(f1, allow_pickle=True, ).item())


        # ??????????????????
        for f in tqdm(sorted(os.listdir(fdir2))):
            # if not '005' in f:
            #     continue
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir2 + f, allow_pickle=True, ).item())
                dic_variables.update(dic_i)

        for pix in tqdm(dic_variables):
            # r,c=pix
            # if c>180:
            #     continue
            SOS_index_series = dic_SOS[pix]
            time_series = dic_variables[pix]
            if len(SOS_index_series)!=15:
                continue
            if len(time_series)!=5475:
                continue
            for year in range(15):
                # SOS=SOS_index_series[year+1]
                SOS = SOS_index_series[year]
                # print(len(SOS_index_series))
                # time_series=time_series.reshape(14,-1)
                time_series = time_series.reshape(15, -1)
                pre_time_series=time_series[year][SOS -90:SOS]
                # print(pre_time_series)
                if np.isnan(np.nanmean(pre_time_series)):
                    continue
                variable_mean=np.nanmean(pre_time_series)
                dic_pre_variables[pix].append(variable_mean)
                # plt.imshow(time_series)
                # plt.show()
        np.save(outdir + 'pre90_GPCP', dic_pre_variables)


    ##//////////////////////////////////////

    def correlation_for15years(self):
        # fdir_all = result_root + 'detrend/extraction_during_early_growing_season_static/'
        # # f2=result_root+'extraction/extraction_anomaly/CSIF_par_during_extraction/during_early_CSIF_par.npy'
        # fdir_CSIF_par=result_root+'detrend/extraction_during_early_growing_season_static/during_early_CSIF_par/'
        # outdir = result_root + 'detrend_correlation/Pre_early/'
        # Tools().mk_dir(outdir)
        # # exit()
        # dic_CSIF_par={}
        # dic_climate = {}
        # spatial_dic = {}
        # spatial_dic_p_value = {}
        # spatial_dic_count = {}
        # for f_CSIF_par in tqdm(sorted(os.listdir(fdir_CSIF_par))):
        #     dic_i = dict(np.load(fdir_CSIF_par + f_CSIF_par, allow_pickle=True, ).item())
        #     dic_CSIF_par.update(dic_i)
        # # dic_CSIF_par=dict(np.load(f2, allow_pickle=True, ).item())
        # for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
        #     outdir_1 = outdir + fdir_1.split('_')[1] + '_' + fdir_1.split('_')[2] + '/'
        #     print(outdir_1)
        #     Tools().mk_dir(outdir_1)
        #
        #     print(fdir_1)
        #     if fdir_1.startswith('20%_Pre_CO2_extraction'):
        #         continue
        #     if fdir_1.startswith('20%_Pre_LST_extraction'):
        #         continue
        #     for f_1 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
        #         dic = {}
        #         print(f_1)
        #         dic_climate = dict(np.load(fdir_all + fdir_1 + '/' + f_1 , allow_pickle=True, ).item())
        #
        #         f_name1 = fdir_all.split('/')[-2].split('_')[-2]
        #         # print(f_name1)
        #         f_name = 'CSIF_par_and_' + f_name1 + '_' + f_1
        #         print(f_name)
        #         # outf=outdir+
        #         for pix in tqdm(dic_CSIF_par):
        #             if pix not in dic_climate:
        #                 continue
        #             val_climate = dic_climate[pix]
        #             val_CSIF_par=dic_CSIF_par[pix]
        #             if len(val_climate) != 15:
        #                 continue
        #             if len(val_CSIF_par) != 15:
        #                 continue
        #             val_climate = np.array(val_climate)
        #             val_CSIF_par = np.array(val_CSIF_par)
        #             val_climate[val_climate < -99] = np.nan
        #             val_CSIF_par[val_CSIF_par < -99] = np.nan
        #             if np.isnan(np.nanmean(val_climate)):
        #                 continue
        #             if np.isnan(np.nanmean(val_CSIF_par)):
        #                 continue
        #             try:
        #                 # a, b, r = KDE_plot().linefit(xaxis, val)
        #                 r, p = stats.pearsonr(val_climate, val_CSIF_par)
        #                 # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
        #                 # print(k)
        #                 # spatial_dic_count[pix] = len(val_climate)
        #
        #             except Exception as e:
        #                 print(val_climate,val_CSIF_par)
        #                 r = np.nan
        #                 p = np.nan
        #             spatial_dic[pix] =r   #
        #             spatial_dic_p_value[pix] = p
        #     # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
        #     # plt.imshow(count_arr)
        #     # plt.colorbar()
        #     # plt.show()
        #         correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        #         correlation_arr = np.array(correlation_arr)
        #         p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
        #         p_value_arr = np.array(p_value_arr)
        #         # trend_arr[trend_arr < -10] = np.nan
        #         # trend_arr[trend_arr > 10] = np.nan
        #
        #         hist = []
        #         for i in correlation_arr:
        #             for j in i:
        #                 if np.isnan(j):
        #                     continue
        #                 hist.append(j)
        #
        #         plt.hist(hist, bins=80)
        #         plt.figure()
        #         plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
        #         plt.title(f_name.split('.')[0])
        #         plt.colorbar()
        #         plt.show()
        #         outf=outdir_1+f_name
        #         #     # save arr to tif
        #         DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
        #         DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
        #         np.save(outf+'_correlation', correlation_arr)
        #         np.save(outf + '_p_value', p_value_arr)
    ###################?????????????????????per_pix???
        # period='late'

        # # fdir_all = result_root + 'detrend/extraction_pre_{}_static/'.format(period)
        # # fdir_CSIF_par=result_root+'detrend/extraction_during_{}_growing_season_static/during_{}_CSIF_par/'.format(period,period)
        # # outdir = result_root + 'detrend_correlation/Pre_{}/'.format(period)
        # Tools().mk_dir(outdir)
        # # exit()
        # dic_CSIF_par={}
        # dic_climate = {}
        # spatial_dic = {}
        # spatial_dic_p_value = {}
        # spatial_dic_count = {}
        # for f_CSIF_par in tqdm(sorted(os.listdir(fdir_CSIF_par))):
        #     dic_i = dict(np.load(fdir_CSIF_par + f_CSIF_par, allow_pickle=True, ).item())
        #     dic_CSIF_par.update(dic_i)
        #
        # for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
        #     outdir_1 = outdir + fdir_1.split('_')[0] + '_' + fdir_1.split('_')[1] + '/'
        #     print(outdir_1)
        #     Tools().mk_dir(outdir_1)
        #
        #     # print(fdir_1)
        #     if fdir_1.startswith('Pre_CO2'):
        #         continue
        #     if fdir_1.startswith('Pre_LST'):
        #         continue
        #     for fdir_2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
        #         print(fdir_2)
        #         for f_1 in  tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'+fdir_2+'/'))):
        #             dic_ii = dict(np.load(fdir_all + fdir_1 + '/'+fdir_2+'/'+f_1, allow_pickle=True, ).item())
        #             print(fdir_all + fdir_1 + '/'+fdir_2+'/')
        #             dic_climate.update(dic_ii)
        #
        #         f_name1 = fdir_all.split('/')[-2].split('_')[-2]
        #         print(f_name1)
        #         f_name = 'CSIF_par_and_' + f_name1 + '_' + fdir_2
        #         print(f_name)
        #         # outf=outdir+
        #         for pix in tqdm(dic_CSIF_par):
        #             if pix not in dic_climate:
        #                 continue
        #             val_climate = dic_climate[pix]
        #             val_CSIF_par=dic_CSIF_par[pix]
        #             if len(val_climate) != 15:
        #                 continue
        #             if len(val_CSIF_par) != 15:
        #                 continue
        #             val_climate = np.array(val_climate)
        #             val_CSIF_par = np.array(val_CSIF_par)
        #             val_climate[val_climate < -99] = np.nan
        #             val_CSIF_par[val_CSIF_par < -99] = np.nan
        #             if np.isnan(np.nanmean(val_climate)):
        #                 continue
        #             if np.isnan(np.nanmean(val_CSIF_par)):
        #                 continue
        #             try:
        #                 # a, b, r = KDE_plot().linefit(xaxis, val)
        #                 r, p = stats.pearsonr(val_climate, val_CSIF_par)
        #                 # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
        #                 # print(k)
        #                 # spatial_dic_count[pix] = len(val_climate)
        #
        #             except Exception as e:
        #                 print(val_climate,val_CSIF_par)
        #                 r = np.nan
        #                 p = np.nan
        #             spatial_dic[pix] =r   #
        #             spatial_dic_p_value[pix] = p
        #     # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
        #     # plt.imshow(count_arr)
        #     # plt.colorbar()
        #     # plt.show()
        #         correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        #         correlation_arr = np.array(correlation_arr)
        #         p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
        #         p_value_arr = np.array(p_value_arr)
        #         # trend_arr[trend_arr < -10] = np.nan
        #         # trend_arr[trend_arr > 10] = np.nan
        #
        #         hist = []
        #         for i in correlation_arr:
        #             for j in i:
        #                 if np.isnan(j):
        #                     continue
        #                 hist.append(j)
        #
        #         plt.hist(hist, bins=80)
        #         plt.figure()
        #         plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
        #         plt.title(f_name.split('.')[0])
        #         plt.colorbar()
        #         plt.show()
        #         outf=outdir_1+f_name
        #         #     # save arr to tif
        #         DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
        #         DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
        #         np.save(outf+'_correlation', correlation_arr)
        #         np.save(outf + '_p_value', p_value_arr)

    ####during for per pix/////////////////

        period = 'late'
        fdir_all = result_root + 'detrend/extraction_during_{}_growing_season_static/'.format(period)
        fdir_CSIF_par = result_root + 'detrend/extraction_during_{}_growing_season_static/during_{}_CSIF_par/'.format(
            period, period)
        outdir = result_root + 'detrend_correlation/During_{}/'.format(period)
        # fdir_all = result_root + 'detrend/extraction_pre_{}_static/'.format(period)
        # fdir_CSIF_par=result_root+'detrend/extraction_during_{}_growing_season_static/during_{}_CSIF_par/'.format(period,period)
        # outdir = result_root + 'detrend_correlation/Pre_{}/'.format(period)
        Tools().mk_dir(outdir)
        # exit()
        dic_CSIF_par = {}
        dic_climate = {}
        spatial_dic = {}
        spatial_dic_p_value = {}
        spatial_dic_count = {}
        for f_CSIF_par in tqdm(sorted(os.listdir(fdir_CSIF_par))):
            dic_i = dict(np.load(fdir_CSIF_par + f_CSIF_par, allow_pickle=True, ).item())
            dic_CSIF_par.update(dic_i)

        for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):

            if fdir_1.startswith('during_{}_CO2'.format(period)):
                continue
            if fdir_1.startswith('during_{}_LST'.format(period)):
                continue
            if fdir_1.startswith('during_{}_CSIF_par'.format(period)):
                continue
            for f2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
                print(f2)
                dic_ii = dict(np.load(fdir_all + fdir_1 + '/' + f2, allow_pickle=True, ).item())
                print(fdir_all + fdir_1 + '/' + f2)
                dic_climate.update(dic_ii)

            f_name = 'CSIF_par_and_' +fdir_1.split('_')[2]
            print(f_name)
            # outf=outdir+
            for pix in tqdm(dic_CSIF_par):
                if pix not in dic_climate:
                    continue
                val_climate = dic_climate[pix]
                val_CSIF_par = dic_CSIF_par[pix]
                if len(val_climate) != 15:
                    continue
                if len(val_CSIF_par) != 15:
                    continue
                val_climate = np.array(val_climate)
                val_CSIF_par = np.array(val_CSIF_par)
                val_climate[val_climate < -99] = np.nan
                val_CSIF_par[val_CSIF_par < -99] = np.nan
                if np.isnan(np.nanmean(val_climate)):
                    continue
                if np.isnan(np.nanmean(val_CSIF_par)):
                    continue
                try:
                    # a, b, r = KDE_plot().linefit(xaxis, val)
                    r, p = stats.pearsonr(val_climate, val_CSIF_par)
                    # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
                    # print(k)
                    # spatial_dic_count[pix] = len(val_climate)

                except Exception as e:
                    print(val_climate, val_CSIF_par)
                    r = np.nan
                    p = np.nan
                spatial_dic[pix] = r  #
                spatial_dic_p_value[pix] = p
            # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
            # plt.imshow(count_arr)
            # plt.colorbar()
            # plt.show()
            correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            correlation_arr = np.array(correlation_arr)
            p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
            p_value_arr = np.array(p_value_arr)
            # trend_arr[trend_arr < -10] = np.nan
            # trend_arr[trend_arr > 10] = np.nan

            hist = []
            for i in correlation_arr:
                for j in i:
                    if np.isnan(j):
                        continue
                    hist.append(j)

            plt.hist(hist, bins=80)
            plt.figure()
            plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
            plt.title(f_name.split('.')[0])
            plt.colorbar()
            plt.show()
            outf = outdir+ f_name
            #     # save arr to tif
            DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
            DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
            np.save(outf + '_correlation', correlation_arr)
            np.save(outf + '_p_value', p_value_arr)

    #//////////////////////////////////////during////////////////////
        # fdir_all = result_root + '/Volumes/SSD_sumsang/project_greening/Result/detrend/extraction_pre_early_static/'
        # f2=result_root+'extraction/extraction_anomaly/CSIF_par_during_extraction/during_late_CSIF_par.npy'
        # outdir = result_root + 'correlation/During_late/'
        # Tools().mk_dir(outdir)
        # dic_CSIF_par=dict(np.load(f2, allow_pickle=True, ).item())
        # spatial_dic = {}
        # spatial_dic_p_value = {}
        #
        # for f in tqdm(sorted(os.listdir(fdir_all))):
        #     if f == 'during_late_LST.npy':
        #         continue
        #     if f=='during_late_CO2.npy':
        #         continue
        #     # if f=='during_peak_CSIF_par.npy':
        #     #     continue
        #     outf= outdir+f
        #     dic_climate = dict(np.load(fdir_all+f, allow_pickle=True, ).item())
        #
        #     for pix in tqdm(dic_climate):
        #         val_climate = dic_climate[pix]
        #         val_CSIF_par=dic_CSIF_par[pix]
        #         if len(val_climate) != 15:
        #             continue
        #         if len(val_CSIF_par) != 15:
        #             continue
        #         val_climate = np.array(val_climate)
        #         val_CSIF_par = np.array(val_CSIF_par)
        #         val_climate[val_climate < -99] = np.nan
        #         val_CSIF_par[val_CSIF_par < -99] = np.nan
        #         if np.isnan(np.nanmean(val_climate)):
        #             continue
        #         if np.isnan(np.nanmean(val_CSIF_par)):
        #             continue
        #         try:
        #             # a, b, r = KDE_plot().linefit(xaxis, val)
        #             r, p = stats.pearsonr(val_climate, val_CSIF_par)
        #             # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
        #             # print(k)
        #             # spatial_dic_count[pix] = len(val_climate)
        #
        #         except Exception as e:
        #             print('error')
        #
        #             r = np.nan
        #             p = np.nan
        #         spatial_dic[pix] =r   #
        #         spatial_dic_p_value[pix] = p
        # # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
        # # plt.imshow(count_arr)
        # # plt.colorbar()
        # # plt.show()
        #     correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        #     correlation_arr = np.array(correlation_arr)
        #     p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
        #     p_value_arr = np.array(p_value_arr)
        #     # trend_arr[trend_arr < -10] = np.nan
        #     # trend_arr[trend_arr > 10] = np.nan
        #
        #     hist = []
        #     for i in correlation_arr:
        #         for j in i:
        #             if np.isnan(j):
        #                 continue
        #             hist.append(j)
        #
        #     plt.hist(hist, bins=80)
        #     plt.figure()
        #     plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
        #     plt.title(f)
        #     plt.colorbar()
        #     plt.show()
        #
        #     #     # save arr to tif
        #     DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
        #     DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
        #     np.save(outf+'_correlation', correlation_arr)
        #     np.save(outf + '_p_value', p_value_arr)


    def correlation_for14years(self):

    ##################?????????????????????per_pix???   pre
        period = 'peak'
        fdir_all = result_root + 'detrend/extraction_pre_{}_static/'.format(period)
        fdir_CSIF_par = result_root + 'detrend/extraction_during_{}_growing_season_static/during_{}_CSIF_par/'.format(period,
                                                                                                                      period)
        outdir = result_root + 'detrend_correlation/Pre_{}/'.format(period)
        Tools().mk_dir(outdir)
        # exit()
        dic_CSIF_par={}
        dic_climate = {}
        spatial_dic = {}
        spatial_dic_p_value = {}
        spatial_dic_count = {}
        for f_CSIF_par in tqdm(sorted(os.listdir(fdir_CSIF_par))):
            dic_i = dict(np.load(fdir_CSIF_par + f_CSIF_par, allow_pickle=True, ).item())
            dic_CSIF_par.update(dic_i)

        for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
            outdir_1 = outdir + fdir_1.split('_')[0] + '_' + fdir_1.split('_')[1] + '/'
            print(outdir_1)
            Tools().mk_dir(outdir_1)

            # print(fdir_1)
            if fdir_1.startswith('Pre_PAR'):
                continue
            if fdir_1.startswith('Pre_GPCP'):
                continue
            if fdir_1.startswith('Pre_SPEI'):
                continue
            for fdir_2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
                print(fdir_2)
                for f_1 in  tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'+fdir_2+'/'))):
                    dic_ii = dict(np.load(fdir_all + fdir_1 + '/'+fdir_2+'/'+f_1, allow_pickle=True, ).item())
                    print(fdir_all + fdir_1 + '/'+fdir_2+'/')
                    dic_climate.update(dic_ii)

                f_name1 = fdir_all.split('/')[-2].split('_')[-2]
                print(f_name1)
                f_name = 'CSIF_par_and_' + f_name1 + '_' + fdir_2
                print(f_name)
                # outf=outdir+
                for pix in tqdm(dic_CSIF_par):
                    if pix not in dic_climate:
                        continue
                    val_climate = dic_climate[pix]
                    val_CSIF_par=dic_CSIF_par[pix]
                    val_CSIF_par = val_CSIF_par[1:15]
                    # print(len(val_CSIF_par))
                    if len(val_climate) != 14:
                        continue
                    if len(val_CSIF_par) != 14:
                        continue
                    val_climate = np.array(val_climate)
                    val_CSIF_par = np.array(val_CSIF_par)
                    val_climate[val_climate < -99] = np.nan
                    val_CSIF_par[val_CSIF_par < -99] = np.nan
                    if np.isnan(np.nanmean(val_climate)):
                        continue
                    if np.isnan(np.nanmean(val_CSIF_par)):
                        continue
                    try:
                        # a, b, r = KDE_plot().linefit(xaxis, val)
                        r, p = stats.pearsonr(val_climate, val_CSIF_par)
                        # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
                        # print(k)
                        # spatial_dic_count[pix] = len(val_climate)

                    except Exception as e:
                        print(val_climate,val_CSIF_par)
                        r = np.nan
                        p = np.nan
                    spatial_dic[pix] =r   #
                    spatial_dic_p_value[pix] = p
            # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
            # plt.imshow(count_arr)
            # plt.colorbar()
            # plt.show()
                correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                correlation_arr = np.array(correlation_arr)
                p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
                p_value_arr = np.array(p_value_arr)
                # trend_arr[trend_arr < -10] = np.nan
                # trend_arr[trend_arr > 10] = np.nan

                hist = []
                for i in correlation_arr:
                    for j in i:
                        if np.isnan(j):
                            continue
                        hist.append(j)

                plt.hist(hist, bins=80)
                plt.figure()
                plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
                plt.title(f_name.split('.')[0])
                plt.colorbar()
                plt.show()
                outf=outdir_1+f_name
                #     # save arr to tif
                DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
                DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
                np.save(outf+'_correlation', correlation_arr)
                np.save(outf + '_p_value', p_value_arr)


    # ####during for per pix/////////////////
    #
    #     period = 'early'
    #     fdir_all = result_root + 'detrend/extraction_during_{}_growing_season_static/'.format(period)
    #     fdir_CSIF_par = result_root + 'detrend/extraction_during_{}_growing_season_static/during_{}_CSIF_par/'.format(
    #         period, period)
    #     outdir = result_root + 'detrend_correlation/During_{}/'.format(period)
    #     # fdir_all = result_root + 'detrend/extraction_pre_{}_static/'.format(period)
    #     # fdir_CSIF_par=result_root+'detrend/extraction_during_{}_growing_season_static/during_{}_CSIF_par/'.format(period,period)
    #     # outdir = result_root + 'detrend_correlation/Pre_{}/'.format(period)
    #     Tools().mk_dir(outdir)
    #     # exit()
    #     dic_CSIF_par = {}
    #     dic_climate = {}
    #     spatial_dic = {}
    #     spatial_dic_p_value = {}
    #     spatial_dic_count = {}
    #     for f_CSIF_par in tqdm(sorted(os.listdir(fdir_CSIF_par))):
    #         dic_i = dict(np.load(fdir_CSIF_par + f_CSIF_par, allow_pickle=True, ).item())
    #         dic_CSIF_par.update(dic_i)
    #
    #     for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
    #
    #         if fdir_1.startswith('during_{}_PAR'.format(period)):
    #             continue
    #         if fdir_1.startswith('during_{}_GPCP'.format(period)):
    #             continue
    #         if fdir_1.startswith('during_early_CSIF_par'.format(period)):
    #             continue
    #         if fdir_1.startswith('during_{}_SPEI'.format(period)):
    #             continue
    #         if fdir_1.startswith('during_peak_CSIF_par'.format(period)):
    #             continue
    #         if fdir_1.startswith('during_{}_CSIF_par'.format(period)):
    #             continue
    #         for f2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
    #             print(f2)
    #             dic_ii = dict(np.load(fdir_all + fdir_1 + '/' + f2, allow_pickle=True, ).item())
    #             print(fdir_all + fdir_1 + '/' + f2)
    #             dic_climate.update(dic_ii)
    #
    #         f_name = 'CSIF_par_and_' +fdir_1.split('_')[2]
    #         print(f_name)
    #         # outf=outdir+
    #         for pix in tqdm(dic_CSIF_par):
    #             if pix not in dic_climate:
    #                 continue
    #             val_climate = dic_climate[pix]
    #             val_CSIF_par = dic_CSIF_par[pix]
    #             val_CSIF_par = val_CSIF_par[1:15]
    #             # print(len(val_CSIF_par))
    #             if len(val_climate) != 14:
    #                 continue
    #             if len(val_CSIF_par) != 14:
    #                 continue
    #             val_climate = np.array(val_climate)
    #             val_CSIF_par = np.array(val_CSIF_par)
    #             val_climate[val_climate < -99] = np.nan
    #             val_CSIF_par[val_CSIF_par < -99] = np.nan
    #             if np.isnan(np.nanmean(val_climate)):
    #                 continue
    #             if np.isnan(np.nanmean(val_CSIF_par)):
    #                 continue
    #             try:
    #                 # a, b, r = KDE_plot().linefit(xaxis, val)
    #                 r, p = stats.pearsonr(val_climate, val_CSIF_par)
    #                 # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
    #                 # print(k)
    #                 # spatial_dic_count[pix] = len(val_climate)
    #
    #             except Exception as e:
    #                 print(val_climate, val_CSIF_par)
    #                 r = np.nan
    #                 p = np.nan
    #             spatial_dic[pix] = r  #
    #             spatial_dic_p_value[pix] = p
    #         # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
    #         # plt.imshow(count_arr)
    #         # plt.colorbar()
    #         # plt.show()
    #         correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
    #         correlation_arr = np.array(correlation_arr)
    #         p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
    #         p_value_arr = np.array(p_value_arr)
    #         # trend_arr[trend_arr < -10] = np.nan
    #         # trend_arr[trend_arr > 10] = np.nan
    #
    #         hist = []
    #         for i in correlation_arr:
    #             for j in i:
    #                 if np.isnan(j):
    #                     continue
    #                 hist.append(j)
    #
    #         plt.hist(hist, bins=80)
    #         plt.figure()
    #         plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
    #         plt.title(f_name.split('.')[0])
    #         plt.colorbar()
    #         plt.show()
    #         outf = outdir+ f_name
    #         #     # save arr to tif
    #         DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
    #         DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
    #         np.save(outf + '_correlation', correlation_arr)
    #         np.save(outf + '_p_value', p_value_arr)

    #//////////////////////////////////////???????????????????????????pix 14 ?????? ??????????????????
        # fdir_all = result_root + 'extraction/extraction_anomaly/extraction_pre_early_static/'
        # f2=result_root+'extraction/extraction_anomaly/CSIF_par_during_extraction/during_early_CSIF_par.npy'
        # outdir = result_root + 'correlation/Pre_early/'
        # Tools().mk_dir(outdir)
        # # exit()
        # dic_CSIF_par={}
        # dic_climate = {}
        # spatial_dic = {}
        # spatial_dic_p_value = {}
        # spatial_dic_count = {}
        # dic_CSIF_par=dict(np.load(f2, allow_pickle=True, ).item())
        # for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
        #     outdir_1 = outdir + fdir_1.split('_')[1] + '_' + fdir_1.split('_')[2] + '/'
        #     # print(outdir_1)
        #     Tools().mk_dir(outdir_1)
        #
        #     print(fdir_1)
        #     if fdir_1=='20%_Pre_PAR_extraction':
        #         continue
        #     if fdir_1=='20%_Pre_SPEI_extraction':
        #         continue
        #     if fdir_1=='20%_Pre_GPCP_extraction':
        #         continue
        #     for f_1 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
        #
        #         print(f_1)
        #         dic_climate = dict(np.load(fdir_all + fdir_1 + '/' + f_1 , allow_pickle=True, ).item())
        #
        #         f_name1 = fdir_all.split('/')[-2].split('_')[-2]
        #         # print(f_name1)
        #         f_name = 'CSIF_par_and_' + f_name1 + '_' + f_1
        #         print(f_name)
        #         # outf=outdir+
        #         for pix in tqdm(dic_climate):
        #             val_climate = dic_climate[pix]
        #             val_CSIF_par=dic_CSIF_par[pix]
        #             # print(len(val_CSIF_par))
        #             val_CSIF_par = val_CSIF_par[1:15]
        #             # print(len(val_CSIF_par))
        #             if len(val_climate) != 14:
        #                 continue
        #             if len(val_CSIF_par) != 14:
        #                 continue
        #             val_climate = np.array(val_climate)
        #             val_CSIF_par = np.array(val_CSIF_par)
        #             val_climate[val_climate < -99] = np.nan
        #             val_CSIF_par[val_CSIF_par < -99] = np.nan
        #             if np.isnan(np.nanmean(val_climate)):
        #                 continue
        #             if np.isnan(np.nanmean(val_CSIF_par)):
        #                 continue
        #             try:
        #                 # a, b, r = KDE_plot().linefit(xaxis, val)
        #                 r, p = stats.pearsonr(val_climate, val_CSIF_par)
        #                 # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
        #                 # print(k)
        #                 # spatial_dic_count[pix] = len(val_climate)
        #
        #             except Exception as e:
        #                 print('error')
        #
        #                 r = np.nan
        #                 p = np.nan
        #             spatial_dic[pix] =r   #
        #             spatial_dic_p_value[pix] = p
        #     # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
        #     # plt.imshow(count_arr)
        #     # plt.colorbar()
        #     # plt.show()
        #         correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        #         correlation_arr = np.array(correlation_arr)
        #         p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
        #         p_value_arr = np.array(p_value_arr)
        #         # trend_arr[trend_arr < -10] = np.nan
        #         # trend_arr[trend_arr > 10] = np.nan
        #
        #         hist = []
        #         for i in correlation_arr:
        #             for j in i:
        #                 if np.isnan(j):
        #                     continue
        #                 hist.append(j)
        #
        #         plt.hist(hist, bins=80)
        #         plt.figure()
        #         plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
        #         plt.title(f_name.split('.')[0])
        #         plt.colorbar()
        #         plt.show()
        #         outf=outdir_1+f_name
        #         #     # save arr to tif
        #         DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
        #         DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
        #         np.save(outf+'_correlation', correlation_arr)
        #         np.save(outf + '_p_value', p_value_arr)

    #///////////////////////////////////////////////////////////during/////////////////
        # fdir_all = result_root + 'extraction/extraction_anomaly/extraction_during_early_growing_season_static/'
        # f2=result_root+'extraction/extraction_anomaly/CSIF_par_during_extraction/during_early_CSIF_par.npy'
        # outdir = result_root + 'correlation/During_early/'
        # Tools().mk_dir(outdir)
        # dic_CSIF_par=dict(np.load(f2, allow_pickle=True, ).item())
        # spatial_dic = {}
        # spatial_dic_p_value = {}
        #
        #
        #
        #
        # for f in tqdm(sorted(os.listdir(fdir_all))):
        #     if f == 'during_early_SPEI3.npy':
        #         continue
        #     if f=='during_early_GPCP.npy':
        #         continue
        #     if f=='during_early_PAR.npy':
        #         continue
        #     if f=='during_early_CSIF_par.npy':
        #         continue
        #     outf= outdir+f
        #     dic_climate = dict(np.load(fdir_all+f, allow_pickle=True, ).item())
        #
        #     for pix in tqdm(dic_climate):
        #         val_climate = dic_climate[pix]
        #         val_CSIF_par=dic_CSIF_par[pix]
        #         # print(len(val_CSIF_par))
        #         val_CSIF_par = val_CSIF_par[1:15]
        #         # print(len(val_CSIF_par))
        #         if len(val_climate) != 14:
        #             continue
        #         if len(val_CSIF_par) != 14:
        #             continue
        #         val_climate = np.array(val_climate)
        #         val_CSIF_par = np.array(val_CSIF_par)
        #         val_climate[val_climate < -99] = np.nan
        #         val_CSIF_par[val_CSIF_par < -99] = np.nan
        #         if np.isnan(np.nanmean(val_climate)):
        #             continue
        #         if np.isnan(np.nanmean(val_CSIF_par)):
        #             continue
        #         try:
        #             # a, b, r = KDE_plot().linefit(xaxis, val)
        #             r, p = stats.pearsonr(val_climate, val_CSIF_par)
        #             # k, b = np.polyfit(val_climate, val_CSIF_par, 1)
        #             # print(k)
        #             # spatial_dic_count[pix] = len(val_climate)
        #
        #         except Exception as e:
        #             print('error')
        #
        #             r = np.nan
        #             p = np.nan
        #         spatial_dic[pix] =r   #
        #         spatial_dic_p_value[pix] = p
        # # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
        # # plt.imshow(count_arr)
        # # plt.colorbar()
        # # plt.show()
        #     correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        #     correlation_arr = np.array(correlation_arr)
        #     p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
        #     p_value_arr = np.array(p_value_arr)
        #     # trend_arr[trend_arr < -10] = np.nan
        #     # trend_arr[trend_arr > 10] = np.nan
        #
        #     hist = []
        #     for i in correlation_arr:
        #         for j in i:
        #             if np.isnan(j):
        #                 continue
        #             hist.append(j)
        #
        #     plt.hist(hist, bins=80)
        #     plt.figure()
        #     plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
        #     plt.title(f)
        #     plt.colorbar()
        #     plt.show()
        #
        #     #     # save arr to tif
        #     DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
        #     DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
        #     np.save(outf+'_correlation', correlation_arr)
        #     np.save(outf + '_p_value', p_value_arr)


    def contribution(self): ## Yang helps
        period='late'
        fdir_trend=result_root+'trend_calculation/during_{}_1982-2015/'.format(period)
        fdir_coeffient=result_root+'correlation_detrend/during_{}/'.format(period)
        outdir=result_root+'contribution/{}_attribution_GIMMS_NDVI/'.format(period)

        T.mk_dir(outdir,force=True)
        for f_trend in tqdm(sorted(os.listdir(fdir_trend))):
            if not f_trend.endswith('_trend.tif'):
                continue
            if f_trend.endswith('during_{}_GIMMS_NDVI_trend.tif'.format(period)):
                continue
            if f_trend.startswith('._'):
                continue
            var_name_1 = f_trend.split('.')[0][:-6]
            print(var_name_1)
            print(f_trend)

            for f_coeffient in tqdm(sorted(os.listdir(fdir_coeffient))):
                if not f_coeffient.endswith('correlation.tif'):
                    continue
                if f_coeffient.startswith('._'):
                    continue
                var_name_2 = f_coeffient.split('.')[0][:-23]
                print(var_name_2)
                if var_name_1!=var_name_2:
                    continue

                dataset1 = gdal.Open(fdir_coeffient + f_coeffient)
                dataset2 = gdal.Open(fdir_trend + f_trend)
                bandf1 = dataset1.GetRasterBand(1)
                bandf1_array = bandf1.ReadAsArray()

                bandf2 = dataset2.GetRasterBand(1)
                bandf2_array = bandf2.ReadAsArray()
                # mean_array = np.zeros(range(len(band2_array)), range(len(band2_array)[0]))
                # mean_array = [[0] * len(band2_array[0]) for row in range(len(band2_array))]
                mean_array = [[0 for col in range(len(bandf1_array[0]))] for row in range(len(bandf1_array))]
                for r in range(len(bandf1_array)):
                    for c in range(len(bandf1_array[0])):
                        mean_array[r][c] = (bandf1_array[r][c] * bandf2_array[r][c])
                mean_array = np.array(mean_array)
                # ????????????
                geotransform = dataset1.GetGeoTransform()
                longitude_start = geotransform[0]
                latitude_start = geotransform[3]
                pixelWidth = geotransform[1]
                pixelHeight = geotransform[5]


                newRasterfn = outdir + 'contribution_of_{}'.format(var_name_1)+'.tif'
                print(newRasterfn)

                to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight,
                                          mean_array, ndv=-999999)

                plt.imshow(mean_array, cmap='jet', vmin=-0.001, vmax=0.001)
                plt.colorbar()
                plt.show()


    def max_correlation(self):
        period='peak'
        val_list = [ 'GPCP', 'LST', 'PAR', ]
        # val_list=['CO2','SPEI3']
        for val in val_list:
            fdir1 = result_root + 'detrend_correlation/Pre_{}/Pre_{}/correlation/'.format(period,val)
            fdir2 = result_root + 'detrend_correlation/Pre_{}/Pre_{}/p_value/'.format(period,val)
            f3 = result_root + 'detrend_correlation/During_{}/CSIF_par_and_{}_correlation.tif'.format(period, val)
            print(f3)
            f4=result_root+'detrend_correlation/During_{}/CSIF_par_and_{}_p_value.tif'.format(period,val)
            outdir = result_root + 'detrend_max_correlation/{}/'.format(period)

            Tools().mk_dir(outdir)

            arr_climate_during,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(f3)
            arr_p_value_during, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f4)

            all_arr_climate=[]
            all_arr_p_value = []

            for f1 in tqdm(sorted(os.listdir(fdir1))):
                if not f1.endswith('.tif'):
                    continue
                outf = outdir + val
                print(f1)
                arr_climate_pre,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir1 + f1)
                all_arr_climate.append(arr_climate_pre)
            all_arr_climate.append(arr_climate_during)

            for f2 in tqdm(sorted(os.listdir(fdir2))):
                if not f2.endswith('.tif'):
                    continue
                outf = outdir + val
                print(f2)
                arr_p_value_pre,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir2 + f2)
                all_arr_p_value.append(arr_p_value_pre)
            all_arr_p_value.append(arr_p_value_during)
            # for i in range(len(all_arr_climate)):
            #     arr=all_arr_climate[i]
            #     arr[arr<-999]=np.nan
            #     plt.figure()
            #     plt.imshow(arr,vmin=-0.5,vmax=0.5)
            #     plt.title(str(i))
            # plt.show()

            correlation_matrix = np.ones_like(arr_climate_during)*np.nan
            max_arr=np.ones_like(arr_climate_during)*np.nan
            p_value_arr=np.ones_like(arr_climate_during)*np.nan
            print(len(all_arr_climate[0]))

            all_arr_climate[0][all_arr_climate[0] < -99] = np.nan
            all_arr_climate[1][all_arr_climate[1] < -99] = np.nan
            all_arr_climate[2][all_arr_climate[2] < -99] = np.nan
            all_arr_climate[3][all_arr_climate[3] < -99] = np.nan
            all_arr_climate[4][all_arr_climate[4] < -99] = np.nan

            for r in range(len(all_arr_climate[0])):
                for c in range(len(all_arr_climate[0][0])):

                    v1 = (all_arr_climate[0][r][c])
                    v2 = (all_arr_climate[1][r][c])
                    v3 = (all_arr_climate[2][r][c])
                    v4 = (all_arr_climate[3][r][c])
                    v5 = (all_arr_climate[4][r][c])

                    v1_abs = abs(all_arr_climate[0][r][c])
                    v2_abs = abs(all_arr_climate[1][r][c])
                    v3_abs = abs(all_arr_climate[2][r][c])
                    v4_abs = abs(all_arr_climate[3][r][c])
                    v5_abs = abs(all_arr_climate[4][r][c])

                    p1 = (all_arr_p_value[0][r][c])
                    p2 = (all_arr_p_value[1][r][c])
                    p3 = (all_arr_p_value[2][r][c])
                    p4 = (all_arr_p_value[3][r][c])
                    p5 = (all_arr_p_value[4][r][c])

                    # v_list = [v1, v2, v3, v4]
                    # v_abs_list = [v1_abs, v2_abs, v3_abs, v4_abs]
                    # p_list=[p1,p2,p3,p4]

                    v_list = [v1, v2, v3, v4,v5]
                    v_abs_list = [v1_abs, v2_abs, v3_abs, v4_abs,v5_abs]
                    p_list = [p1, p2, p3, p4, p5]

                    matrix=np.isnan(v_list)
                    if True in matrix:
                        continue

                    if np.nanmax(v_list)<0:
                        correlation_matrix[r][c]=np.nanmin(v_list)
                        max_arr[r][c] = np.argmin(v_list)
                        p_value_arr[r][c]=p_list[int(max_arr[r][c])]

                    elif np.nanmin(v_list)>0:
                        correlation_matrix[r][c] = np.nanmax(v_list)
                        max_arr[r][c] = np.argmax(v_list)
                        p_value_arr[r][c] = p_list[int(max_arr[r][c])]
                    else:

                        max_arg = np.argmax(v_abs_list)
                        selected_v = v_list[max_arg]
                        max_arr[r][c] = max_arg
                        correlation_matrix[r][c]=selected_v
                        p_value_arr[r][c] = p_list[max_arg]
                    # if not np.isnan(max_correlation[r][c]):
                    #     continue
                    # if not np.isnan(p_value_arr[r][c]):
                    #     continue
                    # print(r,c)
                    # print(v_list)

                    # if np.isnan(max_correlation[r][c]):
                    #     continue
                    # if np.isnan(min_correlation[r][c]):
                    #     continue



                # correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                # correlation_arr = np.array(correlation_arr)
                # p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
                # p_value_arr = np.array(p_value_arr)
                # trend_arr[trend_arr < -10] = np.nan
                # trend_arr[trend_arr > 10] = np.nan
            # print(type(max_correlation))
            plt.figure()
            plt.imshow(correlation_matrix, cmap='jet', vmin=-1, vmax=1)
            plt.figure()
            plt.imshow(p_value_arr,cmap='jet', vmin=-1, vmax=1)
            plt.figure()
            plt.imshow(max_arr, cmap='jet', vmin=-1, vmax=4)
            plt.title(val)
            plt.colorbar()
            plt.show()

                # save arr to tif
            DIC_and_TIF().arr_to_tif(correlation_matrix, outf + '_max_correlation.tif')
            np.save(outf + '_max_correlation', correlation_matrix)
            DIC_and_TIF().arr_to_tif(max_arr, outf + '_max_index.tif')
            np.save(outf + '_max_index', max_arr)
            DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_max_p_value.tif')
            np.save(outf + '_max_p_value', p_value_arr)

    def max_correlation_among_all_variables(self):

        # period=['early','peak','late']

        # fdir1 = result_root + 'detrend_max_correlation/{}_withoutCO2/'.format(period)
        # outdir = result_root + 'detrend_max_correlation/{}_among_all_variables_withoutCO2/'.format(period)
        fdir1 = result_root + f'/lc_trend/'
        outdir = result_root + f'lc_trend/'

        Tools().mk_dir(outdir)

        all_arr_climate=[]
        all_arr_p_value = []

        for f1 in tqdm(sorted(os.listdir(fdir1))):
            if not f1.endswith('_trend.tif'):
                continue
            print(f1)
            arr_climate_pre,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir1 + f1)
            all_arr_climate.append(arr_climate_pre)

        for f2 in tqdm(sorted(os.listdir(fdir1))):
            if not f2.endswith('_p_value.tif'):
                continue
            print(f2)
            arr_p_value_pre,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir1 + f2)
            all_arr_p_value.append(arr_p_value_pre)


        max_arr=np.ones_like(all_arr_climate[0])*np.nan
        p_value_arr=np.ones_like(all_arr_climate[0])*np.nan
        print(len(all_arr_climate[0]))
        print(len(all_arr_climate[0][0]))

        for r in range(len(all_arr_climate[0])):
            for c in range(len(all_arr_climate[0][0])):
                trend_value_list = []
                trend_value_array=np.ones_like(11)*np.nan
                p_value_list = []
                for climate in all_arr_climate:

                    trend_value_list.append(abs(climate[r][c]))
                trend_value_array=np.array(trend_value_list)
                trend_value_array[trend_value_array>99]=np.nan

                matrix=np.isnan(trend_value_array)
                # if True in matrix:
                #     continue
                # max_arr[r][c]=np.argmax(trend_value_list)
                max_arr[r][c] = np.nanmax(trend_value_array)*20   # per_year %*20 years in totoal
                print(max_arr[r][c])
                # exit()



        # plt.figure()
        #
        # plt.imshow(p_value_arr,cmap='jet', vmin=-1, vmax=1)
        # plt.figure()
        # plt.imshow(max_arr, cmap='jet', vmin=0, vmax=3)
        # # plt.title(outf)
        # plt.colorbar()
        # plt.show()
        #
            # save arr to tif
        # DIC_and_TIF().arr_to_tif(correlation_matrix, outdir + 'max_correlation.tif')
        # np.save(outdir + 'max_correlation', correlation_matrix)
        DIC_and_TIF().arr_to_tif(max_arr, outdir + 'max_trend.tif')
        np.save(outdir + 'max_trend', max_arr)
        # DIC_and_TIF().arr_to_tif(p_value_arr, outdir + 'max_p_value.tif')
        # np.save(outdir + 'max_p_value', max_correlation)

    def extraction_variables_static_during_daily(self):  # ????????????during multiyear

        # variable_list = ['PAR',
        #                 'Temp','VPD','CCI_SM'] # '??????'
        # variable_list=['CCI_SM'] #  ?????????39
        # variable_list=['GIMMS_NDVI'] #  ?????????34
        # variable_list=['VOD'] #  ??????29 348
        # variable_list = ['NIRv']  # ??????37
        # variable_list=['Precip'] # ?????????????????? ??????37
        # variable_list=['Aridity'] # ?????????????????? ??????37
        # variable_list=['MODIS_NDVI'] #  ?????????168
        # variable_list=['MODIS_LAI'] #  240 20yr
        # variable_list = ['CSIF_fpar']  # ?????????
        # variable_list = ['CSIF']  # ????????? 16
        # variable_list = ['LAI4g'] #??????39 468
        variable_list = ['LAI3g']  # ??????37 444

        for variable in variable_list:
            phenology_df = T.load_df(result_root + f'Main_flow/arr/Phenology/pick_daily_phenology/{variable}/pick_daily.df')

            fdir2=results_root+f'/Main_flow/arr/DIC_Daily/{variable}/'

            dic_variables = {}

            # dic_pre_variables = DIC_and_TIF().void_spatial_dic()

            # ??????????????????
            for f in tqdm(sorted(os.listdir(fdir2))):
                # if not '005' in f:
                #     continue
                if not f.startswith('p'):
                    continue
                if f.endswith('.npy'):
                    dic_i = dict(np.load(fdir2 + f, allow_pickle=True, encoding='latin1' ).item())
                    dic_variables.update(dic_i)

            period_list=['early','peak','late']


            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = result_root + 'extraction_original_val/extraction_original_val_daily_test/extraction_during_{}_growing_season_static/'.format(
                    period)

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):

                    if pix not in dic_period:
                        continue
                    picked_daily = dic_period[pix]   #  ????????????
                    # r,c=pix
                    # if c>180:
                    #     continue

                    time_series = dic_variables[pix]
                    time_series_flatten=time_series.flatten()
                    # print(time_series_flatten)
                    time_series_flatten=np.array(time_series_flatten)
                    print(len(time_series_flatten))

                    # if len(time_series_flatten) !=13505:   #(365*37)
                    #     continue
                    # if len(time_series_flatten) !=7300:   #(365*20)
                    #     continue
                    # plt.plot(time_series_flatten)
                    # plt.show()
                    # time_series = time_series_flatten.reshape(-1, 365)  # ??????
                    picked_daily = np.array(picked_daily, dtype=int)


                    for year in range(37):  # ??????


                        during_time_series = time_series[year][picked_daily]
                        # print(picked_month)#!!!!!

                        during_time_series=np.array(during_time_series, dtype=float)

                        during_time_series[during_time_series < -99.] = np.nan

                        # if np.isnan(np.nanmean(during_time_series)):  # ??????
                        #     continue

                        # variable_sum = np.nansum(during_time_series)
                        # dic_during_variables[pix].append(variable_sum)
                        variable_mean = np.nanmean(during_time_series)  # !!! ???????????????sum  # ???????????????????????? nanmean
                        dic_during_variables[pix].append(variable_mean)

                    dic_spatial_count[pix] = len(dic_during_variables[pix])
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                plt.imshow(arr, cmap='jet')
                plt.colorbar()
                plt.title('')
                plt.show()
                np.save(outdir + 'during_{}_{}'.format(period,variable), dic_during_variables)  # ??????


    def extraction_variables_static_during_daily_climate_variables(self):  # ????????????during multiyear

        # variable_list = ['PAR',
        #                 'Temp','VPD','CCI_SM'] # '??????'
        variable_list=['LAI3g']



        for variable in variable_list:
            phenology_df = T.load_df(result_root + 'Main_flow/arr/Phenology/pick_daily_phenology/LAI3g/pick_daily.df')

            f=results_root+f'/Main_flow/arr/DIC_Daily/LAI3g/'

            dic_variables = {}


            dic_variables = dict(np.load(f, allow_pickle=True, encoding='latin1' ).item())

            period_list=['early','peak','late']


            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = result_root + 'extraction_original_val/extraction_original_val_daily_test/extraction_during_{}_growing_season_static/'.format(
                    period)

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):

                    if pix not in dic_period:
                        continue
                    picked_daily = dic_period[pix]   #  ????????????
                    # r,c=pix
                    # if c>180:
                    #     continue

                    time_series = dic_variables[pix]


                    # if len(time_series_flatten) !=13505:   #(365*37)
                    #     continue
                    if len(time_series) != 14235:  # (365*39)
                        continue
                    # if len(time_series_flatten) !=7300:   #(365*20)
                    #     continue
                    # plt.plot(time_series_flatten)
                    # plt.show()
                    time_series_reshape = time_series.reshape(-1, 365)  # ??????
                    picked_daily = np.array(picked_daily, dtype=int)


                    for year in range(39):  # ??????


                        during_time_series = time_series_reshape[year][picked_daily]
                        # print(picked_month)#!!!!!

                        during_time_series=np.array(during_time_series, dtype=float)

                        during_time_series[during_time_series < -99.] = np.nan

                        # if np.isnan(np.nanmean(during_time_series)):  # ??????
                        #     continue

                        # variable_sum = np.nansum(during_time_series)
                        # dic_during_variables[pix].append(variable_sum)
                        variable_mean = np.nanmean(during_time_series)  # !!! ???????????????sum  # ???????????????????????? nanmean
                        dic_during_variables[pix].append(variable_mean)

                    dic_spatial_count[pix] = len(dic_during_variables[pix])
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                plt.imshow(arr, cmap='jet')
                plt.colorbar()
                plt.title('')
                plt.show()
                np.save(outdir + 'during_{}_{}'.format(period,variable), dic_during_variables)  # ??????

    def extraction_variables_static_during_month(self):  # ????????????during multiyear

        # variable_list = ['CO2','PAR',
        #                 'Temp','VPD'] # '??????'
        # variable_list=['CCI_SM'] #  ?????????39
        # variable_list=['GIMMS_NDVI'] #  ?????????34
        # variable_list=['VOD'] #  ??????29 348
        # variable_list = ['NIRv']  # ??????37
        # variable_list=['Precip'] # ?????????????????? ??????37
        # variable_list=['Aridity'] # ?????????????????? ??????37
        # variable_list=['MODIS_NDVI'] #  ?????????168
        # variable_list = ['MODIS_LAI']  # 240 20yr
        # variable_list = ['CSIF_fpar']  # ?????????
        variable_list = ['Trendy_ensemble']  # ????????? 16
        # variable_list = ['LAI4g'] #??????39 468
        # variable_list = ['LAI3g']  # ??????37 444

        for variable in variable_list:
            phenology_df = T.load_df(
                results_root + f'Main_flow/arr/Phenology/Get_Monthly_Early_Peak_Late/MODIS_LAI/Monthly_Early_Peak_Late_via_DOY.df')

            fdir2 = data_root + f'original_dataset/{variable}_dic/'

            dic_variables = {}

            # dic_pre_variables = DIC_and_TIF().void_spatial_dic()

            # ??????????????????
            for f in tqdm(sorted(os.listdir(fdir2))):
                # if not '005' in f:
                #     continue
                if not f.startswith('p'):
                    continue
                if f.endswith('.npy'):
                    dic_i = dict(np.load(fdir2 + f, allow_pickle=True, encoding='latin1').item())
                    dic_variables.update(dic_i)

            period_list = ['early', 'peak', 'late']

            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = result_root + 'extraction_original_val/extraction_original_val_monthly/extraction_during_{}_growing_season_static/'.format(
                    period)

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):

                    if pix not in dic_period:
                        continue
                    picked_month = dic_period[pix]  # ????????????
                    # r,c=pix
                    # if c>180:
                    #     continue

                    time_series = dic_variables[pix]
                    if len(time_series) != 240:  # (12*20)/12*37=444
                        continue
                    # plt.plot(time_series)
                    # plt.show()
                    time_series = time_series.reshape(-1, 12)  # ??????
                    picked_month = np.array(picked_month, dtype=int)
                    picked_month = picked_month-1


                    for year in range(20):  # ??????

                        during_time_series = time_series[year][picked_month]

                        # print(picked_month)#!!!!!

                        during_time_series = np.array(during_time_series, dtype=float)

                        during_time_series[during_time_series < -99.] = np.nan

                        # if np.isnan(np.nanmean(during_time_series)):  # ??????
                        #     continue

                        # variable_sum = np.nansum(during_time_series)
                        # dic_during_variables[pix].append(variable_sum)
                        variable_mean = np.nanmean(during_time_series)  # !!! ???????????????sum  # ???????????????????????? nanmean
                        dic_during_variables[pix].append(variable_mean)

                    dic_spatial_count[pix] = len(dic_during_variables[pix])
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                plt.imshow(arr, cmap='jet')
                plt.colorbar()
                plt.title('')
                plt.show()
                # np.save(outdir + 'pre_{}mo_CO2_original'.format(N), dic_pre_variables) #??????
                np.save(outdir + 'during_{}_{}'.format(period, variable), dic_during_variables)  # ??????

    def extraction_variables_static_during_Trendy(self):  # ????????????during multiyear


        variables_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai',
                          'ISAM_S2_LAI',
                          'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                          'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai',
                          'ISBA-CTRIP_S2_lai','Trendy_ensemble']

        phenology_df = T.load_df(
            data_root + f'Get_Monthly_Early_Peak_Late/Monthly_Early_Peak_Late.df')

        for variable in variables_list:

            fdir2 = data_root + f'DIC2/{variable}/'

            dic_variables = {}

            # dic_pre_variables = DIC_and_TIF().void_spatial_dic()

            # ??????????????????
            for f in tqdm(sorted(os.listdir(fdir2))):

                if not f.startswith('p'):
                    continue
                if f.endswith('.npy'):
                    dic_i = dict(np.load(fdir2 + f, allow_pickle=True, encoding='latin1').item())
                    dic_variables.update(dic_i)

            period_list = ['early', 'peak', 'late']

            for period in period_list:
                dic_during_variables = DIC_and_TIF().void_spatial_dic()
                outdir = results_root + 'extraction_original_val/extraction_original_val_Trendy2/extraction_during_{}_growing_season_static/'.format(
                    period)

                Tools().mk_dir(outdir, True)
                dic_period = T.df_to_spatial_dic(phenology_df, period)

                dic_spatial_count = {}
                spatial_dic = {}
                for pix in tqdm(dic_variables):

                    if pix not in dic_period:
                        continue
                    picked_month = dic_period[pix]  # ????????????
                    # r,c=pix
                    # if c>180:
                    #     continue

                    time_series = dic_variables[pix]
                    if len(time_series) != 468:  # (12*20)/12*37=444
                        continue
                    # plt.plot(time_series)
                    # plt.show()
                    time_series = time_series.reshape(-1, 12)  # ??????
                    picked_month = np.array(picked_month, dtype=int)
                    picked_month = picked_month-1


                    for year in range(39):  # ??????

                        during_time_series = time_series[year][picked_month]

                        # print(picked_month)#!!!!!

                        during_time_series = np.array(during_time_series, dtype=float)

                        during_time_series[during_time_series < -99.] = np.nan

                        # if np.isnan(np.nanmean(during_time_series)):  # ??????
                        #     continue

                        # variable_sum = np.nansum(during_time_series)
                        # dic_during_variables[pix].append(variable_sum)
                        variable_mean = np.nanmean(during_time_series)  # !!! ???????????????sum  # ???????????????????????? nanmean
                        dic_during_variables[pix].append(variable_mean)

                    dic_spatial_count[pix] = len(dic_during_variables[pix])
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
                # plt.imshow(arr, cmap='jet')
                # plt.colorbar()
                # plt.title('')
                # plt.show()
                # np.save(outdir + 'pre_{}mo_CO2_original'.format(N), dic_pre_variables) #??????
                np.save(outdir + 'during_{}_{}'.format(period, variable), dic_during_variables)  # ??????

    def extraction_winter_during(self):  # ????????????winter

        f1 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/winter.npy'  # !!??????


        period='winter'

        fdir=data_root+'original_dataset/temperature_dic/'
        outdir = result_root + 'extraction_original_val/1982-2015_original_extraction_all_seasons/'


        Tools().mk_dir(outdir,True)


        dic_winter= dict(np.load(f1, allow_pickle=True, ).item())

        dic_variables=T.load_npy_dir(fdir)
        dic_during_variables={}


        for pix in tqdm(dic_variables):
            if pix not in dic_winter:
                continue


            winter_index = dic_winter[pix]

            time_series = dic_variables[pix]
            # print(len(time_series))
            # exit()
            time_series=np.array(time_series)

            if len(time_series) != 444:
                continue

            time_series = time_series.reshape(-1, 12)
            time_series=time_series[:34]
            # print(time_series)
            # exit()
            time_series_T = time_series.T# ??????


            during_time_series_part1 = T.pick_vals_from_1darray(time_series_T,winter_index[0])
            during_time_series_part2 = T.pick_vals_from_1darray(time_series_T, winter_index[1])
            # print(winter_index[0])
            # print(winter_index[1])

            during_time_series_part1_T=during_time_series_part1.T[:-1].T
            during_time_series_part2_T = during_time_series_part2.T[1:].T
            if len(winter_index[0])==0:
                continue
            if len(winter_index[1])==0:
                continue
            during_winter_time_series=np.concatenate((during_time_series_part1_T,during_time_series_part2_T))
            during_winter_time_series_T=during_winter_time_series.T
            sum_list=[]
            for i in during_winter_time_series_T:
                # variable_sum = np.nansum(i)
                variable_sum = np.nanmean(i)
                sum_list.append(variable_sum)
            dic_during_variables[pix] = sum_list


        np.save(outdir + 'winter_temperature_extraction', dic_during_variables)  # ??????

    def extraction_winter_index(self):  # ????????????during multiyear
        outdir=result_root+'20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/'
        f1 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/late_end_mon.npy'  # !!??????
        f2 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/early_start_mon.npy'

        dic_start = dict(np.load(f1, allow_pickle=True, ).item())
        dic_end = dict(np.load(f2, allow_pickle=True, ).item())
        dic_winter={}
        dic_spatial_count={}

        for pix in dic_start:
            r,c=pix
            if r>120:
                continue
            if r<50:
                continue
            start=dic_start[pix]
            end = dic_end[pix]
            if len(start) == 0:
                continue

            if start[0]<end[0]:
                continue
            annual_winter_list=[]

            start=start[0]-1+1#
            end=end[0]-1#
            part1=list(range(start,12))
            part2=list(range(0,end))
            winter_month=[part1,part2]

            dic_spatial_count[pix]=len(winter_month[0]+winter_month[1])
            dic_winter[pix]=winter_month
        count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
        # plt.imshow(count_arr)
        # plt.colorbar()
        #
        # plt.show()
        np.save(outdir + 'winter',dic_winter)  # ??????

    def plot_phenology(self):

        f1 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/late_end_mon.npy'  # !!??????
        f2 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/early_start_mon.npy'

        dic_start = dict(np.load(f1, allow_pickle=True, ).item())
        dic_end = dict(np.load(f2, allow_pickle=True, ).item())
        dic_SOS={}
        dic_EOS = {}

        for pix in dic_start:
            r,c=pix

            start=dic_start[pix]
            end = dic_end[pix]
            if len(start) == 0:
                continue
            dic_SOS[pix]=end[0]
            dic_EOS[pix]=start[0]
        arr_SOS=DIC_and_TIF().pix_dic_to_spatial_arr(dic_SOS)
        arr_EOS=DIC_and_TIF().pix_dic_to_spatial_arr(dic_EOS)

        plt.imshow(arr_SOS)
        plt.figure()
        plt.imshow(arr_EOS)
        plt.colorbar()
        plt.show()


    def calculate_growing_season_length(self):  # ????????????during multiyear

        period='late'
        f1 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/early_start.npy'  # !!??????
        f2 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/early_end.npy'

        outdir=result_root+'growing_season_length_calculation/'
        Tools().mk_dir(outdir,True)
        dic_variables = {}

        dic_start = dict(np.load(f1, allow_pickle=True, ).item())
        dic_end = dict(np.load(f2, allow_pickle=True, ).item())

        growing_length = {}


        for pix in tqdm(dic_start):
            if pix not in dic_end:
                continue

            start_index = dic_start[pix]
            end_index = dic_end[pix]
            # print(start_index)
            # print(end_index)
            if len(start_index)==0:
                continue
            if len(end_index)==0:
                continue
            start_index = start_index[0]
            end_index = end_index[0]
            growing_length[pix]=end_index-start_index
            length=np.array(end_index-start_index)
            # print(length)
            if length <= 0:
                continue
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(growing_length)
        plt.imshow(arr, cmap='jet')
        plt.colorbar()
        plt.title('')
        plt.show()
        np.save(outdir + '{}_growing_season_length'.format(period), growing_length)  # ??????

    def extraction_variables_static_pre_month(self):  # ????????????during multiyear
        N=3
        variable='PAR' #'??????'
        period='late'
        f1 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/late_start_mon.npy'
        f2 = result_root + '20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/late_end_mon.npy'
        # fdir2=data_root+'GIMMS_NDVI/per_pix_clean_zscore/'  #'??????'
        fdir2=data_root+'Terraclimate/PAR/PAR_zscore/' #'??????'
        # fdir2=data_root+'GLEAM/surf_soil_moisture_zscore/'
        outdir = result_root + 'extraction_anomaly_val/extraction_pre_{}_{}month_growing_season_static/'.format(period,N) # ??????


        Tools().mk_dir(outdir,True)
        dic_variables = {}
        dic_pre_variables = DIC_and_TIF().void_spatial_dic()

        dic_start = dict(np.load(f1, allow_pickle=True, ).item())
        dic_end = dict(np.load(f2, allow_pickle=True, ).item())

        dic_spatial_count = {}
        spatial_dic = {}

        # ??????????????????
        for f in tqdm(sorted(os.listdir(fdir2))):
            # if not '005' in f:
            #     continue
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir2 + f, allow_pickle=True, encoding='latin1' ).item())
                dic_variables.update(dic_i)

        for pix in tqdm(dic_variables):
            if pix not in dic_start:
                continue
            # r,c=pix
            # if c>180:
            #     continue
            start_index = dic_start[pix]
            end_index = dic_end[pix]
            print(start_index)

            if len(start_index) == 0:
                continue
            start_index = start_index[0]
            end_index = end_index[0]
            time_series = dic_variables[pix]
            time_series=np.array(time_series)
            # if len(time_series)!=5475:  #??????
            # if len(time_series)!=5475
            if len(time_series) != 444:
                continue
            # plt.plot(time_series)
            # plt.show()
            time_series = time_series.reshape(37, -1)  # ??????

            for year in range(37):  # ??????
                # print(len(SOS_index_series))
                # time_series = time_series.reshape(15, -1)
                # print(time_series)
                pre_time_series=time_series[year][(start_index-N):start_index]

                # plt.imshow(time_series)
                # plt.show()
                pre_time_series[pre_time_series<-99]=np.nan

                # if np.isnan(np.nansum(during_time_series)):
                #     continue
                if np.isnan(np.nanmean(pre_time_series)):  # ??????
                    continue
                # variable_sum=np.nansum(during_time_series)
                variable_mean = np.nanmean(pre_time_series)  # !!! ???????????????sum  # ???????????????????????? nanmean
                # dic_during_variables[pix].append(variable_mean)
                dic_pre_variables[pix].append(variable_mean)   # ????????????15??????
            dic_spatial_count[pix] = len(dic_pre_variables[pix])
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
        plt.imshow(arr, cmap='jet')
        plt.colorbar()
        plt.title('')
        plt.show()
        # np.save(outdir + 'pre_{}mo_CO2_original'.format(N), dic_pre_variables) #??????
        np.save(outdir + 'pre_{}month_{}_{}'.format(N, period,variable), dic_pre_variables)  # ??????


    def forward_window_extract(self, x, window):
        # ????????????
        # window = window-1
        # ?????????????????????

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_extraction_by_window=[]
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:
                temp = []
                for w in range(window):
                    temp.append(x[i + w])
                new_x_extraction_by_window.append(temp)

        return new_x_extraction_by_window

    def forward_window_extract_anomaly(self, x, window):
        # ????????????
        # window = window-1
        # ?????????????????????

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        # new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        new_x_extraction_by_window=[]
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:
                anomaly = []
                x_vals=[]
                for w in range(window):
                    x_val=(x[i + w])
                    x_vals.append(x_val)
                if np.isnan(np.nanmean(x_vals)):
                    continue

                x_mean=np.nanmean(x_vals)


                for i in range(len(x_vals)):
                    if x_vals[0]==None:
                        continue
                    x_anomaly=x_vals[i]-x_mean

                    anomaly.append(x_anomaly)
                new_x_extraction_by_window.append(anomaly)

        return new_x_extraction_by_window

    def forward_window_extract_mean(self, x, window):
        # ????????????
        # window = window-1
        # ?????????????????????

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        new_x_extraction_by_window=[]
        for i in range(len(x)):
            if i + window >= len(x):
                continue
            else:
                y_mean_list = []
                y_vals_list=[]
                for w in range(window):
                    y_val=(x[i + w])
                    y_vals_list.append(y_val)
                if np.isnan(np.nanmean(y_vals_list)):
                    continue
                y_vals = T.interp_nan(y_vals_list)
                if y_vals[0] == None:
                    continue
                y_mean=np.nanmean(y_vals)

                y_mean_list.append(y_mean)
                new_x_extraction_by_window.append(y_mean_list)

        return new_x_extraction_by_window


    def univariate_correlation_window(self):  # ??????????????????
        period = 'early'
        window=20
        # month=3
        time_range='1982-2018'

        fdir_all = result_root + f'detrend_original/extract_detrend_original_window/{time_range}_during_{period}/{window}_year_window/'
        # fdir_all=result_root+'extraction_original_val_window/{}_during_{}/{}_year_window/'.format(time_range,period,window)
        outdir = result_root + 'univariate_correlation_window_detrend/{}_during_{}_window{}/'.format(time_range,period,window)
        Tools().mk_dir(outdir,force=True)
        dic_y = {}
        dic_climate = {}
        spatial_dic = {}
        spatial_dic_p_value = {}
        spatial_dic_count = {}
        for Y_variables in tqdm(sorted(os.listdir(fdir_all))):

            if Y_variables!='detrend_during_{}_LAI_GIMMS'.format(period):
                continue

            for f_Y in tqdm(sorted(os.listdir(fdir_all+'/'+Y_variables))):
                dic_i = dict(np.load(fdir_all +'/' + Y_variables+'/'+f_Y, allow_pickle=True, ).item())
                dic_y.update(dic_i)

            climate_list=[]
            climate_list_name=[]

            for X_variable in tqdm(sorted(os.listdir(fdir_all))):
                print (X_variable)
                # exit()
                if X_variable != 'detrend_during_{}_temperature'.format(period, ):
                    continue

                # if X_variable == 'detrend_during_{}_root_soil_moisture'.format(period, ):
                #     continue
                # if X_variable == 'detrend_during_{}_surf_soil_moisture'.format(period, ):
                #     continue
                # if X_variable == 'detrend_during_{}_GIMMS_NDVI'.format(period):
                #     continue
                # if X_variable == 'detrend_during_{}_CSIF_fpar'.format(period):
                #     continue
                # if X_variable == 'detrend_during_{}_NIRv'.format(period):
                #     continue
                # if X_variable == 'detrend_during_{}_VOD'.format(period):
                #     continue
                # if X_variable == 'detrend_during_{}_CCI_SM'.format(period):
                #     continue
                # if X_variable == 'detrend_during_{}_Aridity'.format(period):
                #     continue
                # if X_variable == 'detrend_during_{}_Precip'.format(period):
                #     continue

                if X_variable.startswith('.'):
                    continue

                for f_X in tqdm(sorted(os.listdir(fdir_all+'/'+X_variable))):
                    # print(fdir_all +'/' + X_variable+'/'+f_X)
                    # exit()
                    dic_i = dict(np.load(fdir_all +'/' + X_variable+'/'+f_X, allow_pickle=True, ).item())
                    dic_climate.update(dic_i)


                variable_name = X_variable
                print(variable_name)
                f_name =  X_variable.split('.')[0] + '_LAI_GIMMS'
                print(f_name)
                # exit()

                array_len=len(dic_y[(220,404)])
                print(array_len)
                # print(len(dic_y[(220,404)]))
                # exit()
                for w in range(array_len):
                    for pix in tqdm(dic_y):
                        if pix not in dic_climate:
                            continue

                        if len(dic_y[pix])==0:
                            continue
                        if len(dic_climate[pix])!=array_len:  # ?????????CCI?????????????????????
                            continue
                        # if len(dic_climate[pix])!=array_len:
                        #     continue
                        if len(dic_climate[pix])==0:
                            continue
                        if len(dic_y[pix])!=array_len:
                            continue
                        if len(dic_y[pix])!=len(dic_climate[pix]):  # ??????????????????
                            continue
                        val_y_variables = dic_y[pix][w]
                        # print(val_y_variables)
                        # exit()
                        val_climate = dic_climate[pix][w]

                        if len(val_climate) < window:
                            continue
                        if len(val_y_variables)<window:
                            continue

                        val_climate = np.array(val_climate)
                        val_y_variables = np.array(val_y_variables)
                        r,c=pix
                        if r>120:
                            continue
                        # print(pix)
                        # print(np.shape(val_climate))
                        # exit()
                        # plt.figure()
                        # plt.plot(val_climate)
                        # plt.plot(val_y_variables)
                        # plt.grid(1)
                        # plt.show()
                        val_climate[val_climate < -99] = np.nan
                        val_y_variables[val_y_variables < -99] = np.nan
                        if np.isnan(np.nanmean(val_climate)):
                            continue
                        if np.isnan(np.nanmean(val_y_variables)):
                            continue
                        try:

                            r, p = stats.pearsonr(val_climate, val_y_variables)

                        except Exception as e:
                            print(val_climate, val_y_variables)
                            r = np.nan
                            p = np.nan
                        spatial_dic[pix] = r  #
                        spatial_dic_p_value[pix] = p
                    # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
                    # plt.imshow(count_arr)
                    # plt.colorbar()
                    # plt.show()
                    correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                    correlation_arr = np.array(correlation_arr)
                    p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
                    p_value_arr = np.array(p_value_arr)


                    hist = []
                    for i in correlation_arr:
                        for j in i:
                            if np.isnan(j):
                                continue
                            hist.append(j)

                    # plt.hist(hist, bins=80)
                    # plt.figure()
                    # plt.imshow(correlation_arr, cmap='jet', vmin=-1, vmax=1)
                    # plt.title(f_name+'_'+str(w))
                    # plt.colorbar()
                    # plt.show()
                    #     # save arr to tif
                    outdir2=outdir+f_name
                    T.mk_dir(outdir2,force=True)

                    outf = outdir2+'/'+f_name+'_'+'{:02d}'.format(w)
                    print(str(w))
                    print(f_name)
                    print(outf)
                    # exit()
                    DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_correlation.tif')
                    DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
                    np.save(outf + '_correlation', correlation_arr)
                    np.save(outf + '_p_value', p_value_arr)

    def multiregression_beta_window(self):  # ??????????????????
        period = 'early'
        window = 15
        slices=37-window
        time_range = '1982-2018'

        fdir_all = result_root + 'extraction_original_window/{}_during_{}/{}_year_window/'.format(time_range,
                                                                                                             period,                                                                                                window)
        outdir = result_root + 'multiregression_beta_window/{}_during_{}_window{}/'.format(time_range, period,
                                                                                                     window)
        f_y_mean= result_root + 'extraction_original_val_window_mean/{}_during_{}/{}_year_window/during_{}_GIMMS_NDVI.npy'\
            .format(time_range,period,window,period)

        mean_dic = T.load_npy(f_y_mean)
        Tools().mk_dir(outdir, force=True)
        dic_y = {}
        # dic_climate = {}

        for Y_variables in tqdm(sorted(os.listdir(fdir_all))):

            if Y_variables != 'during_{}_GIMMS_NDVI.npy'.format(period):
                continue

            dic_y = dict(np.load(fdir_all + '/' + Y_variables, allow_pickle=True, ).item())


            climate_all_variables_dic = {}
            climate_name_list = []
            for X_variable in tqdm(sorted(os.listdir(fdir_all))):
                print(X_variable)
                # exit()

                if X_variable == 'during_{}_root_soil_moisture.npy'.format(period, ):
                    continue
                if X_variable == 'during_{}_surf_soil_moisture.npy'.format(period, ):
                    continue
                if X_variable == 'during_{}_LAI_GIMMS_NDVI.npy'.format(period):
                    continue
                if X_variable == 'during_{}_CSIF_fpar.npy'.format(period):
                    continue
                if X_variable == 'during_{}_NIRv.npy'.format(period):
                    continue
                if X_variable == 'during_{}_VOD.npy'.format(period):
                    continue
                # if X_variable == 'during_{}_SPEI3.npy'.format(period):
                #     continue
                if X_variable == 'during_{}_Aridity.npy'.format(period):
                    continue
                if X_variable == 'during_{}_Precip.npy'.format(period):
                    continue


                if X_variable.startswith('.'):
                    continue

                dic_climate={}

                dic_climate = dict(np.load(fdir_all + '/' + X_variable, allow_pickle=True, ).item())

                climate_all_variables_dic[X_variable]=dic_climate
                climate_name_list.append(X_variable)


            for w in range(slices):
                outf = outdir + 'multiregression_{}_{}'.format(period, time_range)+'_window'+ '{:02d}'.format(w)
                print(outf)

                multi_derivative={}

                for pix in tqdm(dic_y):
                    x_val_list=[]
                    for v_ in climate_all_variables_dic:
                        if pix not in climate_all_variables_dic[v_]:  ##
                            continue

                        x_val=climate_all_variables_dic[v_][pix][w]

                        if not len(x_val) == window:  ##
                            continue
                        if len(x_val) == 0:
                            continue
                        if np.isnan(np.nanmean(x_val)):
                            continue
                        x_vals = T.interp_nan(x_val)

                        if x_vals[0] == None:
                            continue
                        x_val_list.append(x_val)

                        # ???y?????????
                    if len(dic_y[pix]) != slices:
                        continue
                    val_y_variable = dic_y[pix][w]
                    if len(val_y_variable) == 0:
                        continue

                    val_y_variable = T.interp_nan(val_y_variable)
                    if val_y_variable[0]== None:
                        continue
                    if np.isnan(np.nanmean(val_y_variable)):
                        continue


                    val_climate = np.array(x_val_list)
                    val_climate_T = val_climate.T

                    val_y_variables = np.array(val_y_variable)

                    if pix not in mean_dic:  ##
                        continue

                    if not len(mean_dic[pix]) == slices:  ##
                        continue

                    y_mean = mean_dic[pix][w][0]

                    r, c = pix
                    if r > 120:
                        continue

                    val_climate_T[val_climate_T < -9999] = np.nan
                    val_y_variables[val_y_variables < -9999] = np.nan
                    if np.isnan(np.nanmean(val_climate_T)):
                        continue
                    if np.isnan(np.nanmean(val_y_variables)):
                        continue
                    try:

                        linear_model = LinearRegression()
                        linear_model.fit(val_climate_T, val_y_variables)
                        coef_ = np.array(linear_model.coef_) / y_mean
                        coef_dic = dict(zip(climate_name_list, coef_))
                        # print(coef_dic)

                        multi_derivative[pix] = coef_dic

                    except Exception as e:
                        print('error')
                        # print(x_val_list, val_y_variable)

                    # spatial_dic[pix] = multi_derivative #
                    # print(spatial_dic)
                    # exit()

                # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
                # plt.imshow(count_arr)
                # plt.colorbar()
                # plt.show()
                # correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                # correlation_arr = np.array(correlation_arr)
                #
                #
                # hist = []
                # for i in correlation_arr:
                #     for j in i:
                #         if np.isnan(j):
                #             continue
                #         hist.append(j)

                np.save(outf + '_Beta', multi_derivative)

    def partial_correlation_window(self):  # ??????????????????
        period = 'peak'
        window = 15
        slices=39-window
        time_range = '1982-2018'



        fdir_Y = results_root + f'extract_relative_change_window/15_year_window_1982/Y/'
        fdir_X = results_root +f'extract_relative_change_window/15_year_window_1982/X/'

        climate_all_variables_dic = {}
        climate_name_list = []


        for f_x in tqdm(os.listdir(fdir_X)):
            # exit()
            if not period in f_x:
                continue
            if 'LAI3g' in f_x:
                continue




            dic_climate = {}

            dic_ii = dict(np.load(fdir_X + f_x, allow_pickle=True, ).item())
            dic_climate.update(dic_ii)

            climate_all_variables_dic[f_x.split('.')[0]] = dic_climate
            climate_name_list.append(f_x.split('.')[0])
        print(climate_name_list)



        for f_y in os.listdir(fdir_Y):

            f_name= f_y.split('.')[0]
            if not period in f_y:
                continue


            outdir = results_root + f'partial_window/{time_range}_{f_name}_{window}/'


            Tools().mk_dir(outdir, force=True)
            dic_y = {}
            # dic_climate = {}
            dic_i = dict(np.load(fdir_Y +f_y, allow_pickle=True, ).item())
            dic_y.update(dic_i)


            for w in range(slices):

                outf = outdir + f_y.split('.')[0]+f'_{w:02d}'
                print(outf)

                partial_correlation_dic={}
                partial_p_value_dic={}

                for pix in tqdm(dic_y):
                    x_val_list_valid=[]

                    df_new = pd.DataFrame()
                    for v_ in climate_all_variables_dic:
                        if pix not in climate_all_variables_dic[v_]:  ##
                            continue

                        x_val=climate_all_variables_dic[v_][pix][w]

                        if not len(x_val) == window:  ##
                            continue
                        if len(x_val) == 0:
                            continue
                        if np.isnan(np.nanmean(x_val)):
                            continue


                        matix = np.isnan(x_val)  # ????????????time series ??????
                        matix = list(matix)
                        valid_number = matix.count(False)
                        # print(matix)

                        if valid_number / len(x_val) < 0.70:
                            continue


                        x_vals = T.interp_nan(x_val)
                        if x_vals[0] == None:
                            continue

                        df_new[v_] = x_vals
                        x_val_list_valid.append(v_)

                        # ???y?????????
                    if len(dic_y[pix]) != slices:
                        continue
                    val_y_variable = dic_y[pix][w]

                    if len(val_y_variable) == 0:
                        continue
                    # print(val_y_variable)

                    if np.isnan(np.nanmean(val_y_variable)):
                        continue

                    matix = np.isnan(val_y_variable)  # ????????????time series ??????
                    matix = list(matix)
                    valid_number = matix.count(False)
                    # print(matix)

                    if valid_number / len(val_y_variable) < 0.70:
                        continue

                    val_y_variable = T.interp_nan(val_y_variable)
                    if val_y_variable[0]== None:
                        continue
                    if np.isnan(np.nanmean(val_y_variable)):
                        continue

                    val_y_variables = np.array(val_y_variable)
                    df_new['y']=val_y_variables
                    # T.print_head_n(df_new)


                    df_new = df_new.dropna(axis=1, how='all')
                    x_var_list_valid_new = []
     ############################
                    for v_ in x_val_list_valid:
                        if not v_ in df_new:
                            continue
                        else:
                            x_var_list_valid_new.append(v_)

                    r, c = pix
                    if r > 120:
                        continue
                    try:
                        partial_correlation = {}
                        partial_correlation_p_value = {}
                        for x in x_var_list_valid_new:

                            x_var_list_valid_new_cov = copy.copy(x_var_list_valid_new)
                            x_var_list_valid_new_cov.remove(x)
                            corr, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                            partial_correlation[x] = corr
                            partial_correlation_p_value[x] = p


                    except Exception as e:
                            print('error')
                        # print(x_val_list, val_y_variable)val_y_variables

                    partial_correlation_dic[pix] = partial_correlation
                    partial_p_value_dic[pix] = partial_correlation_p_value
                T.save_npy(partial_correlation_dic, outf + '_correlation')

                T.save_npy(partial_p_value_dic, outf + '_p_value')

    def partial_correlation_window_Trendy(self):  # ??????????????????
        period = 'early'
        window = 15
        slices=39-window
        time_range = '1982-2020'



        fdir_Y = results_root + f'extract_relative_change_window/{window}_year_window_1982/Y/'
        fdir_X = results_root + f'extract_relative_change_window/{window}_year_window_1982/X/'

        climate_all_variables_dic = {}
        climate_name_list = []


        for f_x in tqdm(os.listdir(fdir_X)):
            # exit()
            if not period in f_x:
                continue

            dic_climate = {}

            dic_ii = dict(np.load(fdir_X + f_x, allow_pickle=True, ).item())
            dic_climate.update(dic_ii)

            climate_all_variables_dic[f_x.split('.')[0]] = dic_climate
            climate_name_list.append(f_x.split('.')[0])
        print(climate_name_list)



        for f_y in os.listdir(fdir_Y):
            f_name= f_y.split('.')[0]
            if not period in f_y:
                continue

            outdir = results_root + f'partial_window/{time_range}_{f_name}_{window}/'


            Tools().mk_dir(outdir, force=True)
            dic_y = {}
            # dic_climate = {}
            dic_i = dict(np.load(fdir_Y +f_y, allow_pickle=True, ).item())
            dic_y.update(dic_i)


            for w in range(slices):
                outf = outdir + f_y.split('.')[0]+f'_{w:02d}'
                print(outf)

                partial_correlation_dic={}
                partial_p_value_dic={}

                for pix in tqdm(dic_y):
                    x_val_list_valid=[]

                    df_new = pd.DataFrame()
                    for v_ in climate_all_variables_dic:
                        if pix not in climate_all_variables_dic[v_]:  ##
                            continue

                        x_val=climate_all_variables_dic[v_][pix][w]

                        if not len(x_val) == window:  ##
                            continue
                        if len(x_val) == 0:
                            continue
                        if np.isnan(np.nanmean(x_val)):
                            continue
                        x_vals = T.interp_nan(x_val)

                        if x_vals[0] == None:
                            continue
                        df_new[v_] = x_vals
                        x_val_list_valid.append(v_)

                        # ???y?????????
                    if len(dic_y[pix]) != slices:
                        continue
                    val_y_variable = dic_y[pix][w]

                    if len(val_y_variable) == 0:
                        continue

                    val_y_variable = T.interp_nan(val_y_variable)
                    if val_y_variable[0]== None:
                        continue
                    if np.isnan(np.nanmean(val_y_variable)):
                        continue

                    val_y_variables = np.array(val_y_variable)
                    df_new['y']=val_y_variables
                    # T.print_head_n(df_new)


                    df_new = df_new.dropna(axis=1, how='all')
                    x_var_list_valid_new = []
     ############################
                    for v_ in x_val_list_valid:
                        if not v_ in df_new:
                            continue
                        else:
                            x_var_list_valid_new.append(v_)

                    r, c = pix
                    if r > 120:
                        continue
                    try:
                        partial_correlation = {}
                        partial_correlation_p_value = {}
                        for x in x_var_list_valid_new:

                            x_var_list_valid_new_cov = copy.copy(x_var_list_valid_new)
                            x_var_list_valid_new_cov.remove(x)
                            corr, p = self.partial_corr(df_new, x, 'y', x_var_list_valid_new_cov)
                            partial_correlation[x] = corr
                            partial_correlation_p_value[x] = p


                    except Exception as e:
                            print('error')
                        # print(x_val_list, val_y_variable)val_y_variables

                    partial_correlation_dic[pix] = partial_correlation
                    partial_p_value_dic[pix] = partial_correlation_p_value
                T.save_npy(partial_correlation_dic, outf + '_correlation')

                T.save_npy(partial_p_value_dic, outf + '_p_value')




    def partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        # print(df)
        df = df.dropna()
        # try:
        # print(x)
        # print(y)
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p


    def trend_window_LAI(self):  # ??????trend only for LAI4g ??????????????????????????????????????????

        period_list = ['early','peak','late']

        variable_list= ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                             'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai', 'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai',
                             'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai', 'Trendy_ensemble',]

        # variable_list=[ 'LAI3g_monthly']
        # variable_list=['LAI3g_monthly']
        window = 15
        slices = 39 - window
        time_range = '1982-2018'

        fdir_all = results_root + 'extract_relative_change_window/15_year_window_1982/Y/'

        for period in period_list:
            for variable in variable_list:

                f=f'{variable}_{period}_relative_change.npy'
                outdir = results_root + 'trend_window/trend_window/{}_during_{}_window{}_{}/'.format(time_range, period,window,variable)
                print(f)

                Tools().mk_dir(outdir, force=True)

                dic_climate = {}
                dic_i = dict(np.load(fdir_all  + f, allow_pickle=True, ).item())
                dic_climate.update(dic_i)

                for w in range(slices):

                    spatial_trend = {}
                    spatial_trend_p_value = {}

                    outf=outdir+f'{w:02d}_{variable}'
                    for pix in tqdm(dic_climate):
                        x_val = dic_climate[pix][w]
                        if not len(x_val) == window:  ##
                            continue
                        if len(x_val) == 0:
                            continue
                        if np.isnan(np.nanmean(x_val)):
                            continue

                        matix = np.isnan(x_val)  # ????????????time series ??????
                        matix = list(matix)
                        valid_number = matix.count(False)
                        # print(matix)

                        if valid_number / len(x_val) < 0.70:
                            continue

                        x_vals = T.interp_nan(x_val)

                        if x_vals[0] == None:
                            continue

                        try:
                            xaxis = range(len(x_vals))
                            # a, b, r = KDE_plot().linefit(xaxis, val)
                            r, p = stats.pearsonr(xaxis, x_vals)
                            k, b = np.polyfit(xaxis, x_vals, 1)


                        except Exception as e:
                            print(x_vals)
                            k = np.nan
                            p = np.nan

                        spatial_trend[pix] = k  # ???trend
                        spatial_trend_p_value[pix] = p

                    correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_trend)
                    correlation_arr = np.array(correlation_arr)
                    p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_trend_p_value)
                    p_value_arr = np.array(p_value_arr)
                    # trend_arr[trend_arr < -10] = np.nan
                    # trend_arr[trend_arr > 10] = np.nan

                    hist = []
                    for i in correlation_arr:
                        for j in i:
                            if np.isnan(j):
                                continue
                            hist.append(j)

                    # plt.hist(hist, bins=80)
                    # plt.figure()
                    # plt.imshow(correlation_arr, cmap='jet', vmin=-0.1, vmax=0.1)
                    # plt.title('')
                    # plt.colorbar()
                    # plt.show()

                    # #     # save arr to tif
                    DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_trend.tif')
                    DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
                    np.save(outf + '_trend.tif', correlation_arr)
                    np.save(outf + '_p_value.tif', p_value_arr)


    def mean_window(self):  # ????????? window ????????????

        period_list = ['early','peak','late']

        # variable_list= ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
        #                      'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai', 'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai',
        #                      'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai', 'Trendy_ensemble',]
        # variable_list=['LAI3g_daily', 'LAI3g_monthly']
        variable_list=['CO2', 'VPD','CCI_SM','Temp','PAR']

        window = 15
        slices = 39 - window
        time_range = '1982-2018'

        fdir_all = results_root + 'extract_relative_change_window/15_year_window_1982/X/'

        for period in period_list:
            for variable in variable_list:


                f=f'{variable}_{period}_relative_change.npy'
                outdir = results_root + 'mean_window/{}_during_{}_window{}_{}/'.format(time_range, period,window,variable)
                print(f)

                Tools().mk_dir(outdir, force=True)

                dic_climate = {}

                dic_i = dict(np.load(fdir_all  + f, allow_pickle=True, ).item())
                dic_climate.update(dic_i)

                for w in range(slices):
                    spatial_average = {}

                    outf=outdir+f'{w:02d}_{variable}'
                    for pix in tqdm(dic_climate):
                        x_val = dic_climate[pix][w]
                        if not len(x_val) == window:  ##
                            continue
                        if len(x_val) == 0:
                            continue
                        if np.isnan(np.nanmean(x_val)):
                            continue

                        matix = np.isnan(x_val)  # ????????????time series ??????
                        matix = list(matix)
                        valid_number = matix.count(False)
                        # print(matix)

                        if valid_number / len(x_val) < 0.70:
                            continue

                        x_vals = T.interp_nan(x_val)

                        if x_vals[0] == None:
                            continue

                        try:
                            window_average=np.nanmean(x_vals)



                        except Exception as e:
                            print(x_vals)
                            window_average = np.nan


                        spatial_average[pix] = window_average  # ???trend


                    correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_average)
                    correlation_arr = np.array(correlation_arr)

                    # trend_arr[trend_arr < -10] = np.nan
                    # trend_arr[trend_arr > 10] = np.nan

                    hist = []
                    for i in correlation_arr:
                        for j in i:
                            if np.isnan(j):
                                continue
                            hist.append(j)

                    # plt.hist(hist, bins=80)
                    # plt.figure()
                    # plt.imshow(correlation_arr, cmap='jet', vmin=-0.1, vmax=0.1)
                    # plt.title('')
                    # plt.colorbar()
                    # plt.show()

                    # #     # save arr to tif
                    DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_trend.tif')

                    np.save(outf + '_trend.tif', correlation_arr)




    def trend_window_trend(self):  # ??????window_trend_trend

        window = 15
        slices = 37 - window
        time_range = '1982-2018'


        periods = ['early', 'peak', 'late']
        for period in periods:

            fdir_X = result_root + f'trend_window/{time_range}_during_{period}_window15/'

            outdir = result_root + f'trend_window/trend_{time_range}_during_{period}_window15/'
            Tools().mk_dir(outdir, force=True)
            # exit()

            for f_X in tqdm(sorted(os.listdir(fdir_X))):
                if 'p_value' in f_X:
                    continue

                dic_climate = dict(np.load(fdir_X + f_X, allow_pickle=True, ).item())


                outf = outdir +f_X.split('.')[0]
                print(outf)
                # exit()

                # /////////////////////////////// ??????????????????/////////////////////////////
                spatial_dic = {}
                spatial_dic_p_value = {}
                spatial_dic_count = {}

                for pix in tqdm(dic_climate):
                    val = dic_climate[pix]
                    val = np.array(val)

                    val[val < -99999] = np.nan
                    if np.isnan(np.nanmean(val)):
                        continue
                    try:
                        xaxis = list(range(len(val)))
                        # a, b, r = KDE_plot().linefit(xaxis, val)
                        r, p = stats.pearsonr(xaxis, val)
                        k, b = np.polyfit(xaxis, val, 1)
                        # print(k)
                        spatial_dic_count[pix] = len(val)

                    except Exception as e:
                        k = np.nan
                        b = np.nan
                    spatial_dic[pix] = k  #
                    # spatial_dic[pix] = b  #
                    spatial_dic_p_value[pix] = p
                count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
                # plt.imshow(count_arr)
                # plt.colorbar()
                # plt.title(f_X)
                # plt.show()
                correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                correlation_arr = np.array(correlation_arr)
                p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
                p_value_arr = np.array(p_value_arr)
                # trend_arr[trend_arr < -10] = np.nan
                # trend_arr[trend_arr > 10] = np.nan

                hist = []
                for i in correlation_arr:
                    for j in i:
                        if np.isnan(j):
                            continue
                        hist.append(j)

                # plt.hist(hist, bins=80)
                # plt.figure()
                # plt.imshow(correlation_arr, cmap='jet', vmin=-0.1, vmax=0.1)
                # plt.title(variable_name_list[ii])
                # plt.colorbar()
                # plt.show()

                # #     # save arr to tif
                DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_trend.tif')
                DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
                np.save(outf + '', correlation_arr)
                np.save(outf + '', p_value_arr)

    def variables_contribution_window(self):  #  ???trend_window ????????? 19???-?????????????????????
        period = 'early'
        fdir = result_root + f'partial_window/1982-2000_during_{period}_window15/'

        var_name_list = [f'during_{period}_CCI_SM', f'during_{period}_CO2',
                        f'during_{period}_PAR',
                       f'during_{period}_VPD', f'during_{period}_Temp']

        window_list=list(range(0,24))

        for x in var_name_list:
            spatial_dic_time_series = {}
            ourdir = result_root + f'partial_window/conversion/1982-2000_during_{period}_{x}_partial_correlation_window15/'
            T.mk_dir(ourdir,force=True)
            for window in window_list:
                f = fdir + f'partial_correlation_{period}_1982-2000_window{window:02d}_correlation.npy'

                result_dic = T.load_npy(f)

                # color_map = dict(zip(x_var_list, list(range(len(x_var_list)))))
                # print(color_map)
                # exit()
                spatial_dic = {}

                for pix in result_dic:
                    if x not in result_dic[pix]:
                        continue
                    spatial_dic[pix] = result_dic[pix][x]
            #
                # spatial_dic_time_series[window]=spatial_dic

            # tif_template = '/Volumes/SSD_sumsang/project_greening/Data/NIRv_tif_05/1982-2018/198205.tif'
            # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_time_series[1])[:120]
            # # DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
            # cmap = sns.color_palette('hls',as_cmap=True)
            # plt.imshow(arr,cmap='jet')
            # plt.colorbar()
            # plt.show()

                outf_npy = f'{x}_{window:02d}_partial_correlation.npy'

                np.save(ourdir + outf_npy, spatial_dic)
            pass

    def conversion(self):  ## trend_window ??????????????????19?????????
        period_list = ['early', ]
        slice_list = list(range(0, 23))

        time = '1982-2020'
        for period in period_list:

            fdir = result_root + f'extract_relative_change_window/trend_window/1982-2020_during_{period}_window15/'
            outdir = result_root + f'extract_relative_change_window/trend_window/1982-2020_during_{period}_window15_conversion/'
            T.mk_dir(outdir,force=True)

            for f in (os.listdir(fdir)):
                outf_npy=f.split('.')[0]

                if not f.endswith('.npy'):
                    continue

                # val_dic = T.load_npy(fdir + f)
                val_array = np.load(fdir+f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)


                fname = f.split('.')[0]
                print(fname)

                spatial_dic={}
                pix_list=val_dic[0]

                for pix in pix_list:

                    val_list = []
                    for slice in slice_list:
                        if pix not in val_dic[slice]:
                            continue
                        val = val_dic[slice][pix]
                        if val < -99:
                            val_list.append(np.nan)

                        val_list.append(val)
                    if len(val_list) != 23:
                        continue
                    spatial_dic[pix]=val_list

                np.save(outdir + outf_npy, spatial_dic)
            pass

    def conversion_mean(self):  ## mean_window ??????????????????19?????????
        period_list = ['early','peak','late']
        slice_list = list(range(0, 24))
        variable_list = ['VPD','CCI_SM','CO2','PAR','Temp']
        time = '1982-2018'
        for period in period_list:
            for variable in variable_list:

                fdir = results_root + f'/mean_window/mean_window/{time}_during_{period}_window15_{variable}/'
                outdir = results_root + f'/mean_window/mean_window_conversion/{time}_during_{period}_window15_conversion_{variable}/'
                T.mk_dir(outdir,force=True)
                val_dic_list=[]


                for f in (os.listdir(fdir)):


                    if not f.endswith('.npy'):
                        continue

                    print(f)

                    # val_dic = T.load_npy(fdir + f)
                    val_array = np.load(fdir+f)
                    val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                    val_dic_list.append(val_dic)

                    spatial_dic={}

                for pix in val_dic_list[0]:

                    val_list = []
                    for slice in slice_list:
                        if pix not in val_dic_list[slice]:
                            continue
                        val = val_dic_list[slice][pix]
                        if val < -99:
                            val_list.append(np.nan)

                        val_list.append(val)
                    if len(val_list) != 24:
                        continue
                    spatial_dic[pix] = val_list

                np.save(outdir + f'{period}_{variable}_15_window_mean', spatial_dic)
            pass

    def conversion_trendy(self):  ## trend_window ??????????????????19?????????
        period_list = ['early','peak','late']
        slice_list = list(range(0, 24))
        variable_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                         'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai', 'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai',
                         'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai', 'Trendy_ensemble']
        time = '1982-2018'
        for period in period_list:
            for variable in variable_list:

                fdir = results_root + f'/trend_window/trend_window/{time}_during_{period}_window15_{variable}/'
                outdir = results_root + f'/trend_window/Trend_window_conversion/{time}_during_{period}_window15_conversion_{variable}/'
                T.mk_dir(outdir,force=True)
                val_dic_list=[]


                for f in (os.listdir(fdir)):


                    if not f.endswith('.npy'):
                        continue
                    # if 'p_value' in f:
                    #     continue
                    if 'trend' in f:
                        continue
                    print(f)

                    # val_dic = T.load_npy(fdir + f)
                    val_array = np.load(fdir+f)
                    val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                    val_dic_list.append(val_dic)

                    spatial_dic={}

                for pix in val_dic_list[0]:

                    val_list = []
                    for slice in slice_list:
                        if pix not in val_dic_list[slice]:
                            continue
                        val = val_dic_list[slice][pix]
                        if val < -99:
                            val_list.append(np.nan)

                        val_list.append(val)
                    if len(val_list) != 24:
                        continue
                    spatial_dic[pix] = val_list

                np.save(outdir + f'{period}_{variable}_15_window_p_value', spatial_dic)
            pass


    def save_moving_window_correlation(self):
        time_range={}

        outdir = result_root + 'trend_window/{}_during_early_window15/'
        Tools().mk_dir(outdir, force=True)
        fdir_all_seasons=result_root+'trend_window/save_result/'
        # fdir=result_root+'univariate_correlation/during_{}/15_year_window/'.format(period)
        dic_mean = {}
        for fdir_1 in tqdm(sorted(os.listdir(fdir_all_seasons))):
            # print(fdir_1)

            for fdir_2 in sorted(os.listdir(fdir_all_seasons + '/'+fdir_1 )):
                # print(fdir_2)
                # exit()

                mean_array_list = []
                CI_arry_list = []
                for f in sorted(os.listdir(fdir_all_seasons + '/'+fdir_1 +'/'+ fdir_2+'/')):

                    # print(f)
                    # exit()
                    if not 'LAI4g' in f:
                        continue
                    if not f.endswith('correlation.npy'):
                        continue
                    array = np.load(fdir_all_seasons + '/'+fdir_1 +'/'+ fdir_2+'/'+f)  # ????????????????????????
                    # print(fdir_all_seasons + '/'+fdir_1 +'/'+ fdir_2+'/'+f)

                    array[array<-99]=np.nan
                    array=array[:120]

                    mean_array=np.nanmean(array)

                    array_flatten=array.flatten()
                    array_valid=[]
                    for j in array_flatten:
                        if np.isnan(j):
                            continue
                        array_valid.append(j)
                    # print(array_valid)

                    mean_array_list.append(mean_array)
                    n = len(array_valid)
                    se = stats.sem(array_valid)
                    h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
                # print(h)
                    CI_arry_list.append(h)
                mean_array_list=np.array(mean_array_list)
                CI_arry_list=np.array(CI_arry_list)
                # print(len(mean_array_list))
                fname=fdir_2.split('_')[0]+'_'+fdir_2.split('_')[1]+'_'+fdir_2.split('_')[2]
                print(fname)

                dic_mean[fname]=[mean_array_list,CI_arry_list]

        np.save(outdir +'plot_moving_{}'.format(time_range),dic_mean)

    def save_moving_window_multi_regression(self):
        period='early'

        HI_ratio_dic = Build_trend_dataframe().P_PET_ratio(data_root+'original_dataset/aridity_P_PET_dic/')
        HI_reclass_dic=Build_trend_dataframe().P_PET_reclass(HI_ratio_dic)
        HI_reclass_dic={'HI_reclass':HI_reclass_dic}
        HI_reclass_df=T.spatial_dics_to_df(HI_reclass_dic)
        HI_reclass_df.loc[HI_reclass_df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Non Humid'
        HI_reclass_list=T.get_df_unique_val_list(HI_reclass_df,'HI_reclass')
        HI_reclass_dic_reverse = {}
        for reclass in HI_reclass_list:
            df_selected=HI_reclass_df[HI_reclass_df['HI_reclass']==reclass]
            pix_list=df_selected['pix'].to_list()
            pix_list = set(pix_list)
            HI_reclass_dic_reverse[reclass] = pix_list
        outdir = result_root + 'partial_window/plot_moving_window_partial_correlation/'
        Tools().mk_dir(outdir, force=True)
        fdir_all_seasons=result_root+'partial_window/1982-2015_during_{}_window15/'.format(period)


        # var_name_list=[f'during_{period}_CCI_SM.npy', f'during_{period}_CO2.npy', f'during_{period}_PAR.npy', f'during_{period}_VPD.npy', f'during_{period}_temperature.npy']
        var_name_list=['CCI_SM', 'CO2', 'PAR', 'VPD', 'temperature']
        # var_name_list = [f'during_{period}_Aridity', f'during_{period}_CCI_SM', f'during_{period}_CO2', f'during_{period}_GIMMS_NDVI',f'during_{period}_NIRv', f'during_{period}_PAR',
        #                  f'during_{period}_Precip',  f'during_{period}_SPEI3',
        #                  f'during_{period}_VPD', f'during_{period}_temperature']

        for HI_reclass in HI_reclass_dic_reverse:
            selected_pix = HI_reclass_dic_reverse[HI_reclass]
            average_contribution_dic ={}

            for x in var_name_list:
                mean_array_list = []
                CI_arry_list = []

                for f in tqdm(sorted(os.listdir(fdir_all_seasons))):
                    # print(fdir_1)
                    if 'p_value' in f:
                        continue
                    vals_list = []
                    result_dic = T.load_npy(fdir_all_seasons + f)  # ????????????????????????
                    mean_array=0
                    for pix in result_dic:
                        if not pix in selected_pix:
                            continue
                        r, c = pix
                        if r > 120:
                            continue
                        dic_i=result_dic[pix]
                        if not x in dic_i:
                            continue
                        vals = result_dic[pix][x]
                        vals_list.append(vals)

                    mean_array=np.nanmean(vals_list)
                    std_array = np.nanstd(vals_list)
                    vals_list=np.array(vals_list)

                    up=mean_array+std_array
                    bottom=mean_array-std_array
                    array_valid = []

                    for j in vals_list:
                        if np.isnan(j):
                            continue
                        if j>up:
                            continue
                        if j<bottom:
                            continue
                        array_valid.append(j)
                    # print(array_valid)

                    mean_array_list.append(np.nanmean(array_valid))
                    n = len(array_valid)
                    se = stats.sem(array_valid)
                    h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
                    CI_arry_list.append(h)

                mean_array_list=np.array(mean_array_list)
                CI_arry_list=np.array(CI_arry_list)
                print(len(mean_array_list))
                print(x)
                average_contribution_dic[x]=[mean_array_list,CI_arry_list]

            np.save(outdir +'moving_partial_correlation_{}_{}'.format(period,HI_reclass),average_contribution_dic)


    def plot_moving_window_correlation(self):

        dic=dict(np.load(result_root+'partial_window/plot_moving_window_partial_correlation/moving_partial_correlation_peak_Non Humid.npy',allow_pickle=True,).item())
        print(dic)

        early_key_list=[]
        peak_key_list = []
        late_key_list = []
        season_list=[]
        for key in dic:
            if 'early' in key:
                early_key_list.append(key)
            elif 'peak' in key:
                peak_key_list.append(key)
            elif 'late' in key:
                late_key_list.append(key)
            else:
                continue
        early_key_list.sort()
        peak_key_list.sort()
        peak_key_list.sort()

        # print(early_key_list)
        # print(peak_key_list)
        # print(late_key_list)
        season_list=[early_key_list,peak_key_list,late_key_list]
        plt.figure()
        for season in season_list:

            flag = 1
            for key in season:
                plt.subplot(1, 4, flag)
                val=dic[key]
                print(val)
                mean_array_list=val[0]
                CI_array_list=val[1]
                plt.plot(range(len(mean_array_list)),mean_array_list,zorder=99)
                plt.fill_between(range(len(mean_array_list)),y1=mean_array_list-CI_array_list,y2=mean_array_list+CI_array_list,alpha=0.3)
                plt.title(key)
                flag=flag+1
                plt.title('non-humid')
        plt.show()



    def simple_linear_regression(self): # detrend_correlation and trend correlation
        period = 'peak'
        time='1982-1998'
        fdir = result_root + 'extraction_original_val/{}_original_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time,time,period)
        fdir_Y=result_root+'%NDVI/'

        outdir = result_root + 'simple_linear_regression_withtrend/during_{}_{}/'.format(period,time)
        Tools().mk_dir(outdir,force=True)
        # exit()

        spatial_dic = {}
        spatial_dic_p_value = {}
        spatial_dic_count = {}
        for Y_variables in tqdm(sorted(os.listdir(fdir_Y))):
            if Y_variables != '{}_{}.npy'.format(time,period):
                continue

            dic_y = dict(np.load(fdir_Y + Y_variables, allow_pickle=True, ).item())

            # climate_variables = {}
            for X_variable in tqdm(sorted(os.listdir(fdir))):

                if X_variable == 'during_{}_GIMMS_NDVI.npy'.format(period):
                    continue
                if X_variable == 'during_{}_CSIF_fpar.npy'.format(period):
                    continue
                if X_variable == 'during_{}_NIRv.npy'.format(period):
                    continue
                if X_variable == 'during_{}_VOD.npy'.format(period):
                    continue

                dic_climate = dict(np.load(fdir + X_variable, allow_pickle=True, ).item())
                variable_name = X_variable
                print(variable_name)
                # climate_variables[variable_name] = dic_climate
                # print(len(dic_climate))
                outf=outdir+X_variable.split('.')[0]+'_GIMMS_NDVI'
                print(outf)
                # exit()
                for pix in tqdm(dic_y):
                    if pix not in dic_climate:
                        continue
                    val_climate = dic_climate[pix]
                    # print(val_climate)
                    # exit()

                    val_y = dic_y[pix]

                    if len(val_climate) != 17:
                        continue
                    if len(val_y) != 17:
                        continue
                    val_climate = np.array(val_climate)
                    val_CSIF_par = np.array(val_y)
                    val_climate[val_climate < -99] = np.nan
                    val_CSIF_par[val_CSIF_par < -99] = np.nan
                    if np.isnan(np.nanmean(val_climate)):
                        continue
                    if np.isnan(np.nanmean(val_CSIF_par)):
                        continue
                    try:
                        xaxis=list(range(len(val_climate)))
                        # a, b, r = KDE_plot().linefit(val_climate, val_y)
                        # r, p = stats.pearsonr(val_climate, val_y)
                        k, b = np.polyfit(val_climate, val_y, 1)
                        # print(k)
                        # spatial_dic_count[pix] = len(val_climate)

                    except Exception as e:
                        print(val_climate, val_CSIF_par)
                        r = np.nan
                        p = np.nan
                    # spatial_dic[pix] = r  #
                    spatial_dic[pix] = k  #
                    # spatial_dic_p_value[pix] = p
                # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
                # plt.imshow(count_arr)
                # plt.colorbar()
                # plt.show()
                correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                correlation_arr = np.array(correlation_arr)
                p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
                p_value_arr = np.array(p_value_arr)
                # trend_arr[trend_arr < -10] = np.nan
                # trend_arr[trend_arr > 10] = np.nan

                hist = []
                for i in correlation_arr:
                    for j in i:
                        if np.isnan(j):
                            continue
                        hist.append(j)

                # plt.hist(hist, bins=80)
                # plt.figure()
                # plt.imshow(correlation_arr, cmap='jet', vmin=-0.5, vmax=0.5)
                # plt.title(f_X.split('.')[0])
                # plt.colorbar()
                # plt.show()

                # #     # save arr to tif
                DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_beta.tif')

                np.save(outf + '_beta', correlation_arr)

        pass

    def simple_correlation(self): #
        periods = ['early','peak','late']
        time='2000-2018'
        # fdir = result_root + 'extraction_original_val/{}_original_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time,time,period)
        fdir_X=results_root+f'/zscore/2000-2018_X/'
        fdir_Y = results_root + f'/zscore/2000-2018_Y/'
        outdir = result_root + 'zscore_simple_correlation/during_{}/'.format(time)
        Tools().mk_dir(outdir, force=True)

        # exit()

        spatial_dic = {}
        spatial_dic_p_value = {}
        spatial_dic_count = {}
        for period in periods:

            for Y_variable in tqdm(sorted(os.listdir(fdir_Y))):
                # if Y_variables != 'during_{}_LAI_GIMMS.npy'.format(period):
                #     continue
                if Y_variable!=f'LAI4g_{period}_zscore.npy':
                    continue

                dic_y = dict(np.load(fdir_Y + Y_variable, allow_pickle=True, ).item())
                dic_GIMMS={}
                for pix in dic_y:
                    val = dic_y[pix]
                    val = np.array(val)

                    dic_GIMMS[pix] = val


                # climate_variables = {}
                for X_variable in tqdm(sorted(os.listdir(fdir_X))):



                    dic_climate = dict(np.load(fdir_X + X_variable, allow_pickle=True, ).item())
                    variable_name = X_variable
                    print(variable_name)
                    # climate_variables[variable_name] = dic_climate
                    # print(len(dic_climate))
                    outf=outdir+X_variable.split('.')[0]+'_LAI4g'
                    print(outf)
                    # exit()
                    for pix in tqdm(dic_GIMMS):
                        if pix not in dic_climate:
                            continue
                        val_climate = dic_climate[pix]
                        # print(val_climate)
                        # exit()

                        val_y = dic_GIMMS[pix]


                        if len(val_climate) != 19:
                            continue

                        if len(val_y) != 19:
                            continue
                        val_climate = np.array(val_climate)
                        val_CSIF_par = np.array(val_y)
                        val_climate[val_climate < -99] = np.nan
                        val_CSIF_par[val_CSIF_par < -99] = np.nan
                        if np.isnan(np.nanmean(val_climate)):
                            continue
                        if np.isnan(np.nanmean(val_CSIF_par)):
                            continue
                        try:


                            r, p = T.nan_correlation(val_climate, val_CSIF_par)

                            # print(k)
                            # spatial_dic_count[pix] = len(val_climate)

                        except Exception as e:
                            print(val_climate, val_CSIF_par)
                            r = np.nan
                            p = np.nan
                        spatial_dic[pix] = r  #

                    # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
                    # plt.imshow(count_arr)
                    # plt.colorbar()
                    # plt.show()
                    correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                    correlation_arr = np.array(correlation_arr)
                    p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
                    p_value_arr = np.array(p_value_arr)
                    # trend_arr[trend_arr < -10] = np.nan
                    # trend_arr[trend_arr > 10] = np.nan

                    # hist = []
                    # for i in correlation_arr:
                    #     for j in i:
                    #         if np.isnan(j):
                    #             continue
                    #         hist.append(j)

                    # plt.hist(hist, bins=80)
                    # plt.figure()
                    # plt.imshow(correlation_arr, cmap='jet', vmin=-0.5, vmax=0.5)
                    # # plt.title(f_X.split('.')[0])
                    # plt.colorbar()
                    # plt.show()

                    # #     # save arr to tif
                    DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_r.tif')
                    DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p.tif')

                    np.save(outf + '_r', correlation_arr)
                    np.save(outf + '_p', p_value_arr)

        pass

    def plot_simple_moving_window_correlation(self):

        time_range = {}

        # outdir = result_root + 'trend_window/{}_during_early_window15/'
        # Tools().mk_dir(outdir, force=True)
        # fdir_all_seasons = result_root + '/univariate_correlation_window_detrend/1982-2018_during_early_window20/'
        fdir_all_seasons=result_root+'univariate_correlation_window/1982-2018_during_late_window20/'
        dic_mean = {}
        for fdir_1 in tqdm(sorted(os.listdir(fdir_all_seasons))):
            # print(fdir_1)

                mean_array_list = []
                CI_arry_list = []
                for f in sorted(os.listdir(fdir_all_seasons + '/' + fdir_1 )):
                    if not 'correlation' in f:
                        continue
                    if  'tif' in f:
                        continue
                    array = np.load(fdir_all_seasons + '/' + fdir_1 + '/'  + '/' + f)  # ????????????????????????
                    # print(fdir_all_seasons + '/'+fdir_1 +'/'+ fdir_2+'/'+f)

                    array[array < -99] = np.nan
                    array = array[:120]

                    mean_array = np.nanmean(array)

                    array_flatten = array.flatten()
                    array_valid = []
                    for j in array_flatten:
                        if np.isnan(j):
                            continue
                        array_valid.append(j)
                    # print(array_valid)

                    mean_array_list.append(mean_array)
                    n = len(array_valid)
                    se = stats.sem(array_valid)
                    h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
                    # print(h)
                    CI_arry_list.append(h)
                mean_array_list = np.array(mean_array_list)
                CI_arry_list = np.array(CI_arry_list)
                # print(len(mean_array_list))
                # fname = fdir_2.split('_')[0] + '_' + fdir_2.split('_')[1] + '_' + fdir_2.split('_')[2]
                # print(fname)

                plt.plot(mean_array_list)
                plt.title(fdir_1)
                plt.show()


                # dic_mean[fname] = [mean_array_list, CI_arry_list]

        # np.save(outdir + 'plot_moving_{}'.format(time_range), dic_mean)

    def difference_correlation_between_two_period(self): #
        variable_list=['CO2','PAR','SPEI3','VPD','temperature','CCI_SM']
        period = 'peak'
        time_list=['1982-2001','2002-2018']
        for variable in variable_list:
            array_two_period = []
            outdir = result_root + 'test_original/during_{}_difference/'.format(period)
            Tools().mk_dir(outdir, force=True)

            for time in time_list:

                fdir = results_root+'test_original/during_{}_{}/'.format(period,time)

                f=f'during_{period}_{variable}_LAI_GIMMS_r.npy'
                print(f)
                outf = outdir +f'{period}_{variable}_correlation_difference'
                # print(outf)

                val_array = np.load(fdir + f)
                val_array[val_array <-99] = np.nan


                array_two_period.append(val_array)
            difference_array=array_two_period[1]-array_two_period[0]
            # difference_array[difference_array ] = np.nan


            DIC_and_TIF().arr_to_tif(difference_array, outf + '.tif')

            np.save(outf + '', difference_array)


            pass

    def save_trend(self): #### 11/12/2021 # ????????????????????????????????????????????????????????????????????????mask?????????

        outdir = result_root + 'save_trend/three_season_results/'
        Tools().mk_dir(outdir, force=True)
        fdir_all_seasons=result_root+'/trend_calculation/'

        dic_trend_P = {}

        for fdir_1 in tqdm(sorted(os.listdir(fdir_all_seasons))):
            print(fdir_1)
            var_name_trend = []
            var_name_p_value = []

            for f in sorted(os.listdir(fdir_all_seasons + '/' + fdir_1)):
                # print(f)
                if f.endswith('trend.npy'):
                    var_name_trend.append(f)
                if f.endswith('p_value.npy'):
                    var_name_p_value.append(f)

            for i in range(len(var_name_p_value)):
                p_file_name=var_name_p_value[i]
                trend_file_name = var_name_trend[i]

                array_p = np.load(fdir_all_seasons + '/'+fdir_1+'/'+p_file_name)
                array_p=np.array(array_p)
                print(fdir_all_seasons + '/'+fdir_1+'/'+p_file_name)

                array_trend=np.load(fdir_all_seasons + '/'+fdir_1+'/'+trend_file_name)
                array_trend = np.array(array_trend)

                # print(np.shape(array_p))
                # print(np.shape(array_trend))
                array_p[array_p<-99]=np.nan
                array_p=array_p[:120]

                array_trend = array_trend[:120]
                mask_p_value=array_p<0.1
                array_trend[array_trend < -99] = np.nan
                array_trend[~mask_p_value]=np.nan

                mean_array_trend=np.nanmean(array_trend)


    def save_results_for_three_seasons(self):
        outdir=result_root+'save_results_for_three_seasons/'  # anomaly /original
        Tools().mk_dir(outdir, force=True)
        fdir_all_seasons = result_root + 'extraction_anomaly_val/1982-2015_anomaly_extraction_all_seasons/'
        year_list_1 = list(range(1982, 2016))
        year_list_2 = list(range(1988, 2016))
        # print(year_list)
        mean_variable={}
        for fdir_1 in tqdm(sorted(os.listdir(fdir_all_seasons))):
            print(fdir_1)
            if fdir_1.startswith(('.')):
                continue

            for f in sorted(os.listdir(fdir_all_seasons + fdir_1)):
                if f.startswith(('.')):
                    continue
                print(f)
                if 'VOD' in f:
                    year_list=year_list_2
                else:
                    year_list=year_list_1

                dic = dict(np.load(fdir_all_seasons + fdir_1 + '/' + f, allow_pickle=True, ).item())
                mean_list=[]
                for ii in range(len(year_list)):
                    spatial_list = []

                    for pix in dic:
                        r, c = pix
                        val = dic[pix]
                        if r > 120:
                            continue
                        if len(val) == 0:
                            continue
                        if np.nanmean(val) < -99:
                            continue
                        spatial_list.append(val[ii])
                    mean_val = np.nanmean(spatial_list)
                    # print(mean_val)
                    mean_list.append(mean_val)
                f=f.split('.')[0]
                # print(f)
                mean_variable[f]=mean_list
            print(len(mean_variable))

            np.save(outdir + 'save_anomaly_for_all', mean_variable)  # ???????????????????????????????????????????????????34 ?????????

    def plot_anomaly_for_three_seasons(self):
        f=result_root+'save_results_for_three_seasons/save_anomaly_for_all.npy'
        year_list_1=list(range(1982,2016))
        year_list_2 = list(range(1988, 2016))

        # print(year_list_1)

        dic = dict(np.load(f,allow_pickle=True, ).item())
        early_key_list = []
        peak_key_list = []
        late_key_list = []

        for key in dic:
            if 'early' in key:
                early_key_list.append(key)
            elif 'peak' in key:
                peak_key_list.append(key)
            elif 'late' in key:
                late_key_list.append(key)
            else:
                continue
        early_key_list.sort()
        peak_key_list.sort()
        peak_key_list.sort()

        print(early_key_list)
        print(peak_key_list)
        print(late_key_list)

        season_list = [early_key_list, peak_key_list, late_key_list]
        print(season_list)
        # exit()
        plt.figure()
        color_list=['g','r','b']
        for i in range(len(season_list)):
            season=season_list[i]
            print(season)
            color=color_list[i]

            flag = 1
            for key in season:
                if 'VOD' in key:
                    year_list=year_list_2
                else:
                    year_list=year_list_1
                plt.subplot(4, 3, flag)
                val = dic[key]
                print(len(val))
                mean_array_list = val

                plt.plot(year_list, mean_array_list, zorder=99,color=color)

                plt.title(key)
                flag = flag + 1
        plt.legend()
        plt.show()


    def mean_calculation(self):  # ?????????????????????
        period = 'early'
        time = '2000-2018'
        fdir_X = results_root + f'extraction_original_val/{time}_Trendy/'
        # exit()
        # fdir_Y = result_root + 'extraction_anomaly_val/{}_during_{}_growing_season/Y_{}/'.format(time,period,time)

        outdir = results_root + 'mean_calculation_original/during_{}_{}/'.format(period, time)
        Tools().mk_dir(outdir, force=True)
        # exit()
        all_list = []
        variable_name_list = []

        for f_X in tqdm(sorted(os.listdir(fdir_X))):

            dic_climate = dict(np.load(fdir_X + f_X, allow_pickle=True, ).item())

            all_list.append(dic_climate)
            variable_name_list.append(f_X)

            # print(variable_name_list)
            # print(len(all_list))

        for ii in range(len(variable_name_list)):
            outf = outdir + variable_name_list[ii].split('.')[0]
            print(outf)

            # /////////////////////////////// ??????????????????/////////////////////////////
            spatial_dic = {}

            spatial_dic_count = {}

            for pix in tqdm(all_list[ii]):
                val = all_list[ii][pix]
                val = np.array(val)

                val[val < -99999] = np.nan
                if np.isnan(np.nanmean(val)):
                    continue
                try:

                    # spatial_dic[pix] = np.mean(val[-5:])
                    # spatial_dic[pix] = np.mean(val[:5]) # ?????????????????????
                    spatial_dic[pix] = np.mean(val)

                    spatial_dic_count[pix] = len(val)

                except Exception as e:
                   print('error')

            # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(count_arr)
            # plt.colorbar()
            # plt.title(variable_name_list[ii])
            # plt.show()
            mean_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            mean_arr = np.array(mean_arr)


            # hist = []
            # for i in mean_arr:
            #     for j in i:
            #         if np.isnan(j):
            #             continue
            #         hist.append(j)

            # plt.hist(hist, bins=80)
            # plt.figure()
            # plt.imshow(correlation_arr, cmap='jet', vmin=-0.1, vmax=0.1)
            # plt.title(variable_name_list[ii])
            # plt.colorbar()
            # plt.show()

            # #     # save arr to tif
            DIC_and_TIF().arr_to_tif(mean_arr, outf + '_mean.tif')

            np.save(outf + '_mean', mean_arr)

        pass

    def CV_calculation(self):
        period = 'winter'
        time = '1982-2015'
        fdir_X = result_root + 'extraction_original_val/{}_original_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(
            time, time, period)
        # print(fdir_X)
        # exit()
        # fdir_Y = result_root + 'extraction_anomaly_val/{}_during_{}_growing_season/Y_{}/'.format(time,period,time)

        outdir = result_root + 'CV_calculation_original/during_{}_{}/'.format(period, time)
        Tools().mk_dir(outdir, force=True)
        # exit()
        all_list = []
        variable_name_list = []

        for f_X in tqdm(sorted(os.listdir(fdir_X))):


            dic_climate = dict(np.load(fdir_X + f_X, allow_pickle=True, ).item())

            all_list.append(dic_climate)
            variable_name_list.append(f_X)

            # print(variable_name_list)
            # print(len(all_list))

        for ii in range(len(variable_name_list)):
            outf = outdir + variable_name_list[ii].split('.')[0]
            print(outf)

            # /////////////////////////////// ??????????????????/////////////////////////////
            spatial_dic = {}
            spatial_dic_p_value = {}
            spatial_dic_count = {}

            for pix in tqdm(all_list[ii]):
                val = all_list[ii][pix]
                val = np.array(val)

                val[val < -99999] = np.nan
                if np.isnan(np.nanmean(val)):
                    continue
                try:

                    spatial_dic[pix]=(np.std(val)/np.mean(val))*100


                    spatial_dic_count[pix] = len(val)

                except Exception as e:
                   print('error')

            # count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            # plt.imshow(count_arr)
            # plt.colorbar()
            # plt.title(variable_name_list[ii])
            # plt.show()
            CV_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            CV_arr = np.array(CV_arr)


            # hist = []
            # for i in mean_arr:
            #     for j in i:
            #         if np.isnan(j):
            #             continue
            #         hist.append(j)

            # plt.hist(hist, bins=80)
            # plt.figure()
            # plt.imshow(correlation_arr, cmap='jet', vmin=-0.1, vmax=0.1)
            # plt.title(variable_name_list[ii])
            # plt.colorbar()
            # plt.show()

            # #     # save arr to tif
            DIC_and_TIF().arr_to_tif(CV_arr, outf + '_CV.tif')

            np.save(outf + '_CV', CV_arr)

        pass



    def trend_calculation(self):

        # time = '1982-2015'
        # variable='GIMMS_NDVI'
        periods = ['early', 'peak', 'late']

        for period in periods:

            fdir_X = results_root + f'zscore/2000-2018_daily/2000-2018_Y/'
            outdir = result_root + f'trend_zscore/2000-2018_daily/2000-2018_Y/'
            # lc_list = ['water', 'grass', 'shrub', 'crop', 'EBF', 'ENF', 'DBF', 'DNF', 'savanna', 'urban', 'nonveg']
            # for lc in lc_list:
            #     fdir_X=data_root+f'original_dataset/landcover/{lc}_dic/'
            #     outdir = result_root + f'lc_trend/'

            # fdir_X = result_root + f'Pierre_relative_change/2000-2018_Y/'
            # outdir = result_root + f'trend_relative_change/2000-2018_Y/'

            # fdir_X = result_root + 'Pierre_relative_change/1982-/'
            # outdir = result_root + f'trend_calculation_relative_change_1982_/'
            Tools().mk_dir(outdir, force=True)

            dic_climate = {}
            spatial_dic = {}
            spatial_dic_p_value = {}
            spatial_dic_count = {}

            for f_X in tqdm(sorted(os.listdir(fdir_X))):

                dic_i = dict(np.load(fdir_X + f_X, allow_pickle=True, ).item())
                dic_climate.update(dic_i)

                # dic_climate = dict(np.load(fdir_X + f_X, allow_pickle=True, ).item())

                # outf=outdir+f_X.split('.')[0]
                # split1 = f_X.split('.')[0].split('_')[0:]
                # split2='_'.join(split1)
                outf = outdir + f_X.split('.')[0]
                # outf=outdir+split2
                print(outf)
                # exit()

                # /////////////////////////////// ??????????????????/////////////////////////////

                for pix in tqdm(dic_climate):
                    val = dic_climate[pix]
                    val = np.array(val)

                    val[val < -99999] = np.nan
                    if np.isnan(np.nanmean(val)):
                        continue
                    try:
                        xaxis = list(range(len(val)))
                        # a, b, r = KDE_plot().linefit(xaxis, val)
                        r, p = stats.pearsonr(xaxis, val)
                        k, b = np.polyfit(xaxis, val, 1)
                        # print(k)
                        spatial_dic_count[pix] = len(val)
                        spatial_dic[pix] = k  #
                        # spatial_dic[pix] = b  #
                        spatial_dic_p_value[pix] = p

                    except Exception as e:
                        k = np.nan
                        b = np.nan

                count_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
                # plt.imshow(count_arr)
                # plt.colorbar()
                # plt.title(variable_name_list[ii])
                # plt.show()
                correlation_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
                correlation_arr = np.array(correlation_arr)
                p_value_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_p_value)
                p_value_arr = np.array(p_value_arr)
                # trend_arr[trend_arr < -10] = np.nan
                # trend_arr[trend_arr > 10] = np.nan

                hist = []
                for i in correlation_arr:
                    for j in i:
                        if np.isnan(j):
                            continue
                        hist.append(j)

                # plt.hist(hist, bins=80)
                # plt.figure()
                # plt.imshow(correlation_arr, cmap='jet', vmin=-0.2, vmax=0.2)
                # plt.imshow(p_value_arr, cmap='jet', vmin=0, vmax=0.1)
                # plt.title('')
                # plt.colorbar()
                # plt.show()

                # #     # save arr to tif
                DIC_and_TIF().arr_to_tif(correlation_arr, outf + '_trend.tif')
                DIC_and_TIF().arr_to_tif(p_value_arr, outf + '_p_value.tif')
                np.save(outf + '_trend', correlation_arr)
                np.save(outf + '_p_value', p_value_arr)

    def max_trend_among_all_variables(self):

        fdir1 = result_root + f'/lc_trend/'
        outdir = result_root + f'lc_trend/'

        Tools().mk_dir(outdir)

        all_arr_climate=[]
        all_arr_p_value = []

        for f1 in tqdm(sorted(os.listdir(fdir1))):
            if not f1.endswith('_trend.tif'):
                continue
            print(f1)
            arr_climate_pre,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir1 + f1)
            all_arr_climate.append(arr_climate_pre)

        for f2 in tqdm(sorted(os.listdir(fdir1))):
            if not f2.endswith('_p_value.tif'):
                continue
            print(f2)
            arr_p_value_pre,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir1 + f2)
            all_arr_p_value.append(arr_p_value_pre)


        max_arr=np.ones_like(all_arr_climate[0])*np.nan
        p_value_arr=np.ones_like(all_arr_climate[0])*np.nan
        print(len(all_arr_climate[0]))
        print(len(all_arr_climate[0][0]))

        for r in range(len(all_arr_climate[0])):
            for c in range(len(all_arr_climate[0][0])):
                trend_value_list = []

                p_value_list = []
                for climate in all_arr_climate:

                    trend_value_list.append(abs(climate[r][c]))
                trend_value_array=np.array(trend_value_list)
                trend_value_array[trend_value_array>99]=np.nan

                # matrix=np.isnan(trend_value_array)
                # if True in matrix:
                #     continue
                max_arr[r][c] = np.nanmax(trend_value_array)   # per_year %*20 years in total
                print(max_arr[r][c])
                # exit()

        plt.figure()

        plt.imshow(p_value_arr,cmap='jet', vmin=-1, vmax=1)
        plt.figure()
        plt.imshow(max_arr, cmap='jet', vmin=0, vmax=3)
        # plt.title(outf)
        plt.colorbar()
        plt.show()
        #
            # save arr to tif
        # DIC_and_TIF().arr_to_tif(correlation_matrix, outdir + 'max_correlation.tif')
        # np.save(outdir + 'max_correlation', correlation_matrix)
        DIC_and_TIF().arr_to_tif(max_arr, outdir + 'max_trend.tif')
        np.save(outdir + 'max_trend', max_arr)
        # DIC_and_TIF().arr_to_tif(p_value_arr, outdir + 'max_p_value.tif')
        # np.save(outdir + 'max_p_value', max_correlation)




    def mask_with_p_trend_correlation(self):
            period = 'late'
            p_tif = result_root + 'trend/CSIF_par_trend_new/during_{}_CSIF_par_p_value.tif'.format(period)
            fdir_tif = result_root + 'trend_correlation/mask_with_p_original/during_{}/'.format(period)
            threshold = 0.05
            outmask_dir = result_root + 'trend_correlation/mask_with_p_tif/during_{}/'.format(period)
            T.mk_dir(outmask_dir, force=True)

            p_arr = to_raster.raster2array(p_tif)[0]
            matrix = p_arr > threshold

            # matrix[p_arr>threshold]=True
            print(matrix)

            for mask_tif in os.listdir(fdir_tif):
                if not mask_tif.endswith('correlation.tif'):
                    continue
                mask_arr = to_raster.raster2array(fdir_tif + mask_tif)[0]
                # plt.imshow(mask_arr)
                # plt.show()
                mask_arr[matrix] = np.nan
                outf = mask_tif.split('.')[0]
                print(outf)
                DIC_and_TIF().arr_to_tif(mask_arr, outmask_dir + 'mask_p_0.05_' + outf + '.tif')

    def mask_with_p_trend_trend(self):
        period = 'late'
        p_tif = result_root + 'trend/CSIF_par_trend_new/during_{}_CSIF_par_p_value.tif'.format(period)
        mask_arr_tif = result_root + 'trend/CSIF_par_trend_new/during_{}_CSIF_par_trend.tif'.format(period)
        threshold = 0.05
        outmask_dir = result_root + 'trend/mask_with_p_tif/'
        T.mk_dir(outmask_dir, force=True)

        p_arr = to_raster.raster2array(p_tif)[0]
        matrix = p_arr > threshold
        # matrix[p_arr>threshold]=True
        print(matrix)

        mask_arr = to_raster.raster2array(mask_arr_tif)[0]

        # plt.imshow(mask_arr)
        # plt.show()
        mask_arr[matrix] = np.nan
        mask_arr[mask_arr < -999] = np.nan
        print(len(mask_arr[0]))

        # plt.imshow(mask_arr)
        # plt.show()

        outf = (mask_arr_tif.split('.')[0]).split('/')[-1]

        DIC_and_TIF().arr_to_tif(mask_arr, outmask_dir + 'mask_p_0.05_' + outf + '.tif')

    def mask_with_p_trend_max_contribution(self):
        period = 'early'
        p_tif = result_root + 'trend/CSIF_par_trend_new/during_{}_CSIF_par_p_value.tif'.format(period)
        mask_arr_tif = result_root + 'trend_max_correlation/{}_among_all_variables/max_index.tif'.format(period)

        threshold = 0.05
        outmask_dir = result_root + 'trend_max_correlation/mask_with_p_tif/'
        T.mk_dir(outmask_dir, force=True)

        p_arr = to_raster.raster2array(p_tif)[0]
        matrix = p_arr > threshold
        # matrix[p_arr>threshold]=True
        print(matrix)

        mask_arr = to_raster.raster2array(mask_arr_tif)[0]

        # plt.imshow(mask_arr)
        # plt.show()
        mask_arr[matrix] = np.nan
        mask_arr[mask_arr < -999] = np.nan
        print(len(mask_arr[0]))
        print(mask_arr.shape())

        # plt.imshow(mask_arr)
        # plt.show()

        outf = period + '_' + (mask_arr_tif.split('.')[0]).split('/')[-1]

        np.save(mask_arr, outmask_dir + 'mask_p_0.1_' + outf + '.npy')
        DIC_and_TIF().arr_to_tif(outmask_dir + 'mask_p_0.05_' + outf + '.tif', mask_arr)

    def climate_constrain_greenning(self):
        # 1 ??????SOS ?????????????????? 2 extract SOS 14 ??????SOS??? 3 extract ??? 15 ??????PAR ?????????
        # ??????6?????????
        # outdir = result_root + 'anomaly/PAR/'
        dic_SOS = {}
        result_dic_r = DIC_and_TIF().void_spatial_dic_nan()
        result_dic_p = DIC_and_TIF().void_spatial_dic_nan()
        SOS_anomaly = []
        pre_variable = []

        outdir = result_root + 'climate_constrain_greenning_leaf_out/PAR/'
        # outdir = result_root + 'climate_constrain_greenning_EOS/LST_mean/'
        Tools().mk_dir(outdir)
        f1 = result_root + 'contribution/Max_contribution_index_threshold_20%.npy'

        dic_trend = dict(np.load(f1, allow_pickle=True, ).item())

        fdir1 = result_root + 'anomaly/CSIF_par/'
        for f_sos in tqdm(sorted(os.listdir(fdir1))):
            # if not '005' in f:
            #     continue
            if not f_sos.startswith('p'):
                continue
            if f_sos.endswith('.npy'):
                dic_i = dict(np.load(fdir1 + f_sos, allow_pickle=True, ).item())
                dic_SOS.update(dic_i)

        # f_variable = result_root + 'extraction_pre_EOS/20%_Pre_LST_mean_extraction/pre15_LST_mean.npy'
        f_variable = result_root + 'extraction_pre_SOS/20%_Pre_PAR_extraction/pre15_PAR.npy'
        # f_variable = result_root + 'extraction/20%_Pre_LST_mean_extraction/pre30_LST_mean.npy'
        # f_variable = result_root + 'extraction/20%_Pre_LST_mean_extraction/pre30_LST_mean.npy'
        # f_variable = result_root + 'extraction/20%_Pre_LST_mean_extraction/pre30_LST_mean.npy'
        dic_variable = dict(np.load(f_variable, allow_pickle=True, ).item())

        for pix in tqdm(dic_SOS):
            # r,c=pix
            # china_r = list(range(75, 150))
            # china_c = list(range(550, 620))
            # if not r in china_r:
            #     continue
            # if not c in china_c:
            #     continue
            SOS_index_series = dic_SOS[pix]
            pre_variable = dic_variable[pix]
            print(len(pre_variable))
            if len(SOS_index_series) != 15:
                continue
            if len(pre_variable) != 15:
                print('pre_variable error')
                continue
            trend = dic_trend[pix]
            # ///??????LST ??????????????????
            # SOS_index_series=SOS_index_series[1:]
            matrix = np.isnan(SOS_index_series)
            if True in matrix:
                continue
            if trend < 0:
                r, p = stats.pearsonr(pre_variable, SOS_index_series)
                result_dic_r[pix] = r
                result_dic_p[pix] = p
        relationship_arr = DIC_and_TIF().pix_dic_to_spatial_arr(result_dic_r)
        plt.imshow(relationship_arr, vmin=0, vmax=0.7, cmap='jet')
        plt.show()
        DIC_and_TIF().pix_dic_to_tif(result_dic_r, outdir + 'SOS_pre15_r.tif')
        DIC_and_TIF().pix_dic_to_tif(result_dic_p, outdir + 'SOS_pre15_p.tif')

    def product_comparison(self):
        fdir=result_root+'trend_relative_change/2000-2018/'
        outdir = result_root + 'trend_relative_change/product_comparison_2000-2018/'
        T.mk_dir(outdir, force=True)
        all_products={}
        marked={}
        period='peak'

        for f in tqdm(sorted(os.listdir(fdir))):

            if period not in f:
                continue

            if not f.endswith('trend.npy'):
                continue
            print(f)
            outf=outdir+period
            print(outf)
            arr_climate = np.load(fdir + f)
            arr_climate[arr_climate<-999]=np.nan
            dic_climate=DIC_and_TIF().spatial_arr_to_dic(arr_climate)# ????????????????????????
            products_name = f.split('_')[0]
            all_products[products_name]=dic_climate

        mark_list=[]

        for pix in tqdm(dic_climate):
            val_all=[]
            for product in all_products:

                val_all.append(all_products[product][pix])

            sig_list=[]
            for val in val_all:
                if np.isnan(val):
                    continue
                if val>0:
                    sig='+'
                elif val<0:
                    sig='-'
                else:
                    continue
                sig_list.append(sig)
            set_sig_list=list(set(sig_list))
            if len(set_sig_list)==1:
                sig=set_sig_list[0]
                if sig=='-':
                    mark=-len(val_all)
                else:
                    mark=len(val_all)

            elif len(set_sig_list) == 2:
                negative_sig_number=T.count_num(sig_list,'-')
                positive_sig_number = T.count_num(sig_list, '+')
                if negative_sig_number>positive_sig_number:
                    mark=-negative_sig_number
                else:
                    mark=positive_sig_number
            else:
                # print(len(set_sig_list))
                continue
            marked[pix]=mark
        arr=DIC_and_TIF().pix_dic_to_spatial_arr(marked)
        plt.imshow(arr,cmap='jet')
        plt.show()
        DIC_and_TIF().arr_to_tif(arr, outf + '.tif')




def rename():
    variable='CCI_SM'
    period='late'
    fdir=result_root+'univariate_correlation_detrend/during_{}_window15/during_{}_{}_NIRv_window15/'.format(period,period,variable)
    outdir=result_root+'univariate_correlation_detrend/during_{}_window15/during_{}_new_{}_NIRv_window15/'.format(period,period,variable)
    T.mk_dir(outdir,force=True)
    for f in tqdm(sorted(os.listdir(fdir))):

        if not f.endswith('correlation.npy'):
            continue
        print(f)
        dic = np.load(fdir + f)  # ????????????????????????
        i=f.split('.')[0].split('_')[-2]
        # print(f)
        i=int(i)
        outf=outdir+f.split('.')[0][0:-14]+'_{:02d}'.format(i)+'_correlation.npy'
        print(outf)
        np.save(outf, dic)

class plot_results:


    def __init__(self):
        self.this_class_arr='/Volumes/SSD_sumsang/project_greening/Result/new_result/'
        pass


    def run(self):
        # self.extraction_pre_window()
        self.plot_greening_bar()


    def plot_greening_bar(self):
        fdir_all_seasons=self.this_class_arr+'/trend_calculation/'
        # ???????????????????????? 2 ???P mask trend

        for fdir_1 in tqdm(sorted(os.listdir(fdir_all_seasons))):
            print(fdir_1)
            var_name_trend = []
            var_name_p_value = []
            if not '_Y' in fdir_1:
                continue

            for f in sorted(os.listdir(fdir_all_seasons + '/' + fdir_1)):
                print(f)

                if f.endswith('trend.npy'):
                    var_name_trend.append(f)
                if f.endswith('p_value.npy'):
                    var_name_p_value.append(f)
            print(var_name_p_value)
            print(var_name_trend)
            exit()

            for i in range(len(var_name_p_value)):
                p_file_name = var_name_p_value[i]
                trend_file_name = var_name_trend[i]

                array_p = np.load(fdir_all_seasons + '/' + fdir_1 + '/' + p_file_name)
                array_p = np.array(array_p)
                print(fdir_all_seasons + '/' + fdir_1 + '/' + p_file_name)

                array_trend = np.load(fdir_all_seasons + '/' + fdir_1 + '/' + trend_file_name)
                array_trend = np.array(array_trend)

                array_p[array_p < -99] = np.nan
                array_p = array_p[:120]

                array_trend = array_trend[:120]
                mask_p_value = array_p < 0.1
                array_trend[array_trend < -99] = np.nan
                array_trend[~mask_p_value] = np.nan
                array_trend = array_trend[:120]
                mean_array_trend = np.nanmean(array_trend)

    pass

    def plot_bar_greening_percentage(self):
        count_no_trend=0
        count_greening_0_1=0
        count_browning_0_1=0
        count_greening_0_05 = 0
        count_browning_0_05 = 0


        # for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
        #     trend = row['CSIF_par_trend_{}'.format(period)]
        #     p_val=row['CSIF_par_p_value_{}'.format(period)]
        #     if p_val>0.1:
        #         count_no_trend=count_no_trend+1
        #     elif 0.05<p_val<0.1:
        #         if trend>0:
        #             count_greening_0_1=count_greening_0_1+1
        #         else:
        #             count_browning_0_1 = count_browning_0_1 + 1
        #     else:
        #         if trend>0:
        #             count_greening_0_05=count_greening_0_05+1
        #         else:
        #             count_browning_0_05=count_browning_0_05+1
        # greening_0_1=count_greening_0_1/len(df_pick)*100
        # browning_0_1=count_browning_0_1 / len(df_pick)*100
        # greening_0_05 = count_greening_0_05 / len(df_pick)*100
        # browning_0_05 = count_browning_0_05 / len(df_pick)*100
        # no_trend=count_no_trend/len(df_pick)*100
        #
        # y1 = np.array([browning_0_05])
        # y2=np.array([browning_0_1])
        # y3 = np.array([no_trend])
        # y4=np.array([greening_0_1])
        # y5=np.array([greening_0_05])
        #
        # # plot bars in stack manner
        #
        # plt.bar(koppen, y1, color='sienna')
        # plt.bar(koppen, y2, bottom=y1, color='peru')
        # plt.bar(koppen, y3, bottom=y1 + y2, color='gray')
        # plt.bar(koppen, y4, bottom=y1 + y2 + y3, color='limegreen')
        # plt.bar(koppen, y5, bottom=y1 + y2 + y3+y4, color='forestgreen')
        #
        #
        # plt.text(koppen, y1 / 2., round(browning_0_05), fontsize=10, color='w', ha='center', va='center')
        # plt.text(koppen, y1 + y2 / 2., round(browning_0_1), fontsize=10, color='w', ha='center', va='center')
        # plt.text(koppen, y1 + y2 + y3 / 2., round(no_trend), fontsize=10, color='w', ha='center', va='center')
        # plt.text(koppen, y1 + y2 + y3 + y4 / 2., round(greening_0_1), fontsize=10, color='w', ha='center',
        #          va='center')
        # plt.text(koppen, y1 + y2 + y3 + y4 + y5 / 2, round(greening_0_05), fontsize=10, color='w', ha='center',
        #          va='center')
        # plt.text(koppen, 102, len(df_pick), fontsize=10, color='k', ha='center', va='center')
        #
        # plt.xlabel("landcover")
        # plt.ylabel("Percentage")

def plot_show():
    fdir= '/Volumes/1T/wen_prj/Result/%CSIF/'

    dic = {}

    # dic = dict(np.load(fdir, allow_pickle=True, ).item())

    spatial_dic = {}
    spatial_dic_count={}

    for f in tqdm(os.listdir(fdir)):
        if f.endswith('.npy'):
            # if not '005' in f:
            #     continue
            dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
            dic.update(dic_i)

    for pix in tqdm(dic):
        val = dic[pix]
        if len(val) == 0:
            continue
        # spatial_dic[pix]=val[0]
        spatial_dic_count[pix] = len(val)
        # print(val)
    arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_count)
    plt.imshow(arr)
    plt.title('')
    plt.show()


    def plot_anomaly_ly():
        # f = result_root + 'extraction/extraction_during_whole_growing_season_static/during_whole_CSIF_par.npy'
        # fdir = '/Volumes/1T/wen_prj/Result/plot_anomaly/whole/'
        # fdir = '/Volumes/1T/wen_prj/Result/plot_anomaly/peak/'
        # fdir = '/Volumes/1T/wen_prj/Result/plot_anomaly/late/'
        for i in ['whole','peak','late','early',]:
            fdir = '/Volumes/1T/wen_prj/Result/plot_anomaly/{}/'.format(i)
            line = []
            std_list = []
            for year in tqdm(os.listdir(fdir),desc=i):
                dic = T.load_npy(fdir+year)
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic)
                arr = arr[:120]
                mean = np.nanmean(arr)
                std = np.nanstd(arr)
                line.append(mean)
                std_list.append(std)
            line = np.array(line)
            std_list = np.array(std_list)/8.
            plt.plot(line,label=i)
            x = range(len(line))
            plt.fill_between(x, y1=line-std_list, y2=line+std_list,alpha=0.3,interpolate=True,zorder=-99)
        plt.legend()
        plt.show()

    def plot_anomaly():
        # ?????????
        f1 = result_root + 'extraction/extraction_during_whole_growing_season_static/during_whole_CSIF_par.npy'
        f2 = result_root + 'extraction/extraction_during_early_growing_season_static/during_early_CSIF_par.npy'
        f3 = result_root + 'extraction/extraction_during_peak_growing_season_static/during_peak_CSIF_par.npy'
        f4 = result_root + 'extraction/extraction_during_late_growing_season_static/during_late_CSIF_par.npy'
        # outdir=result_root + 'plot_anomaly/whole/'
        # Tools().mk_dir(outdir)
        # dic_spring = {}
        # dic_peak = {}
        # dic_late = {}
        # dic_whole = {}
        #
        # year_list = list(range(2002, 2017))
        # # print(len(year_list))
        # # exit()
        # dic_early = dict(np.load(f2, allow_pickle=True, ).item())
        # dic_peak = dict(np.load(f3, allow_pickle=True, ).item())
        # dic_late = dict(np.load(f4, allow_pickle=True, ).item())
        # dic_whole = dict(np.load(f1, allow_pickle=True, ).item())
        # early_yearly_array_dic={}
        # peak_yearly_array_dic={}
        # late_yearly_array_dic={}
        # whole_yearly_array_dic={}
        #
        # spatial_dic = {}
        #
        # for year in year_list:  # ?????????????????????????????????????????????????????????
        #     early_yearly_array_dic[year] = []
        #     peak_yearly_array_dic[year] = []
        #     late_yearly_array_dic[year] = []
        #
        # i=0
        # for year in year_list:
        #     outf = outdir + '{}'.format(year)
        #     result_dic_early = DIC_and_TIF().void_spatial_dic_nan()
        #     result_dic_peak = DIC_and_TIF().void_spatial_dic_nan()
        #     result_dic_late = DIC_and_TIF().void_spatial_dic_nan()
        #     result_dic_whole = DIC_and_TIF().void_spatial_dic_nan()
        #     for pix in tqdm(dic_whole, desc='{}'.format(year)):
        #         # print(pix)
        #         val_early = dic_early[pix]
        #         # print(val_early)
        #         val_peak = dic_peak[pix]
        #         val_late = dic_late[pix]
        #         val_whole = dic_whole[pix]
        #         if len(val_early) != 15:
        #             continue
        #         if len(val_late) != 15:
        #             continue
        #         if len(val_peak) != 15:
        #             continue
        #         if len(val_whole) != 15:
        #             continue
        #         # print(len(val_early))
        #         result_dic_early[pix]=val_early[i]
        #         # print(val_early[i])
        #         result_dic_peak[pix] = val_peak[i]
        #         # print(val_peak[i])
        #         result_dic_late[pix] = val_late[i]
        #         # print(val_late[i])
        #         result_dic_whole[pix] = val_whole[i]
        #         # print(val_whole[i])
        #     early_yearly_array_dic[year]=result_dic_early
        #     peak_yearly_array_dic[year]=result_dic_peak
        #     late_yearly_array_dic[year]=result_dic_late
        #     whole_yearly_array_dic[year]=result_dic_whole
        #     # for pix in early_yearly_array_dic[year]:
        #     #     print(pix,early_yearly_array_dic[year][pix])
        #     # arr = DIC_and_TIF().pix_dic_to_spatial_arr(whole_yearly_array_dic[year])
        #     # plt.imshow(arr)
        #     # plt.title('')
        #     # plt.show()
        #     i += 1
        #     np.save(outf, whole_yearly_array_dic[year])
        pass


    def plot_latitude_bar_ly():

        tif = '/Volumes/1T/wen_prj/Result/contribution/Max_contribution_index_threshold_20%.tif'
        arr = raster2array.raster2array(tif)[0]
        spring_list = []
        summer_list = []
        autumn_list = []
        for i in range(len(arr)):
            if i > 120:
                continue
            vals = arr[i]
            vals = np.array(vals)
            vals = LY_Tools_newest.Tools().remove_np_nan(vals)

            if len(vals) == 0:
                continue

            vals = list(vals)
            spring_n = vals.count(1)
            summer_n = vals.count(2)
            autumn_n = vals.count(3)
            total = float(len(vals))

            spring_ratio = spring_n / total
            summer_ratio = summer_n / total
            autumn_ratio = autumn_n / total

            spring_list.append(spring_ratio)
            summer_list.append(summer_ratio)
            autumn_list.append(autumn_ratio)

            # plt.bar(spring_ratio,bottom=0)
            # plt.bar(summer_ratio,bottom=0)
            # plt.bar(autumn_ratio,bottom=0)
        spring_list = np.array(spring_list)
        summer_list = np.array(summer_list)
        autumn_list = np.array(autumn_list)

        spring_list_integrate = integrate_latitude(spring_list)
        summer_list_integrate = integrate_latitude(summer_list)
        autumn_list_integrate = integrate_latitude(autumn_list)


        plt.bar(range(len(spring_list_integrate)),spring_list_integrate,color='g')
        plt.bar(range(len(spring_list_integrate)),summer_list_integrate,bottom=spring_list_integrate,color='r')
        plt.bar(range(len(spring_list_integrate)),autumn_list_integrate,bottom=spring_list_integrate+summer_list_integrate,color='b')
        # plt.imshow(arr)
        plt.show()
        pass


    def integrate_latitude(in_list):
        in_list_reshape = np.reshape(in_list,(-1,4))
        inte_list = []
        for i in in_list_reshape:
            mean = np.mean(i)
            inte_list.append(mean)
        inte_list = np.array(inte_list)
        return inte_list


    def extraction_monthly_data():
        # N =1
        # f1 = result_root + '/early_peak_late_dormant_period_multiyear/20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/early_start.npy'
        f2 = result_root + 'early_peak_late_dormant_period_multiyear/20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/early_start_mon.npy'
        f3 = result_root + 'early_peak_late_dormant_period_multiyear/20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/late_end_mon.npy'
        fdir2 = data_root + 'SPEI/SPEI3/dic_SPEI3/'  # ??????
        outdir = result_root + 'extraction/extraction_during_whole_growing_season_static/'  # ??????
        # outf=result_root + 'extraction/test_pre_15_PAR.npy'


        Tools().mk_dir(outdir)
        dic_variables = {}
        dic_pre_variables = DIC_and_TIF().void_spatial_dic()
        dic_during_variables = DIC_and_TIF().void_spatial_dic()
        # dic_start_day = dict(np.load(f1, allow_pickle=True, ).item())
        dic_start_month = dict(np.load(f2, allow_pickle=True, ).item())
        dic_end_month = dict(np.load(f3, allow_pickle=True, ).item())

        dic_spatial_count = {}
        spatial_dic = {}

        # ??????????????????
        for f in tqdm(sorted(os.listdir(fdir2))):
            # if not '005' in f:
            #     continue
            if not f.startswith('p'):
                continue
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir2 + f, allow_pickle=True, ).item())
                dic_variables.update(dic_i)

        for pix in tqdm(dic_variables):
            # r,c=pix
            # if c>180:
            #     continue

            # start_day_index = dic_start_day[pix]
            start_month_index = dic_start_month[pix]
            end_month_index = dic_end_month[pix]
            if len(start_month_index) == 0:
                continue
            if len(end_month_index) == 0:
                continue
            # start_day_index = start_day_index[0]
            start_month_index=start_month_index[0]
            end_month_index = end_month_index[0]
            time_series = dic_variables[pix]
            if len(time_series) != 180:  # ??????
                # if len(time_series) != 5110:
                continue
            # plt.plot(time_series)
            # plt.show()
            time_series = time_series.reshape(15, -1)  # ??????

            for year in range(15):  # ??????
                # print(len(SOS_index_series))
                # time_series = time_series.reshape(15, -1)
                # print(time_series)
                # pre_time_series = time_series[year][(start_month_index - N):start_month_index]
                during_time_series = time_series[year][start_month_index:end_month_index]
                # print(during_time_series)
                # plt.imshow(time_series)
                # plt.show()
                if np.isnan(np.nanmean(during_time_series)):
                    continue
                variable_mean = np.nanmean(during_time_series)
                dic_during_variables[pix].append(variable_mean)
                dic_pre_variables[pix].append(variable_mean)  # ????????????15??????
            dic_spatial_count[pix] = len(dic_pre_variables[pix])
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic_spatial_count)
        plt.imshow(arr, cmap='jet')
        plt.colorbar()
        plt.title('')
        plt.show()
        # np.save(outdir + 'pre_{}mo_SPEI3'.format(N), dic_pre_variables)  # ??????
        np.save(outdir + 'during_whole_SPEI3', dic_during_variables)  # ??????
        # np.save(outf, dic_pre_variables)  # ??????

    def soil_preprocess():
        fdir=data_root+'SOIL/OCD/'
        outf = data_root + 'SOIL/OCD_mean.tif'
        brandf_all_array=[]
        for f in tqdm(sorted(os.listdir(fdir))):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            dataset = gdal.Open(fdir + f)
            bandf = dataset.GetRasterBand(1)
            bandf_array = bandf.ReadAsArray()
            brandf_all_array.append(bandf_array)
            # mean_array = np.zeros(range(len(band2_array)), range(len(band2_array)[0]))
            # mean_array = [[0] * len(band2_array[0]) for row in range(len(band2_array))]
        mean_array = np.zeros_like(brandf_all_array[0])
        # print(np.shape(mean_array))
        # exit()
        # print(len(bandf_array[0]))
        # exit()
        for r in range(len(bandf_array)):
            for c in range(len(bandf_array[0])):
                # print(brandf_all_array[1][r][c])
                # print(r,c)
                mean_array[r][c]=brandf_all_array[0][r][c]*5/200+brandf_all_array[1][r][c]*10/200+brandf_all_array[2][r][c]*15/200+brandf_all_array[3][r][c] * 30 / 200+brandf_all_array[4][r][c] * 40 / 200 +brandf_all_array[5][r][c] * 100 / 200
                # print(len(brandf_all_array))
                # print(mean_array[r][c])
        mean_array = np.array(mean_array)
            # ????????????
        geotransform = dataset.GetGeoTransform()
        longitude_start = geotransform[0]
        latitude_start = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        newRasterfn = outf

        to_raster.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight,
                                      mean_array, ndv=-999999)
    # plt.imshow(band1_array)
    # plt.show()

class Unify_date_range:

    def __init__(self):

        pass

    def run(self):
        # self.__data_range_index()
        start = 2000
        end = 2018
        period='peak'

        # X_dir = result_root + f'extraction_original_val/extraction_original_val_monthly/extraction_during_{period}_growing_season_static/'
        # outdirX = result_root + f'extraction_original_val/{start}-{end}_monthly/'
        X_dir=results_root+f'extraction_original_val/extraction_original_val_Trendy2/extraction_during_{period}_growing_season_static/'
        outdirX = results_root + f'extraction_original_val/{start}-{end}_Trendy_2/'

        self.unify(X_dir,outdirX,start,end)
        # #
        # Y_dir = result_root+'extraction_anomaly_val/extraction_during_{}_growing_season_static/during_{}_Y/'.format(period,period)
        # outdirY = result_root+'extraction_anomaly_val/{}-{}_during_{}_growing_season/Y_{}-{}/'.format(start,end,period,start,end)
        # Y_dir = result_root + 'detrend_extraction/during_{}_growing_season_static/during_{}_Y/'.format(
        #     period, period)
        # outdirY = result_root + 'detrend_extraction/{}-{}_during_{}_growing_season/Y_{}-{}/'.format(start, end,
        #                                                                                                 period, start,
        #                                                                                                 end)
        # self.unify(Y_dir,outdirY,start,end)


        # self.check_data_length()
        pass

    def __data_range_index(self,start=0,end=0,product='NIRv',isplot=False):
        dic = {


            # 'LAI3g': list(range(1982, 2019)),
            # # 'LAI4g': list(range(1982, 2021)),
            #
            # 'MODIS_LAI': list(range(2000, 2020)),
            #
            #  'CCI_SM':list(range(1982,2021)),
            #
            #
            # 'PAR':list(range(1982,2021)),
            # 'CO2': list(range(1982, 2021)),
            #
            # 'Temp':list(range(1982,2021)),
            # 'VPD':list(range(1982,2021)),

             'CABLE-POP_S2_lai':list(range(1982,2021)),
            'CLASSIC_S2_lai':list(range(1982,2021)),
              'CLASSIC-N_S2_lai':list(range(1982,2021)),
              'CLM5':list(range(1982,2021)),
              'IBIS_S2_lai':list(range(1982,2021)),
            'ISAM_S2_LAI':list(range(1982,2021)),
            'LPJ-GUESS_S2_lai':list(range(1982,2021)),
              'LPX-Bern_S2_lai':list(range(1982,2021)),
              'OCN_S2_lai':list(range(1982,2021)),
            'ORCHIDEE_S2_lai':list(range(1982,2021)),
              'ORCHIDEEv3_S2_lai':list(range(1982,2021)),
              'VISIT_S2_lai':list(range(1982,2021)),
              'YIBs_S2_Monthly_lai':list(range(1982,2021)),
              'ISBA-CTRIP_S2_lai':list(range(1982,2021)),
            'Trendy_ensemble':list(range(1982,2021)),



        }

        def plot():
            plt.figure(figsize=(10, int(len(dic))/2))
            for product in dic:
                # print(dic[product][-1])
                plt.barh(product, len(dic[product]), left=dic[product][0], align='edge')
            plt.xlim(1980, 2020)
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if isplot == True:
            plot()
        if start == 0:
            plot()
        date_range = dic[product]
        if not start in date_range:
            print('the year {} is not in {}'.format(start,product))
            # plot()
            raise UserWarning('the year {} is not in {}'.format(start,product))
        if not end in date_range:
            print('the year {} is not in {}'.format(start,product))
            # plot()
            raise UserWarning('the year {} is not in {}'.format(end,product))
        in_date_range = list(range(start,end+1))
        index_list = []
        for i in in_date_range:
            index = date_range.index(i)
            index_list.append(index)

        return index_list,len(date_range)


    def unify(self,fdir,outdir,start,end):

        T.mk_dir(outdir,force=True)
        for f in os.listdir(fdir):
            # if 'interpolation'not in f:
            #     continue

            f_split = f.split('.')[0]
            f_split1 = f_split.split('_')
            product = '_'.join(f_split1[2:])
            # product = '_'.join(f_split1[2:3])
            print(product)

            try:
                selected_index,data_length = self.__data_range_index(start, end, product)
            except:
                continue
            fname = os.path.join(fdir,f)
            dic = T.load_npy(fname)
            selected_dic = {}
            for pix in tqdm(dic):
                vals = dic[pix]
                # if len(vals) == 0:
                #     continue
                if len(vals) != data_length:
                    continue
                selected_vals = T.pick_vals_from_1darray(vals,selected_index)
                selected_vals = np.array(selected_vals)
                selected_dic[pix] = selected_vals
            T.save_npy(selected_dic,outdir+f)


    def check_data_length(self):
        fdir = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/1982-2015/X_1982-2015/'
        template_file = '/Volumes/SSD/drought_legacy_new/conf/tif_template.tif'
        for f in os.listdir(fdir):
            dic = T.load_npy(fdir+f)
            spatial_dic = {}
            for pix in dic:
                length = len(dic[pix])
                spatial_dic[pix] = length
            arr = DIC_and_TIF(template_file).pix_dic_to_spatial_arr(spatial_dic)
            plt.figure()
            plt.imshow(arr)
            plt.title(f)
        plt.show()

        pass




class Hydrothemal:
    def __init__(self):
        pass
    def gen_map(self):
        pass
    def load_map_mat(self):

        MATf='/Volumes/SSD_sumsang/project_greening/Result/map_mat/MAT.tif'
        MAPf = '/Volumes/SSD_sumsang/project_greening/Result/map_mat/MAP.tif'
        # max_contribution='/Volumes/SSD_sumsang/project_greening/Result/trend_max_correlation/mask_with_p_tif/mask_p_0.1_early_max_index.tif'
        max_contribution='/Volumes/SSD_sumsang/project_greening/Result/trend_max_correlation/early_among_all_variables/max_index.tif'

        MAT_arr=to_raster.raster2array(MATf)[0]
        T.mask_999999_arr(MAT_arr)
        MAP_arr=to_raster.raster2array(MAPf)[0]
        T.mask_999999_arr(MAP_arr)
        MAP_arr=MAP_arr
        max_contribution_arr=to_raster.raster2array(max_contribution)[0]
        T.mask_999999_arr(max_contribution_arr)
        return MAT_arr, MAP_arr, max_contribution_arr
    def bin_generation_T(self,step=30,max_v=None, min_v=None):
        bins=np.linspace(start=min_v,stop=max_v, num=step)
        bins_str=[]
        for i in bins:
            bins_str.append('{:0.1f}'.format(i))
        return bins, bins_str

    def bin_generation_P(self,step=30,max_v=None, min_v=None):
        bins=np.linspace(start=min_v,stop=max_v, num=step)
        bins_str=[]
        for i in bins:
            bins_str.append('{}'.format(int(i)))
        return bins, bins_str

    def plot_matrix(self):
        MAT_arr, MAP_arr,max_contribution_arr= self.load_map_mat()
        # plt.imshow(MAT_arr)
        # plt.figure()
        # plt.imshow(MAP_arr)
        # plt.figure()
        # plt.imshow(max_contribution_arr)
        # plt.show()
        bins_T,bins_str_T=self.bin_generation_T(max_v=30., min_v=-10.)
        # print(bins_T)
        bins_P, bins_str_P = self.bin_generation_T(max_v=1000,min_v=0)
        df=pd.DataFrame()
        mat_dic=DIC_and_TIF().spatial_arr_to_dic(MAT_arr)
        map_dic = DIC_and_TIF().spatial_arr_to_dic(MAP_arr)
        max_contribution_dic = DIC_and_TIF().spatial_arr_to_dic(max_contribution_arr)
        mat_list=[]
        map_list=[]
        max_contribution_list=[]
        pix_list=[]
        for pix in mat_dic:
            mat=mat_dic[pix]
            map=map_dic[pix]
            max_contribution=max_contribution_dic[pix]
            mat_list.append(mat)
            map_list.append(map)
            max_contribution_list.append(max_contribution)
            pix_list.append(pix)
        df['pix']=pix_list
        df['MAT']=mat_list
        df['MAP']=map_list
        df['max_contribution']=max_contribution_list
        x=[]
        y=[]
        z=[]
        maxtrix=[]
        for t in tqdm(range(len(bins_T))):
            if (t+1)>=len(bins_T):
                continue
            df_t=df[df['MAT']>bins_T[t]]
            df_t=df_t[df_t['MAT']<bins_T[t+1]]
            temporary=[]
            for p in range(len(bins_P)):
                if (p + 1) >= len(bins_P):
                    continue
                df_p = df_t[df_t['MAP'] > bins_P[t]]
                df_p = df_p[df_p['MAP'] < bins_P[t + 1]]
                df_p=df_p.dropna(subset=['max_contribution'])
                if len(df_p)==0:
                    temporary.append(np.nan)
                    continue
                max_contribution_set=df_p['max_contribution']
                max_contribution_set_list=df_p['max_contribution'].to_list()

                max_index=max_contribution_set.mode()[0]
                print(max_index)
                print(max_contribution_set_list)
                input()

                temporary.append(max_index)
                x.append(bins_P[p])
                y.append(bins_T[t])
                z.append(max_index)
            maxtrix.append(temporary)
        print(maxtrix)
        plt.imshow(maxtrix)
        plt.show()



def normalization():

    val=range(-50,51)
    print(val)
    val_new=T.normalize(val, 100, 0)
    print(val_new)






def main():

    # statistic_anaysis().extraction_during_window()
    # statistic_anaysis().extraction_variables_static_pre_month()
    # statistic_anaysis().extraction_variables_static_during_daily()  # only for LAI3g, MODIS
    # statistic_anaysis().extraction_variables_static_during_month()
    # statistic_anaysis().extraction_variables_static_during_daily_climate_variables()
    # statistic_anaysis().extraction_variables_static_during_Trendy()



    # statistic_anaysis().multiregression_beta_window()
    # statistic_anaysis().save_moving_window_multi_regression()
    #
    # statistic_anaysis().plot_moving_window_correlation()

    # statistic_anaysis().trend_window_LAI()
    # statistic_anaysis().mean_window()
    # statistic_anaysis().conversion_mean()
    # statistic_anaysis().conversion_trendy()
    # statistic_anaysis().variables_contribution_window()
    # statistic_anaysis().conversion()

    statistic_anaysis().partial_correlation_window()
    # statistic_anaysis().univariate_correlation_window()
    # statistic_anaysis().plot_simple_moving_window_correlation()
    # statistic_anaysis().save_moving_window_correlation()


    # statistic_anaysis().trend_calculation()
    # statistic_anaysis().detrend()
    # statistic_anaysis().mean_calculation()
    # statistic_anaysis().CV_calculation()



    # statistic_anaysis().simple_correlation()
    # statistic_anaysis().difference_correlation_between_two_period()
    # statistic_anaysis().multiregression_beta_window()
    # statistic_anaysis().save_results_for_three_seasons()
    # statistic_anaysis().plot_anomaly_for_three_seasons()
    # statistic_anaysis().save_anomaly_for_three_seasons()
    # statistic_anaysis().plot_anomaly_for_three_seasons()
    # statistic_anaysis().max_trend_among_all_variables()
    # statistic_anaysis().product_comparison()

    # lc_list = ['water', 'grass', 'shrub', 'crop', 'EBF', 'ENF', 'DBF', 'DNF', 'savanna', 'urban', 'nonveg']
    # for lc in lc_list:
    #     fdir = f'/Volumes/SSD_sumsang/project_greening/Data/landcover/landcover_tif/{lc}/'
    #     outdir = f'/Volumes/SSD_sumsang/project_greening/Data/original_dataset/landcover/{lc}_dic/'
    #     # # Pre_Process().data_transform(fdir, outdir)
    #     tif2dict(fdir, outdir)
    # tif2dict_trendy()

    # tif2dict()
    # tif2dict_daily()
    # Phenology_retrieval()
    # average_peak_calculation()

    # Main_flow_Early_Peak_Late_Dormant().annual_phelogy()
    # Main_flow_Early_Peak_Late_Dormant().trend()
    # Main_flow_Early_Peak_Late_Dormant().contribution()
    # Main_flow_Early_Peak_Late_Dormant().changes_NDVI_keenan()

    # Main_flow_Early_Peak_Late_Dormant().anomaly()
    # Main_flow_Early_Peak_Late_Dormant().zscore()
    # Main_flow_Early_Peak_Late_Dormant().Pierre_anomaly_variables()



    # Main_flow_Early_Peak_Late_Dormant().phenology_Yao_method()
    # Main_flow_Early_Peak_Late_Dormant().annual_phenology_retrieval_Yao_method()
    # statistic_anaysis().simple_linear_regression()
    # Main_flow_Early_Peak_Late_Dormant().changes()
    # Main_flow_Early_Peak_Late_Dormant().clean_data()
    # extraction_monthly_data()
    # soil_preprocess()

    # Hydrothemal().plot_matrix()
    # statistic_anaysis().run()
    # Unify_date_range().run()
    # rename()
    # plot_results().run()
    # normalization()
    pass

if __name__ == '__main__':
    main()