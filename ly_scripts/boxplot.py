import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.multivariate.pca

from __init__ import *

T = Tools()


def plot_box():
    # f = '/Volumes/NVME2T/wen_proj/Matrix/data/2002-2015_partial_correlation_p_value_early_anomaly.npy'
    f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_correlation_anomaly/MODIS_NDVI/0105/2002-2015_partial_correlation_late_anomaly_MODIS_NDVI2.npy'
    dic = T.load_npy(f)
    df_temp = T.dic_to_df_different_columns(dic, 'pix')
    df_f = results_root + 'Data_frame_2002-2015/Data_frame_2002-2015_df.df'
    df = T.load_df(df_f)
    df = df[df['2002-2015_during_late_MODIS_NDVI_trend'] > 0]
    pix_list = T.get_df_unique_val_list(df, 'pix')
    r_list = []
    for i, row in df_temp.iterrows():
        pix = row.pix
        if not pix in pix_list:
            r_list.append(np.nan)
            continue
        r, c = row['pix']
        r_list.append(r)
    df_temp['r'] = r_list
    df_temp = df_temp[df_temp['r'] < 120]
    box = []
    labels = []
    for col in df_temp:
        print(col)
        if col == 'r':
            continue
        if col == 'pix':
            continue
        vals = df_temp[col]
        box_i = []
        for i in vals:
            if np.isnan(i):
                continue
            box_i.append(i)
        box.append(box_i)
        labels.append(col)
    plt.boxplot(box, labels=labels, showfliers=False)
    plt.show()


def plot_scatter():
    y_f = '/Volumes/NVME2T/wen_proj/20220104/npy/2002-2015_multi_linearearly_anomaly_MODIS_NDVI.npy'
    x_f = '/Volumes/NVME2T/wen_proj/20220104/npy/during_early_SPEI3_trend.npy'
    dicy = T.load_npy(y_f)
    x = np.load(x_f)
    x[x < -999] = np.nan
    plt.imshow(x)
    plt.show()
    dicx = DIC_and_TIF().spatial_arr_to_dic(x)
    # dicy = DIC_and_TIF().spatial_arr_to_dic(y)
    keys = []
    for pix in dicy:
        vals = dicy[pix]
        for k in vals:
            keys.append(k)
    keys = list(set(keys))
    keys.sort()
    for k in keys:
        print(k)
        x_list = []
        y_list = []
        for pix in dicy:
            r, c = pix
            if r > 120:
                continue
            xx = dicx[pix]
            if not k in dicy[pix]:
                continue
            yy = dicy[pix][k]
            if xx < -9999:
                continue
            if yy < -9999:
                continue
            x_list.append(xx)
            y_list.append(yy)
        # plt.figure()
        KDE_plot().plot_scatter(x_list, y_list)
        plt.xlabel('SPEI3_trend')
        plt.ylabel(f'{k}_vs_MODIS_NDVI')
    plt.show()


def plot_scatter1():
    fdir = results_root + 'mean_calculation_original/2002-2015_first_last_five_years'
    # fdir=results_root+'mean_calculation_original/during_early_2002-2015'
    corr_fdir = results_root + '/multiregression_anomaly/MODIS_NDVI/0104/'
    period_list = ['early', 'peak', 'late', ]
    order_list = ['first', 'last']
    for period in period_list:
        ############## change here ##################
        if not period == 'early':
            continue
        ############## change here ##################
        NDVI_trend_f = results_root + '/trend_calculation_anomaly/during_{}_2002-2015/2002-2015_during_{}_MODIS_NDVI_trend.npy'.format(
            period, period)
        temperature_trend_f = results_root + '/trend_calculation_anomaly/during_{}_2002-2015/2002-2015_during_{}_MODIS_NDVI_trend.npy'.format(
            period, period)
        # corr_f = f'2002-2015_partial_correlation_{period}_anomaly_CSIF.npy'
        corr_f = f'2002-2015_multi_linear{period}_anomaly_CSIF_fpar.npy'
        corr_fpath = join(corr_fdir, corr_f)
        corr_dic = T.load_npy(corr_fpath)
        NDVI_trend_arr = np.load(NDVI_trend_f)
        variable_trend_arr = np.load(temperature_trend_f)
        NDVI_trend_dic = DIC_and_TIF().spatial_arr_to_dic(NDVI_trend_arr)
        variable_trend_dic = DIC_and_TIF().spatial_arr_to_dic(variable_trend_arr)
        product_list = []
        for pix in corr_dic:
            vals = corr_dic[pix]
            for k in vals:
                product_list.append(k)
        product_list.append('{}_surf_soil_moisture'.format(period))
        product_list.append('{}_SPEI3'.format(period))
        product_list = list(set(product_list))
        product_list.sort()
        for product in product_list:
            ## Y axis ##
            ############## change here ##################
            if not product == f'{period}_CO2':
                continue
            ############## change here ##################
            product_dic = {}
            for pix in corr_dic:
                r, c = pix
                if r > 120:
                    continue
                NDVI_trend = NDVI_trend_dic[pix]
                if NDVI_trend > 0:
                    continue
                variable_trend = variable_trend_dic[pix]
                # if variable_trend < 0:
                #     continue
                dic_i = corr_dic[pix]
                if not product in dic_i:
                    continue
                val = dic_i[product]
                product_dic[pix] = val
            order_dic = {}
            for product1 in product_list:
                ## X axis ##
                ############## change here ##################
                if not product1 == f'{period}_CCI_SM':
                    continue
                ############## change here delata as Xaxis##################
                for order in order_list:
                    folder_name = f'during_{period}_2002-2015_{order}_five'
                    fpath = join(fdir, folder_name, f'during_{product1}_mean.npy')

                    arr = np.load(fpath)
                    arr[arr < -9999] = np.nan
                    order_dic[order] = arr
                delta_arr = order_dic[order_list[1]] - order_dic[order_list[0]]
                delta_dic = DIC_and_TIF().spatial_arr_to_dic(delta_arr)
                ################mean as Xaxis###############################

                # fpath = join(fdir, f'during_{product1}_mean.npy')
                # arr = np.load(fpath)
                # arr[arr<-9999]=np.nan
                # delta_dic = DIC_and_TIF().spatial_arr_to_dic(arr)

                x_list = []
                y_list = []
                for pix in delta_dic:
                    r, c = pix
                    if r > 120:
                        continue
                    NDVI_trend = NDVI_trend_dic[pix]
                    if NDVI_trend > 0:
                        continue
                    variable_trend = variable_trend_dic[pix]
                    # if variable_trend < 0:
                    #     continue
                    if not pix in product_dic:
                        continue
                    x = delta_dic[pix]
                    y = product_dic[pix]
                    x_list.append(x)
                    y_list.append(y)
                x_mean = np.nanmean(x_list)
                x_std = np.nanstd(x_list)
                y_mean = np.nanmean(y_list)
                y_std = np.nanstd(y_list)

                x_up = x_mean + 3 * x_std
                x_down = x_mean - 3 * x_std
                y_up = y_mean + 3 * y_std
                y_down = y_mean - 3 * y_std
                x_list_new = []
                y_list_new = []
                for i in range(len(x_list)):
                    x = x_list[i]
                    y = y_list[i]
                    if x > x_up:
                        continue
                    if x < x_down:
                        continue
                    if y > y_up:
                        continue
                    if y < y_down:
                        continue
                    x_list_new.append(x)
                    y_list_new.append(y)
                plt.figure()
                plt.scatter(x_list_new, y_list_new)
                # KDE_plot().plot_scatter(x_list_new,y_list_new)
                plt.xlabel(f'Delta_{product1}')
                plt.ylabel(f'partial_correlation_{product}_anomaly_CSIF')
                # plt.title(product)
                plt.show()


def plot_vectors():
    fdir = '/Volumes/NVME2T/wen_proj/20220107/OneDrive_1_2022-1-9/1982-2015_first_last_five_years'
    water_balance_tif = '/Volumes/NVME2T/wen_proj/20220107/HI_difference.tif'
    limited_area = ['energy_limited', 'water_limited', ]
    period_list = ['early', 'peak', 'late', ]
    order_list = ['first', 'last']
    water_balance_dic = DIC_and_TIF().spatial_tif_to_dic(water_balance_tif)
    for period in period_list:
        data_dic = {'HI': [], 'NDVI': []}
        for order in order_list:
            folder = f'during_{period}_1982-2015_{order}_five'
            HI_f = f'during_{period}_Aridity_mean.npy'
            NDVI_f = f'during_{period}_GIMMS_NDVI_mean.npy'
            fpath_HI = join(fdir, folder, HI_f)
            fpath_NDVI = join(fdir, folder, NDVI_f)
            HI_arr = np.load(fpath_HI)
            NDVI_arr = np.load(fpath_NDVI)
            HI_arr[HI_arr < -9999] = np.nan
            NDVI_arr[NDVI_arr < -9999] = np.nan
            HI_dic = DIC_and_TIF().spatial_arr_to_dic(HI_arr)
            NDVI_dic = DIC_and_TIF().spatial_arr_to_dic(NDVI_arr)
            data_dic['HI'].append(HI_dic)
            data_dic['NDVI'].append(NDVI_dic)
        x1_dic = data_dic['HI'][0]
        x2_dic = data_dic['HI'][1]
        y1_dic = data_dic['NDVI'][0]
        y2_dic = data_dic['NDVI'][1]

        key_list = []
        r_list = []
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        wb_list = []
        for key in x1_dic:
            r, c = key
            key_list.append(key)
            x1 = x1_dic[key]
            x2 = x2_dic[key]
            y1 = y1_dic[key]
            y2 = y2_dic[key]
            wb = water_balance_dic[key]

            x1_list.append(x1)
            x2_list.append(x2)
            y1_list.append(y1)
            y2_list.append(y2)
            r_list.append(r)
            wb_list.append(wb)

        df = pd.DataFrame()
        df['pix'] = key_list
        df['r'] = r_list
        df['x1'] = x1_list
        df['x2'] = x2_list
        df['y1'] = y1_list
        df['y2'] = y2_list
        df['wb'] = wb_list

        df = df.dropna()
        df_copy = copy.copy(df)
        for limited in limited_area:
            if limited == 'energy_limited':
                df_ltd = df_copy[df_copy['wb'] > 0]
            else:
                df_ltd = df_copy[df_copy['wb'] < 0]
            df = df_ltd
            df = df[df['r'] < 120]
            df = df[df['x1'] < 3]
            df = df[df['x1'] < 3]
            df = df[df['x1'] != 0]
            df = df[df['x2'] != 0]
            df = df.sample(n=1000)

            plt.figure()
            for i, row in df.iterrows():
                x = row.x1
                x2 = row.x2
                y = row.y1
                y2 = row.y2
                dx = x2 - x
                dy = y2 - y
                if dy > 0 and dx > 0:
                    plt.arrow(x, y, dx, dy, ec='g', fc='g', alpha=0.8, head_width=0)
                elif dy > 0 and dx < 0:
                    plt.arrow(x, y, dx, dy, ec='cyan', fc='cyan', alpha=0.8, head_width=0)
                elif dy < 0 and dx > 0:
                    plt.arrow(x, y, dx, dy, ec='purple', fc='purple', alpha=0.8, head_width=0)
                elif dy < 0 and dx < 0:
                    plt.arrow(x, y, dx, dy, ec='r', fc='r', alpha=0.8, head_width=0)
            plt.title(limited)
        plt.show()


def plot_pie_chart():
    ################## change area ##################
    # fdir = '/Volumes/NVME2T/wen_proj/20220107/OneDrive_1_2022-1-9/1982-2015_first_last_five_years'
    fdir = '/Volumes/NVME2T/wen_proj/20220107/2002-2015_first_last_five_years'
    water_balance_tif = '/Volumes/NVME2T/wen_proj/20220107/HI_difference.tif'
    year_range = '2002-2015'
    # x_variable = 'Aridity'
    x_variable = 'VPD'
    # y_variable = 'MODIS_NDVI'
    y_variable = 'GIMMS_NDVI'

    # labels = [
    #     'wetter greening',
    #     'dryer greening',
    #     'wetter browning',
    #     'dryer browning',
    # ]

    labels = [
        'dryer greening',
        'wetter greening',
        'dryer browning',
        'wetter browning',
    ]
    ################## change area ##################
    suptitle = f'{year_range} {x_variable} {y_variable}'
    limited_area = ['energy_limited', 'water_limited', ]
    period_list = ['early', 'peak', 'late', ]
    order_list = ['first', 'last']
    water_balance_dic = DIC_and_TIF().spatial_tif_to_dic(water_balance_tif)
    flag = 0
    plt.figure()
    for period in period_list:
        data_dic = {'x': [], 'y': []}
        for order in order_list:
            folder = f'during_{period}_{year_range}_{order}_five'
            x_f = f'during_{period}_{x_variable}_mean.npy'
            y_f = f'during_{period}_{y_variable}_mean.npy'
            fpath_x = join(fdir, folder, x_f)
            fpath_y = join(fdir, folder, y_f)
            x_arr = np.load(fpath_x)
            y_arr = np.load(fpath_y)
            x_arr[x_arr < -9999] = np.nan
            y_arr[y_arr < -9999] = np.nan
            x_dic = DIC_and_TIF().spatial_arr_to_dic(x_arr)
            y_dic = DIC_and_TIF().spatial_arr_to_dic(y_arr)
            data_dic['x'].append(x_dic)
            data_dic['y'].append(y_dic)
        x1_dic = data_dic['x'][0]
        x2_dic = data_dic['x'][1]
        y1_dic = data_dic['y'][0]
        y2_dic = data_dic['y'][1]

        key_list = []
        r_list = []
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        wb_list = []
        for key in x1_dic:
            r, c = key
            key_list.append(key)
            x1 = x1_dic[key]
            x2 = x2_dic[key]
            y1 = y1_dic[key]
            y2 = y2_dic[key]
            wb = water_balance_dic[key]

            x1_list.append(x1)
            x2_list.append(x2)
            y1_list.append(y1)
            y2_list.append(y2)
            r_list.append(r)
            wb_list.append(wb)

        df = pd.DataFrame()
        df['pix'] = key_list
        df['r'] = r_list
        df['x1'] = x1_list
        df['x2'] = x2_list
        df['y1'] = y1_list
        df['y2'] = y2_list
        df['wb'] = wb_list

        df = df.dropna()
        df_copy = copy.copy(df)
        for limited in limited_area:
            flag += 1
            plt.subplot(3, 2, flag)
            if limited == 'energy_limited':
                df_ltd = df_copy[df_copy['wb'] > 0]
            else:
                df_ltd = df_copy[df_copy['wb'] < 0]
            df = df_ltd
            df = df[df['r'] < 120]
            df = df[df['x1'] < 3]
            df = df[df['x1'] < 3]
            df = df[df['x1'] != 0]
            df = df[df['x2'] != 0]
            # df = df.sample(n=1000)

            part1 = 0
            part2 = 0
            part3 = 0
            part4 = 0
            total = 0
            for i, row in df.iterrows():
                total += 1
                x = row.x1
                x2 = row.x2
                y = row.y1
                y2 = row.y2
                dx = x2 - x
                dy = y2 - y
                if dy > 0 and dx > 0:
                    # plt.arrow(x, y, dx, dy, ec='g', fc='g', alpha=0.8, head_width=0)
                    part1 += 1
                elif dy > 0 and dx < 0:
                    # plt.arrow(x, y, dx, dy, ec='cyan', fc='cyan', alpha=0.8, head_width=0)
                    part2 += 1

                elif dy < 0 and dx > 0:
                    # plt.arrow(x, y, dx, dy, ec='purple', fc='purple', alpha=0.8, head_width=0)
                    part3 += 1

                elif dy < 0 and dx < 0:
                    # plt.arrow(x, y, dx, dy, ec='r', fc='r', alpha=0.8, head_width=0)
                    part4 += 1
            ratio1 = part1 / total
            ratio2 = part2 / total
            ratio3 = part3 / total
            ratio4 = part4 / total
            parts = [ratio1, ratio2, ratio3, ratio4]
            plt.pie(parts, labels=labels)
            plt.title(f'{limited} {period}')
    plt.suptitle(suptitle)
    # plt.suptitle('1982-2015')
    plt.show()
    pass


def HI_reclass(water_balance_tif):
    dic = DIC_and_TIF().spatial_tif_to_dic(water_balance_tif)
    dic_reclass = {}
    for pix in dic:
        val = dic[pix]
        label = None
        if val > 50:
            label = 'Humid'
        elif val < -50:
            label = 'Arid'
        elif val > -50 and val < -0:
            label = 'Semi Arid'
        elif val > 0 and val < 50:
            label = 'Semi Humid'
        dic_reclass[pix] = label
    return dic_reclass

def P_PET_reclass_int(dic):
    dic_reclass = {}
    for pix in dic:
        val = dic[pix]
        # label = None
        label = np.nan
        if val > 0.65:
            # label = 'Humid'
            label = 3
        elif val < 0.2:
            # label = 'Arid'
            label = 0
        elif val > 0.2 and val < 0.5:
            # label = 'Semi Arid'
            label = 1
        elif val > 0.5 and val < 0.65:
            # label = 'Semi Humid'
            label = 2
        dic_reclass[pix] = label
    return dic_reclass

def P_PET_reclass(dic):
    dic_reclass = {}
    for pix in dic:
        val = dic[pix]
        label = None
        # label = np.nan
        if val > 0.65:
            label = 'Humid'
            # label = 3
        elif val < 0.2:
            label = 'Arid'
            # label = 0
        elif val > 0.2 and val < 0.5:
            label = 'Semi Arid'
            # label = 1
        elif val > 0.5 and val < 0.65:
            label = 'Semi Humid'
            # label = 2
        dic_reclass[pix] = label
    return dic_reclass


def plot_pie_chart_trend():
    ################## change area ##################
    # fdir = '/Volumes/NVME2T/wen_proj/20220107/OneDrive_1_2022-1-9/1982-2015_first_last_five_years'
    fdir = '/Volumes/NVME2T/wen_proj/20220111/trend_calculation_anomaly'
    water_balance_tif = '/Volumes/NVME2T/wen_proj/20220107/HI_difference.tif'
    year_range = '2002-2015'
    # year_range = '1982-2015'

    x_variable = 'Aridity_trend'
    # x_variable = 'VPD'
    y_variable = 'MODIS_NDVI_trend'
    # y_variable = 'NIRv_trend'
    # y_variable = 'GIMMS_NDVI_trend'

    # labels = [
    #     'wetter greening',
    #     'dryer greening',
    #     'wetter browning',
    #     'dryer browning',
    # ]
    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    # labels = [
    #     'dryer greening',
    #     'wetter greening',
    #     'dryer browning',
    #     'wetter browning',
    # ]
    ################## change area ##################
    # suptitle = f'{year_range} {x_variable} {y_variable}'
    limited_area = ['energy_limited', 'water_limited', ]
    period_list = ['early', 'peak', 'late', ]
    P_PET_long_term_dic = P_PET_ratio()

    HI_zone_class_dic = P_PET_reclass(P_PET_long_term_dic)
    # HI_zone_class_arr = DIC_and_TIF().pix_dic_to_spatial_arr(HI_zone_class_dic)
    # plt.imshow(HI_zone_class_arr)
    # plt.colorbar()
    # plt.show()
    # year_range_list = ['1982-2015', '2002-2015']
    # flag = 0
    # plt.figure()
    for period in period_list:
        folder = f'during_{period}_{year_range}'
        # dic_all = DIC_and_TIF().void_spatial_dic_dic()
        dic_all = {}
        for f in T.listdir(join(fdir, folder)):
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, folder, f)
            arr = np.load(fpath)
            T.mask_999999_arr(arr)
            var_name = f.replace('.npy', '')
            var_name = var_name.replace(f'{year_range}_during_', '')
            var_name = var_name.replace(f'{period}_', '')
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            print(var_name)
            dic_all[var_name] = dic
        df = T.spatial_dics_to_df(dic_all)

        r_list = []
        for i, row in df.iterrows():
            r, c = row.pix
            r_list.append(r)
        df['r'] = r_list
        df = df[df['r'] < 120]
        T.add_dic_to_df(df, HI_zone_class_dic, 'HI_class')
        x_variable_p_value = f'{x_variable.replace("_trend", "_p_value")}'
        y_variable_p_value = f'{y_variable.replace("_trend", "_p_value")}'
        df_new = df[[x_variable, y_variable,
                     x_variable_p_value,
                     y_variable_p_value,
                     'HI_class']]
        df_new = df_new.dropna()
        T.print_head_n(df_new)
        zones_list = T.get_df_unique_val_list(df, 'HI_class')

        # df_greening_non_sig = df_new[df_new[y_variable_p_value]>0.1]
        # df_greening_sig = df_new[df_new[y_variable_p_value]<0.1]
        df_greening_sig = df_new
        df_greening_sig_greening = df_greening_sig[df_greening_sig[y_variable] > 0]
        df_greening_sig_browning = df_greening_sig[df_greening_sig[y_variable] < 0]
        # print(len(df_new))
        # print('non-sig',len(df_greening_non_sig)/len(df_new))
        print('sig greening', len(df_greening_sig_greening) / len(df_new))
        print('sig browning', len(df_greening_sig_browning) / len(df_new))
        parts = []
        labels = []
        flag = 0
        color_list = ['g', 'cyan', 'yellow', 'r', ]
        color_list_all = []
        for y_trend in greening_trend_list:
            if y_trend == 'greening':
                df_select_y = df_greening_sig_greening
            elif y_trend == 'browning':
                df_select_y = df_greening_sig_browning
            else:
                raise UserWarning
            for x_trend in x_trend_list:
                if x_trend == '> 0':
                    df_select_x = df_select_y[df_select_y[x_variable] >= 0]
                elif x_trend == '< 0':
                    df_select_x = df_select_y[df_select_y[x_variable] < 0]
                else:
                    raise UserWarning
                sum_ = 0
                colors_xtrend = color_list[flag]
                flag += 1
                for zone in zones_list:
                    df_zone = df_select_x[df_select_x['HI_class'] == zone]
                    ratio = len(df_zone) / len(df_select_x)
                    ratio_total = len(df_zone) / len(df_greening_sig)
                    parts.append(ratio_total)
                    labels.append(zone + '\n' + str(round(ratio * 100)) + '%')
                    # print(zone,'\n',y_trend,x_variable,x_trend,'\n',ratio,ratio_total)
                    sum_ += ratio
                    color_list_all.append(colors_xtrend)
        wedges, texts = plt.pie(parts, labels=labels, colors=color_list_all, shadow=False)
        # plt.pie(parts,labels=labels)
        for w in wedges:
            w.set_linewidth(2)
            w.set_edgecolor('w')
        plt.show()


def plot_pie_chart_trend_1():
    ################## change area ##################
    # fdir = '/Volumes/NVME2T/wen_proj/20220107/OneDrive_1_2022-1-9/1982-2015_first_last_five_years'
    fdir = '/Volumes/NVME2T/wen_proj/20220111/trend_calculation_anomaly'
    water_balance_tif = '/Volumes/NVME2T/wen_proj/20220107/HI_difference.tif'
    year_range = '2002-2015'
    # year_range = '1982-2015'

    x_variable = 'Aridity_trend'
    # x_variable = 'VPD'
    y_variable = 'MODIS_NDVI_trend'
    # y_variable = 'NIRv_trend'
    # y_variable = 'GIMMS_NDVI_trend'

    # labels = [
    #     'wetter greening',
    #     'dryer greening',
    #     'wetter browning',
    #     'dryer browning',
    # ]
    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    # labels = [
    #     'dryer greening',
    #     'wetter greening',
    #     'dryer browning',
    #     'wetter browning',
    # ]
    ################## change area ##################
    # suptitle = f'{year_range} {x_variable} {y_variable}'
    limited_area = ['energy_limited', 'water_limited', ]
    period_list = ['early', 'peak', 'late', ]
    P_PET_long_term_dic = P_PET_ratio()

    HI_zone_class_dic = P_PET_reclass(P_PET_long_term_dic)
    # HI_zone_class_arr = DIC_and_TIF().pix_dic_to_spatial_arr(HI_zone_class_dic)
    # plt.imshow(HI_zone_class_arr)
    # plt.colorbar()
    # plt.show()
    # year_range_list = ['1982-2015', '2002-2015']
    # flag = 0
    # plt.figure()
    for period in period_list:
        folder = f'during_{period}_{year_range}'
        # dic_all = DIC_and_TIF().void_spatial_dic_dic()
        dic_all = {}
        for f in T.listdir(join(fdir, folder)):
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, folder, f)
            arr = np.load(fpath)
            T.mask_999999_arr(arr)
            var_name = f.replace('.npy', '')
            var_name = var_name.replace(f'{year_range}_during_', '')
            var_name = var_name.replace(f'{period}_', '')
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            dic_all[var_name] = dic
        df = T.spatial_dics_to_df(dic_all)

        r_list = []
        for i, row in df.iterrows():
            r, c = row.pix
            r_list.append(r)
        df['r'] = r_list
        df = df[df['r'] < 120]
        T.add_dic_to_df(df, HI_zone_class_dic, 'HI_class')
        x_variable_p_value = f'{x_variable.replace("_trend", "_p_value")}'
        y_variable_p_value = f'{y_variable.replace("_trend", "_p_value")}'
        df_new = df[[x_variable, y_variable,
                     x_variable_p_value,
                     y_variable_p_value,
                     'HI_class']]
        df_new = df_new.dropna()
        T.print_head_n(df_new)
        zones_list = T.get_df_unique_val_list(df, 'HI_class')

        # df_greening_non_sig = df_new[df_new[y_variable_p_value]>0.1]
        df_greening_sig = df_new[df_new[y_variable_p_value] < 0.1]
        # df_greening_sig = df_new

        # print(len(df_new))
        # print('non-sig',len(df_greening_non_sig)/len(df_new))
        # print('sig greening',len(df_greening_sig_greening)/len(df_new))
        # print('sig browning',len(df_greening_sig_browning)/len(df_new))
        parts = []
        labels = []
        color_list = ['g', 'cyan', 'yellow', 'r', ]
        color_list_all = []
        flag = 0
        for zone in zones_list:
            df_zone = df_greening_sig[df_greening_sig['HI_class'] == zone]
            for y_trend in greening_trend_list:
                if y_trend == 'greening':
                    df_select_y = df_zone[df_zone[y_variable] > 0]
                elif y_trend == 'browning':
                    df_select_y = df_zone[df_zone[y_variable] < 0]
                else:
                    raise UserWarning
                for x_trend in x_trend_list:
                    if x_trend == '> 0':
                        df_select_x = df_select_y[df_select_y[x_variable] >= 0]
                    elif x_trend == '< 0':
                        df_select_x = df_select_y[df_select_y[x_variable] < 0]
                    else:
                        raise UserWarning
                    sum_ = 0
                    colors_xtrend = color_list[flag]
                    print(flag)
                    ratio = len(df_select_x) / len(df_zone)
                    ratio_total = len(df_select_x) / len(df_greening_sig)
                    parts.append(ratio_total)
                    label_i = f'{zone}\n{y_trend}-{x_variable}{x_trend}'
                    labels.append(label_i + '\n' + str(round(ratio * 100)) + '%')
                    # print(zone,'\n',y_trend,x_variable,x_trend,'\n',ratio,ratio_total)
                    sum_ += ratio
                    color_list_all.append(colors_xtrend)
            flag += 1

        wedges, texts = plt.pie(parts, labels=labels, colors=color_list_all, shadow=False)
        # plt.pie(parts,labels=labels)
        for w in wedges:
            w.set_linewidth(2)
            w.set_edgecolor('w')
        plt.show()


def add_dic_to_df(df, dic, key_name):
    val_list = []
    for i, row in df.iterrows():
        pix = row['pix']
        if not pix in dic:
            val = None
        else:
            val = dic[pix]
        val_list.append(val)
    df[key_name] = val_list


def drop_n_std(vals, n=1):
    vals = np.array(vals)
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    up = mean + n * std
    down = mean - n * std
    vals[vals > up] = np.nan
    vals[vals < down] = np.nan
    return vals


def P_PET_ratio(P_PET_fdir):
    # fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
    fdir = P_PET_fdir
    dic = T.load_npy_dir(fdir)
    dic_long_term = {}
    for pix in dic:
        vals = dic[pix]
        vals = np.array(vals)
        T.mask_999999_arr(vals)
        vals[vals == 0] = np.nan
        if np.isnan(np.nanmean(vals)):
            continue
        vals = drop_n_std(vals)
        long_term_vals = np.nanmean(vals)
        dic_long_term[pix] = long_term_vals
    return dic_long_term


def plot_bar_trend_ratio():
    ################## change area ##################
    # fdir = '/Volumes/NVME2T/wen_proj/20220107/OneDrive_1_2022-1-9/1982-2015_first_last_five_years'
    fdir = '/Volumes/NVME2T/wen_proj/20220111/trend_calculation_anomaly'
    P_PET_fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
    P_PET_long_term_dic = P_PET_ratio(P_PET_fdir)
    HI_zone_class_dic = P_PET_reclass(P_PET_long_term_dic)
    year_range = '2002-2015'
    # year_range = '1982-2015'

    x_variable = 'Aridity'
    # x_variable = 'VPD'
    # y_variable = 'MODIS_NDVI'

    y_variable = 'CSIF_fpar'
    # y_variable = 'NIRv'
    # y_variable = 'GIMMS_NDVI'
    x_variable = x_variable + '_trend'
    y_variable = y_variable + '_trend'
    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    ################## change area ##################
    # suptitle = f'{year_range} {x_variable} {y_variable}'
    limited_area = ['energy_limited', 'water_limited', ]
    period_list = ['early', 'peak', 'late', ]

    # HI_zone_class_arr = DIC_and_TIF().pix_dic_to_spatial_arr(HI_zone_class_dic)
    # plt.imshow(HI_zone_class_arr)
    # plt.colorbar()
    # plt.show()
    # year_range_list = ['1982-2015', '2002-2015']
    # flag = 0
    # plt.figure()
    for period in period_list:
        folder = f'during_{period}_{year_range}'
        # dic_all = DIC_and_TIF().void_spatial_dic_dic()
        dic_all = {}
        for f in T.listdir(join(fdir, folder)):
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir, folder, f)
            arr = np.load(fpath)
            T.mask_999999_arr(arr)
            var_name = f.replace('.npy', '')
            var_name = var_name.replace(f'{year_range}_during_', '')
            var_name = var_name.replace(f'{period}_', '')
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            dic_all[var_name] = dic
        df = T.spatial_dics_to_df(dic_all)

        r_list = []
        for i, row in df.iterrows():
            r, c = row.pix
            r_list.append(r)
        df['r'] = r_list
        df = df[df['r'] < 120]
        T.add_dic_to_df(df, HI_zone_class_dic, 'HI_class')
        x_variable_p_value = f'{x_variable.replace("_trend", "_p_value")}'
        y_variable_p_value = f'{y_variable.replace("_trend", "_p_value")}'
        df_new = df[[x_variable, y_variable,
                     x_variable_p_value,
                     y_variable_p_value,
                     'HI_class']]
        df_new = df_new.dropna()
        T.print_head_n(df_new)
        zones_list = T.get_df_unique_val_list(df, 'HI_class')

        # df_greening_non_sig = df_new[df_new[y_variable_p_value]>0.1]
        df_greening_sig = df_new[df_new[y_variable_p_value] < 0.1]
        # df_greening_sig = df_new

        # print(len(df_new))
        # print('non-sig',len(df_greening_non_sig)/len(df_new))
        # print('sig greening',len(df_greening_sig_greening)/len(df_new))
        # print('sig browning',len(df_greening_sig_browning)/len(df_new))
        flag = 0
        for zone in zones_list:
            df_zone = df_greening_sig[df_greening_sig['HI_class'] == zone]
            parts = []
            bottom = 0
            color_list = ['g', 'cyan', 'yellow', 'r', ]
            flag1 = 0
            for y_trend in greening_trend_list:
                if y_trend == 'greening':
                    df_select_y = df_zone[df_zone[y_variable] > 0]
                elif y_trend == 'browning':
                    df_select_y = df_zone[df_zone[y_variable] < 0]
                else:
                    raise UserWarning
                for x_trend in x_trend_list:
                    if x_trend == '> 0':
                        df_select_x = df_select_y[df_select_y[x_variable] >= 0]
                    elif x_trend == '< 0':
                        df_select_x = df_select_y[df_select_y[x_variable] < 0]
                    else:
                        raise UserWarning
                    sum_ = 0
                    ratio = len(df_select_x) / len(df_zone)
                    plt.bar(zones_list[flag], ratio, bottom=bottom, color=color_list[flag1])
                    text = f'{y_trend}\n{x_variable}{x_trend}'
                    plt.text(zones_list[flag], bottom + ratio / 2, text)
                    flag1 += 1
                    bottom += ratio
                    parts.append(ratio)
                    label_i = f'{zone}\n{y_trend}-{x_variable}{x_trend}'
                    # labels.append(label_i+'\n'+str(round(ratio*100))+'%')
                    # print(zone,'\n',y_trend,x_variable,x_trend,'\n',ratio,ratio_total)
                    sum_ += ratio
                    # color_list_all.append(colors_xtrend)
            flag += 1
        plt.show()


def mask_NDVI(ndvi_mask_tif, df):
    # tif = '/Volumes/NVME2T/wen_proj/20220111/NDVI_mask.tif'
    tif = ndvi_mask_tif
    arr = ToRaster().raster2array(tif)[0]
    # plt.imshow(arr)
    # plt.show()
    dic = DIC_and_TIF().spatial_arr_to_dic(arr)
    index_drop = []
    spatial_dic = DIC_and_TIF().void_spatial_dic_nan()
    for i, row in df.iterrows():
        pix = row.pix
        val = dic[pix]
        spatial_dic[pix] = 1
        if val < -999:
            index_drop.append(i)
    arr_1 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
    plt.imshow(arr_1)
    plt.show()
    print(index_drop)
    exit()

    return dic


def plot_ratio_trend():
    fdir = '/Volumes/NVME2T/wen_proj/20220111/1982-2015_during_early'
    P_PET_fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'

    x_variable = 'Aridity'
    y_variable = 'GIMMS_NDVI'
    n = 15  # every n year trend

    P_PET_long_term_dic = P_PET_ratio(P_PET_fdir)
    HI_zone_class_dic = P_PET_reclass(P_PET_long_term_dic)

    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    x_fname = f'1982-2015_during_early_{x_variable}.npy'
    y_fname = f'1982-2015_during_early_{y_variable}.npy'
    x_fpath = join(fdir, x_fname)
    y_fpath = join(fdir, y_fname)
    dicx = T.load_npy(x_fpath)
    dicy = T.load_npy(y_fpath)
    vals_len = 9999
    for pix in dicy:
        vals = dicy[pix]
        if len(vals) != 0:
            vals_len = len(vals)
        break

    dic_all = {}
    for pix in dicy:
        dic_all[pix] = {}
    for i in tqdm(range(vals_len)):
        for pix in dicy:
            if not pix in dicx:
                continue
            x_vals = dicx[pix]
            if i + n >= vals_len:
                continue
            y_vals = dicy[pix]
            indexs = list(range(i, i + n))
            x_vals_pick = T.pick_vals_from_1darray(x_vals, indexs)
            y_vals_pick = T.pick_vals_from_1darray(y_vals, indexs)
            try:
                x_trend, _, _ = KDE_plot().linefit(range(len(x_vals_pick)), x_vals_pick)
                y_trend, _, _ = KDE_plot().linefit(range(len(y_vals_pick)), y_vals_pick)
            except:
                x_trend = np.nan
                y_trend = np.nan
            dic_i = {
                f'{i}_{x_variable}_trend': x_trend,
                f'{i}_{y_variable}_trend': y_trend
            }
            dic_all[pix].update(dic_i)
    df = T.dic_to_df(dic_all, 'pix')
    r_list = []
    for i, row in df.iterrows():
        r, c = row.pix
        r_list.append(r)
    df['r'] = r_list
    df = df[df['r'] < 120]
    # df = df.dropna(how='any')
    T.add_dic_to_df(df, HI_zone_class_dic, 'HI_class')
    zones_list = T.get_df_unique_val_list(df, 'HI_class')
    for zone in zones_list:
        df_zone = df[df['HI_class'] == zone]
        y_dic = {}
        plt.figure()
        for i in tqdm(range(vals_len - n), desc=zone):
            x_col_name = f'{i}_{x_variable}_trend'
            y_col_name = f'{i}_{y_variable}_trend'
            y_dic_i = {}
            for y_trend in greening_trend_list:
                if y_trend == 'greening':
                    df_select_y = df_zone[df_zone[y_col_name] > 0]
                elif y_trend == 'browning':
                    df_select_y = df_zone[df_zone[y_col_name] < 0]
                else:
                    raise UserWarning

                for x_trend in x_trend_list:
                    if x_trend == '> 0':
                        df_select_x = df_select_y[df_select_y[x_col_name] >= 0]
                    elif x_trend == '< 0':
                        df_select_x = df_select_y[df_select_y[x_col_name] < 0]
                    else:
                        raise UserWarning
                    ratio = len(df_select_x) / len(df_zone)
                    # y_list.append(ratio)
                    # plt.scatter(i, ratio,color=color_list[flag1])
                    text = f'{y_trend}\n{x_variable}{x_trend}'
                    y_dic_i[text] = ratio

            y_dic[i] = y_dic_i
        keys_list = []
        for i in y_dic:
            y_dic_i = y_dic[i]
            for key in y_dic_i:
                keys_list.append(key)
            break
        flag1 = 0
        color_list = ['g', 'cyan', 'purple', 'r', ]
        for key in keys_list:
            y_list = []
            x_list = []
            for i in range(len(y_dic)):
                y_dic_i = y_dic[i]
                val = y_dic_i[key]
                x_list.append(i)
                y_list.append(val)
            plt.plot(x_list, y_list, color=color_list[flag1], label=key)
            flag1 += 1

        plt.legend()
        plt.title(zone)
    plt.show()
    T.print_head_n(df)


def plot_vectors1():
    fdir = '/Volumes/NVME2T/wen_proj/20220107/OneDrive_1_2022-1-9/1982-2015_first_last_five_years'
    water_balance_tif = '/Volumes/NVME2T/wen_proj/20220107/HI_difference.tif'
    P_PET_fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'

    P_PET_long_term_dic = P_PET_ratio(P_PET_fdir)
    HI_zone_class_dic = P_PET_reclass(P_PET_long_term_dic)
    period_list = ['early', 'peak', 'late', ]
    order_list = ['first', 'last']
    water_balance_dic = DIC_and_TIF().spatial_tif_to_dic(water_balance_tif)
    for period in period_list:
        data_dic = {'HI': [], 'NDVI': []}
        for order in order_list:
            folder = f'during_{period}_1982-2015_{order}_five'
            HI_f = f'during_{period}_VPD_mean.npy'
            NDVI_f = f'during_{period}_GIMMS_NDVI_mean.npy'
            fpath_HI = join(fdir, folder, HI_f)
            fpath_NDVI = join(fdir, folder, NDVI_f)
            HI_arr = np.load(fpath_HI)
            NDVI_arr = np.load(fpath_NDVI)
            HI_arr[HI_arr < -9999] = np.nan
            NDVI_arr[NDVI_arr < -9999] = np.nan
            HI_dic = DIC_and_TIF().spatial_arr_to_dic(HI_arr)
            NDVI_dic = DIC_and_TIF().spatial_arr_to_dic(NDVI_arr)
            data_dic['HI'].append(HI_dic)
            data_dic['NDVI'].append(NDVI_dic)
        x1_dic = data_dic['HI'][0]
        x2_dic = data_dic['HI'][1]
        y1_dic = data_dic['NDVI'][0]
        y2_dic = data_dic['NDVI'][1]

        key_list = []
        r_list = []
        x1_list = []
        x2_list = []
        y1_list = []
        y2_list = []
        wb_list = []
        for key in x1_dic:
            r, c = key
            key_list.append(key)
            x1 = x1_dic[key]
            x2 = x2_dic[key]
            y1 = y1_dic[key]
            y2 = y2_dic[key]
            wb = water_balance_dic[key]

            x1_list.append(x1)
            x2_list.append(x2)
            y1_list.append(y1)
            y2_list.append(y2)
            r_list.append(r)
            wb_list.append(wb)

        df = pd.DataFrame()
        df['pix'] = key_list
        df['r'] = r_list
        df['x1'] = x1_list
        df['x2'] = x2_list
        df['y1'] = y1_list
        df['y2'] = y2_list
        df['wb'] = wb_list
        df = T.add_dic_to_df(df, HI_zone_class_dic, 'HI_class')

        limited_area = T.get_df_unique_val_list(df, 'HI_class')
        limited_area = list(limited_area)
        # print(limited_area)
        # limited_area.remove('Humid')
        df = df.dropna()
        df_copy = copy.copy(df)
        # for limited in limited_area:
        df_ltd = df_copy[df_copy['HI_class'] != 'Humid']
        # df_ltd = df_copy[df_copy['HI_class'] == 'Humid']
        df = df_ltd
        df = df[df['r'] < 120]
        df = df[df['x1'] < 3]
        df = df[df['x1'] < 3]
        df = df[df['x1'] != 0]
        df = df[df['x2'] != 0]
        # if len(df) > 1000:
        #     df = df.sample(n=1000)

        plt.figure()
        for i, row in df.iterrows():
            x = row.x1
            x2 = row.x2
            y = row.y1
            y2 = row.y2
            dx = x2 - x
            dy = y2 - y
            if dy > 0 and dx > 0:
                plt.arrow(x, y, dx, dy, ec='g', fc='g', alpha=0.3, head_width=0)
            elif dy > 0 and dx < 0:
                plt.arrow(x, y, dx, dy, ec='cyan', fc='cyan', alpha=0.3, head_width=0)
            elif dy < 0 and dx > 0:
                plt.arrow(x, y, dx, dy, ec='purple', fc='purple', alpha=0.3, head_width=0)
            elif dy < 0 and dx < 0:
                plt.arrow(x, y, dx, dy, ec='r', fc='r', alpha=0.3, head_width=0)
        # plt.title(limited)
        plt.show()


def NDVI_seasonal_compose():
    NDVI_dir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/'
    outdir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal'
    T.mk_dir(outdir)
    year_range = list(range(1982, 2016))
    tif_dir = join(NDVI_dir, 'tif')
    season_dic = {
        'spring': (3, 4, 5),
        'summer': (6, 7, 8),
        'autumn': (9, 10, 11),
        'winter': (12, 1, 2),
    }

    season_fpath_dic = {}

    for season in season_dic:
        season_fpath_dic[season] = {}
        for y in year_range:
            season_fpath_dic[season][y] = []
    for season in season_dic:
        months_list = season_dic[season]
        for f in T.listdir(tif_dir):
            if not f.endswith('.tif'):
                continue
            date = f.split('.')[0]
            year = date[:4]
            month = date[4:]
            month = int(month)
            year = int(year)
            if month in months_list:
                season_fpath_dic[season][year].append(join(tif_dir, f))
    for season in season_fpath_dic:
        outdir_i = join(outdir, season)
        T.mk_dir(outdir_i)
        for year in year_range:
            print(season, year)
            outf = join(outdir_i, str(year) + '.tif')
            flist = season_fpath_dic[season][year]
            Pre_Process().compose_tif_list(flist, outf)


def NDVI_seasonal_transform():
    fdir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal'
    outdir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix'
    T.mk_dir(outdir)
    for season in T.listdir(fdir):
        print(season)
        fdir_i = join(fdir, season)
        outdir_i = join(outdir, season)
        Pre_Process().data_transform(fdir_i, outdir_i)


def NDVI_trend_spatial():
    fdir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix'
    outdir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix_trend_p'
    T.mk_dir(outdir)
    # NDVI_mask_tif = '/Volumes/NVME2T/wen_proj/20220111/NDVI_mask.tif'
    # NDVI_mask_dic = DIC_and_TIF().spatial_tif_to_dic(NDVI_mask_tif)
    season_list = []
    for season in T.listdir(fdir):
        print(season)
        outf = join(outdir, season)
        season_list.append(season)
        fdir_i = join(fdir, season)
        dic = T.load_npy_dir(fdir_i)
        vals_dic = {}
        for pix in tqdm(dic, desc=season):
            vals = dic[pix]
            T.mask_999999_arr(vals)
            try:
                k, _, _ = KDE_plot().linefit(range(len(vals)), vals)
                r, p = T.nan_correlation(range(len(vals)), vals)
                if p > 0.05:
                    continue
                vals_dic[pix] = (k, p)
            except:
                pass
        T.save_npy(vals_dic, outf)


def NDVI_trend_line_and_spatial():
    fdir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix'
    NDVI_mask_tif = '/Volumes/NVME2T/wen_proj/20220111/NDVI_mask.tif'
    NDVI_mask_dic = DIC_and_TIF().spatial_tif_to_dic(NDVI_mask_tif)
    all_dic = {}
    season_list = []
    for season in T.listdir(fdir):
        # if not season == 'spring':
        #     continue
        season_list.append(season)
        fdir_i = join(fdir, season)
        dic = T.load_npy_dir(fdir_i)
        # vals_dic = {}
        # for pix in dic:
        #     vals = dic[pix]
        #     T.mask_999999_arr(vals)
        #     try:
        #         k,_,_ = KDE_plot().linefit(range(len(vals)),vals)
        #         trend_dic[pix] = k
        #     except:
        #         pass
        all_dic[season] = dic
    df = T.spatial_dics_to_df(all_dic)
    r_list = []
    for i, row in df.iterrows():
        pix = row.pix
        r, c = pix
        r_list.append(r)
    df['r'] = r_list
    df = df[df['r'] < 120]
    K = KDE_plot()
    T.add_spatial_dic_to_df(df, NDVI_mask_dic, 'NDVI_mask')
    df = df.dropna()
    for season in season_list:
        print(season)
        vals_all = []
        for i, row in df.iterrows():
            vals = row[season]
            vals_all.append(vals)
        #     x = list(range(len(vals)))
        #     y = vals
        #     try:
        #         k,_,_ = K.linefit(x,y)
        #     except:
        #         k = np.nan
        #     k_list.append(k)
        # df[f'{season}_trend'] = k_list
        vals_all = np.array(vals_all)
        vals_all_T = vals_all.T
        mean_list = []
        std_list = []
        for y in vals_all_T:
            mean = np.nanmean(y)
            std = np.nanstd(y)
            mean_list.append(mean)
            std_list.append(std)
        std_list = np.array(std_list) / 8
        mean_list = np.array(mean_list)
        plt.figure()
        plt.plot(mean_list, label=season)
        plt.fill_between(range(len(mean_list)), mean_list + std_list, mean_list - std_list, alpha=0.2)
        plt.legend()
        plt.figure()
        plt.title(season)
        dic = all_dic[season]
        trend_dic = {}
        for pix in dic:
            vals = dic[pix]
            x = list(range(len(vals)))
            y = vals
            r,p = stats.pearsonr(x,y)
            k,_,_ = K.linefit(x,y)
            if p < 0.05:
                trend_dic[pix] = k
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(trend_dic)
        land_tif = '/Volumes/SSD/drought_response/conf/land.tif'
        DIC_and_TIF().plot_back_ground_arr(land_tif)
        mean_arr = np.nanmean(arr)
        std_arr = np.nanstd(arr)
        up = mean_arr + std_arr
        down = mean_arr - std_arr
        plt.imshow(arr,vmin=down,vmax=up,cmap='jet')
        plt.colorbar()

    plt.show()


def get_variables_trend_df(period, x_variable='Aridity_trend'):
    '''
    period: 'early', 'peak', 'late'
    '''

    ################## change area ##################
    fdir = '/Volumes/NVME2T/wen_proj/20220111/trend_calculation_anomaly'
    P_PET_dir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'

    # year_range = '2002-2015'
    year_range = '1982-2015'

    x_trend_list = ['> 0', '< 0', ]
    ################## change area ##################
    P_PET_long_term_dic = P_PET_ratio(P_PET_dir)

    HI_zone_class_dic = P_PET_reclass(P_PET_long_term_dic)
    folder = f'during_{period}_{year_range}'
    # dic_all = DIC_and_TIF().void_spatial_dic_dic()
    dic_all = {}
    for f in tqdm(T.listdir(join(fdir, folder))):
        if not f.endswith('.npy'):
            continue
        if not x_variable in f:
            continue
        fpath = join(fdir, folder, f)
        arr = np.load(fpath)
        T.mask_999999_arr(arr)
        var_name = f.replace('.npy', '')
        var_name = var_name.replace(f'{year_range}_during_', '')
        var_name = var_name.replace(f'{period}_', '')
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        dic_all[var_name] = dic
    df = T.spatial_dics_to_df(dic_all)
    r_list = []
    for i, row in df.iterrows():
        r, c = row.pix
        r_list.append(r)
    df['r'] = r_list
    df = df[df['r'] < 120]
    T.add_spatial_dic_to_df(df, HI_zone_class_dic, 'HI_class')
    return df


def plot_df_ratio(df, x_variable, y_variable):
    zones_list = T.get_df_unique_val_list(df, 'HI_class')
    parts = []
    labels = []
    color_list = ['g', 'cyan', 'yellow', 'r', ]
    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    color_list_all = []
    flag = 0
    for zone in zones_list:
        df_zone = df[df['HI_class'] == zone]
        for y_trend in greening_trend_list:
            if y_trend == 'greening':
                df_select_y = df_zone[df_zone[y_variable] > 0]
            elif y_trend == 'browning':
                df_select_y = df_zone[df_zone[y_variable] < 0]
            else:
                raise UserWarning
            for x_trend in x_trend_list:
                if x_trend == '> 0':
                    df_select_x = df_select_y[df_select_y[x_variable] >= 0]
                elif x_trend == '< 0':
                    df_select_x = df_select_y[df_select_y[x_variable] < 0]
                else:
                    raise UserWarning
                sum_ = 0
                colors_xtrend = color_list[flag]
                print(flag)
                ratio = len(df_select_x) / len(df_zone)
                # ratio_total = len(df_select_x) / len(df)
                # parts.append(ratio_total)
                label_i = f'{zone}\n{y_trend}-{x_variable}{x_trend}'
                labels.append(label_i + '\n' + str(round(ratio * 100)) + '%')
                # print(zone,'\n',y_trend,x_variable,x_trend,'\n',ratio,ratio_total)
                sum_ += ratio
                color_list_all.append(colors_xtrend)
        flag += 1

    wedges, texts = plt.pie(parts, labels=labels, colors=color_list_all, shadow=False)
    # plt.pie(parts,labels=labels)
    for w in wedges:
        w.set_linewidth(2)
        w.set_edgecolor('w')
    plt.show()


def plot_ratio():
    period = 'early'
    x_variable = 'Aridity_trend'
    period_to_season = {
        'early': 'spring',
        'peak': 'summer',
        'late': 'autum',
    }
    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    NDVI_spatial_trend_dir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix_trend_p'
    df = get_variables_trend_df(period=period, x_variable=x_variable)
    ndvi_trend_dic_f = join(NDVI_spatial_trend_dir, period_to_season[period] + '.npy')
    ndvi_trend_dic = T.load_npy(ndvi_trend_dic_f)
    ndvi_trend_dic_r = {}
    ndvi_trend_dic_p = {}
    for pix in ndvi_trend_dic:
        r, p = ndvi_trend_dic[pix]
        if np.isnan(r):
            continue
        ndvi_trend_dic_p[pix] = p
        ndvi_trend_dic_r[pix] = r
    df = T.add_spatial_dic_to_df(df, ndvi_trend_dic_p, 'NDVI_trend_p')
    df = T.add_spatial_dic_to_df(df, ndvi_trend_dic_r, 'NDVI_trend')
    df = df.dropna()

    df = df[df['NDVI_trend_p'] < 0.05]
    HI_class = T.get_df_unique_val_list(df, 'HI_class')
    for zone in HI_class:
        df_zone = df[df['HI_class'] == zone]
        bottom = 0
        for greening in greening_trend_list:
            if greening == 'greening':
                df_ndvi_trend = df_zone[df_zone['NDVI_trend'] > 0]
            else:
                df_ndvi_trend = df_zone[df_zone['NDVI_trend'] <= 0]
            for x_trend in x_trend_list:
                if x_trend == '> 0':
                    df_x_trend = df_ndvi_trend[df_ndvi_trend[x_variable] > 0]
                else:
                    df_x_trend = df_ndvi_trend[df_ndvi_trend[x_variable] <= 0]
                ratio = len(df_x_trend) / len(df_zone)
                # print(greening,x_variable,x_trend,ratio)
                text = '_'.join((greening, x_variable + x_trend))
                x = zone
                y = ratio
                plt.bar(x, y, bottom=bottom)
                plt.text(x, (y / 2 + bottom), text, ha='left')
                # plt.xticks(rotation=90)
                bottom += y
    plt.show()
    # T.print_head_n(df)


def plot_ratio_trend1():
    fdir = '/Volumes/NVME2T/wen_proj/20220111/1982-2015_during_early'
    NDVI_new_fdir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix/spring'
    P_PET_fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'

    x_variable = 'Aridity'
    y_variable = 'GIMMS_NDVI'
    n = 15  # every n year trend

    P_PET_long_term_dic = P_PET_ratio(P_PET_fdir)
    HI_zone_class_dic = P_PET_reclass(P_PET_long_term_dic)

    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    x_fname = f'1982-2015_during_early_{x_variable}.npy'
    x_fpath = join(fdir, x_fname)
    dicx = T.load_npy(x_fpath)
    dicy = T.load_npy_dir(NDVI_new_fdir)
    vals_len = 9999
    for pix in dicy:
        vals = dicy[pix]
        if len(vals) != 0:
            vals_len = len(vals)
        break

    dic_all = {}
    for pix in dicy:
        dic_all[pix] = {}
    for i in tqdm(range(vals_len)):
        for pix in dicy:
            if not pix in dicx:
                continue
            x_vals = dicx[pix]
            if i + n >= vals_len:
                continue
            y_vals = dicy[pix]
            indexs = list(range(i, i + n))
            x_vals_pick = T.pick_vals_from_1darray(x_vals, indexs)
            y_vals_pick = T.pick_vals_from_1darray(y_vals, indexs)
            try:
                x_trend, _, _ = KDE_plot().linefit(range(len(x_vals_pick)), x_vals_pick)
                y_trend, _, _ = KDE_plot().linefit(range(len(y_vals_pick)), y_vals_pick)
            except:
                x_trend = np.nan
                y_trend = np.nan
            dic_i = {
                f'{i}_{x_variable}_trend': x_trend,
                f'{i}_{y_variable}_trend': y_trend
            }
            dic_all[pix].update(dic_i)
    df = T.dic_to_df(dic_all, 'pix')
    r_list = []
    for i, row in df.iterrows():
        r, c = row.pix
        r_list.append(r)
    df['r'] = r_list
    df = df[df['r'] < 120]
    # df = df.dropna(how='any')
    T.add_spatial_dic_to_df(df, HI_zone_class_dic, 'HI_class')
    zones_list = T.get_df_unique_val_list(df, 'HI_class')
    for zone in zones_list:
        df_zone = df[df['HI_class'] == zone]
        y_dic = {}
        plt.figure()
        for i in tqdm(range(vals_len - n), desc=zone):
            x_col_name = f'{i}_{x_variable}_trend'
            y_col_name = f'{i}_{y_variable}_trend'
            y_dic_i = {}
            for y_trend in greening_trend_list:
                if y_trend == 'greening':
                    df_select_y = df_zone[df_zone[y_col_name] > 0]
                elif y_trend == 'browning':
                    df_select_y = df_zone[df_zone[y_col_name] < 0]
                else:
                    raise UserWarning

                for x_trend in x_trend_list:
                    if x_trend == '> 0':
                        df_select_x = df_select_y[df_select_y[x_col_name] >= 0]
                    elif x_trend == '< 0':
                        df_select_x = df_select_y[df_select_y[x_col_name] < 0]
                    else:
                        raise UserWarning
                    ratio = len(df_select_x) / len(df_zone)
                    # y_list.append(ratio)
                    # plt.scatter(i, ratio,color=color_list[flag1])
                    text = f'{y_trend}\n{x_variable}{x_trend}'
                    y_dic_i[text] = ratio

            y_dic[i] = y_dic_i
        keys_list = []
        for i in y_dic:
            y_dic_i = y_dic[i]
            for key in y_dic_i:
                keys_list.append(key)
            break
        flag1 = 0
        color_list = ['g', 'cyan', 'purple', 'r', ]
        for key in keys_list:
            y_list = []
            x_list = []
            for i in range(len(y_dic)):
                y_dic_i = y_dic[i]
                val = y_dic_i[key]
                x_list.append(i)
                y_list.append(val)
            plt.plot(x_list, y_list, color=color_list[flag1], label=key)
            flag1 += 1

        plt.legend()
        plt.title(zone)
    plt.show()
    # T.print_head_n(df)
def ndvi_spatial_trend_check():
    fdir = '/Volumes/NVME2T/wen_proj/20220111/Archive(2)'
    period = 'early'
    folder = f'during_{period}_1982-2015'
    f = f'1982-2015_during_{period}_GIMMS_NDVI_trend.npy'
    arr_wen = np.load(join(fdir,folder,f))
    arr_wen[arr_wen<-9999]=np.nan
    mean = np.nanmean(arr_wen)
    std = np.nanstd(arr_wen)
    up = mean + std
    down = mean - std
    plt.imshow(arr_wen,vmax=up,vmin=down)
    plt.colorbar()
    plt.show()

    pass
def check_HI_Class():
    P_PET_fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
    P_PET_long_term_dic = P_PET_ratio(P_PET_fdir)
    HI_zone_class_dic = P_PET_reclass_int(P_PET_long_term_dic)
    arr = DIC_and_TIF().pix_dic_to_spatial_arr(HI_zone_class_dic)
    plt.imshow(arr,cmap='jet_r')
    plt.colorbar()
    plt.show()


def co2_ndvi_limited_by_vpd():
    # 横轴 vpd 纵轴 ndvi 和 co2
    # co2_f = '/Volumes/NVME2T/wen_proj/20220111/1982-2015_during_early/1982-2015_during_early_CO2.npy'
    # vpd_f = '/Volumes/NVME2T/wen_proj/20220111/1982-2015_during_early/1982-2015_during_early_VPD.npy'
    # vpd_f = '/Volumes/NVME2T/wen_proj/20220111/origional/during_early_VPD.npy'
    fdir = '/Volumes/NVME2T/wen_proj/20220111/origional/1982-2015_original_extraction_all_seasons'
    multi_reg_f = '/Volumes/NVME2T/wen_proj/20220111/1982-2015_multi_linearearly_anomaly.npy'
    NDVI_trenf_dir = '/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix_trend_p'
    seasons_list = ['early','peak','late',]
    greening_trend_list = ['greening', 'browning']
    x_trend_list = ['> 0', '< 0', ]
    seasons_dic = {
        'early':'spring',
        'peak':'summer',
        'late':'autumn',
    }
    K = KDE_plot()
    multi_reg_dic = T.load_npy(multi_reg_f)
    co2_multi_reg_dic = {}
    for pix in multi_reg_dic:
        dic_i = multi_reg_dic[pix]
        # early_CO2_multi_reg = dic_i['early_CO2']
        if not 'early_temperature' in dic_i:
            continue
        early_CO2_multi_reg = dic_i['early_temperature']
        co2_multi_reg_dic[pix] = early_CO2_multi_reg
    plt.figure(figsize=(20,24))
    flag = 0
    for season in seasons_list:
        folder = f'1982-2015_extraction_during_{season}_growing_season_static'
        ndvi_dir = f'/Volumes/SSD/drought_response_Wen/data/GIMMS_NDVI/seasonal_perpix/{seasons_dic[season]}'
        HI_trend_df = get_variables_trend_df(season)
        # zones_list = T.get_df_unique_val_list(HI_trend_df, 'HI_class')
        NDVI_trend_f = join(NDVI_trenf_dir,f'{seasons_dic[season]}.npy')
        NDVI_trend_dic = T.load_npy(NDVI_trend_f)
        NDVI_trend_dic_k = {}
        NDVI_trend_dic_p = {}
        for pix in NDVI_trend_dic:
            k,p = NDVI_trend_dic[pix]
            NDVI_trend_dic_k[pix] = k
            NDVI_trend_dic_p[pix] = p
        co2_f = join(fdir,folder,f'during_{season}_CO2.npy')
        vpd_f = join(fdir,folder,f'during_{season}_VPD.npy')
        co2_dic = T.load_npy(co2_f)
        vpd_dic = T.load_npy(vpd_f)
        ndvi_dic = T.load_npy_dir(ndvi_dir)

        vpd_trend_dic = {}
        for pix in vpd_dic:
            vals = vpd_dic[pix]
            k,_,_ = K.linefit(range(len(vals)),vals)
            vpd_trend_dic[pix] = k

        df = HI_trend_df
        df = T.add_spatial_dic_to_df(df,co2_dic,'co2')
        df = T.add_spatial_dic_to_df(df,vpd_dic,'vpd')
        df = T.add_spatial_dic_to_df(df,ndvi_dic,'ndvi')
        df = T.add_spatial_dic_to_df(df,NDVI_trend_dic_k,'NDVI_trend_dic_k')
        df = T.add_spatial_dic_to_df(df,NDVI_trend_dic_p,'NDVI_trend_dic_p')
        df = T.add_spatial_dic_to_df(df,vpd_trend_dic,'vpd_trend')
        df = T.add_spatial_dic_to_df(df,co2_multi_reg_dic,'co2_multi_reg')
        HI_class_dic_new = {}
        for i,row in df.iterrows():
            pix = row.pix
            HI_class = row['HI_class']
            if HI_class == None:
                HI_class_new = None
            elif HI_class == 'Humid':
                HI_class_new = 'Humid'
            else:
                HI_class_new = 'Non-Humid'
            HI_class_dic_new[pix] = HI_class_new
        df = T.add_spatial_dic_to_df(df,HI_class_dic_new,'HI_class_new')

        # df = df[df['NDVI_trend_dic_p']<0.05]
        df = df[df['r']<120]
        zones_list = T.get_df_unique_val_list(HI_trend_df, 'HI_class_new')

        T.print_head_n(df)
        for zone in zones_list:
            df_zone = df[df['HI_class_new']==zone]
            # for greening in greening_trend_list:
            #     if greening == 'greening':
            #         df_greening = df_zone[df_zone['NDVI_trend_dic_k']>0]
            #     else:
            #         df_greening = df_zone[df_zone['NDVI_trend_dic_k']>0]
                # for x_trend in x_trend_list:
                #     if x_trend == '> 0':
                #         df_x_trend = df_greening[df_greening['vpd_trend']>0]
                #     else:
                #         df_x_trend = df_greening[df_greening['vpd_trend']<0]
            df_greening = df_zone
            df_x_trend = df_greening
            x_list = []
            y_list = []
            for i,row in df_x_trend.iterrows():
                co2 = row['co2']
                ndvi = row['ndvi']
                vpd = row['vpd']
                co2_multi_reg = row['co2_multi_reg']
                co2 = np.array(co2)
                ndvi = np.array(ndvi)
                vpd = np.array(vpd)
                co2[co2<-999] = np.nan
                ndvi[ndvi<-999] = np.nan
                vpd[vpd<-999] = np.nan

                vpd_mean = np.nanmean(vpd)
                # if vpd_mean > 1:
                #     continue
                try:
                    vpd_trend,vpd_p,_ = K.linefit(list(range(len(vpd))),vpd)
                    # if vpd_p > 0.05:
                    #     continue
                except:
                    # print(vpd)
                    vpd_trend = np.nan
                # r,p = T.nan_correlation(ndvi,co2)
                # x = vpd_mean
                x = vpd_trend
                y = co2_multi_reg
                x_list.append(x)
                y_list.append(y)
            flag += 1
            ax = plt.subplot(8,6,flag)
            # plt.scatter(x_list,y_list)
            cmap_ = K.cmap_with_transparency('Reds')
            K.plot_scatter(x_list,y_list,cmap=cmap_,s=3,ax=ax,plot_fit_line=True)
            # title = f'{season}-{zone}\n{greening}-vpd_etrend{x_trend}'
            # title = f'{season}-{zone}\n{greening}'
            title = f'{season}-{zone}'
            print(title)
            plt.title(title)
            plt.xlabel('Annual VPD mean')
            plt.ylabel('CO2 vs NDVI correlation')
    plt.tight_layout()
    plt.savefig('corr4.pdf')

def main():
    # plot_box()
    # plot_scatter()
    # plot_scatter1()
    # plot_vectors()
    # plot_pie_chart()
    # plot_pie_chart_trend()
    # plot_pie_chart_trend_1()
    # plot_bar_trend_ratio()
    # plot_pie_trend_ratio()
    # plot_vectors1()
    # P_PET_ratio()
    # plot_ratio_trend()
    # mask_NDVI()
    # NDVI_seasonal_transform()
    # NDVI_trend_line_and_spatial()
    # plot_ratio_trend1()
    # NDVI_trend_spatial()
    # get_variables_trend_df()
    # plot_ratio()
    # check_HI_Class()
    # ndvi_spatial_trend_check()
    co2_ndvi_limited_by_vpd()
    pass


if __name__ == '__main__':
    main()
