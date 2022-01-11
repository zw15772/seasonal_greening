import pandas as pd
from __init__ import *

T = Tools()

def plot_box():
    # f = '/Volumes/NVME2T/wen_proj/Matrix/data/2002-2015_partial_correlation_p_value_early_anomaly.npy'
    f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_correlation_anomaly/MODIS_NDVI/0105/2002-2015_partial_correlation_late_anomaly_MODIS_NDVI2.npy'
    dic = T.load_npy(f)
    df_temp= T.dic_to_df_different_columns(dic,'pix')
    df_f= results_root+'Data_frame_2002-2015/Data_frame_2002-2015_df.df'
    df=T.load_df(df_f)
    df=df[df['2002-2015_during_late_MODIS_NDVI_trend']>0]
    pix_list=T.get_df_unique_val_list(df,'pix')
    r_list=[]
    for i,row in df_temp.iterrows():
        pix=row.pix
        if not pix in pix_list:
            r_list.append(np.nan)
            continue
        r,c=row['pix']
        r_list.append(r)
    df_temp['r']=r_list
    df_temp=df_temp[df_temp['r']<120]
    box = []
    labels = []
    for col in df_temp:
        print(col)
        if col=='r':
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
    plt.boxplot(box,labels=labels,showfliers=False)
    plt.show()


def plot_scatter():
    y_f = '/Volumes/NVME2T/wen_proj/20220104/npy/2002-2015_multi_linearearly_anomaly_MODIS_NDVI.npy'
    x_f = '/Volumes/NVME2T/wen_proj/20220104/npy/during_early_SPEI3_trend.npy'
    dicy = T.load_npy(y_f)
    x = np.load(x_f)
    x[x<-999]=np.nan
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
            r,c = pix
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
        KDE_plot().plot_scatter(x_list,y_list)
        plt.xlabel('SPEI3_trend')
        plt.ylabel(f'{k}_vs_MODIS_NDVI')
    plt.show()



def plot_scatter1():
    fdir = results_root+'mean_calculation_original/2002-2015_first_last_five_years'
    # fdir=results_root+'mean_calculation_original/during_early_2002-2015'
    corr_fdir =results_root+'/multiregression_anomaly/MODIS_NDVI/0104/'
    period_list = ['early','peak','late',]
    order_list = ['first','last']
    for period in period_list:
        ############## change here ##################
        if not period == 'early':
            continue
        ############## change here ##################
        NDVI_trend_f = results_root + '/trend_calculation_anomaly/during_{}_2002-2015/2002-2015_during_{}_MODIS_NDVI_trend.npy'.format(period,period)
        temperature_trend_f = results_root + '/trend_calculation_anomaly/during_{}_2002-2015/2002-2015_during_{}_MODIS_NDVI_trend.npy'.format(period,period)
        # corr_f = f'2002-2015_partial_correlation_{period}_anomaly_CSIF.npy'
        corr_f = f'2002-2015_multi_linear{period}_anomaly_CSIF_fpar.npy'
        corr_fpath = join(corr_fdir,corr_f)
        corr_dic = T.load_npy(corr_fpath)
        NDVI_trend_arr=np.load(NDVI_trend_f)
        variable_trend_arr=np.load(temperature_trend_f)
        NDVI_trend_dic=DIC_and_TIF().spatial_arr_to_dic(NDVI_trend_arr)
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
                r,c=pix
                if r>120:
                    continue
                NDVI_trend=NDVI_trend_dic[pix]
                if NDVI_trend>0:
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
                    fpath = join(fdir,folder_name,f'during_{product1}_mean.npy')

                    arr = np.load(fpath)
                    arr[arr<-9999]=np.nan
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

                x_up = x_mean + 3*x_std
                x_down = x_mean - 3*x_std
                y_up = y_mean + 3*y_std
                y_down = y_mean - 3*y_std
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
                plt.scatter(x_list_new,y_list_new)
                # KDE_plot().plot_scatter(x_list_new,y_list_new)
                plt.xlabel(f'Delta_{product1}')
                plt.ylabel(f'partial_correlation_{product}_anomaly_CSIF')
                # plt.title(product)
                plt.show()

def plot_vectors():
    fdir = '/Volumes/NVME2T/wen_proj/20220107/OneDrive_1_2022-1-9/1982-2015_first_last_five_years'
    water_balance_tif = '/Volumes/NVME2T/wen_proj/20220107/HI_difference.tif'
    limited_area = ['energy_limited','water_limited',]
    period_list = ['early', 'peak', 'late', ]
    order_list = ['first', 'last']
    water_balance_dic = DIC_and_TIF().spatial_tif_to_dic(water_balance_tif)
    for period in period_list:
        data_dic = {'HI':[],'NDVI':[]}
        for order in order_list:
            folder = f'during_{period}_1982-2015_{order}_five'
            HI_f = f'during_{period}_Aridity_mean.npy'
            NDVI_f = f'during_{period}_GIMMS_NDVI_mean.npy'
            fpath_HI = join(fdir,folder,HI_f)
            fpath_NDVI = join(fdir,folder,NDVI_f)
            HI_arr = np.load(fpath_HI)
            NDVI_arr = np.load(fpath_NDVI)
            HI_arr[HI_arr<-9999] = np.nan
            NDVI_arr[NDVI_arr<-9999] = np.nan
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
            r,c = key
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
            df = df[df['r']<120]
            df = df[df['x1']<3]
            df = df[df['x1']<3]
            df = df[df['x1']!=0]
            df = df[df['x2']!=0]
            df = df.sample(n=1000)

            plt.figure()
            for i,row in df.iterrows():
                x = row.x1
                x2 = row.x2
                y = row.y1
                y2= row.y2
                dx = x2 - x
                dy = y2 - y
                if dy > 0 and dx > 0:
                    plt.arrow(x,y,dx,dy,ec='g',fc='g',alpha=0.8,head_width=0)
                elif dy > 0 and dx < 0:
                    plt.arrow(x,y,dx,dy,ec='cyan',fc='cyan',alpha=0.8,head_width=0)
                elif dy < 0 and dx > 0:
                    plt.arrow(x,y,dx,dy,ec='purple',fc='purple',alpha=0.8,head_width=0)
                elif dy < 0 and dx < 0:
                    plt.arrow(x, y, dx, dy, ec='r', fc='r',alpha=0.8,head_width=0)
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
            plt.subplot(3,2,flag)
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
            parts = [ratio1,ratio2,ratio3,ratio4]
            plt.pie(parts,labels=labels)
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
    greening_trend_list = ['greening','browning']
    x_trend_list = ['> 0','< 0',]
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
        for f in T.listdir(join(fdir,folder)):
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir,folder,f)
            arr = np.load(fpath)
            T.mask_999999_arr(arr)
            var_name = f.replace('.npy','')
            var_name = var_name.replace(f'{year_range}_during_','')
            var_name = var_name.replace(f'{period}_','')
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            print(var_name)
            dic_all[var_name] = dic
        df = T.spatial_dics_to_df(dic_all)

        r_list = []
        for i,row in df.iterrows():
            r,c = row.pix
            r_list.append(r)
        df['r'] = r_list
        df = df[df['r']<120]
        T.add_dic_to_df(df,HI_zone_class_dic,'HI_class')
        x_variable_p_value = f'{x_variable.replace("_trend","_p_value")}'
        y_variable_p_value = f'{y_variable.replace("_trend","_p_value")}'
        df_new = df[[x_variable,y_variable,
                     x_variable_p_value,
                     y_variable_p_value,
                     'HI_class']]
        df_new = df_new.dropna()
        T.print_head_n(df_new)
        zones_list = T.get_df_unique_val_list(df,'HI_class')

        # df_greening_non_sig = df_new[df_new[y_variable_p_value]>0.1]
        # df_greening_sig = df_new[df_new[y_variable_p_value]<0.1]
        df_greening_sig = df_new
        df_greening_sig_greening = df_greening_sig[df_greening_sig[y_variable]>0]
        df_greening_sig_browning = df_greening_sig[df_greening_sig[y_variable]<0]
        # print(len(df_new))
        # print('non-sig',len(df_greening_non_sig)/len(df_new))
        print('sig greening',len(df_greening_sig_greening)/len(df_new))
        print('sig browning',len(df_greening_sig_browning)/len(df_new))
        parts = []
        labels = []
        flag = 0
        color_list = ['g','cyan','yellow','r',]
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
                    df_zone = df_select_x[df_select_x['HI_class']==zone]
                    ratio = len(df_zone)/len(df_select_x)
                    ratio_total = len(df_zone)/len(df_greening_sig)
                    parts.append(ratio_total)
                    labels.append(zone+'\n'+str(round(ratio*100))+'%')
                    # print(zone,'\n',y_trend,x_variable,x_trend,'\n',ratio,ratio_total)
                    sum_+=ratio
                    color_list_all.append(colors_xtrend)
        wedges, texts = plt.pie(parts,labels=labels,colors=color_list_all, shadow=False)
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
    greening_trend_list = ['greening','browning']
    x_trend_list = ['> 0','< 0',]
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
        for f in T.listdir(join(fdir,folder)):
            if not f.endswith('.npy'):
                continue
            fpath = join(fdir,folder,f)
            arr = np.load(fpath)
            T.mask_999999_arr(arr)
            var_name = f.replace('.npy','')
            var_name = var_name.replace(f'{year_range}_during_','')
            var_name = var_name.replace(f'{period}_','')
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            dic_all[var_name] = dic
        df = T.spatial_dics_to_df(dic_all)

        r_list = []
        for i,row in df.iterrows():
            r,c = row.pix
            r_list.append(r)
        df['r'] = r_list
        df = df[df['r']<120]
        T.add_dic_to_df(df,HI_zone_class_dic,'HI_class')
        x_variable_p_value = f'{x_variable.replace("_trend","_p_value")}'
        y_variable_p_value = f'{y_variable.replace("_trend","_p_value")}'
        df_new = df[[x_variable,y_variable,
                     x_variable_p_value,
                     y_variable_p_value,
                     'HI_class']]
        df_new = df_new.dropna()
        T.print_head_n(df_new)
        zones_list = T.get_df_unique_val_list(df,'HI_class')

        # df_greening_non_sig = df_new[df_new[y_variable_p_value]>0.1]
        df_greening_sig = df_new[df_new[y_variable_p_value]<0.1]
        # df_greening_sig = df_new

        # print(len(df_new))
        # print('non-sig',len(df_greening_non_sig)/len(df_new))
        # print('sig greening',len(df_greening_sig_greening)/len(df_new))
        # print('sig browning',len(df_greening_sig_browning)/len(df_new))
        parts = []
        labels = []
        color_list = ['g','cyan','yellow','r',]
        color_list_all = []
        flag = 0
        for zone in zones_list:
            df_zone = df_greening_sig[df_greening_sig['HI_class']==zone]
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
                    ratio = len(df_select_x)/len(df_zone)
                    ratio_total = len(df_select_x)/len(df_greening_sig)
                    parts.append(ratio_total)
                    label_i = f'{zone}\n{y_trend}-{x_variable}{x_trend}'
                    labels.append(label_i+'\n'+str(round(ratio*100))+'%')
                    # print(zone,'\n',y_trend,x_variable,x_trend,'\n',ratio,ratio_total)
                    sum_+=ratio
                    color_list_all.append(colors_xtrend)
            flag += 1

        wedges, texts = plt.pie(parts,labels=labels,colors=color_list_all, shadow=False)
        # plt.pie(parts,labels=labels)
        for w in wedges:
            w.set_linewidth(2)
            w.set_edgecolor('w')
        plt.show()



def add_dic_to_df(df,dic,key_name):
    val_list = []
    for i,row in df.iterrows():
        pix = row['pix']
        if not pix in dic:
            val = None
        else:
            val = dic[pix]
        val_list.append(val)
    df[key_name] = val_list


def drop_n_std(vals,n=1):
    vals = np.array(vals)
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    up = mean + n * std
    down = mean - n * std
    vals[vals>up] = np.nan
    vals[vals<down] = np.nan
    return vals

def P_PET_ratio():
    fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
    dic = T.load_npy_dir(fdir)
    dic_long_term = {}
    for pix in dic:
        vals = dic[pix]
        vals = np.array(vals)
        T.mask_999999_arr(vals)
        vals[vals==0]=np.nan
        if np.isnan(np.nanmean(vals)):
            continue
        vals = drop_n_std(vals)
        long_term_vals = np.nanmean(vals)
        dic_long_term[pix] = long_term_vals
    return dic_long_term

def main():
    # plot_box()
    # plot_scatter()
    # plot_scatter1()
    # plot_vectors()
    # plot_pie_chart()
    # plot_pie_chart_trend()
    plot_pie_chart_trend_1()
    # P_PET_ratio()
    pass


if __name__ == '__main__':
    main()