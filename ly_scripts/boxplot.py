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

def main():
    # plot_box()
    # plot_scatter()
    plot_scatter1()
    pass


if __name__ == '__main__':
    main()