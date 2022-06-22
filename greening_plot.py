# coding='utf-8'
import numpy as np

from __init__ import *
import pandas as pd
import plotly.graph_objects as go

import plotly.express as px

project_root='/Volumes/SSD_sumsang/project_greening/'
# project_root='/Volumes/NVME2T/greening_project_old/'
data_root=project_root+'Data/'
results_root=project_root+'Result/new_result/'

# project_root='D:/Greening/'
# data_root=project_root+'Data/'
# results_root=project_root+'Result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)



class Plot_dataframe:
    def __init__(self):
        self.this_class_arr = results_root + '/Data_frame_2000-2018/relative_change/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff = self.this_class_arr + 'Data_frame_2000-2018.df'





    def run(self):
        df = self.__gen_df_init()
        df=self.__clean_df(df)

        # self.call_greening_trend_bar(df)
        # self.call_greening_area_statistic_bar_products(df)
        # self.call_correlation_bar(df)
        # self.plot_multi_window_correlation(df)
        # self.plot_multi_window_trend(df)
        # self.call_multi_correlation_bar(df)
        # self.call_plot_line_for_three_seasons(df)
        # self.call_plot_line_NDVI_three_seasons(df)
        self.call_plot_LST_for_three_seasons(df)
        # self.call_plot_trendy_for_three_seasons(df)
        # self.call_plot_trendy_for_three_seasons(df)
        # self.call_plot_GIMMS_NDVI_for_three_seasons_two_product(df)
        # self.plot_product_bar(df)




    def __df_to_excel(self,df,dff,n=1000,random=True):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass
    def __load_df(self,):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def __gen_df_init(self,):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __clean_df(self,df):


        df = df[df['row'] < 120]
        df = df[df['NDVI_MASK'] == 1]
        df = df[df['koppen'] != 'E']
        return df

    def __load_df_wen(self,file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __gen_df_init_wen(self,file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df= self.__load_df_wen(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def call_plot_line_for_three_seasons(self,df):
        # period_name=['early','peak','late']
        period_name = ['early','peak','late']
        # period_name = [ 'peak']
        # variable_list=['1982-2020_LAI4g','1982-2018_LAI3g','2000-2019_MODIS_LAI']
        variable_list = ['LAI4g']
        # variable_list=['2000-2019_MODIS_LAI']

        # variable_list = ['LAI4g', 'LAI3g', 'MODIS_LAI', 'VOD']

        # variable_list = [
        #                   '2001-2016_CSIF', '1982-2018_NIRv']

        # variable_list=['1982-2015_GIMMS_NDVI']
        # variable_list=['1988-2016_VOD']

        for period in period_name:
            for variable in variable_list:

                # column_name=f'2000-2016_{variable}_{period}_relative_change'
                column_name=f'{variable}_relative_change_{period}_trend_window'
                print(column_name)
                self.plot_line_for_three_seasons(df,column_name)
        plt.legend(fontsize=15)
        plt.title('Humid')
        plt.show()

    def call_plot_line_NDVI_three_seasons(self,df):
        period_name=['early','peak','late']
        for period in period_name:

            column_name='2002-2018_'.format(period)
            # column_name='%CSIF_par_{}'.format(period)
            print(column_name)
            self.plot_line_NDVI(df,column_name)
        plt.legend()

        plt.show()


    def plot_line_for_three_seasons(self,df,y_name):  # 画anomaly

        df=df[df['HI_class']=='Humid']
        # pix_list=T.get_df_unique_val_list(df,'pix')
        # spatial_dic={}
        # for pix in pix_list:
        #     spatial_dic[pix]=1
        # arr=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr)
        # plt.show()

        # plt.hist(df[y_name], bins=80)
        # plt.show()

        dic = {}
        mean_val = {}
        confidence_value={}
        std_val = {}
        year_list = df['year'].to_list()
        year_list = set(year_list)  # 取唯一
        year_list = list(year_list)
        year_list.sort()

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            confidence_value[year]=[]


        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[y_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            # val_list[val_list>1000]=np.nan

            n=len(val_list)
            mean_val_i=np.nanmean(val_list)
            se=stats.sem(val_list)
            h=se * stats.t.ppf((1 + 0.95) / 2., n-1)
            confidence_value[year]=h
            mean_val[year] = mean_val_i


        mean_value_yearly = []
        up_list = []
        bottom_list = []

        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            up_list.append(mean_val[year] + confidence_value[year])
            bottom_list.append(mean_val[year]-confidence_value[year])
        plt.plot(year_list,mean_value_yearly, label=y_name)
        plt.fill_between(range(len(mean_value_yearly)), up_list, bottom_list, alpha=0.3, zorder=-1)
        plt.xticks(range(len(mean_value_yearly)), year_list)





    def call_plot_variables_for_greening_and_browning(self):  ######### 6月
        period_name=['early','peak','late']
        # period_name = ['peak', 'late']
        variables=['CO2_anomaly','GPCP','LST','PAR','SPEI3']
        # variables = ['CO2_anomaly',]
        color_list=['g','r']
        flag=0
        i=0
        plt.figure(figsize=(8, 8))
        for period in period_name:
            j=1
            f = self.this_class_arr + '{}_df.df'.format(period)
            df = self.__gen_df_init_wen(f)
            for variable in variables:
                plt.subplot(3, 5, (5*i+j))
                if i ==0:
                    plt.title(variable)
                column_name='during_{}_{}'.format(period,variable)
                color = color_list[flag]
                print(column_name)
                df_greening = df[df['trend_during_{}_CSIF_par'.format(period)] > 0]
                df_greening = df_greening[df_greening['CSIF_par_p_value_{}'.format(period)] <= 0.1]
                self.plot_variables_for_greening_and_browning(df_greening, column_name, variables, color)
                flag += 1
                color = color_list[flag]
                df_browning = df[df['trend_during_{}_CSIF_par'.format(period)] < 0]
                df_browning = df_browning[df_browning['CSIF_par_p_value_{}'.format(period)] <= 0.1]
                self.plot_variables_for_greening_and_browning(df_browning,column_name,variables,color)
                flag=0
                j=j+1
            i = i + 1
        plt.show()

            # df1 = df1['%CSIF_par_early']
            # df1 = df1[df1['%CSIF_par_peak'] > -1]
            # df1 = df1[df1['%CSIF_par_peak'] < 1]
            # df2 = self.__gen_df_init()
            # df2 = df2['%CSIF_par_peak']
            # df2 = df2[df2['%CSIF_par_peak'] > -1]
            # df2 = df2[df2['%CSIF_par_peak'] < 1]
            # df3 = self.__gen_df_init()
            # df3 = df3['%CSIF_par_late']
            # df3 = df3[df3['%CSIF_par_late'] > -1]
            # df3 = df3[df3['%CSIF_par_late'] < 1]
            # list = [df1, df2, df3]



    def plot_variables_for_greening_and_browning(self,df,column_name,variable,color):
        df = df[df['r_list'] < 120]
        df = df[df[column_name] > -2]
        df = df[df[column_name] < 2]
        print(len(df))

        dic = {}
        mean_val = {}
        confidence_value = {}

        year_list = df['year'].to_list()
        year_list = set(year_list)  # 取唯一
        year_list = list(year_list)
        year_list.sort()

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            confidence_value[year] = []


        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            n = len(val_list)
            mean_val_i = np.nanmean(val_list)
            se = stats.sem(val_list)
            h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
            confidence_value[year] = h
            mean_val[year] = mean_val_i

        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list = []  # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
        xaxis = range(len(mean_val_list))
        print(len(mean_val_list))
        r, p_value = stats.pearsonr(xaxis, mean_val_list)
        k_value, b_value = np.polyfit(xaxis, mean_val_list, 1)

        mean_value_yearly = []
        up_list = []
        bottom_list = []
        fit_value_yearly = []
        p_value_yearly = []

        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            up_list.append(mean_val[year] + confidence_value[year])
            bottom_list.append(mean_val[year] - confidence_value[year])
            fit_value_yearly.append(k_value * (year - 2002) + b_value)


        plt.plot(mean_value_yearly, label=column_name, c=color)
        plt.plot(fit_value_yearly, linestyle='--', label='k={:0.2f},p={:0.3f}'.format(k_value, p_value), c=color)
        plt.legend()
        plt.fill_between(range(len(mean_value_yearly)), up_list, bottom_list, alpha=0.3, zorder=-1, color=color)
        plt.xticks(range(len(mean_value_yearly)), year_list, )



    def plot_line(self,df):
        df = df[df['r_list'] < 120]
        df = df.drop_duplicates(subset=['pix', '%CSIF_par_late'], keep='first')
        df = df[df['%CSIF_par_late'] > -1]
        df = df[df['%CSIF_par_late'] < 1]
        koppen_list=['B','Cf','Csw','Df','Dsw','E']
        landcover_list = ['BF', 'NF', 'shrubs', 'Grassland', 'Savanna', 'Cropland']
        for koppen in koppen_list:
        # for landcover in landcover_list:
            df_pick = df[df['koppen'] == koppen]  # 修改
            # df_pick = df[df['landcover'] == landcover]  # 修改
            self.plot_line_CSIF_par(df_pick, koppen)
        #     # self.plot_LST(df_pick, landcover)
        #     self.plot_SPEI(df_pick, landcover)
        # self.plot_LST(df, 'all')
        # self.plot_SPEI(df,'all')
        self.plot_line_CSIF_par(df,'all')
        plt.legend()
        # plt.title('late_pre_SPEI3')
        # plt.savefig('%CSIF_par_late_koppen', dpi=300)
        plt.show()

    def call_plot_GIMMS_NDVI_for_three_seasons_two_product(self,df):  # 实现GIMMS——NDVI 和MODIS_NDVI 画在一起

        period_name = ['early', 'peak', 'late']

        color_list = ['r', 'darkorange', 'g','lime','b','aqua']
        flag = 0
        df = df[df['row'] < 120]
        df = df[df['NDVI_MASK'] ==1]
        # df=df[df['HI_class']=='Humid']


        for period in period_name:
            column_name_list = ['1982-2015_GIMMS_NDVI_{}_change%_keenan'.format(period),
                '2002-2015_MODIS_NDVI_{}_change%_keenan'.format(period)
                                ]

            for column_name in column_name_list:

            # column_name = '2002-2015_{}_MODIS_NDVI_anomaly'.format(period)
            # column_name='mean_during_{}_GIMMS_NDVI_window'.format(period)
            # column_name = 'during_{}_GIMMS_NDVI_trend_window'.format(period)
            # column_name='anomaly_1982-2015_during_{}_Aridity'.format(period)
            # column_name='{}_pre_15_LST'.format(period)
            # column_name='%CSIF_par_{}'.format(period)
                print(column_name)
                color=color_list[flag]
                self.plot_GIMMS_MODIS_NDVI_for_three_seasons(df,column_name,color)
                flag+=1
        plt.legend()
        plt.show()


    def plot_GIMMS_MODIS_NDVI_for_three_seasons(self,df,column_name,color):  ## 时间序列不一样的

        dic = {}
        mean_val = {}
        confidence_value = {}
        # p_value = {}
        # k_value = {}
        # b_value = {}

        std_val = {}
        if 'GIMMS_NDVI' in column_name:
            year_list = df['year'].to_list()
            year_list = set(year_list)  # 取唯一
            year_list = list(year_list)
            year_list.sort()

        elif 'MODIS_NDVI' in column_name:
            year_list=[]
            for i in range(2002,2016):
                year_list.append(i)
            print(year_list)

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            std_val[year]=[]
            confidence_value[year] = []
            # p_value[year] = []
            # k_value[year] = []
            # b_value[year] = []

        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            val_list=T.remove_np_nan(val_list)
            n = len(val_list)
            mean_val_i = np.nanmean(val_list)
            std_val_i= np.nanstd(val_list)
            se = stats.sem(val_list)
            h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
            confidence_value[year] = h
            mean_val[year] = mean_val_i
            std_val[year]=std_val_i



        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list=[]    # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
        xaxis = range(len(mean_val_list))
        print(len(mean_val_list))
        r, p_value = stats.pearsonr(xaxis, mean_val_list)
        k_value, b_value = np.polyfit(xaxis, mean_val_list, 1)

        mean_value_yearly = []
        up_list = []
        bottom_list = []
        fit_value_yearly=[]
        p_value_yearly=[]


        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            # up_list.append(mean_val[year] + confidence_value[year])
            # bottom_list.append(mean_val[year] - confidence_value[year])
            up_list.append(mean_val[year] + 0.125 * std_val[year])
            bottom_list.append(mean_val[year] - 0.125 * std_val[year])
            if len(year_list)==34:
                xaxis = list(range(1982, 2016))
                fit_value_yearly.append(k_value*(year-1982)+b_value)
            elif len(year_list)==14:
                fit_value_yearly.append(k_value * (year - 2002) + b_value)
                xaxis = list(range(2002, 2016))

        print(up_list)
        print(bottom_list)


        plt.plot(xaxis,mean_value_yearly, label=column_name,c=color)
        plt.plot(xaxis,fit_value_yearly,linestyle='--',label='k={:0.2f},p={:0.3f}'.format(k_value,p_value),c=color)
        # plt.fill_between(xaxis, up_list, bottom_list, alpha=0.1, zorder=-1,color=color)
        plt.title(column_name.split('%')[0]+'_relative')
        #
        # plt.show()
        # exit()


    def call_plot_LST_for_three_seasons(self,df): # 实现变量的三个季节画在一起

        df=df[df['HI_class']=='Humid']
        df=df[df['max_trend']<10]
        df = df[df['landcover'] !='cropland']

        period_name=['early','peak','late']

        color_list=['r','g','b']
        # variable_list=['LAI3g','MODIS_LAI']
        variable_list = ['LAI3g', 'MODIS_LAI', 'Trendy_ensemble']
        # variable_list = ['LAI3g']


        fig = plt.figure()


        i=1
        for period in period_name:
            flag=0
            ax = fig.add_subplot(2, 3, i)

            for variable in variable_list:


                # column_name=f'2000-2018_{variable}_{period}_anomaly_daily'
                column_name = f'2000-2018_{variable}_{period}_relative_change_daily'

                # column_name=f'2000-2018_{variable}_{period}_zscore_monthly'

                print(column_name)
                color=color_list[flag]
                self.plot_LST_for_three_seasons(df,column_name,color)
                flag+=1

            plt.legend()
            plt.ylabel('relative change %')
            plt.title(f'{period}_Humid')
            major_xticks = np.arange(0, 20, 5)
            # major_yticks = np.arange(-10, 15, 5)
            major_yticks = np.arange(-15, 15, 5)
            ax.set_ylim(-15, 15,)
            # major_yticks = np.arange(-0.2, 0.2, 0.05)
            # major_ticks = np.arange(0, 40, 5)  ### 根据数据长度修改这里
            ax.set_xticks(major_xticks)
            ax.set_yticks(major_yticks)
            plt.grid(which='major', alpha=0.5)
            i = i + 1
        plt.show()


    def plot_LST_for_three_seasons(self,df,column_name,color):

        dic = {}
        mean_val = {}
        confidence_value = {}
        std_val = {}
        # year_list = df['year'].to_list()
        # year_list = set(year_list)  # 取唯一
        # year_list = list(year_list)
        # year_list.sort()

        year_list = []
        for i in range(2000, 2019):
            year_list.append(i)
        print(year_list)

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            confidence_value[year] = []

        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            # val_list[val_list>1000]=np.nan

            n = len(val_list)
            mean_val_i = np.nanmean(val_list)
            std_val_i=np.nanstd(val_list)
            se = stats.sem(val_list)
            h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
            confidence_value[year] = h
            mean_val[year] = mean_val_i
            std_val[year]=std_val_i

        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list=[]    # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
        xaxis = range(len(mean_val_list))
        xaxis=list(xaxis)
        print(len(mean_val_list))
        # r, p_value = stats.pearsonr(xaxis, mean_val_list)
        # k_value, b_value = np.polyfit(xaxis, mean_val_list, 1)
        k_value, b_value, r, p_value=T.nan_line_fit(xaxis,mean_val_list)
        print(k_value)

        mean_value_yearly = []
        up_list = []
        bottom_list = []
        fit_value_yearly=[]
        p_value_yearly=[]


        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            # up_list.append(mean_val[year] + confidence_value[year])
            # bottom_list.append(mean_val[year] - confidence_value[year])
            up_list.append(mean_val[year] + 0.125 * std_val[year])
            bottom_list.append(mean_val[year] - 0.125 * std_val[year])

            fit_value_yearly.append(k_value * (year - year_list[0]) + b_value)

        print(up_list)
        print(bottom_list)


        plt.plot(mean_value_yearly, label=column_name,c=color)
        plt.plot(fit_value_yearly,linestyle='--',label='k={:0.2f},p={:0.4f}'.format(k_value,p_value),c=color)
        plt.fill_between(range(len(mean_value_yearly)), up_list, bottom_list, alpha=0.1, zorder=-1,color=color)
        plt.xticks(range(len(mean_value_yearly)), year_list,rotation=45)
        plt.title(f'LAI3g_moving_window_Dryland')
        #
        # plt.show()
        # exit()


    def call_plot_trendy_for_three_seasons(self,df):

        # 实现变量的三个季节画在一起

        df = df[df['HI_class'] == 'Humid']


        period_name=['early','peak','late']
        # period_name = ['late']

        color_list=['r','g','b','y','c','m','k','orange','purple','brown','pink','gray','olive','lime','black']
        width_list=[0.5]*14
        width_list.append(2)


        variables_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                     'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                     'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai']
        # variables_list = ['CABLE-POP_S2_lai','LPX-Bern_S2_lai',
        #                   ]

        # 'SDGVM_S2_lai'
        k_value={}
        b_value={}
        r={}
        p_value={}
        all_products={}
        for period in period_name:
            all_products[period]={}



        for period in period_name:

            for variable in variables_list:

                column_name=f'2000-2018_{variable}_{period}_relative_change_monthly'
                # print(column_name)
                mean_val_list=self.calculate_trendy_products(df,column_name)
                all_products[period][variable]=mean_val_list


        all_products_list=[]


        for period in period_name:
            for variable in all_products[period]:
                vals=all_products[period][variable]
                all_products_list.append(vals)
            all_products_array=np.array(all_products_list)

            all_products_array_T=all_products_array.T

            row = len(all_products_array_T)
            # print(row)
            product_list=[]

            for i in range(row):
                product_mean = np.nanmean(all_products_array_T[i])
                product_list.append(product_mean)
            all_products[period]['mean'] = product_list

            vals=all_products[period]['mean']
            k_value, b_value, r, p_value = T.nan_line_fit(range(len(vals)), vals)
            print(k_value)




####       Start plot print(all_products)
        year_list=list(range(2000,2019))
        fit_value_yearly=[]

        for year in year_list:

            fit_value_yearly.append(k_value * (year - year_list[0]) + b_value)

        fig = plt.figure()

        i = 1

        for period in period_name:
            flag=0
            ax = fig.add_subplot(2, 3, i)

            for variable in all_products[period]:
                vals=all_products[period][variable]
                print(vals)


                color=color_list[flag]
                width=width_list[flag]
                plt.plot(vals, label=variable, c=color, linewidth=width)

                plt.xticks(range(len(vals)), year_list, rotation=45)
                flag += 1

            plt.plot(fit_value_yearly, linestyle='--', label='k={:0.2f},p={:0.4f}'.format(k_value, p_value), c='black')

            plt.legend()
            plt.ylabel('relative change %')
            plt.title(f'{period}_Humid')
            plt.ylim(-20, 30)
            major_ticks = np.arange(0, 20, 5)
            # major_ticks = np.arange(0, 40, 5)  ### 根据数据长度修改这里
            ax.set_xticks(major_ticks)
            plt.grid(which='major', alpha=0.5)
            i = i + 1
        plt.show()


    def calculate_trendy_products(self,df,column_name):

        dic = {}
        mean_val = {}



        year_list = []
        for i in range(2000, 2019):
            year_list.append(i)
        print(year_list)

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []


        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            # val_list[val_list>1000]=np.nan
            mean_val_i = np.nanmean(val_list)
            mean_val[year] = mean_val_i


        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list=[]    # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
            print(len(mean_val_list))


        return mean_val_list

    def plot_trendy_products(self, df, column_name):

        dic = {}
        mean_val = {}
        confidence_value = {}
        std_val = {}
        mean_val_all_products = {}
        # year_list = df['year'].to_list()
        # year_list = set(year_list)  # 取唯一
        # year_list = list(year_list)
        # year_list.sort()

        year_list = []
        for i in range(2000, 2019):
            year_list.append(i)
        print(year_list)

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            confidence_value[year] = []

        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row[column_name]
                dic[year].append(val)
            val_list = np.array(dic[year])
            # val_list[val_list>1000]=np.nan
            mean_val_i = np.nanmean(val_list)
            mean_val[year] = mean_val_i

        # a, b, r = KDE_plot().linefit(xaxis, val)
        mean_val_list = []  # mean_val_list=下面的mean_value_yearly

        for year in year_list:
            mean_val_list.append(mean_val[year])
        xaxis = range(len(mean_val_list))
        xaxis = list(xaxis)
        print(len(mean_val_list))
        # r, p_value = stats.pearsonr(xaxis, mean_val_list)
        # k_value, b_value = np.polyfit(xaxis, mean_val_list, 1)
        mean_val_all_products.append(mean_val_list)
        k_value, b_value, r, p_value = T.nan_line_fit(xaxis, mean_val_list)

        mean_value_yearly = []
        up_list = []
        bottom_list = []
        fit_value_yearly = []
        p_value_yearly = []

        for year in year_list:
            mean_value_yearly.append(mean_val[year])

            fit_value_yearly.append(k_value * (year - year_list[0]) + b_value)

        return mean_value_yearly






    def call_greening_trend_bar(self,df):  # 实现的功能是greening_trend_percentage_bar

        outdir =results_root+'Figure_April/greening_bar_classfication/'
        T.mk_dir(outdir)

        df = df[df['HI_class'] != 'Humid']


        variable_list=['LAI4g','LAI3g','MODIS_LAI']
        period_list=['early','peak','late']
        time='2000-2018'

        # outf = outdir + 'greening_trend_' + period + '_'+time+ '_koppen'+'_'+variable


        koppen_list = ['B', 'Cf', 'Csw', 'Df', 'Dsw']
        # landcover_list = ['BF', 'NF', 'shrub', 'Grassland', 'Savanna', 'Cropland']
        flag = 0
        i = 0
        plt.figure(figsize=(8, 10))


        for period in period_list:
            j=1

            for variable in variable_list:
                plt.subplot(3, 4, (4 * i + j))

                # for landcover in landcover_list:
                for koppen in koppen_list:
                    df_pick = df[df['koppen'] == koppen]  # 修改
                    # df_pick = df[df['landcover'] == landcover]
                    self.plot_greening_trend_bar(df_pick, koppen,period,variable,time)
                    # self.plot_greening_trend_bar(df_pick, landcover, period,variable,time)
                self.plot_greening_trend_bar(df,'all',period,variable,time)

                # plt.title(period + '_landcover_'+time+'_'+variable+'_Dryland')
                # plt.xticks(landcover_list, rotation=45)
                plt.xticks(koppen_list, rotation=45)
                plt.title(variable)
                flag=0
                j=j+1
            i=i+1


        plt.legend(["Negative_0.05", "Negative_0.1", "no trend", "Positive_0.1", "Positive_0.05"])

        plt.tight_layout()
        plt.show()
        plt.savefig(outf+'.pdf', dpi=300,)
        plt.close()


    def plot_greening_trend_bar(self, df_pick,koppen,period,variable,time):# greening and browning percentage
        df_pick=df_pick.dropna(how='any', subset=[f'{time}_{variable}_{period}_relative_change_trend'] )
        print (df_pick)
        # exit()

        count_no_trend=0
        count_greening_0_1=0
        count_browning_0_1=0
        count_greening_0_05 = 0
        count_browning_0_05 = 0


        for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
            trend = row[f'{time}_{variable}_{period}_relative_change_trend']
            p_val=row[f'{time}_{variable}_{period}_relative_change_p_value']
            # trend = row['{}_{}_trend_original'.format(variable,period,)]
            # p_val = row['{}_{}_p_value_original'.format(variable,period)]


            if p_val>0.1:
                count_no_trend=count_no_trend+1
            elif 0.05<p_val<0.1:
                if trend>0:
                    count_greening_0_1=count_greening_0_1+1
                else:
                    count_browning_0_1 = count_browning_0_1 + 1
            else:
                if trend>0:
                    count_greening_0_05=count_greening_0_05+1
                else:
                    count_browning_0_05=count_browning_0_05+1
        greening_0_1=count_greening_0_1/len(df_pick)*100
        browning_0_1=count_browning_0_1 / len(df_pick)*100
        greening_0_05 = count_greening_0_05 / len(df_pick)*100
        browning_0_05 = count_browning_0_05 / len(df_pick)*100
        no_trend=count_no_trend/len(df_pick)*100


        y1 = np.array([browning_0_05])
        y2=np.array([browning_0_1])
        y3 = np.array([no_trend])
        y4=np.array([greening_0_1])
        y5=np.array([greening_0_05])

        # y1 = np.array([browning_0_05]) * len(df_pick)
        # y2 = np.array([browning_0_1]) * len(df_pick)
        # y3 = np.array([no_trend]) * len(df_pick)
        # y4 = np.array([greening_0_1]) * len(df_pick)
        # y5 = np.array([greening_0_05]) * len(df_pick)

        # plot bars in stack manner

        plt.bar(koppen, y1, color='sienna')
        plt.bar(koppen, y2, bottom=y1, color='peru')
        plt.bar(koppen, y3, bottom=y1 + y2, color='gray')
        plt.bar(koppen, y4, bottom=y1 + y2 + y3, color='limegreen')
        plt.bar(koppen, y5, bottom=y1 + y2 + y3+y4, color='forestgreen')


        plt.text(koppen, y1 / 2., round(browning_0_05), fontsize=10, color='w', ha='center', va='center')
        plt.text(koppen, y1 + y2 / 2., round(browning_0_1), fontsize=10, color='w', ha='center', va='center')
        plt.text(koppen, y1 + y2 + y3 / 2., round(no_trend), fontsize=10, color='w', ha='center', va='center')
        plt.text(koppen, y1 + y2 + y3 + y4 / 2., round(greening_0_1), fontsize=10, color='w', ha='center',
                 va='center')
        plt.text(koppen, y1 + y2 + y3 + y4 + y5 / 2, round(greening_0_05), fontsize=10, color='w', ha='center',
                 va='center')
        plt.text(koppen, 102, len(df_pick), fontsize=10, color='k', ha='center', va='center')

        # plt.xlabel("landcover")
        plt.ylabel("Percentage")



    def call_greening_area_statistic_bar_products(self, df):  # 实现的功能是不同产品面积greening sig greening ...

        outdir = results_root + 'Figure_April/greening_bar/'
        T.mk_dir(outdir)


        df = df[df['HI_class'] != 'Humid']


        product_list = ['LAI4g', 'LAI3g', 'MODIS_LAI',]
        period = 'peak'
        time = '2000-2018'

        plt.figure(figsize=(6, 6))

        for product in product_list:
            df = df.dropna(how='any', subset=[f'{time}_{product}_{period}_relative_change_trend'])

            count_no_trend = 0
            count_greening_0_1 = 0
            count_browning_0_1 = 0
            count_greening_0_05 = 0
            count_browning_0_05 = 0
            for i, row in tqdm(df.iterrows(), total=len(df)):
                trend = row[f'{time}_{product}_{period}_relative_change_trend']
                p_val=row[f'{time}_{product}_{period}_relative_change_p_value']

                if p_val > 0.1:
                    count_no_trend = count_no_trend + 1
                elif 0.05 < p_val < 0.1:
                    if trend > 0:
                        count_greening_0_1 = count_greening_0_1 + 1
                    else:
                        count_browning_0_1 = count_browning_0_1 + 1
                else:
                    if trend > 0:
                        count_greening_0_05 = count_greening_0_05 + 1
                    else:
                        count_browning_0_05 = count_browning_0_05 + 1
            greening_0_1 = count_greening_0_1 / len(df) * 100
            browning_0_1 = count_browning_0_1 / len(df) * 100
            greening_0_05 = count_greening_0_05 / len(df) * 100
            browning_0_05 = count_browning_0_05 / len(df) * 100
            no_trend = count_no_trend / len(df) * 100

            y1 = np.array([browning_0_05])
            y2 = np.array([browning_0_1])
            y3 = np.array([no_trend])
            y4 = np.array([greening_0_1])
            y5 = np.array([greening_0_05])

            # y1 = np.array([browning_0_05]) * len(df_pick)
            # y2 = np.array([browning_0_1]) * len(df_pick)
            # y3 = np.array([no_trend]) * len(df_pick)
            # y4 = np.array([greening_0_1]) * len(df_pick)
            # y5 = np.array([greening_0_05]) * len(df_pick)

            # plot bars in stack manner

            plt.bar(product, y1, color='sienna')
            plt.bar(product, y2, bottom=y1, color='peru')
            plt.bar(product, y3, bottom=y1 + y2, color='gray')
            plt.bar(product, y4, bottom=y1 + y2 + y3, color='limegreen')
            plt.bar(product, y5, bottom=y1 + y2 + y3 + y4, color='forestgreen')

            plt.text(product, y1 / 2., round(browning_0_05), fontsize=10, color='w', ha='center', va='center')
            plt.text(product, y1 + y2 / 2., round(browning_0_1), fontsize=10, color='w', ha='center', va='center')
            plt.text(product, y1 + y2 + y3 / 2., round(no_trend), fontsize=10, color='w', ha='center', va='center')
            plt.text(product, y1 + y2 + y3 + y4 / 2., round(greening_0_1), fontsize=10, color='w', ha='center',
                     va='center')
            plt.text(product, y1 + y2 + y3 + y4 + y5 / 2, round(greening_0_05), fontsize=10, color='w', ha='center',
                     va='center')
            plt.text(product, 102, len(df), fontsize=10, color='k', ha='center', va='center')

            # plt.xlabel("landcover")
            plt.ylabel("Percentage")

        # plt.legend()
        plt.legend(["Negative_0.05", "Negative_0.1", "no trend", "Positive_0.1", "Positive_0.05"])
        plt.title('percentage of greening and browning area_' + period + '_'+time+'_Dryland' )

        plt.show()
        outf=outdir+f'{time}_{period}_Dryland'
        plt.savefig(outf + '.pdf', dpi=300, )
        plt.close()

    def greening_area_statistic_bar_products(self, df_pick, koppen, period, variable):  # greening and browning percentage among products
        df_pick = df_pick.dropna(how='any', subset=[f'{period}_{variable}_trend'])
        print(df_pick)
        # exit()


    def call_multi_correlation_bar(self, df):  # 一次实现多个变量个偏相关画图

        variable_list =['CCI_SM','CO2','PAR','VPD','temperature']

        period = 'late'
        time = '2002-2015'

        df = df[df['{}_during_{}_CSIF_trend'.format(time,period)] > 0]
        # df = df[df['{}_during_{}_temperature_trend'.format(time, period)] > 0]


        # y = df['anomaly_{}_during_{}_MODIS_NDVI'.format(time,period)]
        # # df = df[df['GIMMS_NDVI_{}_original'.format(period)] < 1]
        # # df = df[df['GIMMS_NDVI_{}_original'.format(period)] > 0.2]
        # plt.hist(df['anomaly_{}_during_{}_MODIS_NDVI'.format(time,period)], bins=80)
        # plt.show()
        # koppen_list = ['B', 'Cf', 'Csw', 'Df', 'Dsw', 'E']
        landcover_list = ['BF', 'NF', 'shrub', 'Grassland', 'Savanna', 'Cropland']
        plt.figure(figsize=(18, 8))
        flag=1
        for variable in variable_list:
            plt.subplot(2,3,flag)

        # for koppen in koppen_list:
            for landcover in landcover_list:
                # df_pick = df[df['koppen'] == koppen]  # 修改
                df_pick = df[df['landcover'] == landcover]
                # self.plot_greening_trend_bar(df_pick, koppen,period,time)
                self.plot_bar_correlation(df_pick, landcover, time, period,variable)
            self.plot_bar_correlation(df, 'all', time, period,variable)
            # plt.legend()
            plt.legend(["Negative_0.05", "Negative_0.1", "no trend", "Positive_0.1", "Positive_0.05"])
            plt.title('partial_correlation_' + period + '_landcover_' + time + '_' + variable,fontsize=8)
        # plt.title('greening_trend_' + period + '_koppen_'+time+'_'+variable)
            flag+=1
        # plt.show()
        plt.savefig(outf + '.pdf', dpi=300, )
        plt.close()

    def call_multi_correlation_bar_window(self, df):  # 一次实现多个变量个偏相关画图

        outdir = results_root + 'Partial_corr_early/Figure/'
        T.mk_dir(outdir)

        df = df[df['row'] < 120]

        variable_list =['CCI_SM','CO2','PAR','VPD','temperature']

        period = 'early'
        time = '1982-2015'

        outf = outdir + 'partial_correlation_' + period + '_' + time

        print(outf)

        window_list = list(range(1,20))

        plt.figure(figsize=(18, 8))
        flag=1
        for variable in variable_list:

            plt.subplot(2,3,flag)

            for slice in window_list:
                self.plot_bar_correlation(df, slice, time, period,variable)
            # plt.legend()
            plt.legend(["Negative_0.05", "Negative_0.1", "no trend", "Positive_0.1", "Positive_0.05"])
            plt.title('partial_correlation_' + period + '_window_' + time + '_' + variable,fontsize=8)
        # plt.title('greening_trend_' + period + '_koppen_'+time+'_'+variable)
            flag+=1
        plt.show()
        plt.savefig(outf + '.pdf', dpi=300, )
        plt.close()


    def call_correlation_bar(self, df):  # 实现的功能是单个偏相关画图

        outdir = results_root + 'Figure/correlation_bar_2002-2015_browning/'
        T.mk_dir(outdir)

        df = df[df['row'] < 120]

        variable = 'temperature'
        period = 'late'
        time = '2002-2015'
        df = df[df['{}_during_{}_MODIS_NDVI_trend'.format(time,period)] < 0]
        df = df[df['{}_during_{}_temperature_trend'.format(time, period)] > 0]

        outf = outdir + 'partial_correlation_' + period + '_' + time + '_landuse' + '_' + variable

        # outf = outdir + 'greening_trend_' + period + '_'+time+ '_koppen'+'_'+variable
        print(outf)
        # exit()

        # y = df['anomaly_{}_during_{}_MODIS_NDVI'.format(time,period)]
        # # df = df[df['GIMMS_NDVI_{}_original'.format(period)] < 1]
        # # df = df[df['GIMMS_NDVI_{}_original'.format(period)] > 0.2]
        # plt.hist(df['anomaly_{}_during_{}_MODIS_NDVI'.format(time,period)], bins=80)
        # plt.show()
        # koppen_list = ['B', 'Cf', 'Csw', 'Df', 'Dsw', 'E']
        landcover_list = ['BF', 'NF', 'shrub', 'Grassland', 'Savanna', 'Cropland']
        plt.figure(figsize=(6, 6))
        # for koppen in koppen_list:
        for landcover in landcover_list:
            # df_pick = df[df['koppen'] == koppen]  # 修改
            df_pick = df[df['landcover'] == landcover]
            # self.plot_greening_trend_bar(df_pick, koppen,period,time)
            self.plot_bar_correlation(df_pick, landcover, time, period,variable)
        self.plot_bar_correlation(df, 'all', time, period,variable)
        # plt.legend()
        plt.legend(["Negative_0.05", "Negative_0.1", "no trend", "Positive_0.1", "Positive_0.05"])
        plt.title('partial_correlation_' + period + '_landcover_' + time + '_' + variable)
        # plt.title('greening_trend_' + period + '_koppen_'+time+'_'+variable)

        plt.show()
        plt.savefig(outf + '.pdf', dpi=300, )
        plt.close()

    def plot_bar_correlation(self,df_pick,slice,time, period,variable):  # 每个变量 例如CO2 的correlation percentage

        count_no_relationship=0
        positive_relationship_0_1=0
        negative_relationship_0_1=0
        positive_relationship_0_05 = 0
        negative_relationship_0_05 = 0


        for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):

            correlation = row['window_{:02d}_{}_pcorr_{}'.format(slice,variable,period)]
            p_val=row['window_{:02d}_{}_p_value_{}'.format(slice,variable,period)]

            if p_val>0.1:
                count_no_relationship=count_no_relationship+1
            elif 0.05<p_val<0.1:
                if correlation>0:
                    positive_relationship_0_1=positive_relationship_0_1+1
                else:
                    negative_relationship_0_1 = negative_relationship_0_1 + 1
            else:
                if correlation>0:
                    positive_relationship_0_05=positive_relationship_0_05+1
                else:
                    negative_relationship_0_05=negative_relationship_0_05+1
        positive_0_1=positive_relationship_0_1/len(df_pick)*100
        negative_0_1=negative_relationship_0_1 / len(df_pick)*100
        positive_0_05 = positive_relationship_0_05 / len(df_pick)*100
        negative_0_05 = negative_relationship_0_05 / len(df_pick)*100
        no_trend=count_no_relationship/len(df_pick)*100

        y1 = np.array([negative_0_05])
        y2=np.array([negative_0_1])
        y3 = np.array([no_trend])
        y4=np.array([positive_0_1])
        y5=np.array([positive_0_05])

        # plot bars in stack manner

        plt.bar(slice, y1, color='sienna')
        plt.bar(slice, y2, bottom=y1, color='peru')
        plt.bar(slice, y3, bottom=y1 + y2, color='gray')
        plt.bar(slice, y4, bottom=y1 + y2 + y3, color='limegreen')
        plt.bar(slice, y5, bottom=y1 + y2 + y3+y4, color='forestgreen')



        plt.text(slice, y1 / 2., round(negative_0_05), fontsize=8, color='w', ha='center', va='center')
        plt.text(slice, y1 + y2 / 2., round(negative_0_1), fontsize=8, color='w', ha='center', va='center')
        plt.text(slice, y1 + y2 + y3 / 2., round(no_trend), fontsize=8, color='w', ha='center', va='center')
        plt.text(slice, y1 + y2 + y3 + y4 / 2., round(positive_0_1), fontsize=8, color='w', ha='center',
                 va='center')
        plt.text(slice, y1 + y2 + y3 + y4 + y5 / 2, round(positive_0_05), fontsize=8, color='w', ha='center',
                 va='center')
        plt.text(slice, 102, len(df_pick), fontsize=8, color='k', ha='center', va='center')
        # plt.xlabel("landcover")
        plt.ylabel("Percentage")
        # plt.show()

    def plot_multi_window_correlation(self,df):  # 每个变量 例如CO2 的correlation percentage


        df = df[df['HI_class'] != 'Humid']
        variable_list = ['CCI_SM', 'CO2', 'PAR', 'VPD', 'temperature']
        color_list=['forestgreen','limegreen','gray','peru','sienna']
        correlation_label_list=['positive_p<0.05','positive_p<0.1','non_sig','negative_p<0.1', 'negative_p<0.05']
        plot_flag=1
        plt.figure()

        for variable in variable_list:
            plt.subplot(3, 5, plot_flag)

            x_list=[]
            y_list=[]
            for window in range(1,20):
                period='late'
                col_pcorr_name='window_{:02d}_during_{}_{}_pcorr'.format(window,period,variable)
                col_p_value_name='window_{:02d}_during_{}_{}_p_value'.format(window,period,variable)
                # col_pcorr_name = 'window_{:02d}_{}_pcorr_{}'.format(window, variable, period)
                # col_p_value_name = 'window_{:02d}_{}_p_value_{}'.format(window, variable, period)
                df_new=pd.DataFrame()
                df_new[col_p_value_name]=df[col_p_value_name]
                df_new[col_pcorr_name] = df[col_pcorr_name]
                df_new=df_new.dropna()

                ratio_list=[]
                for correlation_label in correlation_label_list:

                    if correlation_label=='positive_p<0.05':
                        df_selected=df_new[df_new[col_pcorr_name]>0]
                        df_selected = df_selected[df_selected[col_p_value_name] < 0.05]

                    elif correlation_label=='positive_p<0.1':
                        df_selected = df_new[df_new[col_pcorr_name] > 0]
                        df_selected = df_selected[df_selected[col_p_value_name] <= 0.1]
                        df_selected = df_selected[df_selected[col_p_value_name] >= 0.05]

                    elif correlation_label=='negative_p<0.05':
                        df_selected = df_new[df_new[col_pcorr_name] < 0]
                        df_selected = df_selected[df_selected[col_p_value_name] < 0.05]
                    elif correlation_label=='negative_p<0.1':
                        df_selected = df_new[df_new[col_pcorr_name] < 0]
                        df_selected = df_selected[df_selected[col_p_value_name] <= 0.1]
                        df_selected = df_selected[df_selected[col_p_value_name] >= 0.05]
                    elif correlation_label=='non_sig':
                        df_selected = df_new[df_new[col_p_value_name] > 0.1]
                    else:
                        raise UserWarning

                    count_selected=len(df_selected)
                    count_total=len(df_new)
                    ratio=count_selected/count_total
                    ratio_list.append(ratio)
                x_list.append(window)
                y_list.append(ratio_list)


            for i in range(len(x_list)):

                x=x_list[i]
                ratio_list=y_list[i]
                bottom=0
                flag=0
                for ratio in ratio_list:
                    plt.bar(x,ratio,bottom=bottom,color=color_list[flag],label=correlation_label_list[flag])
                    bottom+=ratio
                    flag=flag+1
            plt.legend(["positive p<0.05", "positive p<0.1", "no trend", "negative p<0.1", "negative p<0.05"])
            plt.title(period+'_non_humid_'+variable)
            plot_flag=plot_flag+1
        plt.show()

    def plot_multi_window_trend(self, df):  # 每个窗口trend 面积统计
        product='LAI4g'

        df = df[df['HI_class'] != 'Humid']
        period_list = ['early', 'peak', 'late']
        color_list = ['forestgreen', 'limegreen', 'gray', 'peru', 'sienna']
        correlation_label_list = ['positive_p<0.05', 'positive_p<0.1', 'non_sig', 'negative_p<0.1',
                                  'negative_p<0.05']
        plot_flag = 1
        plt.figure()

        for period in period_list:
            plt.subplot(3, 3, plot_flag)

            x_list = []
            y_list = []
            for window in range(0, 24):

                col_pcorr_name = f'{product}_relative_change_{period}_trend_window_{window:02d}'
                col_p_value_name = f'{product}_relative_change_{period}_p_value_window_{window:02d}'

                # col_pcorr_name = 'window_{:02d}_{}_pcorr_{}'.format(window, variable, period)
                # col_p_value_name = 'window_{:02d}_{}_p_value_{}'.format(window, variable, period)
                df_new = pd.DataFrame()
                df_new[col_p_value_name] = df[col_p_value_name]
                df_new[col_pcorr_name] = df[col_pcorr_name]
                df_new = df_new.dropna()

                ratio_list = []
                for correlation_label in correlation_label_list:

                    if correlation_label == 'positive_p<0.05':
                        df_selected = df_new[df_new[col_pcorr_name] > 0]
                        df_selected = df_selected[df_selected[col_p_value_name] < 0.05]

                    elif correlation_label == 'positive_p<0.1':
                        df_selected = df_new[df_new[col_pcorr_name] > 0]
                        df_selected = df_selected[df_selected[col_p_value_name] <= 0.1]
                        df_selected = df_selected[df_selected[col_p_value_name] >= 0.05]

                    elif correlation_label == 'negative_p<0.05':
                        df_selected = df_new[df_new[col_pcorr_name] < 0]
                        df_selected = df_selected[df_selected[col_p_value_name] < 0.05]
                    elif correlation_label == 'negative_p<0.1':
                        df_selected = df_new[df_new[col_pcorr_name] < 0]
                        df_selected = df_selected[df_selected[col_p_value_name] <= 0.1]
                        df_selected = df_selected[df_selected[col_p_value_name] >= 0.05]
                    elif correlation_label == 'non_sig':
                        df_selected = df_new[df_new[col_p_value_name] > 0.1]
                    else:
                        raise UserWarning

                    count_selected = len(df_selected)
                    count_total = len(df_new)
                    ratio = count_selected / count_total
                    ratio_list.append(ratio)
                x_list.append(window)
                y_list.append(ratio_list)

            for i in range(len(x_list)):

                x = x_list[i]
                ratio_list = y_list[i]
                bottom = 0
                flag = 0
                for ratio in ratio_list:
                    plt.bar(x, ratio, bottom=bottom, color=color_list[flag], label=correlation_label_list[flag])
                    bottom += ratio
                    flag = flag + 1
            plt.legend(["positive p<0.05", "positive p<0.1", "no trend", "negative p<0.1", "negative p<0.05"])
            # plt.title(period + '_non_humid_' + variable)
            plot_flag = plot_flag + 1
        plt.show()


    def plot_product_bar(self,df): # 不同的product relative change


        df = df[df['HI_class'] == 'Humid']
        product_list = ['LAI4g', 'LAI3g', 'MODIS_LAI']
        # product_list = ['LAI4g', 'LAI3g', 'MODIS_LAI', 'GIMMS_NDVI', 'VOD', 'NIRv','CSIF']
        # product_list = ['LAI4g', 'LAI3g', 'GIMMS_NDVI', 'VOD', 'NIRv', ]
        color_list = ['forestgreen', 'limegreen', 'gray', 'peru', 'sienna','red','brown']

        plot_flag = 1
        plt.figure()
        period='late'

        x_list = []
        y_list = []
        mean_trend_list=[]



        for product in product_list:
            trend_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                trend = row[f'2000-2018_{product}_{period}_relative_change_trend']


                trend=trend*10
                trend_list.append(trend)
            mean_trend_list=np.nanmean(trend_list)
            x_list.append(product)
            y_list.append(mean_trend_list)

        flag = 0

        for i in range(len(x_list)):

            x=x_list[i]
            trend=y_list[i]
            bottom=0

            plt.bar(x,trend,bottom=bottom,color=color_list[flag],label=product_list[flag])
            bottom+=trend
            flag=flag+1
        # plt.legend()
        plt.title(period+'_Humid_2000-2018',fontsize=15)
        plt.ylabel('relative_change,% decades-1',fontsize=15)
        plot_flag=plot_flag+1
        plt.show()


    pass


class Plot_partial_correlation:
    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame_2000-2018/multi_regression_2000-2018_relative_change_trend/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff = self.this_class_arr + 'Data_frame_2000-2018_multiregression_trend.df'



    def run(self):
        df = self.__gen_df_init()
        # self.call_plot_LST_for_three_seasons()
        # self.call_CSIF_par_trend_bar(df)
        # self.plot_bar(df)


        # self.call_plot_bar_max_contribution(df)

        # self.call_plot_box_correlation(df)
        # self.call_plotbox_Yuan(df)
        # self.plot_increase_decrease_Yuan(df)
        # self.plot_greening_trend_bar(df)

        self.plot_barchartpolar_preprocess(df)
        # self.plot_barchartpolar_preprocess_average_value(df)
        # self.plot_rose()






    def __df_to_excel(self,df,dff,n=1000,random=True):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass
    def __load_df(self,):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def __gen_df_init(self,):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df_wen(self,file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __gen_df_init_wen(self,file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df= self.__load_df_wen(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def call_plot_bar_max_contribution(self,df):
        outdir = results_root+'partial_correlation_anomaly/plot_bar_contribution/greening/'
        T.mk_dir(outdir)
        time_range='2002-2015'
        df = df[df['row'] < 120]
        variable='CSIF'

        period='late'
        df = df.dropna(subset=['{}_{}_{}_max_correlation'.format(variable,period, time_range)])
        df = df[df['{}_during_{}_{}_trend'.format(time_range, period,variable)] > 0]

        # outf =  f'{time_range}_max_contribution_{period}_koppen'
        outf = f'{time_range}_max_contribution1_{period}_landcover_{variable}'

        # df = df[df['during_{}_GIMMS_NDVI_trend_{}'.format(period,time_range)] >= 0]

        # koppen_list = ['B', 'Cf', 'Csw', 'Df', 'Dsw', 'E']
        landcover_list = ['BF', 'NF', 'shrub', 'Grassland', 'Savanna', 'Cropland']
        plt.figure(figsize=(6, 6))
        # for koppen in koppen_list:
        for landcover in landcover_list:
        #     df_pick = df[df['koppen'] == koppen]  # 修改
            df_pick = df[df['landcover'] == landcover]
            self.plot_bar_max_contribution(df_pick, landcover,variable,period,time_range)
        self.plot_bar_max_contribution(df, 'all',variable,period,time_range)

        # plt.title(f'{time_range}_max_contribution_{period}_koppen')
        plt.title(f'{time_range}_max_contribution_{period}_landcover')

        # plt.show()
        plt.savefig(outdir+ outf +'.pdf', dpi=300,)
        plt.close()

    def plot_bar_max_contribution(self,df_pick,koppen,variable,period,time_range):

        count_contribution_SM = 0
        count_contribution_CO2 = 0
        count_contribution_PAR = 0
        count_contribution_VPD = 0
        count_contribution_temperature = 0
        count_contribution_peak_SM = 0
        count_contribution_peak_precip= 0


        contribution_SM = 0
        contribution_CO2 = 0
        contribution_PAR = 0
        contribution_VPD = 0
        contribution_temperature = 0
        contribution_peak_SM = 0
        contribution_peak_precip = 0


        for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):

            contribution = row['{}_{}_{}_max_correlation'.format(variable,period,time_range)]

            if contribution==0:
                count_contribution_SM=count_contribution_SM+1
            if contribution==1:
                count_contribution_CO2 = count_contribution_CO2+1
            if contribution==2:
                count_contribution_PAR = count_contribution_PAR+1
            if contribution==3:
                count_contribution_VPD = count_contribution_VPD+1
            if contribution==4:
                count_contribution_temperature = count_contribution_temperature+1
            # if contribution==5:
            #     count_contribution_peak_SM = count_contribution_peak_SM+1
            # if contribution == 6:
            #     count_contribution_peak_precip = count_contribution_peak_precip + 1

        print(len(df_pick))

        contribution_SM = count_contribution_SM / len(df_pick) * 100
        contribution_CO2 = count_contribution_CO2 / len(df_pick) * 100
        contribution_PAR = count_contribution_PAR / len(df_pick) * 100
        contribution_VPD= count_contribution_VPD / len(df_pick) * 100
        contribution_temperature = count_contribution_temperature / len(df_pick) * 100
        # contribution_peak_SM = count_contribution_peak_SM / len(df_pick) * 100
        # contribution_peak_precip = count_contribution_peak_precip / len(df_pick) * 100

        y1 = np.array([contribution_SM])
        y2 = np.array([contribution_CO2])
        y3 = np.array([contribution_PAR])
        y4 = np.array([contribution_VPD])
        y5 = np.array([contribution_temperature])
        # y6 = np.array([contribution_peak_SM])
        # y7 = np.array([contribution_peak_precip])

        # plot bars in stack manner

        plt.bar(koppen, y1, color='limegreen')
        plt.bar(koppen, y2, bottom=y1, color='paleturquoise')
        plt.bar(koppen, y3, bottom=y1 + y2, color='lightpink')
        plt.bar(koppen, y4, bottom=y1 + y2 + y3, color='orange')
        plt.bar(koppen, y5, bottom=y1 + y2 + y3 + y4, color='rosybrown')
        # plt.bar(koppen, y6, bottom=y1 + y2 + y3 + y4 + y5, color='red')
        # plt.bar(koppen, y7, bottom=y1 + y2 + y3 + y4 + y5 + y6, color='blue')

        plt.text(koppen, y1/2., round(count_contribution_SM), fontsize=12,color='w',ha='center',va='center')
        plt.text(koppen, y1+y2/2., round(count_contribution_CO2), fontsize=12,color='w',ha='center',va='center')
        plt.text(koppen, y1+y2+y3/2., round(count_contribution_PAR), fontsize=12,color='w',ha='center',va='center')
        plt.text(koppen, y1+y2+y3+y4/2., round(count_contribution_VPD), fontsize=12,color='w',ha='center',va='center')
        plt.text(koppen, y1+y2+y3+y4+y5/2., round(count_contribution_temperature), fontsize=12,color='w',ha='center',va='center')
        # plt.text(koppen, y1 + y2 + y3 + y4 + y5 +y6/2., round(count_contribution_peak_SM), fontsize=12, color='w',
        #          ha='center', va='center')
        # plt.text(koppen, y1 + y2 + y3 + y4 + y5 + y6 +y7/2., round(count_contribution_peak_precip), fontsize=12, color='w',
        #          ha='center', va='center')

        plt.text(koppen, 102, len(df_pick), fontsize=12, color='k',ha='center',va='center')
        # plt.xlabel("landcover")
        plt.ylabel("Percentage")
        # plt.show()



    def call_plot_box_correlation(self, df):  # lian_2021 Figure2d

        outdir = results_root + 'Lian_Figure/'

        df = df[df['row'] < 120]

        period = 'early'
        time = '1999-2015'
        variable_list = ['CO2', 'PAR', 'CCI_SM', 'Precip', 'VPD', 'temperature']

        outf = outdir + 'greening_partial_correlation_detrend_' + period + '_' + time + '_koppen'

        print(outf)
        # exit()

        df = df[df['during_{}_GIMMS_NDVI_trend_{}'.format(period, time)] >= 0]

        koppen_list = ['B', 'Cf', 'Csw', 'Df', 'Dsw', 'E']
        # landcover_list = ['BF', 'NF', 'shrub', 'Grassland', 'Savanna', 'Cropland']
        plt.figure(figsize=(18, 18))
        flag = 1
        for variable in variable_list:
            plt.subplot(3, 3, flag)
            for koppen in koppen_list:
                # for landcover in landcover_list:
                df_pick = df[df['koppen'] == koppen]  # 修改
                # df_pick = df[df['landcover'] == landcover]
                self.plot_boxplot_correlation(df_pick, koppen, period, time)
                # self.plot_bar_CSIF_par(df_pick, landcover, period,time)
            self.plot_boxplot_correlation(df, 'all', period, time)
            # plt.legend()
            # plt.legend(["Negative_0.05", "Negative_0.1", "no trend", "Positive_0.1","Positive_0.05"])
            # plt.title('greening_trend_' + period + '_landcover_'+time)
            plt.title(period + '_' + time + '_' + variable)
            flag = flag + 1

        # plt.show()
        plt.savefig(outf + '.pdf', dpi=300, )
        plt.close()


    def plot_boxplot_correlation(self, df_pick, koppen, period, time): ### Wen 2021/12/30

        variable_list = ['PAR', 'Precip', 'VPD', 'temperature']

        mean_variable={}

        for variable in variable_list:
            val_list = []
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):

                val = row['partial_correlation_{}_{}_{}'.format(time,period,variable)]
                val_list.append(val)

            mean_variable[variable]=val_list

        for variable in mean_variable:
            flag = 1

            vals=mean_variable[variable]
            print(vals)
            plt.boxplot(vals)
            plt.title(variable)
            flag=flag+1
            plt.show()


    def plotbox_Yuan(self,df):
        outdir = results_root + 'Yuan_Figure/'

        df = df[df['row'] < 120]
        df = df[df['HI_class'] != 'Humid']


        time_range_list = ['1982-2001','2002-2018']
        variable_list = ['CO2', 'PAR', 'SPEI3', 'VPD', 'temperature','CCI_SM']
        # variable_list = ['SPEI3', 'VPD', ]

        season_list = ['early', 'peak', 'late']
        position_dic={}

        flag=1
        plt.figure(figsize=(18, 18))

        for variable in variable_list:

            for season in season_list:
                for period in time_range_list:

                    # outf = outdir + 'greening_partial_correlation_detrend_' + period + '_' + time + '_koppen'
                    # print(outf)
                    # exit()
                    variable_list = {}
                    val_list = []
                    for i, row in tqdm(df.iterrows(), total=len(df)):

                        # val = row['during_{}_{}_{}_correlation'.format(season, variable,period)]
                        val = row['simple_during_{}_{}_{}_LAI_GIMMS_r'.format(period,season , variable)]

                        val_list.append(val)
                    val_array=np.array(val_list)
                    val_new=T.remove_np_nan(val_array)
                    variable_list[period] = val_new
                    vals = variable_list[period]

                    key = f'{variable}_{season}_{period}'
                    position_dic[key] = (flag,vals)
                    flag += 1

        variable_all_list=[]
        label_list=[]
        position_list=[]
        for key in position_dic:
            vals=position_dic[key][1]
            flag=position_dic[key][0]
            variable_all_list.append(vals)
            label_list.append(key)
            position_list.append(flag)

        plt.boxplot(variable_all_list,positions=position_list, labels=label_list,vert=False)
        plt.tight_layout()
        plt.show()

        #         plt.title(period + '_' + time + '_' + variable)
        #         flag = flag + 1
        #
        #     # plt.show()
        #     plt.savefig(outf + '.pdf', dpi=300, )
        # plt.close()

    def plot_increase_decrease_Yuan(self,df):
        outdir = results_root + 'Yuan_Figure/'

        df = df[df['row'] < 120]
        df = df[df['HI_class'] != 'Humid']

        variable_list = ['CO2', 'PAR', 'SPEI3', 'VPD', 'temperature','CCI_SM']
        # variable_list = ['SPEI3', 'VPD', ]

        season_list = ['early', 'peak', 'late']
        position_dic={}

        flag=1
        plt.figure(figsize=(18, 18))

        for variable in variable_list:
            key = f'{variable}'
            for season in season_list:

                # outf = outdir + 'greening_partial_correlation_detrend_' + period + '_' + time + '_koppen'
                # print(outf)
                # exit()
                val_increase = 0
                val_decrease = 0

                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):

                    # val = row['during_{}_{}_{}_correlation'.format(season, variable,period)]
                    val = row['{}_{}_LAI_GIMMS_r_difference'.format(season, variable)]
                    if val>0:
                        val_increase=val_increase+1
                    if val>0:
                        val_decrease=val_decrease+1
                df_new = df.dropna()

                val_increase_area=val_increase/len(df_new)
                position_dic[key] = ('increase', val_increase_area)
                val_decrease_area = val_decrease / len(df_new)
                position_dic[key] = ('decrease', val_decrease_area)


        variable_all_list=[]
        label_list=[]
        position_list=[]
        for key in position_dic:
            vals=position_dic[key][1]
            flag=position_dic[key][0]
            variable_all_list.append(vals)
            label_list.append(key)
            position_list.append(flag)

        plt.bar(variable_all_list,positions=position_list, labels=label_list,vert=False)
        plt.tight_layout()
        plt.show()

        #         plt.title(period + '_' + time + '_' + variable)
        #         flag = flag + 1
        #
        #     # plt.show()
        #     plt.savefig(outf + '.pdf', dpi=300, )
        # plt.close()


    def plot_greening_trend_bar(self,df): # greening and browning percentage
        df = df[df['row'] < 120]
        df = df[df['HI_class'] != 'Humid']
        time_range='1982-2001'
        period='early'
        print(len(df))

        count_no_trend = 0
        count_greening_0_1 = 0
        count_browning_0_1 = 0
        count_greening_0_05 = 0
        count_browning_0_05 = 0


        for i, row in tqdm(df.iterrows(), total=len(df)):
            trend = row[f'original_during_{period}_LAI_GIMMS_{time_range}_trend']
            p_val=row[f'original_during_{period}_LAI_GIMMS_{time_range}_p_value']

            if p_val>0.1:
                count_no_trend=count_no_trend+1
            elif 0.05<p_val<0.1:
                if trend>0:
                    count_greening_0_1=count_greening_0_1+1
                else:
                    count_browning_0_1 = count_browning_0_1 + 1
            else:
                if trend>0:
                    count_greening_0_05=count_greening_0_05+1
                else:
                    count_browning_0_05=count_browning_0_05+1
        greening_0_1=count_greening_0_1/len(df)*100
        browning_0_1=count_browning_0_1 / len(df)*100
        greening_0_05 = count_greening_0_05 / len(df)*100
        browning_0_05 = count_browning_0_05 / len(df)*100
        no_trend=count_no_trend/len(df)*100

        x=np.arange(5)
        y=[greening_0_05,greening_0_1,browning_0_05,browning_0_1,no_trend]

        plt.bar(x, y)
        plt.xticks(x, ('Sig_Increase','Increase', 'Sig_Decrease', 'Decrease', 'no_change'))
        plt.show()

    def plot_barchartpolar(self,df): # rose 图
        df = df[df['row'] < 120]
        df = df[df['HI_class'] == 'Humid']
        df = df[df['NDVI_MASK'] == 1]


        # df=df.filter(regex='VPD$')
        # print(df.columns)
        #
        variable_list= ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                             'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai', 'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai',
                             'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai', 'LAI3g']
        # variable_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',  'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai'
        #                  ]


        period_list=['early','peak','late']
        dic_products = {}

        for period in period_list:
            print(period)
            dic_products[period] = {}

            for variable in variable_list:
                values_for_all_pixels=[]
                flag=0
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    pix = row.pix
                    column_name_r=f'2000-2018_partial_correlation_{period}_{variable}_VPD'
                    column_name_p_value = f'2000-2018_partial_correlation_p_value_result_{period}_{variable}_VPD'
                    if row[column_name_p_value]>0.1:
                        continue
                    val_r = row[column_name_r]
                    flag+=1
                    values_for_all_pixels.append(val_r)
                print(flag)

                n = len(values_for_all_pixels)
                mean_val_i = np.nanmean(values_for_all_pixels)
                std_val_i=np.nanstd(values_for_all_pixels)
                dic_products[period][variable] = mean_val_i

        # Here we draw the directions graph with 8 point slices
        N = len(variable_list) # Length of our hypothetical dataset contained in a pandas series
        # theta is used to get the x-axis values (the small part near the center of the chart)
        # the np.linspace function is a nifty thing which evenly divides a number range by as many parts as you want
        # in the example below we start at 0, stop at 2 * pi, and exclude the last number
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        #
        for period in period_list:
            vals_list=[]

            for variable in variable_list:
                    vals = dic_products[period][variable]
                    vals_list.append(vals)
                    vals_array=np.array(vals_list)
            print(vals_array)

        # width is used to get the widths of the top of the bars
        # the value to divide by may need a little tinkering with, depending on the data
        # so that the bars don't overlap each other

            width = np.pi / 60 * vals_array
        # this quick update is to avoid low value bars being so skinny that you can't see them!
            for i in range(len(width)):
                if width[i] < 0.2:
                    width[i] = 0.2
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, projection='polar')
            # so here we specify theta for the x-axis, our series summ8 for the heights, the width is what we calculated above
            bars = ax.bar(theta, vals_array, width=width, bottom=0.0, color='g', alpha=0.8)
            # ax.set_xticks(theta,variable_list)
            ax.set_xticks(theta)
            ax.set_xticklabels(variable_list)

            # this function makes sure that 0 degrees is set to North (the default is East for some strange reason)
            ax.set_theta_zero_location("N")
            # this function makes sure the degrees increase in a clockwise direction
            ax.set_theta_direction(-1)
            # and then these 2 options control how many radii there are and how they are labeled
            # this may also need some tinkering depending on the data
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks(np.arange(0, 0.1, 0.8))
            # and finally we do something simple and give it a title!
            plt.title(period, fontsize=15)
            # Use custom colors and opacity
            # for r, bar in zip(summ8.values, bars):
            for r, bar in zip(vals_array, bars):
                bar.set_facecolor(plt.cm.viridis(r / 13))
                bar.set_alpha(0.8)
            plt.show()
            pass


    def plot_barchartpolar_preprocess(self,df): #
        df = df[df['row'] < 120]
        df = df[df['HI_class']== 'Humid']
        df = df[df['NDVI_MASK'] == 1]
        df=df[df['max_trend']<10]
        df = df[df['landcover'] !='cropland']

        regions='Humid'

        product_list= ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                             'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai', 'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai',
                             'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai', 'Trendy_ensemble','LAI3g_monthly', 'MODIS_LAI_monthly','LAI3g_daily', 'MODIS_LAI_daily']
        # product_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',  'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai'
        #                  ]
        variable_list=['CO2','CCI_SM','VPD','Temp','PAR']


        period_list=['early','peak','late']
        dic_products = {}

        for period in period_list:
            print(period)
            dic_products[period] = {}
            for variable in variable_list:
                dic_products[period][variable]={}

                for product in product_list:
                    values_for_all_pixels=[]
                    flag=0
                    for i, row in tqdm(df.iterrows(), total=len(df)):
                        pix = row.pix
                        column_name_r=f'2000-2018_multi_regression_{period}_{product}_{variable}'

                        # column_name_r=f'2000-2018_partial_correlation_{period}_{product}_{variable}_{period}'
                        # column_name_p_value = f'2000-2018_partial_correlation_p_value_result_{period}_{product}_{variable}_{period}'


                        # if row[column_name_p_value]>0.1:
                        #     continue
                        val_r = row[column_name_r]
                        flag+=1
                        values_for_all_pixels.append(val_r)
                    print(flag)

                    n = len(values_for_all_pixels)
                    mean_val_i = np.nanmean(values_for_all_pixels)
                    std_val_i=np.nanstd(values_for_all_pixels)
                    dic_products[period][variable][product] = mean_val_i
        new_dict = {}
        for key1 in dic_products:
            new_dict_i = {}
            for key2 in dic_products[key1]:
                if key2 == '__key__':
                    continue
                for key3 in dic_products[key1][key2]:
                    val = dic_products[key1][key2][key3]
                    new_key = key1 + '_' + key2
                    new_dict_i[new_key] = val
                    if not key3 in new_dict:
                        new_dict[key3] = {}
                    new_dict[key3][new_key] = val
        df1 = T.dic_to_df(new_dict, 'model_name')
        print(df1)

        # T.save_df(df1,results_root+f'polar_bar_plot/drivers_all_models_{regions}_{title}.df')# 17 * 12 列
        # Tools().df_to_excel(df1,results_root+f'polar_bar_plot/drivers_all_models_{regions}_{title}.xlsx')

        T.save_df(df1, results_root + f'polar_bar_plot/multiregression_drivers_all_models_{regions}.df')  # 17 * 12 列
        Tools().df_to_excel(df1, results_root + f'polar_bar_plot/multiregression_drivers_all_models_{regions}.xlsx')


    def plot_barchartpolar_preprocess_average_value(self,df): # 求所有模型每个factor的avarage value
        df = df[df['row'] < 120]
        df = df[df['HI_class'] == 'Humid']
        df = df[df['NDVI_MASK'] == 1]
        df = df[df['max_trend'] < 10]
        df=df[df['landcover']!='cropland']
        regions='Humid'
        title='detrend'

        product_list= ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                             'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai', 'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai',
                             'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai', 'Trendy_ensemble', 'LAI3g','MODIS_LAI']
        # product_list = ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai',  'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai'
        #                  ]
        variable_list=['CO2','CCI_SM','VPD','Temp','PAR']

        period_list=['early','peak','late']
        dic_products = {}

        for period in period_list:
            print(period)
            dic_products[period] = {}
            for variable in variable_list:
                dic_products[period][variable]={}

                for product in product_list:
                    values_for_all_pixels=[]
                    flag=0
                    for i, row in tqdm(df.iterrows(), total=len(df)):
                        pix = row.pix

                        # column_name_r=f'2000-2018_partial_correlation_{period}_{product}_{variable}_{period}'
                        # column_name_p_value = f'2000-2018_partial_correlation_p_value_result_{period}_{product}_{variable}_{period}'
                        column_name_r=f'2000-2018_partial_correlation_{period}_{product}_detrend_{variable}_{period}'
                        column_name_p_value = f'2000-2018_partial_correlation_p_value_result_{period}_{product}_detrend_{variable}_{period}'
                        if row[column_name_p_value]<0.1:

                            flag+=1

                    print(flag)

                    # n = len(flag)
                    controling_area = flag/len(df)
                    print(controling_area)
                    dic_products[period][variable][product] = controling_area
        new_dict = {}
        for key1 in dic_products:
            new_dict_i = {}
            for key2 in dic_products[key1]:
                if key2 == '__key__':
                    continue
                for key3 in dic_products[key1][key2]:
                    val = dic_products[key1][key2][key3]
                    new_key = key1 + '_' + key2
                    new_dict_i[new_key] = val
                    if not key3 in new_dict:
                        new_dict[key3] = {}
                    new_dict[key3][new_key] = val
        df1 = T.dic_to_df(new_dict, 'model_name')
        print(df1)

        T.save_df(df1, results_root + f'polar_bar_plot/drivers_area_models_{regions}_{title}.df')  # 17 * 12 列
        Tools().df_to_excel(df1, results_root + f'polar_bar_plot/drivers_area_models_{regions}_{title}.xlsx')






    def __get_value_from_df(self,df,col,model_name):
        for i, row in df.iterrows():
            if row.model_name==model_name:
                return row[col]

    def __get_bar_color(self,period,varname):
        color_dict = {
            'early_CO2': '#000000',
            'early_CCI_SM': '#007429',
            'early_VPD': '#00c044',
            'early_PAR': '#3aff80',
            'early_Temp': '#b0ffcc',
            'peak_CO2':'#000000',
            'peak_CCI_SM': '#510000',
            'peak_VPD': '#bf0000',
            'peak_PAR': '#ff6868',
            'peak_Temp': '#ffc3c3',
            'late_CO2':'#000000',
            'late_CCI_SM': '#000b75',
            'late_VPD': '#0014d3',
            'late_PAR': '#5565ff',
            'late_Temp': '#b7beff'
        }
        return color_dict[f'{period}_{varname}']

        pass

    def plot_rose(self):

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/polar_bar_plot/drivers_all_models_Humid_detrend.df'
        # f=results_root+'polar_bar_plot/drivers_all_models_humid.df'
        df=T.load_df(f)
        T.print_head_n(df,5)
        period_list = ['early', 'peak', 'late']
        variable_list=['CO2','CCI_SM','VPD','PAR','Temp']
        model_name_list = T.get_df_unique_val_list(df, 'model_name')
        cols_list = []
        for period in period_list:
            for variable in variable_list:
                cols_list.append(f'{period}_{variable}')
        all_bar_n = len(period_list) * len(variable_list) * len(model_name_list)
        print(all_bar_n)
        # exit()
        print(cols_list)
        period_theta_interval = 6
        theta_i_interval = 1.5
        theta_i = (360-period_theta_interval*3)/ (len(period_list) * (len(variable_list)+theta_i_interval) * len(model_name_list))
        color_dict = {}

        fig = go.Figure()
        labels_list = []
        value_list = []
        theta_list = []
        color_list = []
        theta_init = 33
        for i,period in enumerate(period_list):
            for j,model in enumerate(model_name_list):
                for k,variable in enumerate(variable_list):
                    col = f'{period}_{variable}'
                    val = self.__get_value_from_df(df,col,model)
                    labels_list.append(f'{model}_{variable}')
                    value_list.append(val)
                    theta = theta_init + theta_i
                    theta_list.append(theta)
                    color_list.append(self.__get_bar_color(period,variable))
                    theta_init = theta
                    if k==len(variable_list)-1:
                        value_list.append(-2)
                        theta = theta_init + theta_i*theta_i_interval
                        theta_list.append(theta)
                        color_list.append('#ffffff')
                        theta_init = theta
                if j == len(model_name_list)-1:
                    value_list.append(0.6)
                    theta = theta_init + period_theta_interval
                    theta_list.append(theta)
                    color_list.append('#ffffff')
                    theta_init = theta
                    print(theta)
        print(theta_list)
        # exit()
        print(len(theta_list))
        print(len(color_list))
        # exit()
        # value_list.append(-.1)
        # value_list.append(.1)
        # theta_list.append(0)
        # theta_list.append(180)
        value_list = np.array(value_list)
        fig.add_trace(go.Barpolar(r=value_list, theta=theta_list, marker_color=color_list,
                                    #   marker_line_color="black",
                                    # marker_line_width=1,
                                    opacity=0.8,
                                  ))
        fig.update_layout(
            template=None,
            polar=dict(
                # range_r=[0, 1],
                # radialaxis=dict(range=[-0.4, 0.5]),
                # angularaxis=dict(showticklabels=False, ticks='')
            )
        )
        fig.show()
        # fig.write_image(f'test123.pdf')



class Plot_partial_moving_window:
    def __init__(self):
        self.this_class_arr = '/Volumes/SSD_sumsang/project_greening_redo/results/Main_flow/Moving_window/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff = self.this_class_arr + 'mean/mean.df'




    def run(self):
        df = self.__gen_df_init()
        # self.call_plot_LST_for_three_seasons()
        # self.call_CSIF_par_trend_bar(df)
        # self.plot_bar(df)

        self.plot_mean_moving_window(self,df)




    def __df_to_excel(self,df,dff,n=1000,random=True):
        if n == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            if random:
                df = df.sample(n=n, random_state=1)
                df.to_excel('{}.xlsx'.format(dff))
            else:
                df = df.head(n)
                df.to_excel('{}.xlsx'.format(dff))

        pass
    def __load_df(self,):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def __gen_df_init(self,):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df_wen(self,file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __gen_df_init_wen(self,file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df= self.__load_df_wen(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))




def main():

    # Plot_dataframe().run()
    Plot_partial_correlation().run()
    # Plot_partial_moving_window().run()



if __name__ == '__main__':
    main()