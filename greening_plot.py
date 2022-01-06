# coding='utf-8'
from __init__ import *

project_root='/Volumes/SSD_sumsang/project_greening/'
data_root=project_root+'Data/'
results_root=project_root+'Result/new_result/'

def mk_dir(outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

class Plot_dataframe:
    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame_2002-2015/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff = self.this_class_arr + 'Data_frame_2002-2015_df.df'




    def run(self):
        df = self.__gen_df_init()

        # self.call_greening_trend_bar(df)
        self.call_correlation_bar(df)




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

    def call_plot_line_for_three_seasons(self):
        period_name=['early','peak','late']
        for period in period_name:
            f=self.this_class_arr+'{}_df.df'.format(period)
            df = self.__gen_df_init_wen(f)
            column_name='during_{}_CSIF_par'.format(period)
            # column_name='%CSIF_par_{}'.format(period)
            print(column_name)
            self.plot_line_for_three_seasons(df,column_name)
        plt.legend()
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



    def plot_line_for_three_seasons(self,df,y_name):  # 画anomaly
        df = df[df['row'] < 120]
        df = df[df[y_name] > -2]
        df = df[df[y_name] < 2]
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
        plt.plot(mean_value_yearly, label=y_name)
        plt.fill_between(range(len(mean_value_yearly)), up_list, bottom_list, alpha=0.3, zorder=-1)
        plt.xticks(range(len(mean_value_yearly)), year_list)
        # plt.show()

    def call_plot_variables_for_greening_and_browning(self):
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



    def call_plot_LST_for_three_seasons(self):
        period_name=['early','peak','late']
        color_list=['r','g','b']
        flag=0
        for period in period_name:
            f=self.this_class_arr+'{}_df.df'.format(period)
            df = self.__gen_df_init_wen(f)
            column_name='{}_pre_30_LST'.format(period)
            # column_name='{}_pre_15_LST'.format(period)
            # column_name='%CSIF_par_{}'.format(period)
            print(column_name)
            color=color_list[flag]
            self.plot_LST_for_three_seasons(df,column_name,color)
            flag+=1
        plt.legend()
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


    def plot_LST_for_three_seasons(self,df,column_name,color):

        df = df[df['r_list'] < 120]
        df = df[df[column_name] > -2]
        df = df[df[column_name] < 2]
        # plt.hist(df[column_name], bins=80)
        # plt.show()

        dic = {}
        mean_val = {}
        confidence_value = {}
        # p_value = {}
        # k_value = {}
        # b_value = {}

        std_val = {}
        year_list = df['year'].to_list()
        year_list = set(year_list)  # 取唯一
        year_list = list(year_list)
        year_list.sort()

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
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
            n = len(val_list)
            mean_val_i = np.nanmean(val_list)
            se = stats.sem(val_list)
            h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
            confidence_value[year] = h
            mean_val[year] = mean_val_i



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
            up_list.append(mean_val[year] + confidence_value[year])
            bottom_list.append(mean_val[year] - confidence_value[year])
            fit_value_yearly.append(k_value*(year-2002)+b_value)


        plt.plot(mean_value_yearly, label=column_name,c=color)
        plt.plot(fit_value_yearly,linestyle='--',label='k={:0.2f},p={:0.3f}'.format(k_value,p_value),c=color)
        plt.fill_between(range(len(mean_value_yearly)), up_list, bottom_list, alpha=0.3, zorder=-1,color=color)
        plt.xticks(range(len(mean_value_yearly)), year_list,)
        # plt.show()

    def plot_SPEI(self,df,koppen):
        dic = {}
        mean_val = {}
        std_val = {}
        year_list = df['year'].to_list()
        year_list = set(year_list)  # 取唯一
        year_list = list(year_list)
        year_list.sort()

        for year in tqdm(year_list):  # 构造字典的键值，并且字典的键：值初始化
            dic[year] = []
            mean_val[year] = []
            std_val[year] = []

        for year in year_list:
            df_pick = df[df['year'] == year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row['late_pre_1mo_SPEI3']
                dic[year].append(val)
            val_list = np.array(dic[year])
            mean_val[year] = np.nanmean(val_list)
            std_val[year] = np.nanstd(val_list)
        mean_value_yearly = []
        up_list = []
        bottom_list = []
        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            up_list.append(mean_val[year] + std_val[year])
            bottom_list.append(mean_val[year] - std_val[year])
        plt.plot(mean_value_yearly, label=koppen)
        # plt.fill_between(range(len(mean_value_yearly)),up_list,bottom_list,alpha=0.3,zorder=-1)
        plt.xticks(range(len(mean_value_yearly)), year_list)
        # plt.show()


    def plot_line_CSIF_par(self,df,koppen):

        dic={}
        mean_val={}
        std_val={}
        year_list = df['year'].to_list()
        year_list=set(year_list)  # 取唯一
        year_list=list(year_list)
        year_list.sort()

        for year in tqdm(year_list): #构造字典的键值，并且字典的键：值初始化
            dic[year]=[]
            mean_val[year] = []
            std_val[year]=[]

        for year in year_list:
            df_pick = df[df['year']==year]
            for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
                pix = row.pix
                val = row['%CSIF_par_late']
                dic[year].append(val)
            val_list = np.array(dic[year])
            mean_val[year]=np.nanmean(val_list)
            std_val[year]=np.nanstd(val_list)
        mean_value_yearly = []
        up_list=[]
        bottom_list=[]
        for year in year_list:
            mean_value_yearly.append(mean_val[year])
            up_list.append(mean_val[year]+0.25*std_val[year])
            bottom_list.append(mean_val[year]-0.25*std_val[year])
        plt.plot(mean_value_yearly,label=koppen)
        plt.fill_between(range(len(mean_value_yearly)),up_list,bottom_list,alpha=0.3,zorder=-1)
        plt.xticks(range(len(mean_value_yearly)),year_list)
        plt.show()

    def call_greening_trend_bar(self,df):  # 实现的功能是greening_trend_percentage_bar

        outdir =results_root+'Figure/2002-2015/'
        T.mk_dir(outdir)

        df = df[df['row'] < 120]

        variable='CSIF'
        period='late'
        time='2002-2015'
        outf = outdir + 'greening_trend_' + period + '_'+time+'_landuse'+'_'+variable

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
            self.plot_greening_trend_bar(df_pick, landcover, period,time)
        self.plot_greening_trend_bar(df, 'all',period,time)
        # plt.legend()
        plt.legend(["Negative_0.05", "Negative_0.1", "no trend", "Positive_0.1","Positive_0.05"])
        plt.title('greening_trend_' + period + '_landcover_'+time+'_'+variable)
        # plt.title('greening_trend_' + period + '_koppen_'+time+'_'+variable)

        # plt.show()
        plt.savefig(outf+'.pdf', dpi=300,)
        plt.close()


    def plot_greening_trend_bar(self,df_pick,koppen,period,time): # greening and browning percentage

        count_no_trend=0
        count_greening_0_1=0
        count_browning_0_1=0
        count_greening_0_05 = 0
        count_browning_0_05 = 0


        for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):
            trend = row['{}_during_{}_CSIF_trend_{}'.format(time,period,time)]
            p_val=row['{}_during_{}_CSIF_p_value_{}'.format(time,period,time)]


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

    def call_correlation_bar(self, df):  # 实现的功能是

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

    def plot_bar_correlation(self,df_pick,koppen,time, period,variable):  # 每个变量 例如CO2 的correlation percentage

        count_no_relationship=0
        positive_relationship_0_1=0
        negative_relationship_0_1=0
        positive_relationship_0_05 = 0
        negative_relationship_0_05 = 0


        for i, row in tqdm(df_pick.iterrows(), total=len(df_pick)):

            correlation = row['MODIS_NDVI_{}_{}_{}'.format(time, period,variable)]
            p_val=row['MODIS_NDVI_{}_{}_{}_p_value'.format(time,period,variable)]

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

        plt.bar(koppen, y1, color='sienna')
        plt.bar(koppen, y2, bottom=y1, color='peru')
        plt.bar(koppen, y3, bottom=y1 + y2, color='gray')
        plt.bar(koppen, y4, bottom=y1 + y2 + y3, color='limegreen')
        plt.bar(koppen, y5, bottom=y1 + y2 + y3+y4, color='forestgreen')



        plt.text(koppen, y1 / 2., round(negative_0_05), fontsize=12, color='w', ha='center', va='center')
        plt.text(koppen, y1 + y2 / 2., round(negative_0_1), fontsize=12, color='w', ha='center', va='center')
        plt.text(koppen, y1 + y2 + y3 / 2., round(no_trend), fontsize=12, color='w', ha='center', va='center')
        plt.text(koppen, y1 + y2 + y3 + y4 / 2., round(positive_0_1), fontsize=12, color='w', ha='center',
                 va='center')
        plt.text(koppen, y1 + y2 + y3 + y4 + y5 / 2, round(positive_0_05), fontsize=12, color='w', ha='center',
                 va='center')
        plt.text(koppen, 102, len(df_pick), fontsize=12, color='k', ha='center', va='center')
        # plt.xlabel("landcover")
        plt.ylabel("Percentage")
        # plt.show()


class Plot_partial_correlation:
    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame_2002-2015/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff = self.this_class_arr + 'Data_frame_2002-2015_df.df'



    def run(self):
        df = self.__gen_df_init()
        # self.call_plot_LST_for_three_seasons()
        # self.call_CSIF_par_trend_bar(df)
        # self.plot_bar(df)


        self.call_plot_bar_max_contribution(df)

        # self.call_plot_box_correlation(df)



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

        variable_list = ['CO2', 'PAR', 'Precip', 'VPD', 'temperature']

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




def main():

    Plot_dataframe().run()
    # Plot_partial_correlation().run()


if __name__ == '__main__':
    main()