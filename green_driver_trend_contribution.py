# coding=utf-8

from __init__ import *
from pingouin import partial_corr
## for 1982-2015 greening project_dataframe  11/12/2021  Wen

class Build_dataframe:

    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame_2000-2018/'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'Data_frame_2000-2018.df'
        self.P_PET_fdir=data_root+'original_dataset/aridity_P_PET_dic/'
        pass

    def run(self):
        # period = 'early'
        # time = '1982-2015'
        # fdir=results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time, time, period)

        df=self.__gen_df_init(self.dff)


        # df=self.foo1(df)
        # df = self.foo2(df)
        # df=self.add_anomaly_GIMMIS_NDVI_to_df(df)
        # df = self.add_Pierre_GIMMIS_NDVI_to_df(df)
        df=self.add_Keenan_GIMMIS_NDVI_to_df(df)
        # df=self.add_window_trend_to_df(df)
        # df=self.add_window_p_value_to_df(df)
        # df=self.add_row(df)
        # df = self.add_anomaly_to_df(df)
        # df=self.add_trend_to_df(df)
        # df=self.add_p_val_trend_to_df(df)

        # df=self.add_mean_to_df(df)
        # df=self.add_CV_to_df(df)
        # df=self.add_soil_data_to_df(df)
        # df=self.add_MAP_MAT_to_df(df)
        df = self.add_NDVI_mask(df)
        # df=self.add_winter_to_df(df)
        # df=self.add_Koppen_data_to_df(df)
        # df=self.add_landcover_data_to_df(df)
        # df=self.add_max_correlation_to_df(df)
        # df=self.add_partial_correlation_to_df(df)
        # P_PET_dic=self.P_PET_ratio(self.P_PET_fdir)
        # P_PET_reclass_dic=self.P_PET_reclass(P_PET_dic)
        # df=T.add_spatial_dic_to_df(df,P_PET_reclass_dic,'HI_class')


        # df=self.show_field(df)


        # df=self.__rename_dataframe_columns(df)
        # df = self.drop_field_df(df)


        # df=self.add_trend_to_df(df)
        # df=self.add_NDVI_trend_label(df)



        # df=self.add_Koppen_data_to_df(df)


        T.save_df(df,self.dff)
        self.__df_to_excel(df,self.dff)


    def __gen_df_init(self,file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df= self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self,file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self,df,dff,n=1000,random=True):
        dff=dff.split('.')[0]
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

    def foo1(self,df):

        f = results_root+'Pierre_relative_change/2000-2018_daily/LAI3g_early_relative_change.npy'
        dic = {}
        outf = self.dff
        result_dic = {}
        dic = T.load_npy(f)

        pix_list=[]
        change_rate_list=[]
        year=[]
        f_name = 'LAI3g'

        print(f_name)
        for pix in tqdm(dic):
            time_series=dic[pix]
            y=0
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y+1)
                y=y+1
        df['pix']=pix_list
        df[f_name] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df

    def foo2(self, df):  # 新建trend

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/trend_relative_change/2000-2018/MODIS_LAI_peak_relative_change_trend.npy'
        val_array = np.load(f)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)

        # exit()

        pix_list=[]
        for pix in tqdm(val_dic):
            pix_list.append(pix)
        df['pix']=pix_list

        return df
    def add_original_GIMMIS_NDVI_to_df(self,df):
        period='peak'

        f = results_root + 'extraction_original_val/1982-2015_original_extraction_all_seasons/1982-2015_extraction_during_{}_growing_season_static/during_{}_GIMMS_NDVI.npy'.format(period, period)

        NDVI_dic = T.load_npy(f)
        f_name = 'GIMMS_NDVI_{}_original'.format(period)
        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in NDVI_dic:
                NDVI_list.append(np.nan)
                continue
            vals = NDVI_dic[pix]/10000
            if len(vals) != 34:
                NDVI_list.append(np.nan)
                continue
            v1 = vals[year - 1982]
            NDVI_list.append(v1)
        df[f_name] = NDVI_list
        return df


    def add_anomaly_GIMMIS_NDVI_to_df(self, df):
        period = 'late'
        time='2000-2018'
        f = results_root + 'anomaly_variables_independently/{}_during_{}/{}_during_{}_GIMMS_NDVI.npy'.format(time,period,time,period)

        NDVI_dic = T.load_npy(f)
        f_name = 'anomaly_{}_during_{}_GIMMS_NDVI'.format(time,period)
        print(f_name)
        # exit()
        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in NDVI_dic:
                NDVI_list.append(np.nan)
                continue
            vals = NDVI_dic[pix]
            if len(vals) != 17:
                NDVI_list.append(np.nan)
                continue
            v1 = vals[year - 1982]
            NDVI_list.append(v1)
        df[f_name] = NDVI_list
        return df
    def add_Pierre_GIMMIS_NDVI_to_df(self, df):
        periods = ['early', 'peak', 'late']
        time = '1982-2015'
        fdir=results_root+'Pierre_relative_change/2001-2015/'

        for period in periods:
            for f in tqdm(sorted(os.listdir(fdir))):

                NDVI_dic = T.load_npy(fdir+f)
                f_name = f.split('.')[0]+'_relative_change'
                print(f_name)

                NDVI_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']


                    # pix = row.pix
                    pix = row['pix']
                    if not pix in NDVI_dic:
                        NDVI_list.append(np.nan)
                        continue
                    vals = NDVI_dic[pix]
                    if len(vals) != 15:
                        NDVI_list.append(np.nan)
                        continue
                    #
                    v1 = vals[year - 2001]
                    NDVI_list.append(v1)
                df[f_name] = NDVI_list
        return df

    def add_Keenan_GIMMIS_NDVI_to_df(self, df):
        periods = ['early','peak','late']
        # periods = ['peak']
        variable_list=['LAI3g','MODIS_LAI']
        # variable_list=['MODIS_LAI']

        time = '2000-2018'
        # variable='LAI3g'
        for variable in variable_list:
            for period in periods:

                f = results_root + f'Pierre_relative_change/2000-2018_monthly/{variable}_{period}_relative_change.npy'
                # f = results_root + f'Pierre_relative_change/2000-2016_Y/{variable}_{period}_relative_change.npy'
                # f=results_root+f'extraction_original_val/extraction_during_{period}_growing_season_static/during_{period}_{variable}.npy'
                # f=results_root+f'extraction_original_val/2000-2016/during_{period}_{variable}.npy'
                NDVI_dic = T.load_npy(f)
                f_name = f'{time}_{variable}_{period}_relative_change_monthly'
                # f_name = f'{time}_{variable}_{period}_raw'
                print(f_name)

                NDVI_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # if year<2000:
                    #     NDVI_list.append(np.nan)
                    #     continue
                    # #
                    # if year>2018:
                    #     NDVI_list.append(np.nan)
                    #     continue

                    # pix = row.pix
                    pix = row['pix']
                    if not pix in NDVI_dic:
                        NDVI_list.append(np.nan)
                        continue
                    vals = NDVI_dic[pix]
                    # print(vals)
                    if np.isnan(np.nanmean(vals)):
                        NDVI_list.append(np.nan)
                        continue
                    if len(vals) != 19:
                        NDVI_list.append(np.nan)
                        continue
                    #
                    v1 = vals[year - 2000]
                    NDVI_list.append(v1)
                df[f_name] = NDVI_list
        return df

    def add_window_trend_to_df(self, df):
        period_list = ['early','peak','late']
        # period_list = ['early']
        time='1982-2018'
        product='LAI3g'

        window_list = list(range(0, 22))
        print(window_list)

        for period in period_list:
            for slice in window_list:

                new_col_name=f'{product}_relative_change_{period}_trend_window_{slice:02d}'

                print(new_col_name)

                f = results_root + f'extract_relative_change_window/trend_window/{product}/{time}_during_{period}_window15_{product}/{slice:02d}_{product}_trend.tif.npy'

                # f = results_root + f'Partial_corr_{period}/window_{slice:02d}-15.npy'
                print(f)
                # exit()
                val_array = np.load(f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):

                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue

                    vals = val_dic[pix]

                    if vals < -9999:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[new_col_name] = val_list
        return df

    def add_window_p_value_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        # period_list = ['early']
        time = '1982-2018'
        product = 'LAI3g'

        window_list = list(range(0, 22))
        print(window_list)

        for period in period_list:
            for slice in window_list:

                new_col_name = f'{product}_relative_change_{period}_p_value_window_{slice:02d}'

                print(new_col_name)

                f = results_root + f'extract_relative_change_window/trend_window/{product}/{time}_during_{period}_window15_{product}/{slice:02d}_{product}_p_value.tif.npy'

                # f = results_root + f'Partial_corr_{period}/window_{slice:02d}-15.npy'
                print(f)
                # exit()
                val_array = np.load(f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):

                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue

                    vals = val_dic[pix]

                    if vals < -9999:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[new_col_name] = val_list
        return df


    def add_trend_to_df(self, df):

        fdir = results_root + '/lc_trend/'
        for f in (os.listdir(fdir)):
                # print()
            if not f.endswith('.npy'):
                continue
            if 'p_value' in f:
                continue


            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val = val*20
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name]=val_list

        return df


    def add_p_val_trend_to_df(self, df):
        fdir = results_root + 'trend_relative_change/2000-2018_Y/'
        for f in (os.listdir(fdir)):
            # print()
            if not f.endswith('.npy'):
                continue
            if 'trend' in f:
                continue


            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = '2000-2018_'+f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name] = val_list

        return df
    def add_landcover_trend_to_df(self, df):

        fdir = results_root + '/lc_trend/'
        for f in (os.listdir(fdir)):
                # print()
            if not f.endswith('.npy'):
                continue
            if 'p_value' in f:
                continue


            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                val = val_dic[pix]
                val = val*20
                if val < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(val)
            df[f_name]=val_list

        return df
    def add_mean_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        # period_list = ['winter']
        time = '2002-2015'
        df = df[df['row'] < 120]
        for period in period_list:
            fdir = results_root + 'mean_calculation_original/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue

                if f == 'during_{}_root_soil_moisture_mean.npy'.format(period):
                    continue
                if f == 'during_{}_surf_soil_moisture_mean.npy'.format(period):
                    continue
                if f == 'during_{}_VOD_mean.npy'.format(period):
                    continue
                if f == 'during_{}_NIRv_mean.npy'.format(period):
                    continue

                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = 'mean_' + f.split('.')[0] + '_{}'.format(time)
                print(f_name)
                # exit()
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix

                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if vals < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[f_name] = val_list
        return df

    def add_original_values_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        time = '1982-2015'
        df = df[df['row'] < 120]
        for period in period_list:
            fdir = results_root + 'mean_calculation_original/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue

                if f == 'during_{}_root_soil_moisture_mean.npy'.format(period):
                    continue
                if f == 'during_{}_surf_soil_moisture_mean.npy'.format(period):
                    continue
                if f == 'during_{}_VOD.npy'.format(period):
                    continue
                if f == 'during_{}_NIRv.npy'.format(period):
                    continue
                if f == 'during_{}_VOD.npy'.format(period):
                    continue
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = 'mean_' + f.split('.')[0] + '_{}'.format(time)
                print(f_name)
                # exit()
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix

                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if vals < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[f_name] = val_list
        return df

    def add_CV_to_df(self, df):
        # period_list = ['early', 'peak', 'late']
        period_list = ['winter']
        time = '1982-2015'
        df = df[df['row'] < 120]
        for period in period_list:
            fdir = results_root + 'CV_calculation_original/during_{}_{}/'.format(period,time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue

                if f == '{}_during_{}_root_soil_moisture.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_surf_soil_moisture.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_VOD.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_NIRv.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_VOD.npy'.format(time,period):
                    continue
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = 'CV_' + f.split('.')[0] + '_{}'.format(time)
                print(f_name)
                # exit()
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix

                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if vals < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[f_name] = val_list
        return df

    def add_anomaly_to_df(self, df):

        period_list = ['early', 'peak', 'late']
        time = '2002-2015'
        df = df[df['row'] < 120]
        for period in period_list:
            fdir = results_root + 'anomaly_variables_independently/{}_during_{}/'.format(time, period)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue
                if f == '{}_during_{}_MODIS_NDVI.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_root_soil_moisture.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_surf_soil_moisture.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_VOD.npy'.format(time,period):
                    continue
                if f == '{}_during_{}_NIRv.npy'.format(time,period):
                    continue

                val_dic = T.load_npy(fdir+f)

                f_name = 'anomaly_' + f.split('.')[0]
                print(f_name)
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if len(vals) != 14:
                        val_list.append(np.nan)
                        continue
                    v1 = vals[year - 2002]
                    val_list.append(v1)
                df[f_name] = val_list
        return df

    def add_p_val_correlation_to_df(self, df):
        period='early'
        time='2002-2015'
        fdir = results_root + 'trend_calculation_anomaly/during_{}_{}/'.format(period,time)
        for f in (os.listdir(fdir)):
            # print()
            if not f.endswith('.npy'):
                continue
            if f!= 'during_{}_MODIS_NDVI_p_value.npy'.format(period):
                continue
            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = 'anomaly_correlation'+f.split('.')[0]+'_{}'.format(time)
            print(f_name)
            exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                vals = val_dic[pix]
                if vals < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(vals)
            df[f_name] = val_list
        return df

    def add_max_correlation_to_df(self,df):
        variable='CSIF'

        time = '2002-2015'
        period='late'
        fdir = results_root + 'partial_correlation_anomaly/MODIS_NDVI/0105/max_correlation/{}/'.format(variable)
        f='{}_{}_max_correlation.npy'.format(period,time)

        val_dic = T.load_npy(fdir + f)
        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        f_name = variable+'_'+f.split('.')[0]
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_partial_correlation_to_df(self, df):

        time='2002-2015'
        period='late'
        fdir = results_root + 'partial_correlation_anomaly/MODIS_NDVI/0105/variables_contribution_MODIS_NDVI/{}/'.format(period)
        for f in (os.listdir(fdir)):
            if not time in f:
                continue
            if not 'p_value' in f:
                continue
            if not f.endswith('.npy'):
                continue
            val_dic = T.load_npy(fdir + f)
            f_name = 'MODIS_NDVI_'+f.split('.')[0]
            print(f_name)
                # exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                vals = val_dic[pix]
                if vals < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(vals)
            df[f_name] = val_list
        return df

    def add_NDVI_trend_label(self,df):
        early_mark_list=[]
        peak_mark_list = []
        late_mark_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            trend_early=row['during_early_GIMMS_NDVI_trend_1982-2015']
            trend_peak=row['during_peak_GIMMS_NDVI_trend_1982-2015']
            trend_late=row['during_late_GIMMS_NDVI_trend_1982-2015']
            trend_early_p_value=row['during_early_GIMMS_NDVI_p_value_1982-2015']
            trend_peak_p_value = row['during_peak_GIMMS_NDVI_p_value_1982-2015']
            trend_late_p_value =row['during_late_GIMMS_NDVI_p_value_1982-2015']

            if trend_early_p_value>0.1:
                early_mark='notrend'
            else:
                if trend_early>0:
                    early_mark='greening'
                else:
                    early_mark='browning'

            if trend_peak_p_value>0.1:
                peak_mark='notrend'
            else:
                if trend_peak>0:
                    peak_mark='greening'
                else:
                    peak_mark='browning'

            if trend_late_p_value>0.1:
                late_mark='notrend'
            else:
                if trend_late>0:
                    late_mark='greening'
                else:
                    late_mark='browning'

            early_mark_list.append(early_mark)
            peak_mark_list.append(peak_mark)
            late_mark_list.append(late_mark)
        df['early_mark']=early_mark_list
        df['peak_mark'] = peak_mark_list
        df['late_mark'] = late_mark_list

        return df


    def add_MAP_MAT_to_df(self, df):
        f='/Volumes/SSD_sumsang/project_greening/Data/Map_Mat/MAT.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=np.float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'MAT'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df


    def add_landcover_data_to_df(self, df):  #

        lc_dic = {
            1:'ENF',
            2:'EBF',
            3:'DNF',
            4:'DBF',
            6: 'open shrubs',
            7: 'closed shrubs',
            8: 'Woody Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',

        }

        lc_integrate_dic = {
            1:'NF',
            2:'BF',
            3:'NF',
            4:'BF',
            6:'shrub',
            7:'shrub',
            8: 'Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',
        }

        landcover_dic = {}
        fdir = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                landcover_dic.update(dic_i)
        f_name = 'landcover'
        landcover_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in landcover_dic:
                landcover_list.append(np.nan)
                continue
            vals = landcover_dic[pix][0]
            # if vals in lc_dic:
            if vals in lc_integrate_dic:
                lc = lc_integrate_dic[vals]
                # lc=lc_dic[vals]
            else:
                lc = np.nan
            landcover_list.append(lc)
            # landcover_list.append(vals)
        df[f_name] = landcover_list
        return df

    def add_Koppen_data_to_df(self, df):
        f=data_root+'Koppen/koppen_reclass_spatial_dic.npy'
        koppen_dic=T.load_npy(f)
        koppen_list=[]

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in koppen_dic:
                koppen_list.append(np.nan)
                continue
            vals = koppen_dic[pix]

            koppen_list.append(vals)
            # landcover_list.append(vals)
        df['koppen'] = koppen_list
        return df

    def add_soil_data_to_df(self, df):  #

        soil_dic={}
        fdir = data_root + 'SOIL/DIC/'
        for fdir2 in tqdm(os.listdir(fdir)):
            for f in tqdm(os.listdir(fdir+fdir2)):
                if f.endswith('.npy'):
                    dic_i = dict(np.load(fdir + fdir2+'/'+f, allow_pickle=True, ).item())
                    soil_dic.update(dic_i)
            f_name =fdir2.split('_')[1]
            print(f_name)
            soil_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                # pix = row.pix
                pix = row['pix']
                if not pix in soil_dic:
                    soil_list.append(np.nan)
                    continue
                vals = soil_dic[pix]
                # lc = lc_integrate_dic[vals]
                # soil_list.append(lc)
                soil_list.append(vals[0])
            df[f_name] = soil_list
        return df

    def add_landcover_data_to_df(self, df):  #

        lc_dic = {
            1: 'ENF',
            2: 'EBF',
            3: 'DNF',
            4: 'DBF',
            6: 'open shrubs',
            7: 'closed shrubs',
            8: 'Woody Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',

        }

        lc_integrate_dic = {
            1: 'NF',
            2: 'BF',
            3: 'NF',
            4: 'BF',
            6: 'shrub',
            7: 'shrub',
            8: 'Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',
        }

        landcover_dic = {}
        fdir = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                landcover_dic.update(dic_i)
        f_name = 'landcover'
        landcover_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            # year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in landcover_dic:
                landcover_list.append(np.nan)
                continue
            vals = landcover_dic[pix][0]
            # if vals in lc_dic:
            if vals in lc_integrate_dic:
                lc = lc_integrate_dic[vals]
                # lc=lc_dic[vals]
            else:
                lc = np.nan
            landcover_list.append(lc)
            # landcover_list.append(vals)
        df[f_name] = landcover_list
        return df

    def add_row(self,df):
        r_list=[]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r,c=pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_NDVI_mask(self,df):
        f = '/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=np.float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def drop_n_std(self, vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def P_PET_class(self):
        outdir = join(self.this_class_arr, 'P_PET_class')
        T.mkdir(outdir)
        outf = join(outdir, 'P_PET_class.npy')
        if isfile(outf):
            return T.load_npy(outf)
        dic = self.P_PET_ratio(self.P_PET_fdir)
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
        T.save_npy(dic_reclass, outf)
        return dic_reclass

    def P_PET_reclass(self,dic):
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

    def P_PET_ratio(self, P_PET_fdir):
        # fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'
        fdir = P_PET_fdir
        dic = T.load_npy_dir(fdir)
        dic_long_term = {}
        for pix in dic:
            vals = dic[pix]
            vals = np.array(vals)
            vals = T.mask_999999_arr(vals, warning=False)
            vals[vals == 0] = np.nan
            if T.is_all_nan(vals):
                continue
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term

    def add_Humid_nonhumid(self, df):
        P_PET_dic_reclass = self.P_PET_class()
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_reclass')
        df = T.add_spatial_dic_to_df(df, P_PET_dic_reclass, 'HI_class')
        df = df.dropna(subset=['HI_class'])
        df.loc[df['HI_reclass'] != 'Humid', ['HI_reclass']] = 'Dryland'
        return df

    def __rename_dataframe_columns(self, df):

            # for i in df:
            #     if not '2019'in i:
            #         continue
            #
            #
            #     # new_name_dic = {
            #     #
            #     #     i: f'1982-{i}' ,
            #     #
            #     # }
            new_name_dic = {

                '2016/2018/2019/2020_+_VOD_peak_relative_change_p_value': '2016/2018/2019/2020_VOD_peak_relative_change_p_value',
                '2016/2018/2019/2020__LAI3g_early_relative_change_trend':'2016/2018/2019/2020_LAI3g_early_relative_change_trend'

            }

            df = pd.DataFrame(df)
            df = df.rename(columns=new_name_dic)
            return df

    def show_field(self,df):

       for i in df:
           print(i)
       # T.print_head_n(df)
       return df

    def drop_field_df(self, df):
        # for i in df:
        #     # if 'partial_correlation':
        #     #     df=df.drop(columns=[i])
        #
        #     if i.startswith('anomaly'):
        #         print(i)
        #         df=df.drop(columns=[i])

        # df = df.drop(columns=['1982-2020_LAI3g_peak_raw','1982-2020_LAI3g_early_raw','1982-2020_LAI3g_late_raw'])

        for i in df:
            print(i)

        return df

    def __linearfit(self,x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx,sy,sxx,syy,sxy=0,0,0,0,0
        for i in range(0,int(N)):
            sx  += x[i]
            sy  += y[i]
            sxx += x[i]*x[i]
            syy += y[i]*y[i]
            sxy += x[i]*y[i]
        a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
        b = (sy - a*sx)/N
        r = -(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
        return a,b,r



    def add_Koppen_data_to_df(self, df):
        f = data_root + 'Koppen/koppen_reclass_spatial_dic.npy'
        koppen_dic = T.load_npy(f)
        koppen_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            # year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in koppen_dic:
                koppen_list.append(np.nan)
                continue
            vals = koppen_dic[pix]

            koppen_list.append(vals)
            # landcover_list.append(vals)
        df['koppen'] = koppen_list
        return df




class Build_partial_correlation_dataframe:

    def __init__(self):
        self.__config__()
        self.this_class_arr = results_root + 'Data_frame_1982-2018/partial_correlation_1982-2018/'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'Data_frame_1982-2018_df.df'
        self.P_PET_dir = data_root + 'original_dataset/aridity_P_PET_dic/'
        pass



    def __config__(self):
        period='late'

        # self.x_var_list = ['CO2',
        #               'VPD',
        #               'PAR',
        #               'temperature',
        #               'CCI_SM', ]
        self.x_var_list = [f'during_{period}_CCI_SM.npy',
            f'during_{period}_CO2.npy',
                           f'during_{period}_PAR.npy',
                           f'during_{period}_VPD.npy',
                           f'during_{period}_temperature.npy',

                            ]

    def run(self):
        # period = 'early'
        # time = '1982-2015'
        # fdir=results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time, time, period)

        df = self.__gen_df_init(self.dff)

        # df=self.foo(df)

        # df=self.add_partial_to_df(df)
        df=self.add_trend_to_df(df)
        # df=self.add_single_correlation_to_df(df)
        # df=self.add_difference_correlation_to_df(df)
        # df=self.add_max_correlation_to_df(df)
        # df=self.add_landcover_data_to_df(df)
        # df=self.add_Koppen_data_to_df(df)
        # df=self.add_row(df)
        # df=self.add_correlation_window_to_df(df)
        # P_PET_dic=self.P_PET_ratio(self.P_PET_dir)
        # P_PET_reclass_dic=self.P_PET_reclass(P_PET_dic)
        # df=T.add_spatial_dic_to_df(df,P_PET_reclass_dic,'HI_class')
        # df=self.add_NDVI_mask(df)

        # df=self.__rename_dataframe_columns(df)
        df=self.drop_field_df(df)


        # df=self.add_Koppen_data_to_df(df)

        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff)

    def __gen_df_init(self, file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df = self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self, file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self, df, dff, n=1000, random=False):
        dff = dff.split('.')[0]
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




    def foo(self, df):
        df_pix_list=[]

        pix_list=DIC_and_TIF().void_spatial_dic() # 生成360*720 键（0，0）
        for pix in pix_list:
            df_pix_list.append(pix)
        df_pix_list.sort()
        df['pix']=df_pix_list
        return df

    def add_original_GIMMIS_NDVI_to_df(self, df):
        period = 'late'

        f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_original_val/1982-2015_original_extraction_all_seasons/1982-2015_extraction_during_{}_growing_season_static/during_{}_GIMMS_NDVI.npy'.format(period,period)

        NDVI_dic = T.load_npy(f)
        f_name = 'GIMMS_NDVI_{}_original'.format(period)
        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in NDVI_dic:
                NDVI_list.append(np.nan)
                continue
            vals = NDVI_dic[pix]
            if len(vals) != 34:
                NDVI_list.append(np.nan)
                continue
            v1 = vals[year - 1982]
            NDVI_list.append(v1)
        df[f_name] = NDVI_list
        return df


    def add_anomaly_GIMMIS_NDVI_to_df(self, df):
        period = 'late'
        f = results_root + 'extraction_anomaly_val/1982-2015_anomaly_extraction_all_seasons/1982-2015_extraction_during_{}_growing_season_static/during_{}_GIMMS_NDVI.npy'.format(
            period, period)

        NDVI_dic = T.load_npy(f)
        f_name = 'GIMMS_NDVI_{}_anomaly'.format(period)
        NDVI_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in NDVI_dic:
                NDVI_list.append(np.nan)
                continue
            vals = NDVI_dic[pix]
            if len(vals) != 34:
                NDVI_list.append(np.nan)
                continue
            v1 = vals[year - 1982]
            NDVI_list.append(v1)
        df[f_name] = NDVI_list
        return df

    def add_trend_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        time = '1982-2001'
        for period in period_list:
            fdir = results_root + 'trend_calculation_original/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue
                if f != 'during_{}_LAI_GIMMS_{}_trend.npy'.format(period,time):
                    continue
                # if f != 'during_{}_LAI_GIMMS_{}_p_value.npy'.format(period,time):
                #     continue

                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = 'original_' + f.split('.')[0]
                print(f_name)

                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    # year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if vals < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[f_name] = val_list
        return df

    def add_correlation_window_to_df(self, df):
        window_list = list(range(0,19))
        print(window_list)

        for col in self.x_var_list:

            for slice in window_list:
                period = 'late'
                temp=slice+1
                temp_col=col.split('.')[0]
                new_col_name = f'window_{temp:02d}_{temp_col}_pcorr'
                # new_col_name = f'window_{temp:02d}_{temp_col}_p_value'
                print(new_col_name)
                # f=results_root+f'partial_window/1982-2015_during_{period}_window15/partial_correlation_{period}_1982-2015_window{slice:02d}_correlation.npy'
                f = results_root + f'partial_window/1982-2015_during_{period}_window15/partial_correlation_{period}_1982-2015_window{slice:02d}_correlation.npy'

                # f = results_root + f'Partial_corr_{period}/window_{slice:02d}-15.npy'
                print(f)
                # exit()
                val_dic =T.load_npy(f)
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):

                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    if not col in val_dic[pix] :
                        val_list.append(np.nan)
                        continue
                    vals= val_dic[pix][col]

                    if vals < -9999:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[new_col_name] = val_list
        return df

    def add_p_val_trend_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        time = '1982-2015'
        for period in period_list:
            fdir = results_root + 'trend_calculation_original/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue
                if f != 'during_{}_GIMMS_NDVI_p_value.npy'.format(period):
                    continue
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = 'original_' + f.split('.')[0] + '_{}'.format(time)
                print(f_name)
                # exit()
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if vals < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[f_name] = val_list
        return df

    def add_p_val_correlation_to_df(self, df):
        period = 'early'
        time = '2001-2015'
        fdir = results_root + 'trend_calculation_anomaly/during_{}_{}/'.format(period, time)
        for f in (os.listdir(fdir)):
            # print()
            if not f.endswith('.npy'):
                continue
            if f != 'during_{}_GIMMS_NDVI_p_value.npy'.format(period):
                continue
            val_array = np.load(fdir + f)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = 'anomaly_correlation' + f.split('.')[0] + '_{}'.format(time)
            print(f_name)
            exit()
            val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                vals = val_dic[pix]
                if vals < -99:
                    val_list.append(np.nan)
                    continue
                val_list.append(vals)
            df[f_name] = val_list
        return df

    def add_correlation_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        time = '1999-2015'

        for period in period_list:
            fdir = results_root + 'partial_correlation_anomaly_NDVI/'
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = 'anomaly_with_trend_' + f.split('.')[0] + '_{}'.format(time)
                print(f_name)
                # exit()
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if vals < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[f_name] = val_list
        return df

    def add_mean_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        time = '2002-2015'
        for period in period_list:
            fdir = results_root + 'mean_calculation_original/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):

                if not f.endswith('.npy'):
                    continue
                if f =='during_{}_root_soil_moisture_mean.npy'.format(period):
                    continue
                if f =='during_{}_surf_soil_moisture_mean.npy'.format(period):
                    continue
                if f =='during_{}_VOD.npy'.format(period):
                    continue
                if f =='during_{}_NIRv.npy'.format(period):
                    continue
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = 'mean_' + f.split('.')[0] + '_{}'.format(time)
                print(f_name)
                # exit()
                val_list = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]
                    if vals < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(vals)
                df[f_name] = val_list
        return df

    def add_partial_to_df(self, df):  # 这里

        period_list = ['early' ,'peak', 'late']
        time = '1982-2018'

        for period in period_list:
            fdir = results_root + 'partial_correlation/LAI_GIMMS/'


            # f=f'{time}_partial_correlation_{period}_anomaly_LAI_GIMMS.npy'
            f = f'{time}_partial_correlation_p_value_{period}_anomaly_LAI_GIMMS.npy'

            val_dic = T.load_npy(fdir+f)
            var_name_list = []
            for pix in val_dic:
                # print(pix)
                vals = val_dic[pix]
                for var_i in vals:
                    var_name_list.append(var_i)
            var_name_list = list(set(var_name_list))

            for var_i in var_name_list:
                val_list = []

                # df_column_name=f'{var_i}_{time}_correlation'
                df_column_name = f'{var_i}_{time}_p_value'

                print(df_column_name)
                # exit()
                for i, row in tqdm(df.iterrows(), total=len(df),desc=df_column_name):
                    pix = row['pix']
                    if pix not in val_dic:
                        val_list.append(np.nan)
                        continue
                    dic_i = val_dic[pix]
                    if not var_i in dic_i:
                        val_list.append(np.nan)
                        continue
                    val = dic_i[var_i]
                    if val < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(val)
                df[df_column_name] = val_list
        return df

    def add_single_correlation_to_df(self, df):  # 这里
        variable_list=['CO2','PAR','VPD','SPEI3','temperature','CCI_SM']

        period_list = ['early' ,'peak', 'late']
        time = '2002-2018'

        for period in period_list:
            fdir = results_root + 'test_original/during_{}_{}/'.format(period,time)
            print(fdir)
            for var_i in variable_list:

                # f=f'{time}_partial_correlation_{period}_anomaly_LAI_GIMMS.npy'
                f = f'during_{period}_{var_i}_LAI_GIMMS_r.npy'
                print(f)

                # val_dic = T.load_npy(fdir+f)
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)

                val_list = []

                # df_column_name=f'{var_i}_{time}_correlation'
                df_column_name = f'simple_during_{time}_{period}_{var_i}_LAI_GIMMS_r'

                print(df_column_name)
                # exit()
                for i, row in tqdm(df.iterrows(), total=len(df),desc=df_column_name):
                    pix = row['pix']
                    if pix not in val_dic:
                        val_list.append(np.nan)
                        continue
                    val = val_dic[pix]

                    if val < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(val)
                df[df_column_name] = val_list
        return df

    def add_difference_correlation_to_df(self, df):  # 这里
        variable_list=['CO2','PAR','VPD','SPEI3','temperature','CCI_SM']

        period_list = ['early' ,'peak', 'late']

        for period in period_list:
            fdir = results_root + 'test_original/during_{}_difference/'.format(period)

            print(fdir)
            for var_i in variable_list:

                # f=f'{time}_partial_correlation_{period}_anomaly_LAI_GIMMS.npy'
                f = f'{period}_{var_i}_correlation_difference.npy'

                print(f)

                # val_dic = T.load_npy(fdir+f)
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)

                val_list = []

                # df_column_name=f'{var_i}_{time}_correlation'
                df_column_name = f'{period}_{var_i}_LAI_GIMMS_r_difference'
                print(df_column_name)

                # exit()
                for i, row in tqdm(df.iterrows(), total=len(df),desc=df_column_name):
                    pix = row['pix']
                    if pix not in val_dic:
                        val_list.append(np.nan)
                        continue
                    val = val_dic[pix]

                    if val < -99:
                        val_list.append(np.nan)
                        continue
                    val_list.append(val)
                df[df_column_name] = val_list
        return df

    def add_landcover_data_to_df(self, df):  #

        lc_dic = {
            1: 'ENF',
            2: 'EBF',
            3: 'DNF',
            4: 'DBF',
            6: 'open shrubs',
            7: 'closed shrubs',
            8: 'Woody Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',

        }

        lc_integrate_dic = {
            1: 'NF',
            2: 'BF',
            3: 'NF',
            4: 'BF',
            6: 'shrub',
            7: 'shrub',
            8: 'Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',
        }

        landcover_dic = {}
        fdir = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                landcover_dic.update(dic_i)
        f_name = 'landcover'
        landcover_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in landcover_dic:
                landcover_list.append(np.nan)
                continue
            vals = landcover_dic[pix][0]
            # if vals in lc_dic:
            if vals in lc_integrate_dic:
                lc = lc_integrate_dic[vals]
                # lc=lc_dic[vals]
            else:
                lc = np.nan
            landcover_list.append(lc)
            # landcover_list.append(vals)
        df[f_name] = landcover_list
        return df

    def add_Koppen_data_to_df(self, df):
        f = data_root + 'Koppen/koppen_reclass_spatial_dic.npy'
        koppen_dic = T.load_npy(f)
        koppen_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in koppen_dic:
                koppen_list.append(np.nan)
                continue
            vals = koppen_dic[pix]

            koppen_list.append(vals)
            # landcover_list.append(vals)
        df['koppen'] = koppen_list
        return df



    def add_row(self, df):
        r_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r, c = pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_NDVI_mask(self,df):
        f = '/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=np.float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def __rename_dataframe_columns(self, df):


        new_name_dic = {
            'during_early_CCI_SM_2018_1982-2001_p_value':'during_early_CCI_SM_1982-2001_p_value',
            'during_early_CCI_SM_2018_2002-2018_p_value':'during_early_CCI_SM_2002-2018_p_value',
            'during_peak_CCI_SM_2018_1982-2001_p_value': 'during_peak_CCI_SM_1982-2001_p_value',
            'during_peak_CCI_SM_2018_2002-2018_p_value': 'during_peak_CCI_SM_2002-2018_p_value',
            'during_late_CCI_SM_2018_1982-2001_p_value': 'during_late_CCI_SM_1982-2001_p_value',
            'during_late_CCI_SM_2018_2002-2018_p_value': 'during_late_CCI_SM_2002-2018_p_value',


        }

        df = pd.DataFrame(df)
        df = df.rename(columns=new_name_dic)
        return df


    def P_PET_ratio(self, P_PET_fdir):
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
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term

    def P_PET_reclass(self,dic):
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

    def drop_n_std(self,vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def drop_field_df(self, df):

        # df = df.drop(columns=['during_early_CO2_LAI_GIMMS_r',
        #                       'during_early_PAR_LAI_GIMMS_r',
        #                       'during_early_VPD_LAI_GIMMS_r',
        #                       'during_early_SPEI3_LAI_GIMMS_r',
        #                       'during_early_temperature_LAI_GIMMS_r',
        #                       'during_peak_CO2_LAI_GIMMS_r',
        #                       'during_peak_PAR_LAI_GIMMS_r',
        #                       'during_peak_VPD_LAI_GIMMS_r',
        #                       'during_peak_SPEI3_LAI_GIMMS_r',
        #                       'during_peak_temperature_LAI_GIMMS_r',
        #                       'during_late_CO2_LAI_GIMMS_r',
        #                       'during_late_PAR_LAI_GIMMS_r',
        #                       'during_late_VPD_LAI_GIMMS_r',
        #                       'during_late_SPEI3_LAI_GIMMS_r',
        #                       'during_late_temperature_LAI_GIMMS_r',])
        for i in df:
            print(i)
        return df

    def __linearfit(self, x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
        for i in range(0, int(N)):
            sx += x[i]
            sy += y[i]
            sxx += x[i] * x[i]
            syy += y[i] * y[i]
            sxy += x[i] * y[i]
        a = (sy * sx / N - sxy) / (sx * sx / N - sxx)
        b = (sy - a * sx) / N
        r = -(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
        return a, b, r

    def build_df(self, df, x_f_list, x_var_list):

        period = 'early'
        time = '1982-2015'
        y_f = results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/' \
                             'during_{}_GIMMS_NDVI.npy'.format(time, time, period, period)

        y_arr = T.load_npy(y_f)
        pix_list = []
        y_val_list = []
        for pix in y_arr:
            vals = y_arr[pix]
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        for ii in range(len(x_f_list)):
            print(ii)
            # exit()
            x_val_list = []
            x_dic = T.load_npy(x_f_list[ii])
            for i, row in tqdm(df.iterrows(), total=len(df), desc=x_var_list[ii]):
                pix = row.pix
                if not pix in x_dic:
                    x_val_list.append(np.nan)
                    continue
                vals = x_dic[pix]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append(np.nan)
                    continue
                x_val_list.append(vals)
            df[x_var_list[ii]] = x_val_list
        # T.print_head_n(df)
        # exit()
        return df

    def genetate_x_var_list(self, fdir, period):

        x_f_list = []
        x_var_list = []

        for fdir_1 in tqdm(sorted(os.listdir(fdir))):

            if fdir_1 == 'during_{}_GIMMS_NDVI.npy'.format(period, ):
                continue
            if fdir_1 == 'during_{}_NIRv.npy'.format(period, ):
                continue
            if fdir_1 == 'during_{}_VOD.npy'.format(period, ):
                continue
            # if fdir_1 == 'during_{}_root_soil_moisture.npy'.format(period, ):
            #     continue
            # if fdir_1 == 'during_{}_surf_soil_moisture.npy'.format(period, ):
            #     continue

            x_f_list.append(fdir + fdir_1)

        for x_f in x_f_list:
            print(x_f)
            split1 = x_f.split('/')[-1]
            split2 = split1.split('.')[0]
            var_name = '_'.join(split2.split('_')[2:])
            x_var_list.append(var_name)
            print(var_name)

        return x_f_list, x_var_list



    def cal_contribution(self, df, x_var_list):
        predict_y_list = []
        y_trend_list = []
        contribution_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            y_vals = row['y']
            y_trend, y_b, _ = self.__linearfit(range(len(y_vals)), y_vals)
            y_trend_list.append(y_trend)
            contribution_list = []
            contribution_dic_i = {}
            for x in x_var_list:
                x_vals = row[x]
                try:
                    r, p = stats.pearsonr(x_vals, y_vals)  # partial correlation

                    # partial_corr(data=df, x='X', y='Y', covar=['covar1', 'covar2'], method='pearson')
                    k, b, _ = self.__linearfit(range(len(x_vals)), x_vals)  # annual trend
                    if k > 999:
                        contribution = np.nan
                    else:
                        contribution = k * r
                except:
                    contribution = np.nan
                contribution_list.append(contribution)
                contribution_dic_i[x] = contribution
            contribution_dic[pix] = contribution_dic_i
            predict_y = np.sum(contribution_list)
            predict_y_list.append(predict_y)

        for x in x_var_list:
            x_contribution_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row.pix
                contribution_i = contribution_dic[pix][x]
                x_contribution_list.append(contribution_i)
            var_name = f'{x}_contribution'
            df[var_name] = x_contribution_list
        df['predicted_y'] = predict_y_list
        df['y_trend'] = y_trend_list
        T.print_head_n(df)
        return df

    def performance(self, df):
        predicted_y = df['predicted_y'].to_list()
        y_trend = df['y_trend'].to_list()
        df_new = pd.DataFrame()
        df_new['predicted_y'] = predicted_y
        df_new['y_trend'] = y_trend
        df_new = df_new.dropna()
        X = df_new['predicted_y'].to_list()
        Y = df_new['y_trend'].to_list()
        X = np.array(X)
        Y = np.array(Y)
        X_mask = X[~np.isinf(X)]
        Y_mask = Y[~np.isinf(X)]
        KDE_plot().plot_scatter(X_mask, Y_mask, plot_fit_line=True, is_plot_1_1_line=False)
        plt.show()

    def contribution_bar(self, result_f, x_dir):
        # x_dir = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        result_dic = T.load_npy(result_f)
        x_var_list = self.__get_x_var_list(x_dir)
        y_predicted = []
        y_trend = []

        for pix in result_dic:
            r, c = pix
            if r > 120:
                continue
            dic_i = result_dic[pix]
            y_predicted_i = dic_i['y_predicted']
            y_trend_i = dic_i['y_trend']
            y_predicted.append(y_predicted_i)
            y_trend.append(y_trend_i)
        KDE_plot().plot_scatter(y_predicted, y_trend)

        plt.figure()
        box = []
        err = []
        for x_var in x_var_list:
            x_var_all = []
            for pix in result_dic:
                r, c = pix
                if r > 120:
                    continue
                dic_i = result_dic[pix]
                contribution_dic = dic_i['contribution_dic']
                if not x_var in contribution_dic:
                    continue
                contribution_val = contribution_dic[x_var]
                if np.isnan(contribution_val):
                    continue
                x_var_all.append(contribution_val)
            box.append(np.mean(x_var_all))
            # box.append(x_var_all)
            err.append(np.std(x_var_all))
        # plt.boxplot(box,labels=x_var_list,showfliers=False)
        # plt.bar(x_var_list,box,yerr=err)
        plt.bar(x_var_list, box, )
        plt.show()

class Build_trend_dataframe:

    def __init__(self):
        self.this_class_arr = results_root + 'landcover_trend/'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'landcover_trend.df'
        self.P_PET_dir=data_root+'original_dataset/aridity_P_PET_dic/'
        pass





    def run(self):
        # period = 'early'
        # time = '1982-2015'
        # fdir=results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time, time, period)

        df=self.__gen_df_init(self.dff)

        df=self.foo(df)
        df=self.add_trend_to_df(df)


        # df=self.add_soil_data_to_df(df)
        # df=self.add_MAP_MAT_to_df(df)
        # df = self.add_NDVI_mask(df)
        # df=self.add_row(df)
        # df=self.add_mean_to_df(df)

        # # df=self.add_Koppen_data_to_df(df)
        # df=self.add_landcover_data_to_df(df)

        # P_PET_dic=self.P_PET_ratio(self.P_PET_dir)
        # P_PET_reclass_dic=self.P_PET_reclass(P_PET_dic)
        # df=T.add_spatial_dic_to_df(df,P_PET_reclass_dic,'HI_class')

        # df=self.show_field(df)
        # df=self.drop_field_df(df)

        # df=self.__rename_dataframe_columns(df)


        df=self.add_trend_to_df(df)
        # df=self.add_NDVI_trend_label(df)




        T.save_df(df,self.dff)
        self.__df_to_excel(df,self.dff)


    def __gen_df_init(self,file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, file)
            return df
        else:
            df= self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __load_df(self,file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __df_to_excel(self,df,dff,n=1000,random=False):
        dff=dff.split('.')[0]
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

    def foo(self, df):

        f = results_root + 'trend_calculation_relative_change_2000-/early_LAI4g_trend.npy'
        dic = {}
        outf = self.dff
        result_dic = {}
        # dic = T.load_npy(f)
        array=np.load(f)
        dic=DIC_and_TIF().spatial_arr_to_dic(array)
        pix_list = []
        change_rate_list = []
        year = []
        f_name = 'early_LAI4g_trend'

        print(f_name)
        for pix in tqdm(dic):
            time_series = dic[pix]
            y = 0
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val[0])
                year.append(y + 1)
                y = y + 1
        df['pix'] = pix_list
        df[f_name] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df



    def add_trend_to_df(self, df):
        # period_list=['early', 'peak', 'late']
        period_list = ['peak','late' ]
        slice_list=list(range(0,19))

        # period_list = ['early']
        time='1982-2015'
        for period in period_list:

            fdir = results_root + 'trend_window/variables_contribution_conversion/'

            for f in (os.listdir(fdir)):
                if not f.endswith('.npy'):
                    continue
                if not period in f:
                    continue

                val_dic = T.load_npy(fdir + f)
                val_list = []

                fname=f.split('.')[0]+'_window'
                print(fname)

                for i, row in tqdm(df.iterrows(), total=len(df)):

                    year=row['year']
                    pix = row['pix']
                    if not pix in val_dic:
                        val_list.append(np.nan)
                        continue
                    vals = val_dic[pix]

                    if len(vals)!=19:
                        continue
                    v1 = vals[year - 1]
                    val_list.append(v1)
                df[fname]=val_list

        return df



    def add_mean_to_df(self, df):
        # period_list = ['early', 'peak', 'late']
        period_list = ['peak']
        time = '1982-2015'
        df = df[df['row'] < 120]

        for period in period_list:
            fdir_all = results_root + 'extraction_original_val_window_mean/1982-2015_during_{}/15_year_window/'.format(period)
            for fdir in (os.listdir(fdir_all)):
                if fdir == 'during_{}_root_soil_moisture'.format(period):
                    continue
                if fdir == 'during_{}_surf_soil_moisture'.format(period):
                    continue
                if fdir == 'during_{}_VOD'.format(period):
                    continue
                if fdir == 'during_{}_NIRv'.format(period):
                    continue
                if fdir.endswith('npy'):
                    continue

                dic = {}
                val_list=[]

                for f in (os.listdir(fdir_all+fdir)):

                    dic_i = dict(np.load(fdir_all + fdir+'/'+f, allow_pickle=True, ).item())
                    dic.update(dic_i)


                f_name = 'mean_' + fdir+'_window'
                print(f_name)
                # exit()

                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    pix = row['pix']
                    if not pix in dic:
                        val_list.append(np.nan)
                        continue
                    vals = dic[pix]

                    if len(vals) != 19:
                        val_list.append(np.nan)
                        continue
                    v1 = vals[year - 1][0]
                    val_list.append(v1)
                df[f_name] = val_list
        return df



    def add_MAP_MAT_to_df(self, df):
        f='/Volumes/SSD_sumsang/project_greening/Data/Map_Mat/MAT.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=np.float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'MAT'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df


    def add_landcover_data_to_df(self, df):  #

        lc_dic = {
            1:'ENF',
            2:'EBF',
            3:'DNF',
            4:'DBF',
            6: 'open shrubs',
            7: 'closed shrubs',
            8: 'Woody Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',

        }

        lc_integrate_dic = {
            1:'NF',
            2:'BF',
            3:'NF',
            4:'BF',
            6:'shrub',
            7:'shrub',
            8: 'Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',
        }

        landcover_dic = {}
        fdir = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                landcover_dic.update(dic_i)
        f_name = 'landcover'
        landcover_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in landcover_dic:
                landcover_list.append(np.nan)
                continue
            vals = landcover_dic[pix][0]
            # if vals in lc_dic:
            if vals in lc_integrate_dic:
                lc = lc_integrate_dic[vals]
                # lc=lc_dic[vals]
            else:
                lc = np.nan
            landcover_list.append(lc)
            # landcover_list.append(vals)
        df[f_name] = landcover_list
        return df

    def add_Koppen_data_to_df(self, df):
        f=data_root+'Koppen/koppen_reclass_spatial_dic.npy'
        koppen_dic=T.load_npy(f)
        koppen_list=[]

        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in koppen_dic:
                koppen_list.append(np.nan)
                continue
            vals = koppen_dic[pix]

            koppen_list.append(vals)
            # landcover_list.append(vals)
        df['koppen'] = koppen_list
        return df

    def add_soil_data_to_df(self, df):  #

        soil_dic={}
        fdir = data_root + 'SOIL/DIC/'
        for fdir2 in tqdm(os.listdir(fdir)):
            for f in tqdm(os.listdir(fdir+fdir2)):
                if f.endswith('.npy'):
                    dic_i = dict(np.load(fdir + fdir2+'/'+f, allow_pickle=True, ).item())
                    soil_dic.update(dic_i)
            f_name =fdir2.split('_')[1]
            print(f_name)
            soil_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):

                # pix = row.pix
                pix = row['pix']
                if not pix in soil_dic:
                    soil_list.append(np.nan)
                    continue
                vals = soil_dic[pix]
                # lc = lc_integrate_dic[vals]
                # soil_list.append(lc)
                soil_list.append(vals[0])
            df[f_name] = soil_list
        return df

    def add_landcover_data_to_df(self, df):  #

        lc_dic = {
            1: 'ENF',
            2: 'EBF',
            3: 'DNF',
            4: 'DBF',
            6: 'open shrubs',
            7: 'closed shrubs',
            8: 'Woody Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',

        }

        lc_integrate_dic = {
            1: 'NF',
            2: 'BF',
            3: 'NF',
            4: 'BF',
            6: 'shrub',
            7: 'shrub',
            8: 'Savanna',
            9: 'Savanna',
            10: 'Grassland',
            12: 'Cropland',
        }

        landcover_dic = {}
        fdir = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                landcover_dic.update(dic_i)
        f_name = 'landcover'
        landcover_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in landcover_dic:
                landcover_list.append(np.nan)
                continue
            vals = landcover_dic[pix][0]
            # if vals in lc_dic:
            if vals in lc_integrate_dic:
                lc = lc_integrate_dic[vals]
                # lc=lc_dic[vals]
            else:
                lc = np.nan
            landcover_list.append(lc)
            # landcover_list.append(vals)
        df[f_name] = landcover_list
        return df

    def add_row(self,df):
        r_list=[]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r,c=pix
            r_list.append(r)
        df['row'] = r_list
        return df

    def add_NDVI_mask(self,df):
        f = '/Volumes/SSD_sumsang/project_greening/Data/NDVI_mask.tif'

        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        array = np.array(array, dtype=np.float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def P_PET_ratio(self, P_PET_fdir):
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
            vals = self.drop_n_std(vals)
            long_term_vals = np.nanmean(vals)
            dic_long_term[pix] = long_term_vals
        return dic_long_term

    def P_PET_reclass(self,dic):
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

    def drop_n_std(self,vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def __rename_dataframe_columns(self, df):

            new_name_dic = {

                '2002-2015_during_late_MODIS_NDVI_trend_2002-2015': '2002-2015_during_late_MODIS_NDVI_trend',
                '2002-2015_during_early_MODIS_NDVI_trend_2002-2015': '2002-2015_during_early_MODIS_NDVI_trend',
                '2002-2015_during_peak_MODIS_NDVI_trend_2002-2015': '2002-2015_during_peak_MODIS_NDVI_trend',
                '2002-2015_during_late_PAR_trend_2002-2015': '2002-2015_during_late_PAR_trend',
                '2002-2015_during_early_PAR_trend_2002-2015': '2002-2015_during_early_PAR_trend',
                '2002-2015_during_peak_PAR_trend_2002-2015': '2002-2015_during_peak_PAR_trend',

            }

            df = pd.DataFrame(df)
            df = df.rename(columns=new_name_dic)
            return df

    def show_field(self,df):

       for i in df:
           print(i)
       # T.print_head_n(df)
       return df

    def drop_field_df(self, df):
        # for i in df:
        #     # if 'partial_correlation':
        #     #     df=df.drop(columns=[i])
        #
        #     if i.startswith('anomaly'):
        #         print(i)
        #         df=df.drop(columns=[i])

        # df = df.drop(columns=['1982-2015_early_GIMMS_NDVI_anomaly'])
        for i in df:
            print(i)

        return df

    def __linearfit(self,x, y):
        '''
        最小二乘法拟合直线
        :param x:
        :param y:
        :return:
        '''
        N = float(len(x))
        sx,sy,sxx,syy,sxy=0,0,0,0,0
        for i in range(0,int(N)):
            sx  += x[i]
            sy  += y[i]
            sxx += x[i]*x[i]
            syy += y[i]*y[i]
            sxy += x[i]*y[i]
        a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
        b = (sy - a*sx)/N
        r = -(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
        return a,b,r



    def add_Koppen_data_to_df(self, df):
        f = data_root + 'Koppen/koppen_reclass_spatial_dic.npy'
        koppen_dic = T.load_npy(f)
        koppen_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in koppen_dic:
                koppen_list.append(np.nan)
                continue
            vals = koppen_dic[pix]

            koppen_list.append(vals)
            # landcover_list.append(vals)
        df['koppen'] = koppen_list
        return df



def main():
    Build_dataframe().run()
    # Build_partial_correlation_dataframe().run()
    # Build_trend_dataframe().run()
    pass


if __name__ == '__main__':
    main()