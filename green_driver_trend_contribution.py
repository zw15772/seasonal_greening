# coding=utf-8

from __init__ import *
from pingouin import partial_corr
## for 1982-2015 greening project_dataframe  11/12/2021  Wen

class Build_dataframe:

    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame_2002-2015/'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'Data_frame_2002-2015_df.df'
        pass





    def run(self):
        # period = 'early'
        # time = '1982-2015'
        # fdir=results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time, time, period)

        df=self.__gen_df_init(self.dff)


        # df=self.foo(df)
        # df=self.add_anomaly_GIMMIS_NDVI_to_df(df)
        # df = self.add_Pierre_GIMMIS_NDVI_to_df(df)
        # df=self.add_row(df)
        # df = self.add_anomaly_to_df(df)
        # df=self.add_trend_to_df(df)
        # df=self.add_p_val_trend_to_df(df)
        # df=self.add_mean_to_df(df)
        # df=self.add_CV_to_df(df)
        # df=self.add_soil_data_to_df(df)
        # df=self.add_MAP_MAT_to_df(df)
        # df=self.add_winter_to_df(df)
        # df=self.add_Koppen_data_to_df(df)
        # df=self.add_landcover_data_to_df(df)
        # df=self.add_max_correlation_to_df(df)
        # df=self.add_partial_correlation_to_df(df)

        # df=self.show_field(df)
        # df=self.drop_field_df(df)
        # df=self.__rename_dataframe_columns(df)


        # df=self.add_trend_to_df(df)
        # df=self.add_NDVI_trend_label(df)



        # x_f_list, x_var_list=self.genetate_x_var_list(fdir,period)
        # self.build_df(df,x_f_list, x_var_list)
        # self.cal_contribution(df,x_var_list)
        # self.performance(df)

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

    def foo(self,df):

        f = results_root+'anomaly_variables_independently/2002-2015_during_early/2002-2015_during_early_MODIS_NDVI.npy'
        dic = {}
        outf = self.dff
        result_dic = {}
        dic = T.load_npy(f)
        pix_list=[]
        change_rate_list=[]
        year=[]
        f_name = '2002-2015_early_MODIS_NDVI_anomaly'

        print(f_name)
        for pix in tqdm(dic):
            time_series=dic[pix]
            y=0
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y+2002)
                y=y+1
        df['pix']=pix_list
        df[f_name] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
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
        time='2002-2015'
        f = results_root + 'anomaly_variables_independently/{}_during_{}/{}_during_{}_MODIS_NDVI.npy'.format(time,period,time,period)

        NDVI_dic = T.load_npy(f)
        f_name = '{}_{}_MODIS_NDVI_anomaly'.format(time,period)
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
            if len(vals) != 14:
                NDVI_list.append(np.nan)
                continue
            v1 = vals[year - 2002]
            NDVI_list.append(v1)
        df[f_name] = NDVI_list
        return df
    def add_Pierre_GIMMIS_NDVI_to_df(self, df):
        period = 'late'
        time='1999-2015'
        f = results_root + '%NDVI_Pierre/{}_{}.npy'.format(time,period)

        NDVI_dic = T.load_npy(f)
        f_name = '{}_GIMMS_NDVI_{}_change%'.format(time,period)
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
            v1 = vals[year - 1999]
            NDVI_list.append(v1)
        df[f_name] = NDVI_list
        return df

    def add_winter_to_df(self, df):
        period = 'winter'
        time='1982-2015'
        f = results_root + '/anomaly_variables_independently/1982-2015_winter_temperature.npy'

        NDVI_dic = T.load_npy(f)
        f_name = 'anomaly_1982-2015_during_winter_temperature'
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
            if len(vals) != 33:
                NDVI_list.append(np.nan)
                continue

            if (year - 1983) < 0:
                NDVI_list.append(np.nan)
                continue
            v1 = vals[year - 1983]
            NDVI_list.append(v1)
        df[f_name] = NDVI_list
        return df

    def add_trend_to_df(self, df):
        period_list=['early', 'peak', 'late']
        # period_list = ['early']
        time='2002-2015'
        for period in period_list:
            fdir = results_root + 'trend_calculation_anomaly/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue
                if 'p_value' in f:
                    continue
                if f == '{}_during_{}_root_soil_moisture_trend.npy'.format(time, period):
                    continue
                if f == '{}_during_{}_surf_soil_moisture_trend.npy'.format(time, period):
                    continue
                if f == '{}_during_{}_VOD_trend.npy'.format(time, period):
                    continue
                if f == '{}_during_{}_NIRv_trend.npy'.format(time, period):
                    continue

                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = f.split('.')[0]
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


    def add_p_val_trend_to_df(self, df):
        period_list = ['early', 'peak', 'late']
        time='2002-2015'
        for period in period_list:

            fdir = results_root + 'trend_calculation_anomaly/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue
                # if f!= '{}_during_{}_GIMMS_NDVI_p_value.npy'.format(time,period):
                #     continue
                # if f!= '{}_during_{}_MODIS_NDVI_p_value.npy'.format(time,period):
                #     continue
                if f!= '{}_during_{}_CSIF_p_value.npy'.format(time,period):
                    continue
                # if f!= '{}_during_{}_CSIF_fpar_p_value.npy'.format(time,period):
                #     continue
                val_array = np.load(fdir + f)
                val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
                f_name = f.split('.')[0]
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

        # df = df.drop(columns=['CV_during_peak_VOD_CV_1982-2015','during_peak_root_soil_moisture_trend_1982-2015','mean_during_peak_VOD_mean_1982-2015'])
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




class Build_partial_correlation_dataframe:

    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame_1982-2015/'

        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'Build_partial_correlation_dataframe_df.df'
        pass

    def run(self):
        # period = 'early'
        # time = '1982-2015'
        # fdir=results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time, time, period)

        df = self.__gen_df_init(self.dff)

        # df=self.foo(df)

        # df=self.add_partial_to_df(df)
        # df=self.add_max_correlation_to_df(df)
        # df=self.add_landcover_data_to_df(df)
        # df=self.add_Koppen_data_to_df(df)
        df=self.add_row(df)


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

    def __df_to_excel(self, df, dff, n=1000, random=True):
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
        time = '1999-2015'
        for period in period_list:
            fdir = results_root + 'trend_calculation_original/during_{}_{}/'.format(period, time)
            for f in (os.listdir(fdir)):
                # print()
                if not f.endswith('.npy'):
                    continue
                if f != 'during_{}_GIMMS_NDVI_trend.npy'.format(period):
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
        time = '1982-2015'

        for period in period_list:
            fdir = results_root + 'partial_correlation_anomaly_NDVI/'

            # f=f'{time}_partial_correlation_p_value_{period}_anomaly.npy'
            f=f'{time}_partial_correlation{period}_anomaly.npy'

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
                # df_column_name=var_i+time+period
                # df_column_name=f'{var_i}_{time}_{period}_p_value'
                df_column_name=f'{var_i}_{time}_{period}_correlation'

                # print(df_column_name)
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

    def __rename_dataframe_columns(self, df):

        new_name_dic = {

            'anomaly_during_early_GIMMS_NDVI_trend': 'anomaly_during_early_GIMMS_NDVI_trend_1982-2015',
            'anomaly_during_peak_GIMMS_NDVI_trend': 'anomaly_during_peak_GIMMS_NDVI_trend_1982-2015',
            'anomaly_during_late_GIMMS_NDVI_trend': 'anomaly_during_late_GIMMS_NDVI_trend_1982-2015',
            'anomaly_during_peak_GIMMS_NDVI_p_value': 'anomaly_during_peak_GIMMS_NDVI_p_value_1982-2015',
            'anomaly_during_late_GIMMS_NDVI_p_value': 'anomaly_during_late_GIMMS_NDVI_p_value_1982-2015',
            'anomaly_during_early_GIMMS_NDVI_p_value': 'anomaly_during_early_GIMMS_NDVI_p_value_1982-2015',
        }

        df = pd.DataFrame(df)
        df = df.rename(columns=new_name_dic)
        return df

    def drop_field_df(self, df):
        df = df.drop(columns=['anomaly_during_early_GIMMS_NDVI_p_value_1982-2000',
                              'anomaly_during_peak_GIMMS_NDVI_p_value_1982-2000',
                              'anomaly_during_late_GIMMS_NDVI_p_value_1982-2000',
                              'anomaly_during_early_GIMMS_NDVI_p_value_2001-2015',
                              'anomaly_during_peak_GIMMS_NDVI_p_value_2001-2015',
                              'anomaly_during_late_GIMMS_NDVI_p_value_2001-2015'])
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


def main():
    # Build_dataframe().run()
    Build_partial_correlation_dataframe().run()
    pass


if __name__ == '__main__':
    main()