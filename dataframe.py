# coding='utf-8'
from __init__ import *


class Build_dataframe:


    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff = self.this_class_arr + '_df.df'


    def run(self):
        df = self.__gen_df_init()
        # df = self.foo(df)

        # 2 add landcover to df
        # df = self.add_landcover_to_df(df)
        # df = self.landcover_compose(df)
        # df = self.add_PAR_to_df(df)
        # df=self.add_GPCP_to_df(df)
        # df = self.add_LST_to_df(df)
        # df = self.add_SPEI_to_df(df)
        # df=self.add_during_variable_to_df(df)
        # df=self.add_during_temperature_to_df(df)
        # df=self.add_pre_mean_variables_to_df(df)
        # df=self.add_soil_data_to_df(df)
        # df= self.add_landcover_data_to_df(df)
        # df= self.add_CO2_data_to_df(df)
        # df=self.extraction(df)

        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff, random=False)

        # # self.pick_3_period_df(df)
        # self.__rename_dataframe_columns(df)
        # T.save_df(df, self.dff)
        # self.__df_to_excel(df, self.dff, random=False)

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
    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def foo(self,df):
        # df=pd.DataFrame()
        fdir = '/Volumes/SSD_sumsang/project_greening/Result/detrend/%CSIF_par/early/'
        dic = {}
        outf=self.dff
        result_dic = {}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        pix_list=[]
        change_rate_list=[]
        year=[]
        for pix in tqdm(dic):
            time_series=dic[pix]
            y=0
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y+2002)
                y=y+1
        df['pix']=pix_list
        df['%CSIF_early'] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df

    def add_PAR_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_PAR_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_GPCP_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_GPCP_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            GPCP_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            GPCP_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in GPCP_dic:
                    GPCP_list.append(np.nan)
                    continue
                vals = GPCP_dic[pix]
                if len(vals) !=15:
                    GPCP_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                GPCP_list.append(v1)
            df[f_name] = GPCP_list
        return df

    def add_LST_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_peak_static/20%_Pre_LST_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = 'Original_peak_'+f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year-2003)<0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
        return df

    def add_pre_mean_trend_variables_to_df(self, df):
        fdir = results_root + 'trend/anomaly_trend/trend_during_static/trend_during_late_static/'
        for f in (os.listdir(fdir)):
            # print()
            if not f.endswith('.npy'):
                continue
            val_array = np.load(fdir + f)
            val_dic=DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            # print(f_name)
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

    def add_SPEI_to_df(self, df):
        fdir = results_root + 'extraction/extraction_anomaly/extraction_pre_EOS_static/20%_Pre_SPEI_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            SPEI_dic = T.load_npy(fdir + f)
            f_name = 'late_'+f.split('.')[0]
            SPEI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in SPEI_dic:
                    SPEI_list.append(np.nan)
                    continue
                vals = SPEI_dic[pix]
                if len(vals) != 15:
                    SPEI_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                SPEI_list.append(v1)
            df[f_name] = SPEI_list
        return df

    def add_during_variable_to_df(self, df):  # add 15年序列的变量，during early, peak and late
        fdir = results_root + 'extraction/extraction_anomaly/extraction_during_late_growing_season_static/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name=f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_during_temperature_to_df(self, df):  # add 14年的温度 during early, peak and late
        fdir = results_root + 'extraction/extraction_anomaly/temperature_111/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
        return df

    def add_soil_data_to_df(self, df):  #

        soil_dic={}
        # fdir
        # = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        fdir = data_root + 'SOIL/DIC/dic_SOC/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                soil_dic.update(dic_i)
        f_name ='SOC'
        soil_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
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


    def __3_period_cols(self):

        early_col = ['pix', 'year', '%CSIF_early', 'early_pre_15_PAR', 'early_pre_30_PAR', 'early_pre_60_PAR', 'early_pre_90_PAR',
                     'early_pre_15_GPCP', 'early_pre_30_GPCP', 'early_pre_60_GPCP', 'early_pre_90_GPCP',
                     'early_pre_15_LST', 'early_pre_30_LST', 'early_pre_60_LST', 'early_pre_90_LST',
                     'Original_early_pre_15_PAR_original', 'Original_early_pre_30_PAR_original',
                     'Original_early_pre_60_PAR_original', 'Original_early_pre_90_PAR_original',
                     'Original_early_pre_15_GPCP_original', 'Original_early_pre_30_GPCP_original',
                     'Original_early_pre_60_GPCP_original', 'Original_early_pre_90_GPCP_original',
                     'Original_early_pre_15_LST_original', 'Original_early_pre_30_LST_original',
                     'Original_early_pre_60_LST_original', 'Original_early_pre_90_LST_original',
                     'early_pre_1mo_SPEI3', 'early_pre_2mo_SPEI3', 'early_pre_3mo_SPEI3',
                     'early_pre_3mo_CO2_original', 'early_pre_2mo_CO2_original', 'early_pre_1mo_CO2_original',

                     'during_early_PAR', 'during_early_GPCP', 'during_early_LST', 'during_early_SPEI3', 'during_early_CO2', 'during_early_CSIF_par',

                     'early_mean_pre_15_GPCP_original', 'early_mean_pre_30_GPCP_original', 'early_mean_pre_60_GPCP_original',
                     'early_mean_pre_90_GPCP_original',
                     'early_mean_pre_15_PAR_original', 'early_mean_pre_30_PAR_original', 'early_mean_pre_60_PAR_original',
                     'early_mean_pre_90_PAR_original',
                     'early_mean_pre_15_LST_original', 'early_mean_pre_30_LST_original', 'early_mean_pre_60_LST_original', 'early_mean_pre_90_LST_original',
                     'early_mean_pre_3mo_SPEI3', 'early_mean_pre_2mo_SPEI3', 'early_mean_pre_1mo_SPEI3',
                     'early_mean_pre_3mo_CO2_original', 'early_mean_pre_2mo_CO2_original', 'early_mean_pre_1mo_CO2_original',

                     'mean_during_early_CO2', 'mean_during_early_PAR', 'mean_during_early_GPCP', 'mean_during_early_LST', 'mean_during_early_SPEI3', 'mean_during_early_CSIF_par',

                     'early_trend_pre_15_PAR', 'early_trend_pre_30_PAR', 'early_trend_pre_60_PAR','early_trend_pre_90_PAR',
                     'early_trend_pre_15_GPCP', 'early_trend_pre_30_GPCP', 'early_trend_pre_60_GPCP', 'early_trend_pre_90_GPCP',
                     'early_trend_pre_15_LST', 'early_trend_pre_30_LST', 'early_trend_pre_60_LST','early_trend_pre_90_LST',
                     'early_trend_pre_3mo_SPEI3', 'early_trend_pre_2mo_SPEI3', 'early_trend_pre_1mo_SPEI3',
                     'trend_pre_early_2mo_CO2_original', 'trend_pre_early_1mo_CO2_original', 'trend_pre_early_3mo_CO2_original',

                     'trend_during_early_PAR', 'trend_during_early_GPCP', 'trend_during_early_LST',
                     'trend_during_early_CSIF_par', 'trend_during_early_SPEI3', 'trend_during_early_CO2',

                     'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand', 'landcover','sub_landcover']


        peak_col = ['pix', 'year', '%CSIF_peak',
                                   'peak_pre_15_PAR', 'peak_pre_30_PAR', 'peak_pre_60_PAR', 'peak_pre_90_PAR',
                    'peak_pre_15_GPCP', 'peak_pre_30_GPCP', 'peak_pre_60_GPCP', 'peak_pre_90_GPCP',
                    'peak_pre_15_LST', 'peak_pre_30_LST', 'peak_pre_60_LST', 'peak_pre_90_LST',
                    'Original_peak_pre_15_PAR_original', 'Original_peak_pre_30_PAR_original',
                    'Original_peak_pre_60_PAR_original', 'Original_peak_pre_90_PAR_original',
                    'Original_peak_pre_15_GPCP_original', 'Original_peak_pre_30_GPCP_original',
                    'Original_peak_pre_60_GPCP_original', 'Original_peak_pre_90_GPCP_original',
                    'Original_peak_pre_15_LST_original', 'Original_peak_pre_30_LST_original',
                    'Original_peak_pre_60_LST_original', 'Original_peak_pre_90_LST_original',
                    'peak_pre_1mo_SPEI3', 'peak_pre_2mo_SPEI3', 'peak_pre_3mo_SPEI3',
                    'peak_pre_2mo_CO2_original','peak_pre_3mo_CO2_original', 'peak_pre_1mo_CO2_original',

                    'during_peak_CO2', 'during_peak_PAR', 'during_peak_GPCP', 'during_peak_LST', 'during_peak_SPEI3', 'during_peak_CSIF_par','during_early_CSIF_par',

                    'peak_mean_pre_15_GPCP_original', 'peak_mean_pre_30_GPCP_original',
                    'peak_mean_pre_60_GPCP_original', 'peak_mean_pre_90_GPCP_original',
                    'peak_mean_pre_15_PAR_original', 'peak_mean_pre_30_PAR_original', 'peak_mean_pre_60_PAR_original',
                    'peak_mean_pre_90_PAR_original',
                    'peak_mean_pre_15_LST_original', 'peak_mean_pre_30_LST_original', 'peak_mean_pre_60_LST_original',
                    'peak_mean_pre_90_LST_original',
                    'peak_mean_pre_3mo_SPEI3', 'peak_mean_pre_2mo_SPEI3', 'peak_mean_pre_1mo_SPEI3',
                    'peak_mean_pre_2mo_CO2_original','peak_mean_pre_3mo_CO2_original','peak_mean_pre_1mo_CO2_original',
                    'mean_during_peak_PAR', 'mean_during_peak_GPCP', 'mean_during_peak_LST', 'mean_during_peak_SPEI3',
                    'mean_during_peak_CO2', 'mean_during_peak_CSIF_par','mean_during_early_CSIF_par',

                    'peak_trend_pre_15_PAR', 'peak_trend_pre_30_PAR', 'peak_trend_pre_60_PAR', 'peak_trend_pre_90_PAR',
                    'peak_trend_pre_15_GPCP', 'peak_trend_pre_30_GPCP', 'peak_trend_pre_60_GPCP','peak_trend_pre_90_GPCP',
                    'peak_trend_pre_15_LST', 'peak_trend_pre_30_LST', 'peak_trend_pre_60_LST', 'peak_trend_pre_90_LST',
                    'peak_trend_pre_3mo_SPEI3', 'peak_trend_pre_2mo_SPEI3', 'peak_trend_pre_1mo_SPEI3',
                    'trend_pre_peak_1mo_CO2_original', 'trend_pre_peak_3mo_CO2_original','trend_pre_peak_2mo_CO2_original',
                    'trend_during_peak_PAR', 'trend_during_peak_GPCP', 'trend_during_peak_LST',
                    'trend_during_peak_CSIF_par', 'trend_during_early_CSIF_par','trend_during_peak_SPEI3', 'trend_during_peak_CO2',
                                                  'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand', 'landcover', 'sub_landcover']

        late_col = ['pix', 'year', '%CSIF_late',
                    'late_pre_15_PAR', 'late_pre_30_PAR', 'late_pre_60_PAR', 'late_pre_90_PAR',
                    'late_pre_15_GPCP', 'late_pre_30_GPCP', 'late_pre_60_GPCP', 'late_pre_90_GPCP',
                    'late_pre_15_LST', 'late_pre_30_LST', 'late_pre_60_LST', 'late_pre_90_LST',
                    'Original_late_pre_15_PAR_original', 'Original_late_pre_30_PAR_original',
                    'Original_late_pre_60_PAR_original', 'Original_late_pre_90_PAR_original',
                    'Original_late_pre_15_GPCP_original', 'Original_late_pre_30_GPCP_original',
                    'Original_late_pre_60_GPCP_original', 'Original_late_pre_90_GPCP_original',
                    'Original_late_pre_15_LST_original', 'Original_late_pre_30_LST_original',
                    'Original_late_pre_60_LST_original', 'Original_late_pre_90_LST_original',
                    'late_pre_1mo_SPEI3', 'late_pre_2mo_SPEI3', 'late_pre_3mo_SPEI3',
                    'late_pre_2mo_CO2_original', 'late_pre_3mo_CO2_original', 'late_pre_1mo_CO2_original',

                    'during_late_PAR', 'during_late_GPCP', 'during_late_LST', 'during_late_SPEI3',
                    'during_late_CSIF_par', 'during_late_CO2', 'during_peak_CSIF_par', 'during_early_CSIF_par',

                    'late_mean_pre_15_GPCP_original', 'late_mean_pre_30_GPCP_original',
                    'late_mean_pre_60_GPCP_original', 'late_mean_pre_90_GPCP_original',
                    'late_mean_pre_15_PAR_original', 'late_mean_pre_30_PAR_original', 'late_mean_pre_60_PAR_original',  'late_mean_pre_90_PAR_original',
                    'late_mean_pre_15_LST_original', 'late_mean_pre_30_LST_original', 'late_mean_pre_60_LST_original', 'late_mean_pre_90_LST_original',
                    'late_mean_pre_3mo_SPEI3', 'late_mean_pre_2mo_SPEI3', 'late_mean_pre_1mo_SPEI3',
                    'mean_during_late_PAR', 'mean_during_late_GPCP', 'mean_during_late_LST', 'mean_during_late_SPEI3',
                    'mean_during_late_CSIF_par', 'mean_during_peak_CSIF_par', 'mean_during_peak_CSIF_par',

                    'late_trend_pre_15_PAR', 'late_trend_pre_30_PAR', 'late_trend_pre_60_PAR', 'late_trend_pre_90_PAR',
                    'late_trend_pre_15_GPCP', 'late_trend_pre_30_GPCP', 'late_trend_pre_60_GPCP', 'late_trend_pre_90_GPCP',
                    'late_trend_pre_15_LST', 'late_trend_pre_30_LST', 'late_trend_pre_60_LST', 'late_trend_pre_90_LST',
                    'late_trend_pre_3mo_SPEI3', 'late_trend_pre_2mo_SPEI3', 'late_trend_pre_1mo_SPEI3',
                    'trend_pre_late_1mo_CO2_original', 'trend_pre_late_3mo_CO2_original', 'trend_pre_late_2mo_CO2_original',
                    'trend_during_late_PAR', 'trend_during_late_GPCP', 'trend_during_late_LST',  'trend_during_late_CO2',
                    'trend_during_late_CSIF_par', 'trend_during_peak_CSIF_par', 'trend_during_early_CSIF_par','trend_during_late_SPEI3',

                                                  'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand', 'landcover','sub_landcover']

        # all_cols = '''
        # pix	%CSIF_early	year	early_pre_15_PAR	early_pre_30_PAR	early_pre_60_PAR	early_pre_90_PAR	peak_pre_30_PAR	peak_pre_15_PAR	peak_pre_60_PAR	peak_pre_90_PAR	late_pre_30_PAR	late_pre_60_PAR	late_pre_90_PAR	late_pre_15_PAR	early_pre_60_GPCP	early_pre_90_GPCP	early_pre_30_GPCP	early_pre_15_GPCP	peak_pre_90_GPCP	peak_pre_30_GPCP	peak_pre_60_GPCP	peak_pre_15_GPCP	late_pre_15_GPCP	late_pre_60_GPCP	late_pre_30_GPCP	late_pre_90_GPCP	early_pre_30_LST_mean	early_pre_15_LST_mean	early_pre_60_LST_mean	early_pre_90_LST_mean	peak_pre_90_LST	peak_pre_60_LST	peak_pre_30_LST	peak_pre_15_LST	late_pre_15_LST	late_pre_90_LST	late_pre_30_LST	late_pre_60_LST	Original_early_pre_60_PAR_original	Original_early_pre_30_PAR_original	Original_early_pre_15_PAR_original	Original_early_pre_90_PAR_original	Original_peak_pre_90_PAR_original	Original_peak_pre_30_PAR_original	Original_peak_pre_60_PAR_original	Original_peak_pre_15_PAR_original	Original_late_pre_15_PAR_original	Original_late_pre_30_PAR_original	Original_late_pre_60_PAR_original	Original_late_pre_90_PAR_original	Original_early_pre_15_GPCP_original	Original_early_pre_30_GPCP_original	Original_early_pre_60_GPCP_original	Original_early_pre_90_GPCP_original	Original_peak_pre_15_GPCP_original	Original_peak_pre_60_GPCP_original	Original_peak_pre_30_GPCP_original	Original_peak_pre_90_GPCP_original	Original_late_pre_90_GPCP_original	Original_late_pre_60_GPCP_original	Original_late_pre_30_GPCP_original	Original_late_pre_15_GPCP_original	Original_early_pre_90_LST_original	Original_early_pre_60_LST_original	Original_early_pre_30_LST_original	Original_early_pre_15_LST_original	Original_late_pre_90_LST_original	Original_late_pre_30_LST_original	Original_late_pre_60_LST_original	Original_late_pre_15_LST_original	Original_peak_pre_15_LST_original	Original_peak_pre_30_LST_original	Original_peak_pre_90_LST_original	Original_peak_pre_60_LST_original	early_pre_3mo_SPEI3	early_pre_2mo_SPEI3	early_pre_1mo_SPEI3	peak_pre_1mo_SPEI3	peak_pre_3mo_SPEI3	peak_pre_2mo_SPEI3	late_pre_3mo_SPEI3	late_pre_2mo_SPEI3	late_pre_1mo_SPEI3	during_early_PAR	during_early_CSIF_par	during_early_GPCP	during_early_SPEI3	during_peak_LST	during_early_LST	during_late_LST	during_peak_GPCP	during_peak_CSIF_par	during_peak_PAR	during_peak_SPEI3	during_late_GPCP	during_late_CSIF_par	during_late_PAR	during_late_SPEI3	mean_pre_15_GPCP_original	mean_pre_30_GPCP_original	mean_pre_60_GPCP_original	mean_pre_90_GPCP_original	mean_pre_60_PAR_original	mean_pre_30_PAR_original	mean_pre_15_PAR_original	mean_pre_90_PAR_original	mean_pre_3mo_SPEI3	mean_pre_2mo_SPEI3	mean_pre_1mo_SPEI3	mean_pre_90_LST_original	mean_pre_60_LST_original	mean_pre_30_LST_original	mean_pre_15_LST_original	peakmean_pre_15_LST_original	peakmean_pre_30_LST_original	peakmean_pre_90_LST_original	peakmean_pre_60_LST_original	peak_mean_pre_15_GPCP_original	peak_mean_pre_60_GPCP_original	peak_mean_pre_30_GPCP_original	peak_mean_pre_90_GPCP_original	peak_mean_pre_90_PAR_original	peak_mean_pre_30_PAR_original	peak_mean_pre_60_PAR_original	peak_mean_pre_15_PAR_original	peak_mean_pre_1mo_SPEI3	peak_mean_pre_3mo_SPEI3	peak_mean_pre_2mo_SPEI3	late_mean_pre_3mo_SPEI3	late_mean_pre_2mo_SPEI3	late_mean_pre_1mo_SPEI3	late_mean_pre_15_PAR_original	late_mean_pre_30_PAR_original	late_mean_pre_60_PAR_original	late_mean_pre_90_PAR_original	late_mean_pre_90_LST_original	late_mean_pre_30_LST_original	late_mean_pre_60_LST_original	late_mean_pre_15_LST_original	late_mean_pre_90_GPCP_original	late_mean_pre_60_GPCP_original	late_mean_pre_30_GPCP_original	late_mean_pre_15_GPCP_original	mean_during_early_GPCP	mean_during_early_PAR	mean_during_early_SPEI3	mean_during_early_LST	mean_during_early_CSIF_par	mean_during_peak_GPCP	mean_during_peak_PAR	mean_during_peak_LST	mean_during_peak_SPEI3	mean_during_peak_CSIF_par	mean_during_late_GPCP	mean_during_late_PAR	mean_during_late_LST	mean_during_late_SPEI3	mean_during_late_CSIF_par	earlytrend_pre_60_GPCP	earlytrend_pre_90_GPCP	earlytrend_pre_30_GPCP	earlytrend_pre_15_GPCP	early_trend_pre_30_LST_mean	early_trend_pre_15_LST_mean	early_trend_pre_60_LST_mean	early_trend_pre_90_LST_mean	early_trend_pre_15_PAR	early_trend_pre_30_PAR	early_trend_pre_60_PAR	early_trend_pre_90_PAR	early_trend_pre_3mo_SPEI3	early_trend_pre_2mo_SPEI3	early_trend_pre_1mo_SPEI3	peak_trend_pre_1mo_SPEI3	peak_trend_pre_3mo_SPEI3	peak_trend_pre_2mo_SPEI3	peak_trend_pre_30_PAR	peak_trend_pre_15_PAR	peak_trend_pre_60_PAR	peak_trend_pre_90_PAR	peak_trend_pre_90_LST	peak_trend_pre_60_LST	peak_trend_pre_30_LST	peak_trend_pre_15_LST	peak_trend_pre_90_GPCP	peak_trend_pre_30_GPCP	peak_trend_pre_60_GPCP	peak_trend_pre_15_GPCP	late_trend_pre_15_GPCP	late_trend_pre_60_GPCP	late_trend_pre_30_GPCP	late_trend_pre_90_GPCP	late_trend_pre_15_LST	late_trend_pre_90_LST	late_trend_pre_30_LST	late_trend_pre_60_LST	late_trend_pre_30_PAR	late_trend_pre_60_PAR	late_trend_pre_90_PAR	late_trend_pre_15_PAR	late_trend_pre_3mo_SPEI3	late_trend_pre_2mo_SPEI3	late_trend_pre_1mo_SPEI3	trend_during_early_PAR	trend_during_early_CSIF_par	trend_during_early_GPCP	trend_during_early_SPEI3	trend_during_peak_GPCP	trend_during_peak_CSIF_par	trend_during_peak_PAR	trend_during_peak_SPEI3	trend_during_late_GPCP	trend_during_late_CSIF_par	trend_during_late_PAR	trend_during_late_SPEI3	trend_during_peak_LST	trend_during_early_LST	trend_during_late_LST	BDOD	CEC	Clay	Nitrogen	OCD	PH	Sand	landcover
        # '''
        # all_cols_list = all_cols.split()
        # print(len(all_cols_list))
        # print(all_cols_list)
        # exit()


        return early_col,peak_col,late_col

    def __rename_dataframe_columns(self,df):

        early_new_name_dic={
            'early_pre_15_LST_mean':'early_pre_15_LST',
            'early_pre_30_LST_mean': 'early_pre_30_LST',
            'early_pre_60_LST_mean': 'early_pre_60_LST',
            'early_pre_90_LST_mean': 'early_pre_90_LST',
            'earlytrend_pre_15_GPCP':'early_trend_pre_15_GPCP',
            'earlytrend_pre_30_GPCP': 'early_trend_pre_30_GPCP',
            'earlytrend_pre_60_GPCP': 'early_trend_pre_60_GPCP',
            'earlytrend_pre_90_GPCP': 'early_trend_pre_90_GPCP',
            'early_trend_pre_15_LST_mean':'early_trend_pre_15_LST',
            'early_trend_pre_30_LST_mean': 'early_trend_pre_30_LST',
            'early_trend_pre_60_LST_mean': 'early_trend_pre_60_LST',
            'early_trend_pre_90_LST_mean': 'early_trend_pre_90_LST',
            'mean_pre_15_GPCP_original':'early_mean_pre_15_GPCP_original','mean_pre_30_GPCP_original':'early_mean_pre_30_GPCP_original',
            'mean_pre_60_GPCP_original':'early_mean_pre_60_GPCP_original','mean_pre_90_GPCP_original':'early_mean_pre_90_GPCP_original',
            'mean_pre_15_PAR_original': 'early_mean_pre_15_PAR_original',
            'mean_pre_30_PAR_original': 'early_mean_pre_30_PAR_original',
            'mean_pre_60_PAR_original': 'early_mean_pre_60_PAR_original',
            'mean_pre_90_PAR_original': 'early_mean_pre_90_PAR_original',
            'mean_pre_15_LST_original': 'early_mean_pre_15_LST_original',
            'mean_pre_30_LST_original': 'early_mean_pre_30_LST_original',
            'mean_pre_60_LST_original': 'early_mean_pre_60_LST_original',
            'mean_pre_90_LST_original': 'early_mean_pre_90_LST_original',
    }

        peak_new_name_dic = {
            'peakmean_pre_15_LST_original':'peak_mean_pre_15_LST_original',
            'peakmean_pre_30_LST_original': 'peak_mean_pre_30_LST_original',
            'peakmean_pre_60_LST_original': 'peak_mean_pre_60_LST_original',
            'peakmean_pre_90_LST_original': 'peak_mean_pre_90_LST_original',
        }
        df=pd.DataFrame(df)
        df=df.rename(columns=peak_new_name_dic)
        return df
    def pick_3_period_df(self,df):
        outdir = self.this_class_arr
        # print(df)
        # exit()
        early_col, peak_col, late_col = self.__3_period_cols()
        outf_list = ['early_df', 'peak_df', 'late_df']
        for i in range(len([early_col, peak_col, late_col])):
            # print(len(cols))
            print(i)
            outf = outdir + outf_list[i]
            cols = [early_col, peak_col, late_col][i]
            df_pick = df[cols]
            T.save_df(df_pick,outf)
            self.__df_to_excel(df_pick,outf)
            #
            # # T.print_head_n(df)
            # T.print_head_n(df_pick)
            # exit()
            # print('len(df):',len(df))
        # exit()
        pass


def check_data():
    fdir = results_root + 'extraction/extraction_anomaly/extraction_pre_SOS_static/20%_Pre_SPEI_extraction/'
    dics = []
    for f in os.listdir(fdir):
        dic = Tools().load_npy(fdir+f)
        dics.append(dic)
    for pix in dics[0]:
        vs = []
        for i in range(len(dics)):
            v = dics[i][pix]
            if len(v) == 0:
                vs = []
                break
            vs.append(v)
        if len(vs) == 0:
            continue
        vs = np.array(vs)
        vs = vs.T
        plt.plot(vs)
        plt.show()
        # exit()


def check_dataframe():
    dff = Build_dataframe().dff
    df = T.load_df(dff)
    var_list = '''sub_landcover'''
    var_list = var_list.split()
    # print(var_list)
    # exit()
    for v in var_list:
        print(v)
        spatial_dic = DIC_and_TIF().void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = row[v]
            spatial_dic[pix].append(val)
        mean_arr = DIC_and_TIF().pix_dic_to_spatial_arr_mean(spatial_dic)
        # arr=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(mean_arr, cmap='jet', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()

    pass
def test():
    dff = Build_dataframe().dff
    df = T.load_df(dff)
    print('original:',len(df))

    dff2 = Build_early_dataframe().dff
    df2 = T.load_df(dff2)
    print('early:', len(df2))


class Build_early_dataframe:


    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        # self.dff = self.this_class_arr + 'early_df.df'
        # self.dff = self.this_class_arr + 'extraction_df_via_landuse_early/early_Cropland.df'
        self.dff = self.this_class_arr + 'late_df.df'


    def run(self):
        df = self.__gen_df_init()
        # df = self.foo(df)

        # 2 add landcover to df
        # df = self.add_landcover_to_df(df)
        # df = self.landcover_compose(df)
        # df = self.add_PAR_to_df(df)
        # df = self.add_GPP_changes_to_df(df)
        # df=self.add_GPCP_to_df(df)
        # df = self.add_LST_to_df(df)
        # df = self.add_SPEI_to_df(df)
        # df=self.add_during_variable_to_df(df)
        # df=self.add_during_temperature_to_df(df)
        # df=self.add_pre_mean_trend_variables_to_df(df)
        # df=self.add_soil_data_to_df(df)
        # df= self.add_landcover_data_to_df(df)
        # df=self.add_Koppen_data_to_df(df)
        # df= self.add_CO2_to_df(df)
        # df=self.add_during_CO2_to_df(df)
        # df = self.extraction(df)
        # df=self.drop_field_df(df)
        # df=self.add_pre_early_14_year_to_df(df)
        # df=self.add_detrend_CSIF_change_to_df(df)
        # T.save_df(df, self.dff)
        # self.delete_duplicated_df(df)
        # self.__df_to_excel(df, self.dff, random=True)
        # self.add_during_early_15_year_to_df(df)
        # self.add_during_early_14_year_to_df(df)
        self.add_max_correlation_to_df(df)
        # self.add_p_value_to_df(df)

        # df=self.__rename_dataframe_columns(df)
        # self.extraction_via_landuse_df(df)
        # self.extraction_via_Koppen_df(df)
        # df=self.add_CSIF_par_trend_to_df(df)
        # df=self.add_CSIF_change_rate_trend_to_df(df)
        # self.extraction_drivers_facilitators(df)
        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff, random=False)

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
    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def add_PAR_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_PAR_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def foo(self,df):
        # df=pd.DataFrame()
        fdir = '/Volumes/SSD_sumsang/project_greening/Result/%CSIF/peak/'
        dic = {}
        outf=self.dff
        result_dic = {}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        pix_list=[]
        change_rate_list=[]
        year=[]
        for pix in tqdm(dic):
            time_series=dic[pix]
            y=0
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y+2002)
                y=y+1
        df['pix']=pix_list
        df['%CSIF_peak'] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df
    def add_max_correlation_to_df(self,df):
        period='late'
        f = results_root + 'trend_max_correlation/mask_with_p_tif/mask_p_0.1_{}_max_index.tif'.format(period)
        arr=to_raster.raster2array(f)[0]
        correlation_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        correlation_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in correlation_dic:
                correlation_list.append(np.nan)
                continue
            vals = correlation_dic[pix]

            correlation_list.append(vals)
            # landcover_list.append(vals)
        f_name='signigicant_max_index_among_all_variables_{}'.format(period)
        df[f_name] = correlation_list
        return df

    def add_p_value_to_df(self,df):
        f = results_root + 'max_correlation/early/CO2_max_p_value.tif'
        arr_p_value, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(f)
        p_value_dic = DIC_and_TIF().spatial_arr_to_dic(arr_p_value)

        p_value_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in p_value_dic:
                p_value_list.append(np.nan)
                continue
            vals = p_value_dic[pix]

            p_value_list.append(vals)
            # landcover_list.append(vals)
        df['CO2_max_p_value_early'] = p_value_list
        return df

    def add_pre_early_15_year_to_df(self, df):  # 添加detrend GPCP PAR SPEI
        fdir_all = results_root + 'detrend/extraction_pre_early_static/'
        for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir_1)
            if fdir_1.startswith('Pre_CO2'):
                continue
            if fdir_1.startswith('Pre_LST'):
                continue
            for fdir_2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
                dic = {}
                print(fdir_2)
                for f in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/' + fdir_2 + '/'))):
                    dic_i = dict(np.load(fdir_all + fdir_1 + '/' + fdir_2 + '/' + f, allow_pickle=True, ).item())
                    dic.update(dic_i)

                f_name1 = fdir_all.split('/')[-2].split('_')[-2]
                # print(f_name1)
                f_name = 'detrend_' + f_name1 + '_' + fdir_2
                print(f_name)
                varible_list = []

                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in dic:
                        varible_list.append(np.nan)
                        continue
                    vals = dic[pix]
                    if len(vals) != 15:
                        varible_list.append(np.nan)
                        continue
                    v1 = vals[year - 2002]
                    varible_list.append(v1)
                df[f_name] = varible_list
                T.save_df(df, self.dff)
                self.__df_to_excel(df, self.dff, random=False)

    def add_during_early_15_year_to_df(self, df):  # 添加detrend GPCP PAR SPEI
        fdir_all = results_root + 'detrend/extraction_during_early_growing_season_static/'
        for fdir in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir)
            if fdir.startswith('during_early_CO2'):
                continue
            if fdir.startswith('during_early_LST'):
                continue

            dic = {}
            for f in tqdm(sorted(os.listdir(fdir_all + fdir + '/'))):
                dic_i = dict(np.load(fdir_all + fdir +'/' + f, allow_pickle=True, ).item())
                dic.update(dic_i)

            f_name = 'detrend_' + fdir
            print(f_name)
            varible_list = []

            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in dic:
                    varible_list.append(np.nan)
                    continue
                vals = dic[pix]
                if len(vals) != 15:
                    varible_list.append(np.nan)
                    continue
                v1 = vals[year - 2002]
                varible_list.append(v1)
            df[f_name] = varible_list
            T.save_df(df, self.dff)
            self.__df_to_excel(df, self.dff, random=False)

    def add_during_early_14_year_to_df(self, df):  # 添加detrend LST CO2
        fdir_all = results_root + 'detrend/extraction_during_early_growing_season_static/'
        for fdir in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir)
            if fdir.startswith('during_early_GPCP'):
                continue
            if fdir.startswith('during_early_PAR'):
                continue
            if fdir.startswith('during_early_SPEI3'):
                continue
            if fdir.startswith('during_early_CSIF_par'):
                continue

            dic = {}
            for f in tqdm(sorted(os.listdir(fdir_all + fdir + '/'))):
                dic_i = dict(np.load(fdir_all + fdir +'/' + f, allow_pickle=True, ).item())
                dic.update(dic_i)

            f_name = 'detrend_' + fdir
            print(f_name)
            varible_list = []

            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in dic:
                    varible_list.append(np.nan)
                    continue
                vals = dic[pix]
                if len(vals) != 14:
                    varible_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    varible_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                varible_list.append(v1)
            df[f_name] = varible_list

            T.save_df(df, self.dff)
            self.__df_to_excel(df, self.dff, random=False)


    def add_detrend_CSIF_change_to_df(self, df):  # 添加detrend GPCP PAR SPEI
        fdir = results_root + 'detrend/%CSIF_par/early/'
        dic={}
        for f in tqdm(sorted(os.listdir(fdir))):
            dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
            dic.update(dic_i)
            varible_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in dic:
                varible_list.append(np.nan)
                continue
            vals = dic[pix]
            if len(vals) != 15:
                varible_list.append(np.nan)
                continue
            v1 = vals[year - 2002]
            varible_list.append(v1)
        df['%CSIF_par_early_detrend'] = varible_list
        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff, random=False)

    def add_pre_early_14_year_to_df(self, df):  # 添加detrend LST CO2
        fdir_all = results_root + 'detrend/extraction_pre_early_static/'
        for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir_1)
            if  fdir_1.startswith('Pre_GPCP'):
                continue
            if fdir_1.startswith('Pre_PAR'):
                continue
            if fdir_1.startswith('Pre_SPEI'):
                continue
            for fdir_2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
                dic = {}
                print(fdir_2)
                for f in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/' + fdir_2 + '/'))):
                    dic_i = dict(np.load(fdir_all + fdir_1 + '/' + fdir_2 + '/' + f, allow_pickle=True, ).item())
                    dic.update(dic_i)

                f_name1 = fdir_all.split('/')[-2].split('_')[-2]
                # print(f_name1)
                f_name = 'detrend_' + f_name1 + '_' + fdir_2
                print(f_name)
                varible_list = []

                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in dic:
                        varible_list.append(np.nan)
                        continue
                    vals = dic[pix]
                    if len(vals) != 14:
                        varible_list.append(np.nan)
                        continue
                    if (year - 2003) < 0:
                        varible_list.append(np.nan)
                        continue
                    v1 = vals[year - 2003]
                    varible_list.append(v1)
                df[f_name] = varible_list
                T.save_df(df, self.dff)
                self.__df_to_excel(df, self.dff, random=False)

    def add_CSIF_par_trend_to_df(self, df):
        f = results_root + 'trend/CSIF_par_trend/late_trend_CSIF_par_threshold_20%.npy'
        # trend_dic = T.load_npy(f)
        # trend_dic = dict(np.load(f, allow_pickle=True, ).item())
        trend_array=np.load(f)
        trend_dic = DIC_and_TIF().spatial_arr_to_dic(trend_array)
        trend_list= []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in trend_dic:
                trend_list.append(np.nan)
                continue
            vals = trend_dic[pix]

            trend_list.append(vals)
            # landcover_list.append(vals)
        df['CSIF_par_trend_late'] = trend_list
        return df

    def add_CSIF_change_rate_trend_to_df(self, df):
        f = results_root + 'change_trend/early_change_trend.npy'
        trend_array=np.load(f)
        trend_dic = DIC_and_TIF().spatial_arr_to_dic(trend_array)
        trend_list= []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in trend_dic:
                trend_list.append(np.nan)
                continue
            vals = trend_dic[pix]

            trend_list.append(vals)
            # landcover_list.append(vals)
        df['%CSIF_par_trend_early'] = trend_list
        return df

    def add_GPP_changes_to_df(self, df):
        fdir = results_root + '%CSIF/early/'
        CSIF_dic={}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                CSIF_dic.update(dic_i)
        f_name = '%GPP_early'
        CSIF_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CSIF_dic:
                CSIF_list.append(np.nan)
                continue
            vals = CSIF_dic[pix]
            if len(vals) != 15:
                CSIF_list.append(np.nan)
                continue
            v1 = vals[year - 2002]
            CSIF_list.append(v1)
        df[f_name] = CSIF_list
        return df

    def add_PAR_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_PAR_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_GPCP_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_GPCP_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            GPCP_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            GPCP_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in GPCP_dic:
                    GPCP_list.append(np.nan)
                    continue
                vals = GPCP_dic[pix]
                if len(vals) !=15:
                    GPCP_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                GPCP_list.append(v1)
            df[f_name] = GPCP_list
        return df

    def add_LST_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_peak_static/20%_Pre_LST_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = 'Original_peak_'+f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year-2003)<0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
        return df

    def add_pre_mean_trend_variables_to_df(self, df):
        fdir = results_root + 'trend/original_trend/trend_CO2_during_static/'
        for f in (os.listdir(fdir)):
            # print()
            if not f.endswith('.npy'):
                continue
            val_array = np.load(fdir + f)
            val_dic=DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            # print(f_name)
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

    def add_SPEI_to_df(self, df):
        fdir = results_root + 'extraction/extraction_anomaly/extraction_pre_EOS_static/20%_Pre_SPEI_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            SPEI_dic = T.load_npy(fdir + f)
            f_name = 'late_'+f.split('.')[0]
            SPEI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in SPEI_dic:
                    SPEI_list.append(np.nan)
                    continue
                vals = SPEI_dic[pix]
                if len(vals) != 15:
                    SPEI_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                SPEI_list.append(v1)
            df[f_name] = SPEI_list
        return df

    def add_CO2_to_df(self, df):
        fdir = results_root + 'extraction/extraction_anomaly/extraction_pre_late_static/20%_Pre_CO2_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            CO2_dic = T.load_npy(fdir + f)
            f_name = 'late_'+f.split('.')[0]
            CO2_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in CO2_dic:
                    CO2_list.append(np.nan)
                    continue
                vals = CO2_dic[pix]
                if len(vals) != 14:
                    CO2_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    CO2_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                CO2_list.append(v1)
            df[f_name] = CO2_list
        return df

    def add_during_variable_to_df(self, df):  # add 15年序列的变量，during early, peak and late
        fdir = results_root + 'extraction/extraction_anomaly/extraction_during_late_growing_season_static/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name=f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_during_temperature_to_df(self, df):  # add 14年的温度 during early, peak and late
        fdir = results_root + 'extraction/extraction_original/temperature/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
        return df

    def add_during_CO2_to_df(self, df):  # add 14年的CO2 during early, peak and late
        f = results_root + 'extraction/extraction_anomaly/extraction_during_late_growing_season_static/during_late_CO2.npy'
        CO2_dic = T.load_npy(f)
        f_name = f.split('.')[0]
        f_name = f_name.split('/')[-1]+'_anomaly'
        print(f_name)
        CO2_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CO2_dic:
                CO2_list.append(np.nan)
                continue
            vals = CO2_dic[pix]
            if len(vals) != 14:
                CO2_list.append(np.nan)
                continue
            if (year - 2003) < 0:
                CO2_list.append(np.nan)
                continue
            v1 = vals[year - 2003]
            CO2_list.append(v1)
        df[f_name] = CO2_list
        return df

    def add_soil_data_to_df(self, df):  #

        soil_dic={}
        fdir = data_root + 'SOIL/DIC/dic_SOC/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                soil_dic.update(dic_i)
        f_name ='SOC'
        soil_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
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

    def __rename_dataframe_columns(self,df):

        early_new_name_dic={
            # 'early_pre_15_LST_mean':'early_pre_15_LST',
            # 'early_pre_30_LST_mean': 'early_pre_30_LST',
            # 'early_pre_60_LST_mean': 'early_pre_60_LST',
            # 'early_pre_90_LST_mean': 'early_pre_90_LST',
            # 'earlytrend_pre_15_GPCP':'early_trend_pre_15_GPCP',
            # 'earlytrend_pre_30_GPCP': 'early_trend_pre_30_GPCP',
            # 'earlytrend_pre_60_GPCP': 'early_trend_pre_60_GPCP',
            # 'earlytrend_pre_90_GPCP': 'early_trend_pre_90_GPCP',
            # 'early_trend_pre_15_LST_mean':'early_trend_pre_15_LST',
            # 'early_trend_pre_30_LST_mean': 'early_trend_pre_30_LST',
            # 'early_trend_pre_60_LST_mean': 'early_trend_pre_60_LST',
            # 'early_trend_pre_90_LST_mean': 'early_trend_pre_90_LST',
            # 'mean_pre_15_GPCP_original':'early_mean_pre_15_GPCP_original','mean_pre_30_GPCP_original':'early_mean_pre_30_GPCP_original',
            # 'mean_pre_60_GPCP_original':'early_mean_pre_60_GPCP_original','mean_pre_90_GPCP_original':'early_mean_pre_90_GPCP_original',
            # 'mean_pre_15_PAR_original': 'early_mean_pre_15_PAR_original',
            # 'mean_pre_30_PAR_original': 'early_mean_pre_30_PAR_original',
            # 'mean_pre_60_PAR_original': 'early_mean_pre_60_PAR_original',
            # 'mean_pre_90_PAR_original': 'early_mean_pre_90_PAR_original',
            # 'mean_pre_15_LST_original': 'early_mean_pre_15_LST_original',
            # 'mean_pre_30_LST_original': 'early_mean_pre_30_LST_original',
            # 'mean_pre_60_LST_original': 'early_mean_pre_60_LST_original',
            # 'mean_pre_90_LST_original': 'early_mean_pre_90_LST_original',
            # 'mean_pre_3mo_SPEI3':'early_mean_pre_3mo_SPEI3',
            # 'mean_pre_2mo_SPEI3':'early_mean_pre_2mo_SPEI3',
            # 'early_mean_pre_1mo_SPEI3':'early_mean_pre_2mo_SPEI3',
            # 'mean_pre_1mo_SPEI3': 'early_mean_pre_1mo_SPEI3',
            # 'detrend_early_pre_30_LST_mean':'detrend_early_pre_30_LST',
            # 'detrend_early_pre_15_LST_mean':'detrend_early_pre_15_LST',
            # 'detrend_early_pre_60_LST_mean': 'detrend_early_pre_60_LST',
            # 'detrend_early_pre_90_LST_mean': 'detrend_early_pre_90_LST',
            'max_index_among_all_variables':'max_index_among_all_variables_early'
    }

        df=pd.DataFrame(df)
        df=df.rename(columns=early_new_name_dic)
        return df

    def extraction(self,df):

        r_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            r, c = pix
            r_list.append(r)
        df['r_list'] = r_list
        df = df[df['r_list'] < 120]

        print(len(df))
        return df

    def __drivers_facilitators(self):

        drivers = ['pix', 'year', '%CSIF_par_early','%GPP_early',
                     'early_pre_15_PAR', 'early_pre_30_PAR', 'early_pre_60_PAR', 'early_pre_90_PAR',
                     'early_pre_15_GPCP', 'early_pre_30_GPCP', 'early_pre_60_GPCP', 'early_pre_90_GPCP',
                     'early_pre_15_LST', 'early_pre_30_LST', 'early_pre_60_LST', 'early_pre_90_LST',
                     'early_pre_1mo_SPEI3', 'early_pre_2mo_SPEI3', 'early_pre_3mo_SPEI3',
                     'early_pre_3mo_CO2_original',	'early_pre_2mo_CO2_original', 'early_pre_1mo_CO2_original',
                     'during_early_PAR', 'during_early_GPCP', 'during_early_LST', 'during_early_SPEI3','during_early_CO2',

                     'early_trend_pre_15_PAR', 'early_trend_pre_30_PAR', 'early_trend_pre_60_PAR','early_trend_pre_90_PAR',
                     'early_trend_pre_15_GPCP', 'early_trend_pre_30_GPCP', 'early_trend_pre_60_GPCP', 'early_trend_pre_90_GPCP',
                     'early_trend_pre_15_LST', 'early_trend_pre_30_LST', 'early_trend_pre_60_LST','early_trend_pre_90_LST',
                     'early_trend_pre_3mo_SPEI3', 'early_trend_pre_2mo_SPEI3', 'early_trend_pre_1mo_SPEI3',
                     'trend_pre_early_2mo_CO2_original', 'trend_pre_early_1mo_CO2_original','trend_pre_early_3mo_CO2_original',

                     'trend_during_early_PAR', 'trend_during_early_GPCP', 'trend_during_early_LST',
                     'trend_during_early_SPEI3','trend_during_early_CO2']

        facilitors=['pix', 'year', '%CSIF_par_early', '%GPP_early',
                    'early_mean_pre_15_GPCP_original', 'early_mean_pre_30_GPCP_original', 'early_mean_pre_60_GPCP_original',
                     'early_mean_pre_90_GPCP_original',
                     'early_mean_pre_15_PAR_original', 'early_mean_pre_30_PAR_original', 'early_mean_pre_60_PAR_original',
                     'early_mean_pre_90_PAR_original',
                     'early_mean_pre_15_LST_original', 'early_mean_pre_30_LST_original', 'early_mean_pre_60_LST_original', 'early_mean_pre_90_LST_original',
                     'early_mean_pre_3mo_SPEI3', 'early_mean_pre_2mo_SPEI3', 'early_mean_pre_1mo_SPEI3',
                     'early_pre_3mo_CO2_original',	'early_pre_2mo_CO2_original',	'early_pre_1mo_CO2_original',

                     'mean_during_early_PAR', 'mean_during_early_GPCP', 'mean_during_early_LST', 'mean_during_early_SPEI3', 'mean_during_early_CO2',
                     'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand', 'landcover','sub_landcover','koppen']
        print('early drivers:',len(drivers))
        print('early facilitors:', len(facilitors))
        # exit()
        return drivers,facilitors

    def extraction_drivers_facilitators(self, df):
        outdir = self.this_class_arr + 'extraction_df_drivers_facilitators_early_landuse/'
        print(self.dff)
        fname=self.dff.split('.')[0]
        fname=fname.split('/')[-1]
        T.mk_dir(outdir)
        drivers, facilitators = self.__drivers_facilitators()
        outf_list = ['drivers', 'facilitators']
        # print(df)
        # exit()
        for i in range(len([drivers,facilitators])):
            # print(len(cols))
            print(i)
            outf = outdir + fname +'_'+outf_list[i]
            cols = [drivers, facilitators][i]
            df_pick = df[cols]
            T.save_df(df_pick, outf)
            self.__df_to_excel(df_pick, outf,random=False)
            #


    def extraction_via_landuse_df(self,df):

        outdir = self.this_class_arr+'extraction_df_via_landuse_early/'
        outf=outdir+'early_Grassland'
        T.mk_dir(outdir)
        # print(df)
        # exit()
        df = df[df['r_list'] < 120]
        df=df[df['landcover']=='Grassland']
        print(len(df))
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=True)

    def extraction_via_Koppen_df(self, df):

        outdir = self.this_class_arr + 'extraction_df_via_Koppen_early/'
        outf = outdir + 'early_Dsw'
        T.mk_dir(outdir)
        # print(df)
        # exit()
        df = df[df['koppen'] == 'Dsw']
        print(len(df))
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=True)

    def delete_duplicated_df(self,df):
        outdir = self.this_class_arr + 'dataframe_for_trend/'
        outf = outdir + 'peak'
        T.mk_dir(outdir)
        df = df.drop_duplicates(subset=['pix', '%CSIF_par_trend_peak'],keep='first')
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=False)

    def drop_field_df(self,df):
        # df = df.drop(columns=['during_early_CO2','during_peak_CO2',])
        return df


class Build_peak_dataframe:


    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff = self.this_class_arr + 'peak_df.df'
        # self.dff = self.this_class_arr + 'extraction_df_via_landuse_peak/peak_Savanna.df'


    def run(self):
        df = self.__gen_df_init()
        # df = self.foo(df)

        # 2 add landcover to df
        # df = self.add_landcover_to_df(df)
        # df = self.landcover_compose(df)
        # df=self.add_GPP_changes_to_df(df)
        # df = self.add_PAR_to_df(df)
        # df=self.add_GPCP_to_df(df)
        # df = self.add_LST_to_df(df)
        # df = self.add_SPEI_to_df(df)
        # df=self.add_during_variable_to_df(df)
        # df=self.add_during_temperature_to_df(df)
        # df=self.add_pre_mean_trend_variables_to_df(df)
        # df=self.add_soil_data_to_df(df)
        # df= self.add_landcover_data_to_df(df)
        # self.extraction_via_Koppen_df(df)
        # df=self.add_CSIF_change_rate_to_df(df)
        df=self.add_CO2_to_df(df)
        # df=self.add_during_CO2_to_df(df)
        # df=self.add_Koppen_data_to_df(df)
        # df=self.__rename_dataframe_columns(df)
        # df=self.add_during_CSIF(df)
        # self.add_during_peak_15_year_to_df(df)
        # self.add_during_peak_14_year_to_df(df)
        # df=self.add_pre_peak_14_year_to_df(df)
        # df = self.add_pre_peak_15_year_to_df(df)
        # df=self.add_detrend_CSIF_change_to_df(df)
        # df=self.add_detrend_CSIF_change_to_df(df)
        # df=self.extraction(df)
        # self.extraction_via_landuse_df(df)
        # self.extraction_drivers_facilitators(df)
        # df=self.add_CSIF_par_trend_to_df(df)
        # df=self.add_CSIF_change_rate_trend_to_df(df)
        # df=self.add_contribution_data_to_df(df)
        # df=self.drop_field_df(df)
        # self.delete_duplicated_df(df)

        #
        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff, random=False)




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
    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)

        return df,dff
        # return df_early,dff

    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df, self.dff)
            return df
        else:
            df, dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def foo(self,df):
        # df=pd.DataFrame()
        fdir = '/Volumes/SSD_sumsang/project_greening/Result/%CSIF/peak/'
        dic = {}
        outf=self.dff
        result_dic = {}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        pix_list=[]
        change_rate_list=[]
        year=[]
        for pix in tqdm(dic):
            time_series=dic[pix]
            y=0
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y+2002)
                y=y+1
        df['pix']=pix_list
        df['%CSIF_peak'] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df

    def add_pre_peak_15_year_to_df(self, df):  # 添加detrend GPCP PAR SPEI
        fdir_all = results_root + 'detrend/extraction_pre_late_static/'
        for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir_1)
            if fdir_1.startswith('Pre_CO2'):
                continue
            if fdir_1.startswith('Pre_LST'):
                continue
            for fdir_2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
                dic = {}
                print(fdir_2)
                for f in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/' + fdir_2 + '/'))):
                    dic_i = dict(np.load(fdir_all + fdir_1 + '/' + fdir_2 + '/' + f, allow_pickle=True, ).item())
                    dic.update(dic_i)

                f_name1 = fdir_all.split('/')[-2].split('_')[-2]
                # print(f_name1)
                f_name = 'detrend_' + f_name1 + '_' + fdir_2
                print(f_name)
                varible_list = []

                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in dic:
                        varible_list.append(np.nan)
                        continue
                    vals = dic[pix]
                    if len(vals) != 15:
                        varible_list.append(np.nan)
                        continue
                    v1 = vals[year - 2002]
                    varible_list.append(v1)
                df[f_name] = varible_list
                T.save_df(df, self.dff)
                self.__df_to_excel(df, self.dff, random=False)

    def add_detrend_CSIF_change_to_df(self, df):  #
        fdir = results_root + 'detrend/%CSIF_par/late/'
        dic={}
        for f in tqdm(sorted(os.listdir(fdir))):
            dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
            dic.update(dic_i)
            varible_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in dic:
                varible_list.append(np.nan)
                continue
            vals = dic[pix]
            if len(vals) != 15:
                varible_list.append(np.nan)
                continue
            v1 = vals[year - 2002]
            varible_list.append(v1)
        df['%CSIF_par_late_detrend'] = varible_list
        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff, random=False)

    def add_pre_peak_14_year_to_df(self, df):  # 添加detrend LST CO2
        fdir_all = results_root + 'detrend/extraction_pre_late_static/'
        for fdir_1 in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir_1)
            if  fdir_1.startswith('Pre_GPCP'):
                continue
            if fdir_1.startswith('Pre_PAR'):
                continue
            if fdir_1.startswith('Pre_SPEI'):
                continue
            for fdir_2 in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/'))):
                dic = {}
                print(fdir_2)
                for f in tqdm(sorted(os.listdir(fdir_all + fdir_1 + '/' + fdir_2 + '/'))):
                    dic_i = dict(np.load(fdir_all + fdir_1 + '/' + fdir_2 + '/' + f, allow_pickle=True, ).item())
                    dic.update(dic_i)

                f_name1 = fdir_all.split('/')[-2].split('_')[-2]
                print(f_name1)
                f_name = 'detrend_' + f_name1 + '_' + fdir_2
                print(f_name)
                varible_list = []

                for i, row in tqdm(df.iterrows(), total=len(df)):
                    year = row['year']
                    # pix = row.pix
                    pix = row['pix']
                    if not pix in dic:
                        varible_list.append(np.nan)
                        continue
                    vals = dic[pix]
                    if len(vals) != 14:
                        varible_list.append(np.nan)
                        continue
                    if (year - 2003) < 0:
                        varible_list.append(np.nan)
                        continue
                    v1 = vals[year - 2003]
                    varible_list.append(v1)
                df[f_name] = varible_list
                T.save_df(df, self.dff)
                self.__df_to_excel(df, self.dff, random=False)

    def add_during_peak_15_year_to_df(self, df):  # 添加detrend GPCP PAR SPEI
        fdir_all = results_root + 'detrend/extraction_during_late_growing_season_static/'
        for fdir in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir)
            if fdir.startswith('during_late_CO2'):
                continue
            if fdir.startswith('during_late_LST'):
                continue

            dic = {}
            for f in tqdm(sorted(os.listdir(fdir_all + fdir + '/'))):
                dic_i = dict(np.load(fdir_all + fdir +'/' + f, allow_pickle=True, ).item())
                dic.update(dic_i)

            f_name = 'detrend_' + fdir
            print(f_name)
            varible_list = []

            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in dic:
                    varible_list.append(np.nan)
                    continue
                vals = dic[pix]
                if len(vals) != 15:
                    varible_list.append(np.nan)
                    continue
                v1 = vals[year - 2002]
                varible_list.append(v1)
            df[f_name] = varible_list
            T.save_df(df, self.dff)
            self.__df_to_excel(df, self.dff, random=False)

    def add_during_peak_14_year_to_df(self, df):  # 添加detrend LST CO2
        fdir_all = results_root + 'detrend/extraction_during_late_growing_season_static/'
        for fdir in tqdm(sorted(os.listdir(fdir_all))):
            print(fdir)
            if fdir.startswith('during_late_GPCP'):
                continue
            if fdir.startswith('during_late_PAR'):
                continue
            if fdir.startswith('during_late_SPEI3'):
                continue
            if fdir.startswith('during_early_CSIF_par'):
                continue
            if fdir.startswith('during_peak_CSIF_par'):
                continue
            if fdir.startswith('during_late_CSIF_par'):
                continue

            dic = {}
            for f in tqdm(sorted(os.listdir(fdir_all + fdir + '/'))):
                dic_i = dict(np.load(fdir_all + fdir +'/' + f, allow_pickle=True, ).item())
                dic.update(dic_i)

            f_name = 'detrend_' + fdir
            print(f_name)
            varible_list = []

            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in dic:
                    varible_list.append(np.nan)
                    continue
                vals = dic[pix]
                if len(vals) != 14:
                    varible_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    varible_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                varible_list.append(v1)
            df[f_name] = varible_list

            T.save_df(df, self.dff)
            self.__df_to_excel(df, self.dff, random=False)

    def add_CSIF_par_trend_to_df(self, df):
        f = results_root + 'trend/CSIF_par_trend_new/during_peak_CSIF_par_p_value.npy'
        # trend_dic = T.load_npy(f)
        # trend_dic = dict(np.load(f, allow_pickle=True, ).item())
        trend_array=np.load(f)
        trend_dic = DIC_and_TIF().spatial_arr_to_dic(trend_array)
        trend_list= []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in trend_dic:
                trend_list.append(np.nan)
                continue
            vals = trend_dic[pix]

            trend_list.append(vals)
            # landcover_list.append(vals)
        df['CSIF_par_p_value_peak'] = trend_list
        return df

    def add_CSIF_change_rate_trend_to_df(self, df):
        f = results_root + 'change_trend/early_change_p_value.npy'
        trend_array = np.load(f)
        trend_dic = DIC_and_TIF().spatial_arr_to_dic(trend_array)
        trend_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in trend_dic:
                trend_list.append(np.nan)
                continue
            vals = trend_dic[pix]

            trend_list.append(vals)
            # landcover_list.append(vals)
        df['%CSIF_par_p_value_peak'] = trend_list
        return df

    def add_GPP_changes_to_df(self, df):
        fdir = results_root + '%CSIF/peak/'
        CSIF_dic = {}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                CSIF_dic.update(dic_i)
        f_name = '%GPP_peak'
        CSIF_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CSIF_dic:
                CSIF_list.append(np.nan)
                continue
            vals = CSIF_dic[pix]
            if len(vals) != 15:
                CSIF_list.append(np.nan)
                continue
            v1 = vals[year - 2002]
            CSIF_list.append(v1)
        df[f_name] = CSIF_list
        return df

    def add_during_CSIF(self, df):
        f = results_root + 'mean/mean_during_early_static/mean_during_early_CSIF_par.npy'
        # CSIF_dic = T.load_npy(f)
        CSIF_array = np.load(f)
        CSIF_dic = DIC_and_TIF().spatial_arr_to_dic(CSIF_array)
        f_name=f.split('.')[0]
        f_name = f_name.split('/')[-1]
        CSIF_list= []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CSIF_dic:
                CSIF_list.append(np.nan)
                continue
            vals = CSIF_dic[pix]
            if vals < -99:
                CSIF_list.append(np.nan)
                continue
            CSIF_list.append(vals)
        df[f_name] = CSIF_list
        return df

    def add_PAR_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_PAR_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_GPCP_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_GPCP_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            GPCP_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            GPCP_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in GPCP_dic:
                    GPCP_list.append(np.nan)
                    continue
                vals = GPCP_dic[pix]
                if len(vals) !=15:
                    GPCP_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                GPCP_list.append(v1)
            df[f_name] = GPCP_list
        return df

    def add_LST_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_peak_static/20%_Pre_LST_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = 'Original_peak_'+f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year-2003)<0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
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

    def add_pre_mean_trend_variables_to_df(self, df):
        fdir = results_root + 'trend/original_trend/trend_CO2_during_static/'
        for f in (os.listdir(fdir)):
            # print()
            if not f.endswith('.npy'):
                continue
            val_array = np.load(fdir + f)
            val_dic=DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            # print(f_name)
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

    def add_SPEI_to_df(self, df):
        fdir = results_root + 'extraction/extraction_anomaly/extraction_pre_EOS_static/20%_Pre_SPEI_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            SPEI_dic = T.load_npy(fdir + f)
            f_name = 'late_'+f.split('.')[0]
            SPEI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in SPEI_dic:
                    SPEI_list.append(np.nan)
                    continue
                vals = SPEI_dic[pix]
                if len(vals) != 15:
                    SPEI_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                SPEI_list.append(v1)
            df[f_name] = SPEI_list
        return df

    def add_CO2_to_df(self, df):
        fdir = results_root + 'extraction/extraction_anomaly/extraction_pre_peak_static/20%_Pre_CO2_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            CO2_dic = T.load_npy(fdir + f)
            f_name = 'peak_'+f.split('.')[0]
            CO2_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in CO2_dic:
                    CO2_list.append(np.nan)
                    continue
                vals = CO2_dic[pix]
                if len(vals) != 14:
                    CO2_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    CO2_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                CO2_list.append(v1)
            df[f_name] = CO2_list
        return df

    def add_during_CO2_to_df(self, df):  # add 14年的CO2 during early, peak and late
        print(len(df.columns))
        f = results_root + 'extraction/extraction_original_val/extraction_during_peak_growing_season_static/during_peak_CO2.npy'
        CO2_dic = T.load_npy(f)
        f_name = f.split('.')[0]
        f_name = f_name.split('/')[-1]
        CO2_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CO2_dic:
                CO2_list.append(np.nan)
                continue
            vals = CO2_dic[pix]
            if len(vals) != 14:
                CO2_list.append(np.nan)
                continue
            if (year - 2003) < 0:
                CO2_list.append(np.nan)
                continue
            v1 = vals[year - 2003]
            CO2_list.append(v1)
        df[f_name] = CO2_list
        return df

    def add_during_variable_to_df(self, df):  # add 15年序列的变量，during early, peak and late
        fdir = results_root + 'extraction/extraction_anomaly/extraction_during_early_growing_season_static/during_early_CSIF_par.npy'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name=f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_during_temperature_to_df(self, df):  # add 14年的温度 during early, peak and late
        fdir = results_root + 'extraction/extraction_anomaly/temperature_111/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
        return df

    def add_soil_data_to_df(self, df):  #

        soil_dic={}
        # fdir
        # = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        fdir = data_root + 'SOIL/DIC/dic_SOC/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                soil_dic.update(dic_i)
        f_name ='SOC'
        soil_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
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

    def add_contribution_data_to_df(self, df):  #

        f = results_root + 'contribution/Max_contribution_index_threshold_20%.npy'
        # trend_dic = T.load_npy(f)
        # trend_dic = dict(np.load(f, allow_pickle=True, ).item())
        contribution_array = np.load(f)
        contribution_dic = DIC_and_TIF().spatial_arr_to_dic(contribution_array)
        contribution_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in contribution_dic:
                contribution_list.append(np.nan)
                continue
            vals = contribution_dic[pix]

            contribution_list.append(vals)
            # landcover_list.append(vals)
        df['contribution_index'] = contribution_list
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

    def __rename_dataframe_columns(self,df):

        peak_new_name_dic = {
            # 'peakmean_pre_15_LST_original':'peak_mean_pre_15_LST_original',
            # 'peakmean_pre_30_LST_original': 'peak_mean_pre_30_LST_original',
            # 'peakmean_pre_60_LST_original': 'peak_mean_pre_60_LST_original',
            # 'peakmean_pre_90_LST_original': 'peak_mean_pre_90_LST_original',
            # '%CSIF_par_p_value_peak': '%CSIF_par_p_value_late',
            # '%CSIF_peak': '%CSIF_par_peak'
            'detrend_late_pre_30_LST_mean': 'detrend_late_pre_30_LST',
            'detrend_late_pre_15_LST_mean': 'detrend_late_pre_15_LST',
            'detrend_late_pre_60_LST_mean': 'detrend_late_pre_60_LST',
            'detrend_late_pre_90_LST_mean': 'detrend_late_pre_90_LST',
        }
        df=pd.DataFrame(df)
        df=df.rename(columns=peak_new_name_dic)
        return df

    def extraction(self, df):

        r_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            r, c = pix
            r_list.append(r)
        df['r_list'] = r_list
        df = df[df['r_list'] < 180]

        print(len(df))
        return df

    def extraction_via_landuse_df(self, df):

        outdir = self.this_class_arr + 'extraction_df_via_landuse_late/'
        outf = outdir + 'late_BF'
        T.mk_dir(outdir)
        # print(df)
        # exit()
        df = df[df['r_list'] < 120]
        df = df[df['landcover'] == 'BF']
        print(len(df))
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=True)

    def extraction_via_Koppen_df(self, df):

        outdir = self.this_class_arr + 'extraction_df_via_Koppen_peak/'
        outf = outdir + 'peak_A'
        T.mk_dir(outdir)
        # print(df)
        # exit()
        df = df[df['koppen'] == 'A']
        print(len(df))
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=True)

    def __drivers_facilitators(self):  # 这一步相当于已经做了第一步的drivers 清醒

        drivers = ['pix', 'year', '%CSIF_par_peak', '%GPP_peak',
                    'peak_pre_15_PAR', 'peak_pre_30_PAR', 'peak_pre_60_PAR', 'peak_pre_90_PAR',
                    'peak_pre_15_GPCP', 'peak_pre_30_GPCP', 'peak_pre_60_GPCP', 'peak_pre_90_GPCP',
                    'peak_pre_15_LST', 'peak_pre_30_LST', 'peak_pre_60_LST', 'peak_pre_90_LST',
                    'peak_pre_1mo_SPEI3', 'peak_pre_2mo_SPEI3', 'peak_pre_3mo_SPEI3',
                    'peak_pre_2mo_CO2_original','peak_pre_3mo_CO2_original', 'peak_pre_1mo_CO2_original',
                    'during_peak_PAR', 'during_peak_GPCP', 'during_peak_LST', 'during_peak_SPEI3', 'during_peak_CO2',

                    'peak_trend_pre_15_PAR', 'peak_trend_pre_30_PAR', 'peak_trend_pre_60_PAR', 'peak_trend_pre_90_PAR',
                    'peak_trend_pre_15_GPCP', 'peak_trend_pre_30_GPCP', 'peak_trend_pre_60_GPCP', 'peak_trend_pre_90_GPCP',
                    'peak_trend_pre_15_LST', 'peak_trend_pre_30_LST', 'peak_trend_pre_60_LST', 'peak_trend_pre_90_LST',
                    'peak_trend_pre_3mo_SPEI3', 'peak_trend_pre_2mo_SPEI3', 'peak_trend_pre_1mo_SPEI3',
                    'trend_pre_peak_1mo_CO2_original', 'trend_pre_peak_3mo_CO2_original', 'trend_pre_peak_2mo_CO2_original',

                    'trend_during_peak_PAR', 'trend_during_peak_GPCP', 'trend_during_peak_LST', 'trend_during_peak_CO2',
                    'trend_during_peak_SPEI3','trend_during_early_CSIF_par']

        facilitators=['pix', 'year', '%CSIF_par_peak', '%GPP_peak',
                    'peak_mean_pre_15_GPCP_original', 'peak_mean_pre_30_GPCP_original',
                    'peak_mean_pre_60_GPCP_original', 'peak_mean_pre_90_GPCP_original',
                    'peak_mean_pre_15_PAR_original', 'peak_mean_pre_30_PAR_original', 'peak_mean_pre_60_PAR_original',
                    'peak_mean_pre_90_PAR_original',
                    'peak_mean_pre_15_LST_original', 'peak_mean_pre_30_LST_original', 'peak_mean_pre_60_LST_original',
                    'peak_mean_pre_90_LST_original',
                    'peak_mean_pre_3mo_SPEI3', 'peak_mean_pre_2mo_SPEI3', 'peak_mean_pre_1mo_SPEI3',
                    'peak_mean_pre_2mo_CO2_original', 'peak_mean_pre_3mo_CO2_original',	'peak_mean_pre_1mo_CO2_original',
                    'mean_during_peak_PAR', 'mean_during_peak_GPCP', 'mean_during_peak_LST', 'mean_during_peak_SPEI3','mean_during_peak_CO2',
                    'mean_during_early_CSIF_par',
                    'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand', 'landcover','sub_landcover','koppen']

        print('peak drivers:', len(drivers))
        print('peak facilitors:', len(facilitators))
        # exit()
        return drivers,facilitators

    def extraction_drivers_facilitators(self, df):
        outdir = self.this_class_arr + 'extraction_df_drivers_facilitators_peak_landuse/'
        print(self.dff)
        fname=self.dff.split('.')[0]
        fname=fname.split('/')[-1]
        T.mk_dir(outdir)
        drivers, facilitators = self.__drivers_facilitators()
        outf_list = ['drivers', 'facilitators']
        # print(type(df))
        for i in range(len([drivers,facilitators])):
            # print(len(cols))
            print(i)
            outf = outdir + fname +'_'+outf_list[i]
            cols = [drivers, facilitators][i]
            df_pick = df[cols]
            T.save_df(df_pick, outf)
            self.__df_to_excel(df_pick, outf,random=False)
            #

    def delete_duplicated_df(self, df):
        outdir = self.this_class_arr + 'dataframe_for_trend/'
        outf = outdir + 'peak'
        T.mk_dir(outdir)
        df = df.drop_duplicates(subset=['pix', '%CSIF_par_trend_peak'], keep='first')
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=False)

    def drop_field_df(self,df):
        df = df.drop(columns=['peak_pre_3mo_CO2_original','peak_pre_2mo_CO2_original','peak_pre_1mo_CO2_original'])
        return df

class Build_late_dataframe:


    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame/'
        Tools().mk_dir(self.this_class_arr, force=True)

        self.dff = self.this_class_arr
        # self.dff = self.this_class_arr+'late_df.df'
        # self.dff=self.this_class_arr+'extraction_df_via_landuse_late/'
        # self.dff = self.this_class_arr+'extraction_df_drivers_facilitators_peak_landuse/'


    def run(self):
        for f in tqdm(os.listdir(self.dff)):
            print(f)
            if not f.endswith('late_df.df'):
                continue
            if f.startswith('.'):
                continue
            file=self.dff+f
            df = self.__gen_df_init(file)
            # df = self.add_pre_mean_trend_variables_to_df(df)
            # df=self.add_during_CO2_to_df(df)
            # df=self.add_Koppen_data_to_df(df)
            # df=self.add_during_CSIF(df)
            # df=self.add_CO2_to_df(df)
            # df=self.add_GPP_changes_to_df(df)
            # df=self.__rename_dataframe_columns(df)
            # self.extraction_via_Koppen_df(df)
            # self.extraction_drivers_facilitators_Koppen(df,file)
            # df=self.add_CSIF_par_trend_to_df(df,file)
            # T.save_df(df, file)
            # print(file)
            # self.__df_to_excel(df, file, random=False)
    # df = self.foo(df)

        # 2 add landcover to df
        # df = self.add_landcover_to_df(df)
        # df = self.landcover_compose(df)
        # df = self.add_PAR_to_df(df)
        # df=self.add_GPCP_to_df(df)
        # df = self.add_LST_to_df(df)
        # df = self.add_SPEI_to_df(df)
        # df=self.add_during_variable_to_df(df)
        # df=self.add_during_temperature_to_df(df)
        # df=self.add_pre_mean_variables_to_df(df)
        # df=self.add_soil_data_to_df(df)
        # df= self.add_landcover_data_to_df(df)
        # df=self.__rename_dataframe_columns(df)
        # df=self.add_CSIF_change_rate_to_df(df)
        # df=self.add_during_CSIF(df)
        # self.extraction_via_landuse_df(df)
        # df=self.extraction(df)
        # df = self.drop_field_df(df)
        # self.extraction_drivers_facilitators(df)
        # T.save_df(df, self.dff)
        # self.__df_to_excel(df, self.dff, random=False)

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
    #------------'原版'------------------
    # def __load_df(self):
    #     dff = self.dff
    #     df = T.load_df(dff)

    # def __gen_df_init(self):
    #     df = pd.DataFrame()
    #     if not os.path.isfile(self.dff):
    #         T.save_df(df, self.dff)
    #         return df
    #     else:
    #         df, dff = self.__load_df()
    #         return df

#---------------------------新版-----------------wen
    def __load_df(self,file):
        df = T.load_df(file)
        return df
        # return df_early,dff

    def __gen_df_init(self,file):
        df = pd.DataFrame()
        if not os.path.isfile(file):
            T.save_df(df, self.dff)
            return df
        else:
            df= self.__load_df(file)
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def foo(self,df):
        # df=pd.DataFrame()
        fdir = '/Volumes/SSD_sumsang/project_greening/Result/%CSIF/peak/'
        dic = {}
        outf=self.dff
        result_dic = {}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                dic.update(dic_i)
        pix_list=[]
        change_rate_list=[]
        year=[]
        for pix in tqdm(dic):
            time_series=dic[pix]
            y=0
            for val in time_series:
                pix_list.append(pix)
                change_rate_list.append(val)
                year.append(y+2002)
                y=y+1
        df['pix']=pix_list
        df['%CSIF_peak'] = change_rate_list
        df['year'] = year
        # T.save_df(df, outf)
        # df = df.head(1000)
        # df.to_excel(outf+'.xlsx')
        return df

    def add_CSIF_par_trend_to_df(self, df):
        f = results_root + 'trend/CSIF_par_trend/late_trend_CSIF_par_threshold_20%.npy'
        # trend_dic = T.load_npy(f)
        # trend_dic = dict(np.load(f, allow_pickle=True, ).item())
        trend_array=np.load(f)
        trend_dic = DIC_and_TIF().spatial_arr_to_dic(trend_array)
        trend_list= []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in trend_dic:
                trend_list.append(np.nan)
                continue
            vals = trend_dic[pix]

            trend_list.append(vals)
            # landcover_list.append(vals)
        df['CSIF_par_trend_early']=trend_list
        return df

    def add_GPP_changes_to_df(self, df):
        fdir = results_root + '%CSIF/peak/'
        CSIF_dic = {}
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                CSIF_dic.update(dic_i)
        f_name = '%GPP_peak'
        CSIF_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CSIF_dic:
                CSIF_list.append(np.nan)
                continue
            vals = CSIF_dic[pix]
            if len(vals) != 15:
                CSIF_list.append(np.nan)
                continue
            v1 = vals[year - 2002]
            CSIF_list.append(v1)
        df[f_name] = CSIF_list
        return df

    def add_CSIF_change_rate_to_df(self, df):
        CSIF_change_rate_dic={}
        f = results_root + 'extraction/extraction_anomaly/extraction_during_peak_growing_season_static/during_peak_CSIF_par.npy'
        CSIF_change_rate_dic = dict(np.load(f, allow_pickle=True, ).item())
        # fdir = results_root + 'extraction/extraction_anomaly/extraction_during_early_growing_season_static/during_early_CSIF_par.npy'
        # for f in tqdm(os.listdir(fdir)):
        #     if f.endswith('.npy'):
        #         dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
        #         CSIF_change_rate_dic.update(dic_i)
        # f_name ='%CSIF_late'
        f_name =f.split('.')[0]
        f_name = f_name.split('/')[-1]
        CSIF_change_rate_list= []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CSIF_change_rate_dic:
                CSIF_change_rate_list.append(np.nan)
                continue
            vals = CSIF_change_rate_dic[pix]
            if len(vals) !=15:
                CSIF_change_rate_list.append(np.nan)
                continue
            v1 = vals[year-2002]
            CSIF_change_rate_list.append(v1)
        df[f_name] = CSIF_change_rate_list
        return df

    def add_during_CSIF(self, df):
        f = results_root + 'trend/anomaly_trend/trend_during_static/trend_during_peak_static/trend_during_peak_CSIF_par.npy'
        # CSIF_dic = T.load_npy(f)
        CSIF_array = np.load(f)
        CSIF_dic = DIC_and_TIF().spatial_arr_to_dic(CSIF_array)
        f_name=f.split('.')[0]
        f_name = f_name.split('/')[-1]
        CSIF_list= []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CSIF_dic:
                CSIF_list.append(np.nan)
                continue
            vals = CSIF_dic[pix]
            if vals < -99:
                CSIF_list.append(np.nan)
                continue
            CSIF_list.append(vals)
        df[f_name] = CSIF_list
        return df
    def add_PAR_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_PAR_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_GPCP_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_GPCP_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            GPCP_dic = T.load_npy(fdir+f)
            f_name='Original_late_'+f.split('.')[0]
            GPCP_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in GPCP_dic:
                    GPCP_list.append(np.nan)
                    continue
                vals = GPCP_dic[pix]
                if len(vals) !=15:
                    GPCP_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                GPCP_list.append(v1)
            df[f_name] = GPCP_list
        return df

    def add_LST_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_peak_static/20%_Pre_LST_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = 'Original_peak_'+f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year-2003)<0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
        return df

    def add_pre_mean_trend_variables_to_df(self, df):
        fdir = results_root + 'mean/mean_during_static_CO2/'
        for f in (os.listdir(fdir)):
            # print()
            if not f.endswith('.npy'):
                continue
            val_array = np.load(fdir+f)
            val_dic=DIC_and_TIF().spatial_arr_to_dic(val_array)
            f_name = f.split('.')[0]
            f_name = f_name.split('/')[-1]
            # print(f_name)
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

    def add_SPEI_to_df(self, df):
        fdir = results_root + 'extraction/extraction_anomaly/extraction_pre_EOS_static/20%_Pre_SPEI_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            SPEI_dic = T.load_npy(fdir + f)
            f_name = 'late_'+f.split('.')[0]
            SPEI_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in SPEI_dic:
                    SPEI_list.append(np.nan)
                    continue
                vals = SPEI_dic[pix]
                if len(vals) != 15:
                    SPEI_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                SPEI_list.append(v1)
            df[f_name] = SPEI_list
        return df

    def add_during_variable_to_df(self, df):  # add 15年序列的变量，during early, peak and late
        fdir = results_root + 'extraction/extraction_anomaly/extraction_during_late_growing_season_static/'
        for f in (os.listdir(fdir)):
            # print()
            PAR_dic = T.load_npy(fdir+f)
            f_name=f.split('.')[0]
            PAR_list= []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in PAR_dic:
                    PAR_list.append(np.nan)
                    continue
                vals = PAR_dic[pix]
                if len(vals) !=15:
                    PAR_list.append(np.nan)
                    continue
                v1 = vals[year-2002]
                PAR_list.append(v1)
            df[f_name] = PAR_list
        return df

    def add_during_temperature_to_df(self, df):  # add 14年的温度 during early, peak and late
        fdir = results_root + 'extraction/extraction_anomaly/temperature_111/'
        for f in (os.listdir(fdir)):
            # print()
            LST_dic = T.load_npy(fdir + f)
            f_name = f.split('.')[0]
            LST_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in LST_dic:
                    LST_list.append(np.nan)
                    continue
                vals = LST_dic[pix]
                if len(vals) != 14:
                    LST_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    LST_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                LST_list.append(v1)
            df[f_name] = LST_list
        return df

    def add_soil_data_to_df(self, df):  #

        soil_dic={}
        # fdir
        # = data_root + 'GLC2000_0.5DEG/dic_landcover/'
        fdir = data_root + 'SOIL/DIC/dic_SOC/'
        for f in tqdm(os.listdir(fdir)):
            if f.endswith('.npy'):
                dic_i = dict(np.load(fdir + f, allow_pickle=True, ).item())
                soil_dic.update(dic_i)
        f_name ='SOC'
        soil_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
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


    def add_CO2_to_df(self, df):
        fdir = results_root + 'extraction/extraction_original_val/extraction_pre_late_static/20%_Pre_CO2_extraction/'
        for f in (os.listdir(fdir)):
            # print()
            CO2_dic = T.load_npy(fdir + f)
            f_name = 'late_'+f.split('.')[0]
            CO2_list = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                year = row['year']
                # pix = row.pix
                pix = row['pix']
                if not pix in CO2_dic:
                    CO2_list.append(np.nan)
                    continue
                vals = CO2_dic[pix]
                if len(vals) != 14:
                    CO2_list.append(np.nan)
                    continue
                if (year - 2003) < 0:
                    CO2_list.append(np.nan)
                    continue
                v1 = vals[year - 2003]
                CO2_list.append(v1)
            df[f_name] = CO2_list
        return df

    def add_during_CO2_to_df(self, df):  # add 14年的CO2 during early, peak and late
        f = results_root + 'extraction/extraction_original_val/extraction_during_late_growing_season_static/during_late_CO2.npy'
        CO2_dic = T.load_npy(f)
        f_name = f.split('.')[0]
        f_name = f_name.split('/')[-1]
        CO2_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in CO2_dic:
                CO2_list.append(np.nan)
                continue
            vals = CO2_dic[pix]
            if len(vals) != 14:
                CO2_list.append(np.nan)
                continue
            if (year - 2003) < 0:
                CO2_list.append(np.nan)
                continue
            v1 = vals[year - 2003]
            CO2_list.append(v1)
        df[f_name] = CO2_list
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

    def __rename_dataframe_columns(self,df):
        late_new_name_dic = {
        '%CSIF_par_p_value_peak': '%CSIF_par_p_value_late',

        }

        df=pd.DataFrame(df)
        df=df.rename(columns=late_new_name_dic)
        return df
        pass

    def extraction(self, df):

        r_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            r, c = pix
            r_list.append(r)
        df['r_list'] = r_list
        df = df[df['r_list'] < 180]

        print(len(df))
        return df

    def extraction_via_landuse_df(self, df):

        outdir = self.this_class_arr + 'extraction_df_via_landuse_late/'
        outf = outdir + 'late_Cropland'
        T.mk_dir(outdir)
        # print(df)
        # exit()
        df = df[df['r_list'] < 120]
        df = df[df['landcover'] == 'Cropland']
        print(len(df))
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=True)

    def extraction_via_Koppen_df(self, df):

        outdir = self.this_class_arr + 'extraction_df_via_Koppen_late/'
        outf = outdir + 'late_A.df'
        T.mk_dir(outdir)
        # print(df)
        # exit()
        df = df[df['koppen'] == 'A']
        print(len(df))
        T.save_df(df, outf)
        self.__df_to_excel(df, outf, random=True)

    def extraction_drivers_facilitators_Koppen(self, df,file):

        outdir = self.this_class_arr + 'extraction_df_drivers_facilitators_late_landuse/'
        print(file)
        fname = file.split('.')[0]
        fname = fname.split('/')[-1]
        T.mk_dir(outdir)
        drivers, facilitators = self.__drivers_facilitators()
        outf_list = ['drivers', 'facilitators']
        # print(type(df))
        for i in range(len([drivers, facilitators])):
            # print(len(cols))
            print(i)
            outf = outdir + fname + '_' + outf_list[i]
            cols = [drivers, facilitators][i]
            df_pick = df[cols]
            T.save_df(df_pick, outf)
            self.__df_to_excel(df_pick, outf, random=False)
            #

    def __drivers_facilitators(self):

        drivers=['pix', 'year', '%CSIF_par_late', '%GPP_late',
                    'late_pre_15_PAR', 'late_pre_30_PAR', 'late_pre_60_PAR', 'late_pre_90_PAR',
                    'late_pre_15_GPCP', 'late_pre_30_GPCP', 'late_pre_60_GPCP', 'late_pre_90_GPCP',
                    'late_pre_15_LST', 'late_pre_30_LST', 'late_pre_60_LST', 'late_pre_90_LST',
                    'late_pre_1mo_SPEI3', 'late_pre_2mo_SPEI3', 'late_pre_3mo_SPEI3',
                    'late_pre_2mo_CO2_original', 'late_pre_3mo_CO2_original', 'late_pre_1mo_CO2_original',

                    'during_late_PAR', 'during_late_GPCP', 'during_late_LST', 'during_late_SPEI3', 'during_late_CO2',

                     'late_trend_pre_15_PAR', 'late_trend_pre_30_PAR', 'late_trend_pre_60_PAR', 'late_trend_pre_90_PAR',
                     'late_trend_pre_15_GPCP', 'late_trend_pre_30_GPCP', 'late_trend_pre_60_GPCP', 'late_trend_pre_90_GPCP',
                     'late_trend_pre_15_LST', 'late_trend_pre_30_LST', 'late_trend_pre_60_LST', 'late_trend_pre_90_LST',
                     'late_trend_pre_3mo_SPEI3', 'late_trend_pre_2mo_SPEI3', 'late_trend_pre_1mo_SPEI3',
                     'trend_pre_late_1mo_CO2_original', 'trend_pre_late_3mo_CO2_original', 'trend_pre_late_2mo_CO2_original',

                     'trend_during_late_PAR', 'trend_during_late_GPCP', 'trend_during_late_LST', 'trend_during_late_CO2',
                     'trend_during_peak_CSIF_par', 'trend_during_early_CSIF_par', 'trend_during_late_SPEI3',
                    ]

        facilitators=['pix', 'year', '%CSIF_par_late','%GPP_late',
                    'late_mean_pre_15_GPCP_original', 'late_mean_pre_30_GPCP_original',
                    'late_mean_pre_60_GPCP_original', 'late_mean_pre_90_GPCP_original',
                    'late_mean_pre_15_PAR_original', 'late_mean_pre_30_PAR_original', 'late_mean_pre_60_PAR_original',  'late_mean_pre_90_PAR_original',
                    'late_mean_pre_15_LST_original', 'late_mean_pre_30_LST_original', 'late_mean_pre_60_LST_original', 'late_mean_pre_90_LST_original',
                    'late_mean_pre_3mo_SPEI3', 'late_mean_pre_2mo_SPEI3', 'late_mean_pre_1mo_SPEI3',
                    'late_mean_pre_2mo_CO2_original','late_mean_pre_3mo_CO2_original', 'late_mean_pre_1mo_CO2_original',
                    'mean_during_late_PAR', 'mean_during_late_GPCP', 'mean_during_late_LST', 'mean_during_late_SPEI3', 'mean_during_late_CO2',
                    'mean_during_peak_CSIF_par', 'mean_during_early_CSIF_par',
                     'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand', 'landcover','sub_landcover','koppen'
                    ]

        print('late drivers:', len(drivers))
        print('late facilitors:', len(facilitators))
        # exit()
        return drivers,facilitators

    def extraction_drivers_facilitators_landuse(self, df):
        outdir = self.this_class_arr + 'extraction_df_drivers_facilitators_late/'
        print(self.dff)
        fname = self.dff.split('.')[0]
        fname = fname.split('/')[-1]
        T.mk_dir(outdir)
        drivers, facilitators = self.__drivers_facilitators()
        outf_list = ['drivers', 'facilitators']
        # print(type(df))
        for i in range(len([drivers, facilitators])):
            # print(len(cols))
            print(i)
            outf = outdir + fname + '_' + outf_list[i]
            cols = [drivers, facilitators][i]
            df_pick = df[cols]
            T.save_df(df_pick, outf)
            self.__df_to_excel(df_pick, outf, random=False)
            #

    def drop_field_df(self, df):
        df = df.drop(
            columns=['trend_during_late_SPEI3'])
        return df


def main():

    # Build_dataframe().run()
    Build_early_dataframe().run()
    # Build_peak_dataframe().run()
    # Build_late_dataframe().run()
    # check_data()
    # check_dataframe()
    # test()
if __name__ == '__main__':
    main()


