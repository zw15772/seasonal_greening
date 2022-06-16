# coding=gbk
import matplotlib.pyplot as plt
# import plotly.graph_objs as go
from __init__ import *
import lytools
from lytools import *
T = lytools.Tools()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import green_driver_trend_contribution
from sklearn.metrics import mean_squared_error



class Global_vars:
    def __init__(self):
        self.tif_template = this_root + 'conf/tif_template.tif'
        pass

    def koppen_landuse(self):
        kl_list = [u'Forest.A', u'Forest.B', u'Forest.Cf', u'Forest.Csw', u'Forest.Df', u'Forest.Dsw', u'Forest.E',
         u'Grasslands.A', u'Grasslands.B', u'Grasslands.Cf', u'Grasslands.Csw', u'Grasslands.Df', u'Grasslands.Dsw',
         u'Grasslands.E', u'Shrublands.A', u'Shrublands.B', u'Shrublands.Cf', u'Shrublands.Csw', u'Shrublands.Df',
         u'Shrublands.Dsw', u'Shrublands.E']
        return kl_list

    def koppen_list(self):
        koppen_list = [u'A', u'B', u'Cf', u'Csw', u'Df', u'Dsw', u'E',]
        return koppen_list
        pass


    def marker_dic(self):
        markers_dic = {
                       'Shrublands': "o",
                       'Forest': "X",
                       'Grasslands': "p",
                       }
        return markers_dic
    def color_dic_lc(self):
        markers_dic = {
                       'Shrublands': "b",
                       'Forest': "g",
                       'Grasslands': "r",
                       }
        return markers_dic
    def landuse_list(self):
        lc_list = [
              'Forest',
            'Shrublands',
            'Grasslands',
        ]
        return lc_list

    def line_color_dic(self):
        line_color_dic = {
            'pre': 'g',
            'early': 'r',
            'late': 'b'
        }
        return line_color_dic

    def gs_mons(self):

        gs = list(range(4,10))

        return gs

    def variables(self,n=3):

        X = [
            'isohydricity',
            'NDVI_pre_{}'.format(n),
            'CSIF_pre_{}'.format(n),
            'VPD_previous_{}'.format(n),
            'TMP_previous_{}'.format(n),
            'PRE_previous_{}'.format(n),
        ]
        Y = 'greenness_loss'
        # Y = 'carbon_loss'
        # Y = 'legacy_1'
        # Y = 'legacy_2'
        # Y = 'legacy_3'

        return X,Y

        pass

    def clean_df(self,df):
        ndvi_valid_f = '/Users/wenzhang/project/drought_legacy_new/results_root_main_flow_2002/arr/NDVI/NDVI_invalid_mask.npy'
        ndvi_valid_arr = np.load(ndvi_valid_f)

        spatial_dic = DIC_and_TIF().spatial_arr_to_dic(ndvi_valid_arr)
        valid_ndvi_dic = {}
        for pix in spatial_dic:
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            valid_ndvi_dic[pix]=1
        print(len(df))
        drop_index = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='Cleaning DF'):
            pix = row.pix
            if not pix in valid_ndvi_dic:
                drop_index.append(i)
        df = df.drop(df.index[drop_index])

        # print(len(df))
        # exit()
        # df = df.drop_duplicates(subset=['pix', 'delta_legacy'])
        # self.__df_to_excel(df,dff+'drop')

        # df = df[df['ratio_of_forest'] > 0.90]
        df = df[df['lat'] > 30]
        # df = df[df['lat'] < 60]
        # df = df[df['delta_legacy'] < -0]
        # df = df[df['trend_score'] > 0.2]
        # df = df[df['gs_sif_spei_corr'] > 0]

        # trend = df['trend']
        # trend_mean = np.nanmean(trend)
        # trend_std = np.nanstd(trend)
        # up = trend_mean + trend_std
        # down = trend_mean - trend_std
        # df = df[df['trend'] > down]
        # df = df[df['trend'] < up]

        # quantile = 0.4
        # delta_legacy = df['delta_legacy']
        # delta_legacy = delta_legacy.dropna()
        #
        # # print(delta_legacy)
        # q = np.quantile(delta_legacy,quantile)
        # # print(q)
        # df = df[df['delta_legacy']<q]
        # T.print_head_n(df)
        print(len(df))
        # exit()
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            val = 1
            spatial_dic[pix] = val

        return df,spatial_dic

    def mask_arr_with_NDVI(self, inarr):
        ndvi_valid_f = results_root_main_flow_2002 + 'arr/NDVI/NDVI_invalid_mask.npy'
        ndvi_valid_arr = np.load(ndvi_valid_f)
        grid = np.isnan(ndvi_valid_arr)
        inarr[grid] = np.nan

        pass


        pass

class Make_Dataframe:

    def __init__(self):
        self.this_class_arr = '/Volumes/NVME2T/wen_proj/result/Make_Dataframe/'
        Tools().mk_dir(self.this_class_arr, force=True)
        self.dff = self.this_class_arr + 'data_frame.df'

    def run(self):
        # 0 generate a void dataframe
        df = self.__gen_df_init()
        df = self.__add_void_pix_to_df(df)
        # df = self.add_previous_conditions(df)
        df = self.add_MAT_MAP_to_df(df)
        # df = self.add_lon_lat_to_df(df)
        # df = self.select_max_val_and_pre_length(df)
        # df = self.select_max_product(df)
        df = self.add_max_contribution_index(df)
        df = df.dropna()
        T.save_df(df,self.dff)
        self.__df_to_excel(df,self.dff)
        pass


    def _check_spatial(self,df):
        spatial_dic = {}
        for i,row in df.iterrows():
            pix = row.pix
            spatial_dic[pix] = row.lon
            # spatial_dic[pix] = row.isohydricity
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr()
        plt.imshow(arr)
        plt.show()
        pass


    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def __gen_df_init(self):
        df = pd.DataFrame()
        if not os.path.isfile(self.dff):
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df
            # raise Warning('{} is already existed'.format(self.dff))

    def __add_void_pix_to_df(self,df):
        void_dic = DIC_and_TIF().void_spatial_dic()
        pix_list = []
        for pix in void_dic:
            pix_list.append(pix)
        df['pix'] = pix_list
        return df

        pass


    def __df_to_excel(self,df,dff,head=1000):
        if head == None:
            df.to_excel('{}.xlsx'.format(dff))
        else:
            df = df.head(head)
            df.to_excel('{}.xlsx'.format(dff))

        pass


    def add_previous_conditions(self,df):
        fdir = '/Users/wenzhang/project/wen_proj/result/climate_constrain_greenning_SOS 2/'
        for folder in os.listdir(fdir):
            if folder.startswith('.'):
                continue
            for f in os.listdir(os.path.join(fdir,folder)):
                if f.startswith('.'):
                    continue
                arr = to_raster.raster2array(os.path.join(fdir,folder,f))[0]
                arr = np.array(arr)
                arr[arr<-9999]=np.nan
                spatial_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
                var = '{}_{}'.format(folder,f).replace('.tif','')
                val_list = []
                pix_list = []
                for pix in tqdm(spatial_dic,desc=var):
                    pix_list.append(pix)
                    val = spatial_dic[pix]
                    val_list.append(val)
                df['pix'] = pix_list
                df[var] = val_list

                # plt.imshow(arr)
                # plt.show()
        return df

    def add_MAT_MAP_to_df(self,df):
        tif = data_root + 'Climate_408/PRE/MAPRE.tif'
        arr = to_raster.raster2array(tif)[0]
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in dic:
                tmp_trend = dic[pix]
                if tmp_trend > -999:
                    val_list.append(tmp_trend)
                else:
                    val_list.append(np.nan)

            else:
                val_list.append(np.nan)
        df['MAP'] = val_list

        tif = data_root + 'Climate_408/TMP/MATMP.tif'
        arr = to_raster.raster2array(tif)[0]
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row.pix
            if pix in dic:
                tmp_trend = dic[pix]
                if tmp_trend > -999:
                    val_list.append(tmp_trend)
                else:
                    val_list.append(np.nan)
            else:
                val_list.append(np.nan)
        df['MAT'] = val_list

        return df

    def add_lon_lat_to_df(self, df):
        # DIC_and_TIF().spatial_tif_to_lon_lat_dic()
        pix_to_lon_lat_dic_f = DIC_and_TIF().this_class_arr + 'pix_to_lon_lat_dic.npy'
        lon_lat_dic = T.load_npy(pix_to_lon_lat_dic_f)
        # print(pix)
        lon_list = []
        lat_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc='adding lon lat into df'):
            pix = row.pix
            lon, lat = lon_lat_dic[pix]
            lon_list.append(lon)
            lat_list.append(lat)
        df['lon'] = lon_list
        df['lat'] = lat_list

        return df


    def select_max_val_and_pre_length(self,df):
        product_list = [
            'GPCP',
            'LST_mean',
            'PAR',
                        ]
        pre_length_list = [
            15,
            30,
            60,
            90,
        ]

        for product in product_list:
            maxv_list = []
            maxind_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=product):
                pix = row.pix
                val_list = []
                for length in pre_length_list:
                    var = '{}_SOS_pre{}_p'.format(product, length)
                    val = row[var]
                    val_list.append(val)
                if True in np.isnan(val_list):
                    maxv_list.append(np.nan)
                    maxind_list.append(np.nan)
                    continue
                max_v = np.max(val_list)
                max_arg = np.argmax(val_list)
                max_indx = pre_length_list[max_arg]
                maxv_list.append(max_v)
                maxind_list.append(max_indx)
            df['{}_max_value'.format(product)] = maxv_list
            df['{}_max_pre_length'.format(product)] = maxind_list
        return df


    def select_max_product(self,df):
        product_list = [
            'GPCP',
            'LST_mean',
            'PAR',
        ]
        max_product_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            max_var_list = []
            for product in product_list:
                max_var = '{}_max_value'.format(product)
                val = row[max_var]
                max_var_list.append(val)
            if True in max_var_list:
                print(max_var_list)
                exit()
            max_arg = np.argmax(max_var_list)
            max_product = product_list[max_arg]
            max_product_list.append(max_product)
        df['max_corr_product'] = max_product_list
        return df

    def add_max_contribution_index(self,df):
        fdir = '/Volumes/NVME2T/wen_proj/result/0523/'
        f = fdir + 'Max_contribution_index_threshold_20%.npy'
        arr = np.load(f)
        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        max_ind_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            val = dic[pix]
            max_ind_list.append(val)

        df['max_contrib_index'] = max_ind_list
        return df



        pass

class Main_flow_shui_re:

    def __init__(self):

        pass

    def run(self):
        # self.plot_MAP()
        # self.plot_matrix()
        # self.plot_scatter()
        # self.plot_scatter_pre_n()
        self.plot_dominance()
        pass

    def __divide_MA(self,arr,min_v=None,max_v=None,step=None,n=None):
        if min_v == None:
            min_v = np.min(arr)
        if max_v == None:
            max_v = np.max(arr)
        if n == None:
            d = np.arange(start=min_v,step=step,stop=max_v)
        if step == None:
            d = np.linspace(min_v,max_v,num=n)

        # print d
        # exit()
        # if step >= 10:
        #     d_str = []
        #     for i in d:
        #         d_str.append('{}'.format(int(round(i*12.,0))))
        # else:
        d_str = []
        for i in d:
            d_str.append('{}'.format(int(round(i, 0))))
        # print d_str
        # exit()
        return d,d_str
        pass


    def plot_matrix(self):
        var_name = 'GPCP_SOS_pre30_p'
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        df,valid_spatial_dic = Global_vars().clean_df(df)
        # df = df.drop_duplicates(subset=['pix'])
        vals_dic = DIC_and_TIF().void_spatial_dic()
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            # sens = row.lag
            var = row[var_name]
            vals_dic[pix].append(var)
        MAT_series = df.MAT
        MAP_series = df.MAP * 12.
        df['MAP'] = MAP_series
        d_mat,mat_str = self.__divide_MA(MAT_series,step=1)
        d_map,map_str = self.__divide_MA(MAP_series,min_v=0,max_v=2001,step=100)
        # print map_str
        # print d_map
        # exit()

        shuire_matrix = []
        x = []
        y = []
        z = []
        for t in tqdm(range(len(d_mat))):
            if t + 1 >= len(d_mat):
                continue
            df_t = df[df['MAT']>d_mat[t]]
            df_t = df_t[df_t['MAT']<d_mat[t+1]]
            temp = []
            for p in range(len(d_map)):
                if p + 1 >= len(d_map):
                    continue
                df_p = df_t[df_t['MAP']>d_map[p]]
                df_p = df_p[df_p['MAP']<d_map[p+1]]
                pixs = df_p.pix

                if len(pixs) != 0:
                    vals = []
                    for pix in pixs:
                        val = vals_dic[pix]
                        val = np.nanmean(val)
                        vals.append(val)
                    val_mean = np.nanmean(vals)
                else:
                    val_mean = np.nan
                temp.append(val_mean)
                x.append(d_map[p])
                y.append(d_mat[t])
                z.append(val_mean)
            shuire_matrix.append(temp)
        # plt.imshow(shuire_matrix,vmin=-0.3,vmax=0.3)
        # plt.imshow(shuire_matrix)
        # plt.xticks(range(len(shuire_matrix[0])),map_str,rotation=90)
        # plt.yticks(range(len(shuire_matrix)),mat_str,rotation=0)

        plt.figure(figsize=(4, 6))
        cmap = 'RdBu_r'
        plt.scatter(x, y, c=z, marker='s', cmap=cmap, norm=None,vmin=-0.3,vmax=0.3)
        plt.gca().invert_yaxis()
        plt.subplots_adjust(
            top=0.88,
            bottom=0.11,
            left=0.12,
            right=0.90,
            hspace=0.2,
            wspace=0.2
        )
        plt.title('Lag (months)')
        plt.colorbar()
        plt.xlabel('MAP (mm)')
        plt.ylabel('MAT (°C)')
        plt.title(var_name)
        plt.show()


    def plot_scatter(self):
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        df,_ = Global_vars().clean_df(df)

        # max_corr_product
        map = df['MAP']
        mat = df['MAT']
        max_corr_product = df['max_corr_product']
        color_dic = {
            'PAR':'g',
            'LST_mean':'r',
            'GPCP':'b',
        }
        # x,y,z = []
        flag = 0
        xlist = []
        ylist = []
        clist = []
        for x,y,z in tqdm(zip(map,mat,max_corr_product),total=len(map)):
            if np.isnan(x):
                continue
            if np.isnan(y):
                continue
            # print(z)
            if not z in color_dic:
                print(z)
                exit()
                continue
            # flag += 1
            # if not flag % 3 == 0:
            #     continue
            # print(x,y,z)
            # plt.scatter(x,y,c=color_dic[z],s=1,alpha=0.5,marker='+')
            xlist.append(x)
            ylist.append(y)
            clist.append(color_dic[z])
            # plt.show()
            # pause()
        plt.scatter(xlist,ylist,c=clist,s=2,alpha=0.5,marker='+')
        plt.show()

        # print(flag)


        pass
    def plot_scatter_pre_n(self,n=30):
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        # df,_ = Global_vars().clean_df(df)
        product_list = [
            'GPCP',
            'LST_mean',
            'PAR',
        ]
        color_dic = {
            'PAR': 'g',
            'LST_mean': 'r',
            'GPCP': 'b',
        }
        # max_corr_product
        # map = df['MAP']
        # mat = df['MAT']
        x_list = []
        y_list = []
        c_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            map = row['MAP']
            mat = row['MAT']

            val_list = []
            for product in product_list:
                var_name = '{}_SOS_pre{}_p'.format(product,n)
                val = row[var_name]
                val_list.append(val)
            max_arg = np.argmax(val_list)
            max_product = product_list[max_arg]
            # if max_product == 'GPCP':
            #     continue
            x_list.append(map)
            y_list.append(mat)
            c_list.append(color_dic[max_product])
        plt.scatter(x_list, y_list, c=c_list, s=2, alpha=0.2, marker='+')
        plt.show()

        pass




    def plot_MAP(self):
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        # df,spatial_dic = Global_vars().clean_df(df)

        map_dic = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            map = row.MAP
            map_dic[pix] = map
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(map_dic)
        # arr = arr * 12.
        arr = arr
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
        pass


    def plot_dominance(self):
        dff = Make_Dataframe().dff
        df = T.load_df(dff)
        color_dic = {
            1: 'g',
            2: 'r',
            3: 'b',
        }
        x_list = []
        y_list = []
        c_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 180.:
                continue
            map = row['MAP']
            mat = row['MAT']
            max_ind = row['max_contrib_index']
            # print(max_ind)
            # if max_ind == 1:
            if 1:
                c_list.append(color_dic[max_ind])
                x_list.append(map)
                y_list.append(mat)
        plt.scatter(x_list, y_list, c=c_list, s=20, alpha=0.2, marker='s')
        plt.show()

class Analysis:

    def __init__(self):

        pass

    def run(self):

        pass

    # def


class Greening:

    def __init__(self):

        pass

    def run(self):
        # self.anomaly()
        self.origin()
        # self.phenoelogy()
        # self.trend()
        pass

    def anomaly(self):
        fdir = '/Volumes/NVME2T/wen_proj/result/CSIF_par/'
        dic = T.load_npy_dir(fdir,condition='005')
        return dic
        # for pix in dic:
        #     # print(pix,dic[pix])
        #     plt.plot(dic[pix])
        #     plt.show()

        pass

    def origin(self):

        # fdir = '/Users/wenzhang/project/wen_proj/result/Hants_annually_smooth/Hants_annually_smooth_CSIF_par/'
        fdir = '/Volumes/NVME2T/wen_proj/result/Hants_annually_smooth/Hants_annually_smooth_CSIF_par/'
        f2002 = fdir + '2002.npy'
        f2016 = fdir + '2016.npy'
        dic_2002 = T.load_npy(f2002)
        dic_2016 = T.load_npy(f2016)
        early_start_dic, early_end_dic, late_start_dic, late_end_dic = self.phenology()

        spring_contrib_dic = {}
        summer_contrib_dic = {}
        autumn_contrib_dic = {}
        for pix in tqdm(early_start_dic):
            r,c = pix
            if r > 180:
                continue
            try:
                vals_2002 = dic_2002[pix]
                vals_2016 = dic_2016[pix]
            except:
                continue
            early_start = early_start_dic[pix]
            early_end = early_end_dic[pix]
            # print(early_start)
            # print(early_end)
            # print('*'*80)
            # sleep()
            # continue


            late_start = late_start_dic[pix]
            late_end = late_end_dic[pix]

            early_start_2002 = early_start[0]
            early_end_2002 = early_end[0]
            late_start_2002 = late_start[0]
            late_end_2002 = late_end[0]

            early_start_2016 = early_start[-1]
            early_end_2016 = early_end[-1]
            late_start_2016 = late_start[-1]
            late_end_2016 = late_end[-1]


            spring_indx_2002 = list(range(early_start_2002,early_end_2002))
            summer_indx_2002 = list(range(early_end_2002,late_start_2002))
            autumn_indx_2002 = list(range(late_start_2002,late_end_2002))

            spring_indx_2016 = list(range(early_start_2016, early_end_2016))
            summer_indx_2016 = list(range(early_end_2016, late_start_2016))
            autumn_indx_2016 = list(range(late_start_2016, late_end_2016))


            vals_spring_2002 = T.pick_vals_from_1darray(vals_2002,spring_indx_2002)
            vals_summer_2002 = T.pick_vals_from_1darray(vals_2002,summer_indx_2002)
            vals_autumn_2002 = T.pick_vals_from_1darray(vals_2002,autumn_indx_2002)

            vals_spring_2016 = T.pick_vals_from_1darray(vals_2016, spring_indx_2016)
            vals_summer_2016 = T.pick_vals_from_1darray(vals_2016, summer_indx_2016)
            vals_autumn_2016 = T.pick_vals_from_1darray(vals_2016, autumn_indx_2016)

            total_2002 = np.sum(vals_spring_2002) + np.sum(vals_summer_2002) + np.sum(vals_autumn_2002)
            total_2016 = np.sum(vals_spring_2016) + np.sum(vals_summer_2016) + np.sum(vals_autumn_2016)

            diff_total = total_2016 - total_2002
            if diff_total < 0:
                continue
            diff_spring = np.sum(vals_spring_2016) - np.sum(vals_spring_2002)
            diff_summer = np.sum(vals_summer_2016) - np.sum(vals_summer_2002)
            diff_autumn = np.sum(vals_autumn_2016) - np.sum(vals_autumn_2002)

            spring_contribution = diff_spring / diff_total
            summer_contribution = diff_summer / diff_total
            autumn_contribution = diff_autumn / diff_total

            # print(vals)
            if np.isnan(spring_contribution):
                continue

            spring_contrib_dic[pix] = spring_contribution
            summer_contrib_dic[pix] = summer_contribution
            autumn_contrib_dic[pix] = autumn_contribution

            # print('spring_contribution',spring_contribution)
            # print('summer_contribution',summer_contribution)
            # print('autumn_contribution',autumn_contribution)
            # print(sum([spring_contribution,summer_contribution,autumn_contribution]))
            # print('*'*10)
            # try:
            #     plt.plot(vals_2002)
            #     plt.scatter(early_start_2002,vals_2002[early_start_2002])
            #     plt.scatter(early_end_2002,vals_2002[early_end_2002])
            #     plt.scatter(late_start_2002,vals_2002[late_start_2002])
            #     plt.scatter(late_end_2002,vals_2002[late_end_2002])
            #
            #     plt.scatter(early_start_2016, vals_2016[early_start_2016])
            #     plt.scatter(early_end_2016, vals_2016[early_end_2016])
            #     plt.scatter(late_start_2016, vals_2016[late_start_2016])
            #     plt.scatter(late_end_2016, vals_2016[late_end_2016])
            #     plt.plot(vals_2016)
            #     plt.show()
            # except:
            #     plt.close()
            #     continue

        spring_contrib_arr = DIC_and_TIF().pix_dic_to_spatial_arr(spring_contrib_dic)
        summer_contrib_arr = DIC_and_TIF().pix_dic_to_spatial_arr(summer_contrib_dic)
        autumn_contrib_arr = DIC_and_TIF().pix_dic_to_spatial_arr(autumn_contrib_dic)

        # plt.figure()
        # plt.imshow(spring_contrib_arr,vmin=0,vmax=0.5)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.title('spring')
        #
        # plt.figure()
        # plt.imshow(summer_contrib_arr,vmin=0,vmax=0.5)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.title('summer')
        #
        # plt.figure()
        # plt.imshow(autumn_contrib_arr,vmin=0,vmax=0.5)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.title('autumn')


        max_ind_arr = []
        for r in range(len(spring_contrib_arr)):
            temp = []
            for c in range(len(spring_contrib_arr[0])):
                temp_i = []
                for arr in [spring_contrib_arr,summer_contrib_arr,autumn_contrib_arr]:
                    temp_i.append(arr[r][c])
                if True in np.isnan(temp_i):
                    temp.append(np.nan)
                    continue
                if len(temp_i) > 0:
                    max_indx = np.argmax(temp_i)
                    temp.append(float(max_indx))
                else:
                    temp.append(np.nan)
            max_ind_arr.append(temp)
        plt.imshow(max_ind_arr[:180])
        DIC_and_TIF().plot_back_ground_arr_north_sphere()
        plt.show()



    def phenology(self):

        # fdir = '/Volumes/NVME2T/wen_proj/result/early_peak_late_dormant_period_annually/20%_transform_early_peak_late_dormant_period_annually_CSIF_par/'
        fdir = '/Volumes/NVME2T/wen_proj/result/early_peak_late_dormant_period_annually/20%_transform_early_peak_late_dormant_period_annually_CSIF_par 2/'
        e_e_f = fdir + 'early_end.npy'
        e_s_f = fdir + 'early_start.npy'
        l_e_f = fdir + 'late_end.npy'
        l_s_f = fdir + 'late_start.npy'


        early_start_dic = T.load_npy(e_s_f)
        early_end_dic = T.load_npy(e_e_f)
        late_start_dic = T.load_npy(l_s_f)
        late_end_dic = T.load_npy(l_e_f)

        return early_start_dic,early_end_dic,late_start_dic,late_end_dic


    def trend(self):
        anomaly_dic = self.anomaly()
        early_start_dic, early_end_dic, late_start_dic, late_end_dic = self.phenology()
        for pix in anomaly_dic:
            vals = anomaly_dic[pix]
            vals_reshape = np.reshape(vals,(15,-1))
            plt.imshow(vals_reshape)
            plt.show()
            spring_mean_list = []
            peak_mean_list = []
            fall_mean_list = []
            GS_mean_list = []

            for y in range(len(vals_reshape)):
                one_year_vals = vals_reshape[y]
                early_start = early_start_dic[pix][y]
                early_end = early_end_dic[pix][y]
                late_start = late_start_dic[pix][y]
                late_end = late_end_dic[pix][y]

                spring_range = range(early_start,early_end)
                peak_range = range(early_end,late_start)
                fall_range = range(late_start,late_end)
                GS_range = range(early_start,late_end)

                spring_range_vals = T.pick_vals_from_1darray(one_year_vals,spring_range)
                peak_range_vals = T.pick_vals_from_1darray(one_year_vals,peak_range)
                fall_range_vals = T.pick_vals_from_1darray(one_year_vals,fall_range)
                GS_range_vals = T.pick_vals_from_1darray(one_year_vals,GS_range)

                spring_range_vals_mean = np.mean(spring_range_vals)
                peak_range_vals_mean = np.mean(peak_range_vals)
                fall_range_vals_mean = np.mean(fall_range_vals)
                GS_range_vals_mean = np.mean(GS_range_vals)

                spring_mean_list.append(spring_range_vals_mean)
                peak_mean_list.append(peak_range_vals_mean)
                fall_mean_list.append(fall_range_vals_mean)
                GS_mean_list.append(GS_range_vals_mean)

            spring_mean_list = np.array(spring_mean_list)
            peak_mean_list = np.array(peak_mean_list)
            fall_mean_list = np.array(fall_mean_list)
            GS_mean_list = np.array(GS_mean_list)

            # plt.plot(spring_mean_list,label='spring')
            # plt.plot(peak_mean_list,label='peak')
            # plt.plot(fall_mean_list,label='fall')
            # plt.plot(GS_mean_list,label='GS')

            # plt.plot(spring_mean_list+peak_mean_list+fall_mean_list,label='sum')
            # plt.plot((spring_mean_list+peak_mean_list+fall_mean_list)/3.,label='sum/3')

            df = pd.DataFrame()
            df['GS_mean_list']=GS_mean_list
            df['peak_mean_list']=peak_mean_list
            df['spring_mean_list']=spring_mean_list
            df['fall_mean_list']=fall_mean_list
            sns.pairplot(df)

            # plt.scatter(GS_mean_list,spring_mean_list,label='spring')
            # plt.scatter(GS_mean_list,peak_mean_list,label='peak')
            # plt.scatter(GS_mean_list,fall_mean_list,label='fall')
            # ax = plt.figure()
            # KDE_plot().plot_scatter(GS_mean_list,spring_mean_list,plot_fit_line=True,s=20,is_KDE=False,ax=ax)
            # KDE_plot().plot_scatter(GS_mean_list,peak_mean_list,plot_fit_line=True,s=20,is_KDE=False,ax=ax)
            # KDE_plot().plot_scatter(GS_mean_list,fall_mean_list,plot_fit_line=True,s=20,is_KDE=False,ax=ax)
            # plt.legend()
            plt.tight_layout()
            plt.show()


        pass


class Multi_colormap_spatial_map:

    def __init__(self):

        pass

    def run(self):

        # self.latitude_plot()
        self.muti_variate_map()
        pass

    def muti_variate_map(self):
        import colorsys
        fdir = '/Volumes/NVME2T/wen_proj/result/0523/'
        sos_f = fdir + 'leaf_out_contribution_threshold_20%.npy'
        eos_f = fdir + 'EOS_contribution_threshold_20%.npy'
        peak_f = fdir + 'peak_contribution_threshold_20%.npy'

        sos_arr = np.load(sos_f)
        peak_arr = np.load(peak_f)
        eos_arr = np.load(eos_f)

        sos_arr[sos_arr<-2]=np.nan
        sos_arr[sos_arr>2]=np.nan
        peak_arr[peak_arr < -2] = np.nan
        peak_arr[peak_arr > 2] = np.nan
        eos_arr[eos_arr < -2] = np.nan
        eos_arr[eos_arr > 2] = np.nan

        sos_arr[sos_arr < 0] = 0
        sos_arr[sos_arr > 1] = 1
        peak_arr[peak_arr < 0] = 0
        peak_arr[peak_arr > 1] = 1
        eos_arr[eos_arr < 0] = 0
        eos_arr[eos_arr > 1] = 1

        # sos_dic = DIC_and_TIF().spatial_arr_to_dic(sos_arr)
        # eos_dic = DIC_and_TIF().spatial_arr_to_dic(eos_arr)
        # peak_dic = DIC_and_TIF().spatial_arr_to_dic(peak_arr)

        # plt.imshow(sos_arr,cmap='Greens',alpha=0.3)
        # plt.imshow(peak_arr,cmap='Reds',alpha=0.3)
        # plt.imshow(eos_arr,cmap='Blues',alpha=0.3)
        # plt.show()

        matrix = []
        for i in range(len(eos_arr)):
            temp = []
            for j in range(len(eos_arr[0])):
                if np.isnan(sos_arr[i,j]):
                    data = [0.9,0.9,0.9,1]
                else:
                    # r=colorsys.hls_to_rgb(0,peak_arr[i,j],0.7)
                    # g=colorsys.hls_to_rgb(0,peak_arr[i,j],0.7)
                    # r=colorsys.hls_to_rgb(0,peak_arr[i,j],0.7)
                    if round((peak_arr[i,j] + sos_arr[i,j] + eos_arr[i,j]),0) == 1:
                        data = [peak_arr[i,j],sos_arr[i,j],eos_arr[i,j],1]
                    else:
                        data = [0.9,0.9,0.9,1]

                    # data = [peak_arr[i,j],sos_arr[i,j],eos_arr[i,j],np.std([peak_arr[i,j],sos_arr[i,j],eos_arr[i,j]])]
                    # print(data)
                temp.append(data)
            matrix.append(temp)
        plt.imshow(matrix)
        plt.show()

        pass



    def latitude_plot(self):
        fdir = '/Volumes/NVME2T/wen_proj/result/0523/'
        f = fdir + 'Max_contribution_index_threshold_20%.npy'
        arr = np.load(f)

        spring = []
        summer = []
        autumn = []
        flag = 0
        for i in arr:
            flag += 1
            if flag > 120:
                continue
            i = np.array(i)
            i = T.remove_np_nan(i)
            if len(i) == 0:
                spring.append(np.nan)
                summer.append(np.nan)
                autumn.append(np.nan)
            else:
                i = list(i)
                spring_n = i.count(1)
                summer_n = i.count(2)
                autumn_n = i.count(3)
                total_len = float(len(i))
                spring.append(spring_n/total_len)
                summer.append(summer_n/total_len)
                autumn.append(autumn_n/total_len)

        plt.plot(spring,c='g',label='spring')
        plt.plot(summer,c='r',label='summer')
        plt.plot(autumn,c='b',label='autumn')
        plt.legend()
        plt.show()

        pass


class Linear_contribution:

    def __init__(self):

        self.period='peak'
        time_range='1982-2015'
        self.result_f = results_root+'greening_contribution/{}_greening_contribution_{}.npy'.format(time_range,self.period,)
        self.x_dir = results_root+'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time_range,time_range,self.period)
        self.y_f = results_root+'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/during_{}_GIMMS_NDVI.npy'.format(time_range,time_range,self.period,self.period)
        pass


    def run(self):

        # step 1 build dataframe
        df = self.build_df(self.x_dir,self.y_f,self.period)
        x_var_list = self.__get_x_var_list(self.x_dir,self.period)
        # # # step 2 cal correlation
        self.cal_contribution(df, x_var_list)
        # # check model perfermance
        self.contribution_bar(self.result_f, self.x_dir)
        # self.plot_spatial_max_contribution()


    def __get_x_var_list(self,x_dir,period):
        # x_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        x_f_list = []
        for x_f in T.listdir(x_dir):
            if x_f == 'during_{}_GIMMS_NDVI.npy'.format(period,):
                continue
            if x_f == 'during_{}_NIRv.npy'.format(period,):
                continue
            if x_f == 'during_{}_VOD.npy'.format(period, ):
                continue
            if x_f == 'during_{}_root_soil_moisture.npy'.format(period, ):
                continue
            if x_f == 'during_{}_surf_soil_moisture.npy'.format(period, ):
                continue
            x_f_list.append(x_dir + x_f)
        print(x_f_list)
        x_var_list = []
        for x_f in x_f_list:
            split1 = x_f.split('/')[-1]
            split2 = split1.split('.')[0]
            var_name = '_'.join(split2.split('_')[2:])
            x_var_list.append(var_name)
        return x_var_list


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


    def build_df(self,x_dir,y_f,period):
        x_f_list = []
        for x_f in T.listdir(x_dir):

            if x_f == 'during_{}_GIMMS_NDVI.npy'.format(period,):
                continue
            if x_f == 'during_{}_NIRv.npy'.format(period,):
                continue
            if x_f == 'during_{}_VOD.npy'.format(period, ):
                continue
            if x_f == 'during_{}_root_soil_moisture.npy'.format(period, ):
                continue
            if x_f == 'during_{}_surf_soil_moisture.npy'.format(period, ):
                continue
            x_f_list.append(x_dir + x_f)
        print(x_f_list)
        df = pd.DataFrame()
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
        x_var_list = []
        for x_f in x_f_list:
            # print(x_f)
            split1 = x_f.split('/')[-1]
            split2 = split1.split('.')[0]
            var_name = '_'.join(split2.split('_')[2:])
            x_var_list.append(var_name)
            # print(var_name)
            x_val_list = []
            x_arr = T.load_npy(x_f)
            for i,row in tqdm(df.iterrows(),total=len(df),desc=var_name):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)
            # x_val_list = np.array(x_val_list)
            df[var_name] = x_val_list
        # T.print_head_n(df)
        # outexcel = '/Volumes/NVME2T/wen_proj/greening_contribution/1982-2015_extraction_during_late_growing_season_static/test'
        # T.df_to_excel(df,outexcel,n=10000,random=True)
        # exit()
        return df


    def cal_contribution(self,df,x_var_list):
        outf = self.result_f
        y_trend_list = []
        y_predicted_list = []
        contribution_dic_all = {}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 120:
                continue
            y_vals = row['y']
            try:
                y_trend,y_b,_ = self.__linearfit(range(len(y_vals)),y_vals) # dy/dt y trend, greening obs
            except:
                y_trend = np.nan
            y_trend_list.append(y_trend)
            # if y_trend < 0:
            #     continue
            # 1 calculate slopes of X variables
            slope_dic = {}
            for x in x_var_list:
                x_vals = row[x]
                try:
                    k,b,_ = self.__linearfit(range(len(x_vals)),x_vals) # annual trend
                    if k > 999:
                        slope = np.nan
                    else:
                        slope = k
                except:
                    slope = np.nan
                slope_dic[x] = slope

            # 2 calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []
            for x in x_var_list:
                x_vals = row[x]
                if not len(x_vals) == 34:
                    continue
                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals= T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0]==None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend


                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                y_predicted_list.append(np.nan)
                continue


            df_new['y'] = y_vals   # 不detrend
            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1,how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)
            # outexcel = '/Volumes/NVME2T/wen_proj/greening_contribution/1982-2015_extraction_during_late_growing_season_static/test1'
            # T.df_to_excel(df_new,outexcel,n=10000,random=False)
            # exit()
            df_new = df_new.dropna()
            linear_model = LinearRegression()
            linear_model.fit(df_new[x_var_list_valid_new],df_new['y'])
            coef_ = linear_model.coef_
            coef_dic = dict(zip(x_var_list_valid_new,coef_))

            # 3 calculate contribution
            contribution_dic = {}
            y_predicted_i = []
            for x in x_var_list_valid_new:
                slope_x = slope_dic[x] # dx/dt each x trend
                partial_derivative_x = coef_dic[x] # each x sensitivity to y
                contribution = slope_x * partial_derivative_x
                contribution_dic[x] = contribution
                y_predicted_i.append(contribution)
            y_predicted = np.sum(y_predicted_i)
            contribution_dic_all[pix] = {
                'contribution_dic':contribution_dic,
                'y_predicted':y_predicted,
                'y_trend':y_trend,
            }
            y_predicted_list.append(y_predicted)
        # df['y_obs'] = y_trend_list
        # df['y_pred'] = y_predicted_list
        T.save_npy(contribution_dic_all,outf)

    def performance(self,df):
        pass


    def contribution_bar(self,result_f,x_dir):
        y_predicted=[]
        # x_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        result_dic = T.load_npy(result_f)
        x_var_list = self.__get_x_var_list(x_dir,self.period)
        y_trend = []

        for pix in result_dic:
            r,c = pix
            if r > 120:
                continue
            dic_i = result_dic[pix]
            y_predicted_i = dic_i['y_predicted']
            y_trend_i = dic_i['y_trend']
            y_predicted.append(y_predicted_i)
            y_trend.append(y_trend_i)
        KDE_plot().plot_scatter(y_predicted,y_trend)

        plt.figure()
        box = []
        err = []
        for x_var in x_var_list:
            x_var_all = []
            for pix in result_dic:
                r,c = pix
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
        plt.bar(x_var_list,box,)
        plt.show()



    def plot_spatial_max_contribution(self):
        x_var_list = self.__get_x_var_list(self.x_dir)
        x_var_list.sort()
        result_f = self.result_f
        result_dic = T.load_npy(result_f)
        color_map = dict(zip(x_var_list,list(range(len(x_var_list)))))
        print(color_map)
        # exit()
        spatial_dic = {}
        for pix in result_dic:
            result = result_dic[pix]
            contribution_dic = result['contribution_dic']
            contribution_list = []
            x_var_list_new = []
            for x in x_var_list:
                if not x in contribution_dic:
                    continue
                contribution_list.append(abs(contribution_dic[x]))
                x_var_list_new.append(x)
            argsort = np.argsort(contribution_list)[::-1]
            max_x_var = x_var_list_new[argsort[0]]
            # val = x_var_list.index(max_x_var)
            val = color_map[max_x_var]
            spatial_dic[pix] = val
        tif_template = '/Volumes/NVME2T/project05_redo/temp/200105.tif'
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)[:150]
        DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
        # cmap = sns.color_palette('hls',as_cmap=True)
        plt.imshow(arr,cmap='jet')
        plt.colorbar()
        plt.show()

        pass

class Multi_liner_regression:  # 实现求beta 功能

    def __init__(self):

        self.period='early'
        self.variable='LAI3g'
        self.time_range='2000-2018'
        # self.result_dir=results_root+'multiregression/LAI_GIMMS/'
        self.result_dir = results_root + f'partial_correlation_zscore_detrend/'
        # self.result_f = self.result_dir+'/detrend_{}_multi_linear{}_{}.npy'.format(self.time_range,self.period,self.variable)
        self.partial_correlation_result_f = self.result_dir+'/{}_partial_correlation_{}_{}.npy'.format(self.time_range,self.period,self.variable)
        self.partial_correlation_p_value_result_f = self.result_dir + '/{}_partial_correlation_p_value_result_{}_{}.npy'.format(
            self.time_range, self.period,self.variable)
        # self.partial_correlation_VIP_result_f = self.result_dir + '/{}_partial_correlation_VIP_{}_{}.npy'.format(
        #     self.time_range, self.period, self.variable)
        self.x_dir=results_root+f'detrend_Zscore/detrend_{self.time_range}/detrend_{self.time_range}_during_{self.period}/{self.time_range}_X/'
        # self.x_dir = results_root+f'zscore/2000-2018_daily/2000-2018_X/'
        self.y_f = results_root+f'detrend_Zscore/detrend_{self.time_range}/detrend_{self.time_range}_during_{self.period}/{self.time_range}_Y/detrend_{self.variable}_{self.period}_zscore.npy'
        # self.y_f=results_root+f'zscore/2000-2018_daily/2000-2018_Y/{self.variable}_{self.period}_zscore.npy'
        # self.y_mean = results_root + 'mean_calculation_original/during_{}_{}/during_{}_{}_mean.npy'.format(self.period,self.time_range,self.period,self.variable)
        T.mk_dir(self.result_dir,force=True)
        pass


    def run(self):

        # step 1 build dataframe
        df = self.build_df(self.x_dir,self.y_f,self.period)
        x_var_list = self.__get_x_var_list(self.x_dir,self.period)
        # # # step 2 cal correlation
        # self.cal_multi_regression_beta(df, x_var_list,17)  #修改参数
        self.cal_partial_correlation(df, x_var_list,19)  #修改参数
        # self.cal_PLS(df, x_var_list, 19)  # 修改参数
        # self.max_contribution()
        # self.variables_contribution()



    def __get_x_var_list(self,x_dir,period):
        # x_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        x_f_list = []
        for x_f in T.listdir(x_dir):
            if not period in x_f:
                continue

            x_f_list.append(x_dir + x_f)




        print(x_f_list)
        x_var_list = []
        for x_f in x_f_list:
            split1 = x_f.split('/')[-1]
            split2 = split1.split('.')[0]
            var_name = '_'.join(split2.split('_')[0:-2])
            x_var_list.append(var_name)
        return x_var_list


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


    def build_df(self,x_dir,y_f,period):
        x_f_list = []
        for x_f in T.listdir(x_dir):
            if not period in x_f:
                continue

            x_f_list.append(x_dir + x_f)


        print(x_f_list)
        df = pd.DataFrame()
        y_arr = T.load_npy(y_f)
        pix_list = []
        y_val_list = []
        for pix in y_arr:
            vals = y_arr[pix]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals=vals
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        x_var_list = []
        for x_f in x_f_list:
            # print(x_f)
            split1 = x_f.split('/')[-1]
            split2 = split1.split('.')[0]
            var_name = '_'.join(split2.split('_')[0:-2])
            x_var_list.append(var_name)
            # print(var_name)
            x_val_list = []
            x_arr = T.load_npy(x_f)
            for i,row in tqdm(df.iterrows(),total=len(df),desc=var_name):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)
            # x_val_list = np.array(x_val_list)
            df[var_name] = x_val_list
        # T.print_head_n(df)
        # outexcel = '/Volumes/NVME2T/wen_proj/greening_contribution/1982-2015_extraction_during_late_growing_season_static/test'
        # T.df_to_excel(df,outexcel,n=10000,random=True)
        # exit()
        return df


    def cal_multi_regression_beta(self,df,x_var_list,val_len):
        # mean_dic=T.load_npy(self.y_mean)
        mean_dic= np.load(self.y_mean)

        outf = self.result_f

        multi_derivative={}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 120:
                continue
            y_vals = row['y']
            y_vals=T.remove_np_nan(y_vals)

            if len(y_vals)!=val_len:
                continue
            # print(y_vals)

            y_mean=mean_dic[pix]


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]
                if not len(x_vals) == val_len:  ##
                    continue
                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals= T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0]==None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals   # 不detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1,how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            linear_model = LinearRegression()

            linear_model.fit(df_new[x_var_list_valid_new],df_new['y'])
            coef_ = np.array(linear_model.coef_)/y_mean
            coef_dic = dict(zip(x_var_list_valid_new,coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix]=coef_dic
        T.save_npy(multi_derivative, outf)

    def cal_partial_correlation(self,df,x_var_list,val_len):


        outf1 = self.partial_correlation_result_f
        outf2= self.partial_correlation_p_value_result_f

        partial_correlation_dic={}

        partial_p_value_dic={}

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 120:
                continue
            y_vals = row['y']
            y_vals=T.remove_np_nan(y_vals)

            if len(y_vals)!=val_len:
                continue
            # print(y_vals)


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]
                if not len(x_vals) == val_len:  ##
                    continue
                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals= T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0]==None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals   # 不detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1,how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            partial_correlation={}
            partial_correlation_p_value = {}
            for x in x_var_list_valid_new:
                x_var_list_valid_new_cov=copy.copy(x_var_list_valid_new)
                x_var_list_valid_new_cov.remove(x)
                r,p=self.partial_corr(df_new,x,'y',x_var_list_valid_new_cov)
                partial_correlation[x]=r
                partial_correlation_p_value[x] = p

            partial_correlation_dic[pix]=partial_correlation
            partial_p_value_dic[pix] = partial_correlation_p_value
        T.save_npy(partial_correlation_dic, outf1)
        T.save_npy(partial_p_value_dic, outf2)

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


    def cal_PLS(self,df,x_var_list,val_len):

        outf1 = self.partial_correlation_result_f
        outf2= self.partial_correlation_R2_result_f
        outf3 = self.partial_correlation_VIP_result_f

        partial_correlation_dic={}

        partial_VIP_dic={}

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 120:
                continue
            y_vals = row['y']
            y_vals=T.remove_np_nan(y_vals)

            if len(y_vals)!=val_len:
                continue
            # print(y_vals)


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []
            partial_correlation={}

            partial_correlation_VIP={}

            for x in x_var_list:
                x_vals = row[x]
                if not len(x_vals) == val_len:  ##
                    continue
                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals= T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0]==None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals   # 不detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1,how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            X=df_new[x_var_list_valid_new]
            Y=df_new['y']

            coeff,VIPs=self.PLS_time_series(X,x_var_list_valid_new,Y)
            # print(coeff)
            if coeff is None:
                continue
            coeff=coeff.flatten()
            coeff_dic=dict(zip(x_var_list_valid_new,coeff))
            # print(coeff_dic)
            VIP_dic=dict(zip(x_var_list_valid_new,VIPs))
            # print(VIP_dic)

            partial_correlation_dic[pix]=coeff_dic

            partial_VIP_dic[pix] = VIP_dic


        T.save_npy(partial_correlation_dic, outf1)
        T.save_npy(partial_VIP_dic, outf3)


    def PLS(self, X, x_var_list_valid_new, Y):
        # print(X)
        # print(x_var_list_valid_new)
        # print(Y)
        # RepeatedKFold  p次k折交叉验证
        kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1)


        n_components=2
        mse = []
        n = len(X)

        # Calculate MSE using cross-validation, adding one component at a time
        R2_list=[]
        coef_list=[]
        VIPs_list=[]

        for train_index, test_index in kf.split(X):


            # print('train_index', train_index, 'test_index', test_index)
            if len(test_index)<3:
                continue
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # for i in np.arange(0, len(x_var_list_valid_new)):
        #     pls = PLSRegression(n_components=i)
        #     score = -1 * model_selection.cross_val_score(pls, scale(X), Y, cv=kf,
        #                                                  scoring='neg_mean_squared_error').mean()
        #     mse.append(score)
        #
        #     # plot test MSE vs. number of components
        # plt.plot(mse)
        # plt.xlabel('Number of PLS Components')
        # plt.ylabel('MSE')
        # plt.title('hp')

            pls = PLSRegression(n_components,scale=True, max_iter=500, tol=1e-06, copy=True)
            pls.fit(X_train, Y_train)

            Y_pred = pls.predict(X_test)
            # print(pls.coef_)

            # # Calculate coef
            coef_list.append(pls.coef_)

            # # Calculate scores
            Y_test=np.array(Y_test.tolist())
            Y_test = Y_test.flatten()

            Y_pred = Y_pred.tolist()
            Y_pred = np.array(Y_pred)
            Y_pred = Y_pred.flatten()

            R,p=stats.pearsonr(Y_pred,Y_test)
            R2=R**2


            R2_list.append(R2)

            # # Calculate importance

            x_test_trans=pls.transform(X_test)
            # print(X_test)
            # print(pls.x_rotations_)
            if len(pls.x_rotations_)<2:
                continue
            VIPs=self.compute_VIP(X_test,Y_test,pls.x_rotations_,x_test_trans,n_components)

            VIPs_list.append(VIPs)
            # plt.scatter(np.arange(0,X.shape[1]),VIPs)
            # plt.show()

        VIPs_array = np.array(VIPs_list)
        VIPs_reshape = VIPs_array.reshape(len(x_var_list_valid_new), -1)

        VIPs_majority_list=[]
        for i in VIPs_reshape:
            i[i<1]=0
            i[i > 1] = 1
            count_one=np.count_nonzero(i,axis=0)
            if count_one>=len(i)/2:
                VIPs_majority_list.append(1)
            else:
                VIPs_majority_list.append(0)

        VIPs_majority = np.array(VIPs_majority_list)


        coef_array=np.array(coef_list)
        coef_array_flatten=coef_array.flatten()

        coef_reshape=coef_array_flatten.reshape(len(x_var_list_valid_new),-1)
        print(coef_reshape.shape)

        coeff_mean_list=[]

        for i in coef_reshape:

            mean = np.mean(i)
            print(list(i))
            coeff_mean_list.append(mean)
        coeff_mean = np.array(coeff_mean_list)

        R2=np.mean(R2_list)

        # plt.scatter(Y_test, Y_pred)
        # plt.show()
        # print(R2, coeff_mean, VIPs)
        return R2, coeff_mean, VIPs_majority

    def PLS_time_series(self, X, x_var_list_valid_new, Y): # 不做cross_validation 因为时间序列数据不能拆分
        n_components=2
        pls = PLSRegression(n_components,scale=True, max_iter=500, tol=1e-06, copy=True)
        pls.fit(X, Y)

        x_trans=pls.transform(X)

        if len(pls.x_rotations_)<2:
            return None,None

        VIPs=self.compute_VIP(X,Y,pls.x_rotations_,x_trans,n_components)

        return pls.coef_, VIPs

    def compute_VIP(self,X,Y,R,T,A):
        p=X.shape[1]
        Q2=np.square(np.dot(Y.T,T))
        VIPs=np.zeros(p)
        temp=np.zeros(A)

        for j in range(p):
            for a in range (A):

                temp[a]=Q2[a]*pow(R[j,a]/np.linalg.norm(R[:,a]),2)
            VIPs[j]=np.sqrt(p*np.sum(temp)/np.sum(Q2))
        return VIPs

        pass

    def performance(self,df):
        pass


    def contribution_bar(self,result_f,x_dir):
        y_predicted=[]
        # x_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        result_dic = T.load_npy(result_f)
        x_var_list = self.__get_x_var_list(x_dir,self.period)
        y_trend = []

        for pix in result_dic:
            r,c = pix
            if r > 120:
                continue
            dic_i = result_dic[pix]
            y_predicted_i = dic_i['y_predicted']
            y_trend_i = dic_i['y_trend']
            y_predicted.append(y_predicted_i)
            y_trend.append(y_trend_i)
        KDE_plot().plot_scatter(y_predicted,y_trend)

        plt.figure()
        box = []
        err = []
        for x_var in x_var_list:
            x_var_all = []
            for pix in result_dic:
                r,c = pix
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
        plt.bar(x_var_list,box,)
        plt.show()



    def max_contribution(self):
        x_var_list = self.__get_x_var_list(self.x_dir, self.period, self.time_range)
        x_var_list.sort()
        result_f = self.partial_correlation_result_f
        ourdir =self.result_dir+'/max_correlation/'
        T.mk_dir(ourdir)
        result_dic = T.load_npy(result_f)
        output_dic={}

        color_map = dict(zip(x_var_list,list(range(len(x_var_list)))))
        print(color_map)
        # exit()
        spatial_dic = {}
        var_name_list = []
        for pix in result_dic:
            result = result_dic[pix]
            for var_i in result:
                var_name_list.append(var_i)
            var_name_list = list(set(var_name_list))
            contribution_list = []
            x_var_list_new = []
            for x in var_name_list:
                dic_i = result_dic[pix]
                if not x in dic_i:
                    continue
                contribution_list.append(abs(dic_i[x]))
                x_var_list_new.append(x)
            argsort = np.argsort(contribution_list)[::-1]
            max_x_var = x_var_list_new[argsort[0]]
            # val = x_var_list.index(max_x_var)
            val = color_map[max_x_var]
            spatial_dic[pix] = val
        tif_template = '/Volumes/SSD_sumsang/project_greening/Data/NIRv_tif_05/1982-2018/198205.tif'
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)[:120]
        DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
        # cmap = sns.color_palette('hls',as_cmap=True)
        # plt.imshow(arr,cmap='jet')
        # plt.colorbar()
        # plt.show()

        outf_tif=f'{self.period}_{self.time_range}_max_correlation.tif'
        outf_npy = f'{self.period}_{self.time_range}_max_correlation.npy'

        DIC_and_TIF().arr_to_tif(arr,ourdir+outf_tif)
        output_dic=DIC_and_TIF().spatial_arr_to_dic(arr)
        np.save(ourdir+outf_npy, output_dic)
        pass

    def variables_contribution(self):
        x_var_list = self.__get_x_var_list(self.x_dir, self.period, self.time_range)
        x_var_list.sort()
        # result_f = self.partial_correlation_result_f
        result_f=self.partial_correlation_p_value_result_f
        ourdir = self.result_dir + '/variables_contribution_CSIF/{}/'.format(self.period)
        T.mk_dir(ourdir,force=True)
        result_dic = T.load_npy(result_f)
        output_dic = {}

        # color_map = dict(zip(x_var_list, list(range(len(x_var_list)))))
        # print(color_map)
        # exit()
        spatial_dic = {}
        var_name_list = []
        for pix in result_dic:
            result = result_dic[pix]
            for var_i in result:
                var_name_list.append(var_i)
            var_name_list = list(set(var_name_list))
            var_name_list.sort()


        for x in var_name_list:
            for pix in result_dic:
                if x not in result_dic[pix]:
                    continue
                spatial_dic[pix] = result_dic[pix][x]

            tif_template = '/Volumes/SSD_sumsang/project_greening/Data/NIRv_tif_05/1982-2018/198205.tif'
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)[:120]
            DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
        # cmap = sns.color_palette('hls',as_cmap=True)
        # plt.imshow(arr,cmap='jet')
        # plt.colorbar()
        # plt.show()

            outf_tif = f'{self.time_range}_{x}_p_value.tif'
            outf_npy = f'{self.time_range}_{x}_p_value.npy'

            # outf_tif = f'{self.time_range}_{x}.tif'
            # outf_npy = f'{self.time_range}_{x}.npy'

            DIC_and_TIF().arr_to_tif(arr, ourdir + outf_tif)
            output_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            np.save(ourdir + outf_npy, output_dic)
        pass

    def variables_contribution_window(self):

        fdir=results_root+'trend_window/1982-2015_during_early_window15/'

        for f in (os.listdir(fdir)):
                if not f.endswith('.npy'):
                    continue
                if 'p_value' in f:
                    continue
        # ourdir = self.result_dir + '/variables_contribution_CSIF/{}/'.format(self.period)
        # T.mk_dir(ourdir,force=True)
        result_dic = T.load_npy(fdir+f)
        output_dic = {}

        # color_map = dict(zip(x_var_list, list(range(len(x_var_list)))))
        # print(color_map)
        # exit()
        spatial_dic = {}
        var_name_list = []
        for pix in result_dic:
            result = result_dic[pix]
            for var_i in result:
                var_name_list.append(var_i)
            var_name_list = list(set(var_name_list))
            var_name_list.sort()


        for x in var_name_list:
            for pix in result_dic:
                if x not in result_dic[pix]:
                    continue
                spatial_dic[pix] = result_dic[pix][x]

            tif_template = '/Volumes/SSD_sumsang/project_greening/Data/NIRv_tif_05/1982-2018/198205.tif'
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)[:120]
            DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
        # cmap = sns.color_palette('hls',as_cmap=True)
        # plt.imshow(arr,cmap='jet')
        # plt.colorbar()
        # plt.show()

            outf_tif = f'{self.time_range}_{x}_p_value.tif'
            outf_npy = f'{self.time_range}_{x}_p_value.npy'

            # outf_tif = f'{self.time_range}_{x}.tif'
            # outf_npy = f'{self.time_range}_{x}.npy'

            DIC_and_TIF().arr_to_tif(arr, ourdir + outf_tif)
            output_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            np.save(ourdir + outf_npy, output_dic)
        pass

class Multi_liner_regression_for_Trendy:  # 实现求beta 功能

    def __init__(self):

        self.period='late'
        self.variable_list= ['CABLE-POP_S2_lai', 'CLASSIC_S2_lai', 'CLASSIC-N_S2_lai', 'CLM5', 'IBIS_S2_lai', 'ISAM_S2_LAI',
                     'LPJ-GUESS_S2_lai', 'LPX-Bern_S2_lai', 'OCN_S2_lai',
                     'ORCHIDEE_S2_lai', 'ORCHIDEEv3_S2_lai', 'VISIT_S2_lai', 'YIBs_S2_Monthly_lai', 'ISBA-CTRIP_S2_lai']
        self.time_range='2000-2018'
        self.y_fdir=results_root+f'zscore/monthly/2000-2018_Trendy/'


        self.x_dir = results_root+f'zscore/monthly/2000-2018_X/'



        pass


    def run(self):

        for variable in self.variable_list:
            self.y_f=self.y_fdir+f'{variable}_{self.period}_zscore.npy'
            self.result_dir = results_root + f'partial_correlation_zscore_CO2/{variable}/'
            T.mk_dir(self.result_dir, force=True)

            self.partial_correlation_result_f = self.result_dir+'/{}_partial_correlation_{}_{}.npy'.format(self.time_range,self.period, variable)
            self.partial_correlation_p_value_result_f = self.result_dir + '/{}_partial_correlation_p_value_result_{}_{}.npy'.format(
                self.time_range, self.period, variable)

            # step 1 build dataframe
            df = self.build_df(self.x_dir,self.y_f,self.period)
            x_var_list = self.__get_x_var_list(self.x_dir,self.period)
            # # # step 2 cal correlation
            # self.cal_multi_regression_beta(df, x_var_list,17)  #修改参数
            self.cal_partial_correlation(df, x_var_list,19)  #修改参数
            # self.cal_PLS(df, x_var_list, 19)  # 修改参数
            # self.max_contribution()
            # self.variables_contribution()



    def __get_x_var_list(self,x_dir,period):
        # x_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        x_f_list = []
        for x_f in T.listdir(x_dir):
            if not period in x_f:
                continue

            x_f_list.append(x_dir + x_f)




        print(x_f_list)
        x_var_list = []
        for x_f in x_f_list:
            split1 = x_f.split('/')[-1]
            split2 = split1.split('.')[0]
            var_name = '_'.join(split2.split('_')[0:-2])
            # if 'CO2' in var_name:
            #     continue
            x_var_list.append(var_name)
        return x_var_list


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


    def build_df(self,x_dir,y_f,period):
        x_f_list = []
        for x_f in T.listdir(x_dir):
            if not period in x_f:
                continue

            x_f_list.append(x_dir + x_f)


        print(x_f_list)
        df = pd.DataFrame()
        y_arr = T.load_npy(y_f)
        pix_list = []
        y_val_list = []
        for pix in y_arr:
            vals = y_arr[pix]
            # print(vals)
            # exit()
            if len(vals) == 0:
                continue
            vals = np.array(vals)
            vals=vals
            pix_list.append(pix)
            y_val_list.append(vals)
        df['pix'] = pix_list
        df['y'] = y_val_list

        x_var_list = []
        for x_f in x_f_list:
            # print(x_f)
            split1 = x_f.split('/')[-1]
            split2 = split1.split('.')[0]
            var_name = '_'.join(split2.split('_')[0:-2])
            # if 'CO2' in var_name:
            #     continue
            x_var_list.append(var_name)
            # print(var_name)
            x_val_list = []
            x_arr = T.load_npy(x_f)
            for i,row in tqdm(df.iterrows(),total=len(df),desc=var_name):
                pix = row.pix
                if not pix in x_arr:
                    x_val_list.append([])
                    continue
                vals = x_arr[pix]
                vals = np.array(vals)
                if len(vals) == 0:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)
            # x_val_list = np.array(x_val_list)
            df[var_name] = x_val_list
        # T.print_head_n(df)
        # outexcel = '/Volumes/NVME2T/wen_proj/greening_contribution/1982-2015_extraction_during_late_growing_season_static/test'
        # T.df_to_excel(df,outexcel,n=10000,random=True)
        # exit()
        return df


    def cal_multi_regression_beta(self,df,x_var_list,val_len):
        # mean_dic=T.load_npy(self.y_mean)
        mean_dic= np.load(self.y_mean)

        outf = self.result_f

        multi_derivative={}
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 120:
                continue
            y_vals = row['y']
            y_vals=T.remove_np_nan(y_vals)

            if len(y_vals)!=val_len:
                continue
            # print(y_vals)

            y_mean=mean_dic[pix]


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]
                if not len(x_vals) == val_len:  ##
                    continue
                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals= T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0]==None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals   # 不detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1,how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            linear_model = LinearRegression()

            linear_model.fit(df_new[x_var_list_valid_new],df_new['y'])
            coef_ = np.array(linear_model.coef_)/y_mean
            coef_dic = dict(zip(x_var_list_valid_new,coef_))
            # print(df_new['y'])
            # exit()
            multi_derivative[pix]=coef_dic
        T.save_npy(multi_derivative, outf)

    def cal_partial_correlation(self,df,x_var_list,val_len):


        outf1 = self.partial_correlation_result_f
        outf2= self.partial_correlation_p_value_result_f

        partial_correlation_dic={}

        partial_p_value_dic={}

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 120:
                continue
            y_vals = row['y']
            y_vals=T.remove_np_nan(y_vals)

            if len(y_vals)!=val_len:
                continue
            # print(y_vals)


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []

            for x in x_var_list:
                x_vals = row[x]
                if not len(x_vals) == val_len:  ##
                    continue
                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals= T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0]==None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals   # 不detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1,how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            partial_correlation={}
            partial_correlation_p_value = {}
            for x in x_var_list_valid_new:
                x_var_list_valid_new_cov=copy.copy(x_var_list_valid_new)
                x_var_list_valid_new_cov.remove(x)
                r,p=self.partial_corr(df_new,x,'y',x_var_list_valid_new_cov)
                partial_correlation[x]=r
                partial_correlation_p_value[x] = p

            partial_correlation_dic[pix]=partial_correlation
            partial_p_value_dic[pix] = partial_correlation_p_value
        T.save_npy(partial_correlation_dic, outf1)
        T.save_npy(partial_p_value_dic, outf2)

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


    def cal_PLS(self,df,x_var_list,val_len):

        outf1 = self.partial_correlation_result_f
        outf2= self.partial_correlation_R2_result_f
        outf3 = self.partial_correlation_VIP_result_f

        partial_correlation_dic={}

        partial_VIP_dic={}

        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row.pix
            r,c = pix
            if r > 120:
                continue
            y_vals = row['y']
            y_vals=T.remove_np_nan(y_vals)

            if len(y_vals)!=val_len:
                continue
            # print(y_vals)


            #  calculate partial derivative with multi-regression
            df_new = pd.DataFrame()
            x_var_list_valid = []
            partial_correlation={}

            partial_correlation_VIP={}

            for x in x_var_list:
                x_vals = row[x]
                if not len(x_vals) == val_len:  ##
                    continue
                if len(x_vals) == 0:
                    continue

                if np.isnan(np.nanmean(x_vals)):
                    continue
                x_vals= T.interp_nan(x_vals)
                # print(x_vals)
                if x_vals[0]==None:
                    continue
                # x_vals_detrend = signal.detrend(x_vals) #detrend
                df_new[x] = x_vals
                # df_new[x] = x_vals_detrend   #detrend

                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                continue

            df_new['y'] = y_vals   # 不detrend

            # T.print_head_n(df_new)
            df_new = df_new.dropna(axis=1,how='all')
            x_var_list_valid_new = []
            for v_ in x_var_list_valid:
                if not v_ in df_new:
                    continue
                else:
                    x_var_list_valid_new.append(v_)
            # T.print_head_n(df_new)

            df_new = df_new.dropna()
            X=df_new[x_var_list_valid_new]
            Y=df_new['y']

            coeff,VIPs=self.PLS_time_series(X,x_var_list_valid_new,Y)
            # print(coeff)
            if coeff is None:
                continue
            coeff=coeff.flatten()
            coeff_dic=dict(zip(x_var_list_valid_new,coeff))
            # print(coeff_dic)
            VIP_dic=dict(zip(x_var_list_valid_new,VIPs))
            # print(VIP_dic)

            partial_correlation_dic[pix]=coeff_dic

            partial_VIP_dic[pix] = VIP_dic


        T.save_npy(partial_correlation_dic, outf1)
        T.save_npy(partial_VIP_dic, outf3)


    def PLS(self, X, x_var_list_valid_new, Y):
        # print(X)
        # print(x_var_list_valid_new)
        # print(Y)
        # RepeatedKFold  p次k折交叉验证
        kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=1)


        n_components=2
        mse = []
        n = len(X)

        # Calculate MSE using cross-validation, adding one component at a time
        R2_list=[]
        coef_list=[]
        VIPs_list=[]

        for train_index, test_index in kf.split(X):


            # print('train_index', train_index, 'test_index', test_index)
            if len(test_index)<3:
                continue
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # for i in np.arange(0, len(x_var_list_valid_new)):
        #     pls = PLSRegression(n_components=i)
        #     score = -1 * model_selection.cross_val_score(pls, scale(X), Y, cv=kf,
        #                                                  scoring='neg_mean_squared_error').mean()
        #     mse.append(score)
        #
        #     # plot test MSE vs. number of components
        # plt.plot(mse)
        # plt.xlabel('Number of PLS Components')
        # plt.ylabel('MSE')
        # plt.title('hp')

            pls = PLSRegression(n_components,scale=True, max_iter=500, tol=1e-06, copy=True)
            pls.fit(X_train, Y_train)

            Y_pred = pls.predict(X_test)
            # print(pls.coef_)

            # # Calculate coef
            coef_list.append(pls.coef_)

            # # Calculate scores
            Y_test=np.array(Y_test.tolist())
            Y_test = Y_test.flatten()

            Y_pred = Y_pred.tolist()
            Y_pred = np.array(Y_pred)
            Y_pred = Y_pred.flatten()

            R,p=stats.pearsonr(Y_pred,Y_test)
            R2=R**2


            R2_list.append(R2)

            # # Calculate importance

            x_test_trans=pls.transform(X_test)
            # print(X_test)
            # print(pls.x_rotations_)
            if len(pls.x_rotations_)<2:
                continue
            VIPs=self.compute_VIP(X_test,Y_test,pls.x_rotations_,x_test_trans,n_components)

            VIPs_list.append(VIPs)
            # plt.scatter(np.arange(0,X.shape[1]),VIPs)
            # plt.show()

        VIPs_array = np.array(VIPs_list)
        VIPs_reshape = VIPs_array.reshape(len(x_var_list_valid_new), -1)

        VIPs_majority_list=[]
        for i in VIPs_reshape:
            i[i<1]=0
            i[i > 1] = 1
            count_one=np.count_nonzero(i,axis=0)
            if count_one>=len(i)/2:
                VIPs_majority_list.append(1)
            else:
                VIPs_majority_list.append(0)

        VIPs_majority = np.array(VIPs_majority_list)


        coef_array=np.array(coef_list)
        coef_array_flatten=coef_array.flatten()

        coef_reshape=coef_array_flatten.reshape(len(x_var_list_valid_new),-1)
        print(coef_reshape.shape)

        coeff_mean_list=[]

        for i in coef_reshape:

            mean = np.mean(i)
            print(list(i))
            coeff_mean_list.append(mean)
        coeff_mean = np.array(coeff_mean_list)

        R2=np.mean(R2_list)

        # plt.scatter(Y_test, Y_pred)
        # plt.show()
        # print(R2, coeff_mean, VIPs)
        return R2, coeff_mean, VIPs_majority

    def PLS_time_series(self, X, x_var_list_valid_new, Y): # 不做cross_validation 因为时间序列数据不能拆分
        n_components=2
        pls = PLSRegression(n_components,scale=True, max_iter=500, tol=1e-06, copy=True)
        pls.fit(X, Y)

        x_trans=pls.transform(X)

        if len(pls.x_rotations_)<2:
            return None,None

        VIPs=self.compute_VIP(X,Y,pls.x_rotations_,x_trans,n_components)

        return pls.coef_, VIPs

    def compute_VIP(self,X,Y,R,T,A):
        p=X.shape[1]
        Q2=np.square(np.dot(Y.T,T))
        VIPs=np.zeros(p)
        temp=np.zeros(A)

        for j in range(p):
            for a in range (A):

                temp[a]=Q2[a]*pow(R[j,a]/np.linalg.norm(R[:,a]),2)
            VIPs[j]=np.sqrt(p*np.sum(temp)/np.sum(Q2))
        return VIPs

        pass

    def performance(self,df):
        pass


    def contribution_bar(self,result_f,x_dir):
        y_predicted=[]
        # x_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        result_dic = T.load_npy(result_f)
        x_var_list = self.__get_x_var_list(x_dir,self.period)
        y_trend = []

        for pix in result_dic:
            r,c = pix
            if r > 120:
                continue
            dic_i = result_dic[pix]
            y_predicted_i = dic_i['y_predicted']
            y_trend_i = dic_i['y_trend']
            y_predicted.append(y_predicted_i)
            y_trend.append(y_trend_i)
        KDE_plot().plot_scatter(y_predicted,y_trend)

        plt.figure()
        box = []
        err = []
        for x_var in x_var_list:
            x_var_all = []
            for pix in result_dic:
                r,c = pix
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
        plt.bar(x_var_list,box,)
        plt.show()



    def max_contribution(self):
        x_var_list = self.__get_x_var_list(self.x_dir, self.period, self.time_range)
        x_var_list.sort()
        result_f = self.partial_correlation_result_f
        ourdir =self.result_dir+'/max_correlation/'
        T.mk_dir(ourdir)
        result_dic = T.load_npy(result_f)
        output_dic={}

        color_map = dict(zip(x_var_list,list(range(len(x_var_list)))))
        print(color_map)
        # exit()
        spatial_dic = {}
        var_name_list = []
        for pix in result_dic:
            result = result_dic[pix]
            for var_i in result:
                var_name_list.append(var_i)
            var_name_list = list(set(var_name_list))
            contribution_list = []
            x_var_list_new = []
            for x in var_name_list:
                dic_i = result_dic[pix]
                if not x in dic_i:
                    continue
                contribution_list.append(abs(dic_i[x]))
                x_var_list_new.append(x)
            argsort = np.argsort(contribution_list)[::-1]
            max_x_var = x_var_list_new[argsort[0]]
            # val = x_var_list.index(max_x_var)
            val = color_map[max_x_var]
            spatial_dic[pix] = val
        tif_template = '/Volumes/SSD_sumsang/project_greening/Data/NIRv_tif_05/1982-2018/198205.tif'
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)[:120]
        DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
        # cmap = sns.color_palette('hls',as_cmap=True)
        # plt.imshow(arr,cmap='jet')
        # plt.colorbar()
        # plt.show()

        outf_tif=f'{self.period}_{self.time_range}_max_correlation.tif'
        outf_npy = f'{self.period}_{self.time_range}_max_correlation.npy'

        DIC_and_TIF().arr_to_tif(arr,ourdir+outf_tif)
        output_dic=DIC_and_TIF().spatial_arr_to_dic(arr)
        np.save(ourdir+outf_npy, output_dic)
        pass

    def variables_contribution(self):
        x_var_list = self.__get_x_var_list(self.x_dir, self.period, self.time_range)
        x_var_list.sort()
        # result_f = self.partial_correlation_result_f
        result_f=self.partial_correlation_p_value_result_f
        ourdir = self.result_dir + '/variables_contribution_CSIF/{}/'.format(self.period)
        T.mk_dir(ourdir,force=True)
        result_dic = T.load_npy(result_f)
        output_dic = {}

        # color_map = dict(zip(x_var_list, list(range(len(x_var_list)))))
        # print(color_map)
        # exit()
        spatial_dic = {}
        var_name_list = []
        for pix in result_dic:
            result = result_dic[pix]
            for var_i in result:
                var_name_list.append(var_i)
            var_name_list = list(set(var_name_list))
            var_name_list.sort()


        for x in var_name_list:
            for pix in result_dic:
                if x not in result_dic[pix]:
                    continue
                spatial_dic[pix] = result_dic[pix][x]

            tif_template = '/Volumes/SSD_sumsang/project_greening/Data/NIRv_tif_05/1982-2018/198205.tif'
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)[:120]
            DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
        # cmap = sns.color_palette('hls',as_cmap=True)
        # plt.imshow(arr,cmap='jet')
        # plt.colorbar()
        # plt.show()

            outf_tif = f'{self.time_range}_{x}_p_value.tif'
            outf_npy = f'{self.time_range}_{x}_p_value.npy'

            # outf_tif = f'{self.time_range}_{x}.tif'
            # outf_npy = f'{self.time_range}_{x}.npy'

            DIC_and_TIF().arr_to_tif(arr, ourdir + outf_tif)
            output_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            np.save(ourdir + outf_npy, output_dic)
        pass

    def variables_contribution_window(self):

        fdir=results_root+'trend_window/1982-2015_during_early_window15/'

        for f in (os.listdir(fdir)):
                if not f.endswith('.npy'):
                    continue
                if 'p_value' in f:
                    continue
        # ourdir = self.result_dir + '/variables_contribution_CSIF/{}/'.format(self.period)
        # T.mk_dir(ourdir,force=True)
        result_dic = T.load_npy(fdir+f)
        output_dic = {}

        # color_map = dict(zip(x_var_list, list(range(len(x_var_list)))))
        # print(color_map)
        # exit()
        spatial_dic = {}
        var_name_list = []
        for pix in result_dic:
            result = result_dic[pix]
            for var_i in result:
                var_name_list.append(var_i)
            var_name_list = list(set(var_name_list))
            var_name_list.sort()


        for x in var_name_list:
            for pix in result_dic:
                if x not in result_dic[pix]:
                    continue
                spatial_dic[pix] = result_dic[pix][x]

            tif_template = '/Volumes/SSD_sumsang/project_greening/Data/NIRv_tif_05/1982-2018/198205.tif'
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)[:120]
            DIC_and_TIF().plot_back_ground_arr_north_sphere(tif_template)
        # cmap = sns.color_palette('hls',as_cmap=True)
        # plt.imshow(arr,cmap='jet')
        # plt.colorbar()
        # plt.show()

            outf_tif = f'{self.time_range}_{x}_p_value.tif'
            outf_npy = f'{self.time_range}_{x}_p_value.npy'

            # outf_tif = f'{self.time_range}_{x}.tif'
            # outf_npy = f'{self.time_range}_{x}.npy'

            DIC_and_TIF().arr_to_tif(arr, ourdir + outf_tif)
            output_dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            np.save(ourdir + outf_npy, output_dic)
        pass


class plot_partial_plot():
    class Unify_date_range:
        pass

    def run(self):
        period='early'
        time_range='1982-2001'
        f = results_root+'partial_correlation/LAI_GIMMS/{}_partial_correlation_{}_anomaly_LAI_GIMMS.npy'.format(time_range,period)
        # f=results_root+'partial_correlation/LAI_GIMMS/{}_partial_correlation_p_value_{}_anomaly_LAI_GIMMS.npy'.format(time_range,period)
        self.beta_save_(f,time_range,period)


    def __init__(self):
        # self.check_data_length()
        pass

    def beta_save_(self,f,time_range,period):  # 该功能实现所有因素的beta

        outdir=results_root+'partial_correlation/LAI_GIMMS_partial_correlation_individual/{}_{}/'.format(time_range,period)
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
            np.save(outdir + var_i + '_corr', arr)
            # np.save(outdir + var_i+'_p_value',arr)
            DIC_and_TIF().arr_to_tif(arr, outdir + var_i + '_corr.tif')
            # DIC_and_TIF().arr_to_tif(arr,outdir+var_i+'_p_value.tif')
            std = np.nanstd(arr)
            mean = np.nanmean(arr)
            vmin = mean - std
            vmax = mean + std
        #     plt.figure()
        #     plt.imshow(arr, vmin=vmin, vmax=vmax)
        #     plt.title(var_i)
        #     plt.colorbar()
        # plt.show()



    def check_data_length(self):
        fdir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/1982-2015/X_1982-2015/'
        template_file = '/Volumes/NVME2T/drought_legacy_new/conf/tif_template.tif'
        for f in T.listdir(fdir):
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



class Window_correlation:

    def __init__(self):
        # self.x_f = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/1982-2000/X_1982-2000/'
        # self.y_f = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/1982-2000/Y_1982-2000/during_early_GIMMS_NDVI.npy'

        self.x_f = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/1982-2015/X_1982-2015/during_early_Temperature.npy'
        # self.y_f = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/1982-2015/Y_1982-2015/during_early_NIRv.npy'
        self.y_f = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/1982-2015/Y_1982-2015/during_early_GIMMS_NDVI.npy'
        pass


    def run(self):
        self.foo()
        pass

    def __split_x_var(self,x_f):
        # x_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        split1 = x_f.split('/')[-1]
        split2 = split1.split('.')[0]
        var_name = '_'.join(split2.split('_')[2:])
        return var_name


    def foo(self):
        x_f = self.x_f
        y_f = self.y_f
        x_name = self.__split_x_var(x_f)
        y_name = self.__split_x_var(y_f)
        window = 15
        x_dic = T.load_npy(x_f)
        y_dic = T.load_npy(y_f)

        corr_dic = {}
        last_pix = ''
        one_pix = (15,222)
        for pix in tqdm(x_dic):
            # if not pix == one_pix:
            #     continue
            r,c = pix
            if r > 120:
                continue
            if not pix in y_dic:
                continue
            x = x_dic[pix]
            y = y_dic[pix]
            if np.std(x) == 0:
                continue
            if np.std(y) < 0.000001:
                continue
            x_detrend = signal.detrend(x)
            y_detrend = signal.detrend(y)
            # plt.plot(x_detrend)
            # plt.plot(y_detrend)
            # plt.grid(1)
            # plt.show()
            time_series,corr_list = T.slide_window_correlation(x_detrend,y_detrend,window)
            corr_dic[pix] = (corr_list,time_series)
            last_pix = pix
        # last_pix = one_pix
        corr_mean_list = []
        corr_std_list = []
        for i in tqdm(range(len(corr_dic[last_pix][0]))):
            corr_list = []
            spatial_dic = {}
            # print(corr_dic)
            for pix in tqdm(corr_dic):
                corr = corr_dic[pix][0][i]
                corr_list.append(corr)
                spatial_dic[pix] = corr
                # print(corr)
            arr = DIC_and_TIF(Global_vars().tif_template).pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr,cmap='jet')
            plt.colorbar()
            plt.show()
            corr_mean = np.nanmean(corr_list)
            corr_std = np.nanstd(corr_list)
            corr_mean_list.append(corr_mean)
            corr_std_list.append(corr_std/8.)
        corr_mean_list = np.array(corr_mean_list)
        corr_std_list = np.array(corr_std_list)
        time_series_list = corr_dic[last_pix][1]
        time_series_list = np.array(time_series_list)
        time_series_list = time_series_list + 1982
        time_series_list = [int(j) for j in time_series_list]
        plt.plot(time_series_list,corr_mean_list)
        plt.fill_between(time_series_list,corr_mean_list-corr_std_list,corr_mean_list+corr_std_list,alpha=0.2,zorder=-99)
        plt.title('{} vs {}\n {} year window correlation'.format(x_name,y_name,window))
        plt.xticks(time_series_list[::3],time_series_list[::3],rotation=90)
        plt.xlabel('Year')
        plt.ylabel('Pearson correlation')
        plt.tight_layout()
        plt.show()


class Sankey_plot_PLS:

    def __init__(self):
        # self.Y_name = 'LAI3g'
        self.Y_name = 'LAI4g'
        # self.Y_name = 'MODIS-LAI'
        self.var_list = ['CCI_SM', 'CO2', 'PAR', 'Temp', 'VPD']
        self.period_list = ['early', 'peak', 'late']
        self.fdir = results_root+f'Sankey_plot/Data/{self.Y_name}/'
        self.this_class_arr=results_root+f'Sankey_plot/Results/'
        outdir = join(self.this_class_arr,f'{self.Y_name}')
        Tools().mk_dir(outdir,force=True)
        self.dff = join(outdir,'dataframe.df')
        self.this_class_png=join(outdir,'png')
        Tools().mk_dir(self.this_class_arr, force=True)
        # T.open_path_and_file(outdir)

        pass

    def run(self):
        # self.plot_p_value_spatial()
        # df,var_list = self.join_dataframe(self.fdir)
        # df = self.build_sankey_plot(df,var_list)
        # #
        df = self.__gen_df_init()
        # df = green_driver_trend_contribution.Build_dataframe().add_Humid_nonhumid(df)
        # 加字段 NDVI mask,
        # df=self.add_max_trend_to_df(df)
        # df=self.add_landcover_data_to_df(df)
        #
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)

        ## 加筛选条件
        df=df[df['max_lc_trend']<5]
        # df = df[df['late_CO2_VIP'] >1 ]
        df = df[df['landcover'] != 'Cropland']

        self.plot_Sankey(df,True)

        self.plot_Sankey(df,False)
        # self.plot_max_corr_spatial(df)
        # self.max_contribution_bar()
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        return df,dff

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def plot_max_corr_spatial(self,df):
        outdir = join(self.this_class_tif,'plot_max_corr_spatial')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        period_list = self.period_list
        var_list = self.var_list
        var_color_dict = {'CCI_SM':'#ff7f0e','CO2':'#1f77b4','PAR':'#2ca02c','Temp':'#9467bd','VPD':'#d62728'}
        var_value_dict = {'CCI_SM':0,'CO2':1,'PAR':2,'Temp':3,'VPD':4,'None':np.nan,'nan':np.nan}
        color_list = []
        for var_ in var_list:
            color_list.append(var_color_dict[var_])
        cmap = T.cmap_blend(color_list)
        for period in period_list:
            col_name = f'{period}_max_var'
            spatial_dict = T.df_to_spatial_dic(df,col_name)
            spatial_dict_value = {}
            for pix in spatial_dict:
                var_ = spatial_dict[pix]
                var_ = str(var_)
                var_ = var_.replace(period+'_','')
                # print(var_)
                value = var_value_dict[var_]
                spatial_dict_value[pix] = value
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_value)
            arr = arr[:180]
            # plt.figure()
            # DIC_and_TIF().plot_back_ground_arr(global_land_tif,aspect='auto')
            # plt.imshow(arr, cmap=cmap,aspect='auto')
            # plt.colorbar()
            # plt.title(f'{self.Y_name}_{period}')
            # T.plot_colors_palette(cmap)
            outf = join(outdir,f'{self.Y_name}_{period}.tif')
            DIC_and_TIF().arr_to_tif(arr,outf)
        # plt.show()

    def join_dataframe(self,fdir):

        df_list = []
        var_list = []
        for f in T.listdir(fdir):
            # print(f)
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir,f))
            df_i = T.dic_to_df(dic,key_col_str='pix')
            old_col_list = []
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                old_col_list.append(col)
                var_list.append(col)
            if 'VIP' in f:
                new_col_list = [f'{period}_{col}_VIP' for col in old_col_list]
            else:
                new_col_list = [f'{period}_{col}' for col in old_col_list]
            for i in range(len(old_col_list)):
                new_name = new_col_list[i]
                old_name = old_col_list[i]
                df_i = T.rename_dataframe_columns(df_i,old_name,new_name)
            df_list.append(df_i)
        df = pd.DataFrame()
        df = Tools().join_df_list(df,df_list,'pix')
        ## re-index dataframe
        df = df.reset_index(drop=True)
        return df,var_list

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

    def add_max_trend_to_df(self,df):

        f = results_root + f'lc_trend/max_trend.npy'

        val_array = np.load( f)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        f_name = 'max_lc_trend'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]*20
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df


    def build_sankey_plot(self,df,var_list,p_threshold=0.1):

        period_list = ['early','peak','late']
        for period in period_list:
            var_name_list = []
            for var_ in var_list:
                var_name_corr = f'{period}_{var_}'
                var_name_list.append(var_name_corr)
            for i,row in tqdm(df.iterrows(),total=len(df)):
                value_dict = {}
                for var_ in var_name_list:
                    value = row[var_]
                    value = abs(value)
                    value_dict[var_] = value
                max_key = T.get_max_key_from_dict(value_dict)
                df.loc[i,f'{period}_max_var'] = max_key
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)


    def __get_var_list(self,fdir):
        var_list = []
        for f in T.listdir(fdir):
            period = f.split('.')[0].split('_')[-2]
            dic = T.load_npy(join(fdir, f))
            df_i = T.dic_to_df(dic, key_col_str='pix')
            var_list = []
            for col in df_i.columns:
                if col == 'pix':
                    continue
                var_list.append(col)
            break
        return var_list

    def __add_alpha_to_color(self,hexcolor,alpha=0.8):
        rgb = T.hex_color_to_rgb(hexcolor)
        rgb = list(rgb)
        rgb[-1] = alpha
        rgb = tuple(rgb)
        rgb_str = 'rgba'+str(rgb)
        return rgb_str

    def plot_Sankey(self,df,ishumid):
        if ishumid:
            df = df[df['HI_reclass'] == 'Humid']
            outdir = join(self.this_class_png, f'{self.Y_name}/Humid')
            title = 'Humid'
        else:
            df = df[df['HI_reclass'] == 'Dryland']
            outdir = join(self.this_class_png, f'{self.Y_name}/Dryland')
            title = 'Dryland'
        T.mkdir(outdir,force=True)
        # T.open_path_and_file(outdir)
        var_list = self.__get_var_list(self.fdir)

        period_list = ['early','peak','late']
        # period_list = ['late']
        early_max_var_list = [f'early_{var}' for var in var_list]
        peak_max_var_list = [f'peak_{var}' for var in var_list]
        late_max_var_list = [f'late_{var}' for var in var_list]

        node_list = early_max_var_list+peak_max_var_list+late_max_var_list
        position_dict = dict(zip(node_list, range(len(node_list))))

        color_dict = {'CO2': self.__add_alpha_to_color('#00FF00'),
                      'CCI_SM': self.__add_alpha_to_color('#00E7FF'),
                      'PAR': self.__add_alpha_to_color('#FFFF00'),
                      'Temp': self.__add_alpha_to_color('#FF0000'),
                      'VPD': self.__add_alpha_to_color('#B531AF'),
                      }
        node_color_list = [color_dict[var_] for var_ in var_list]
        node_color_list = node_color_list * 3
        # print(node_color_list)
        # exit()
        early_max_var_col = 'early_max_var'
        peak_max_var_col = 'peak_max_var'
        late_max_var_col = 'late_max_var'

        source = []
        target = []
        value = []
        # color_list = []
        # node_list_anomaly_value_mean = []
        # anomaly_value_list = []
        node_list_with_ratio = []
        node_name_list = []
        for early_status in early_max_var_list:
            # print(early_status)
            # exit()
            df_early = df[df[early_max_var_col] == early_status]
            vals = df_early[early_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_early)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{early_status} {vals_mean:.2f}')
        for peak_status in peak_max_var_list:
            df_peak = df[df[peak_max_var_col] == peak_status]
            vals = df_peak[peak_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_peak)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{peak_status} {vals_mean:.2f}')
        for late_status in late_max_var_list:
            df_late = df[df[late_max_var_col] == late_status]
            vals = df_late[late_status].tolist()
            vals_mean = np.nanmean(vals)
            ratio = len(df_late)/len(df)
            node_list_with_ratio.append(ratio)
            node_name_list.append(f'{late_status} {vals_mean:.2f}')
        node_list_with_ratio = [round(i,3) for i in node_list_with_ratio]

        for early_status in early_max_var_list:
            df_early = df[df[early_max_var_col] == early_status]
            early_count = len(df_early)
            for peak_status in peak_max_var_list:
                df_peak = df_early[df_early[peak_max_var_col] == peak_status]
                peak_count = len(df_peak)
                source.append(position_dict[early_status])
                target.append(position_dict[peak_status])
                value.append(peak_count)
                for late_status in late_max_var_list:
                    df_late = df_peak[df_peak[late_max_var_col] == late_status]
                    late_count = len(df_late)
                    source.append(position_dict[peak_status])
                    target.append(position_dict[late_status])
                    value.append(late_count)
        link = dict(source=source, target=target, value=value,)
        # node = dict(label=node_list_with_ratio, pad=100,
        node = dict(label=node_name_list, pad=100,
                    thickness=15,
                    line=dict(color="black", width=0.5),
                    # x=node_x,
                    # y=node_y,
                    color=node_color_list
                )
        data = go.Sankey(link=link, node=node, arrangement='snap', textfont=dict(color="rgba(0,0,0,1)", size=18))
        fig = go.Figure(data)
        fig.update_layout(title_text=f'{title}')
        fig.write_html(join(outdir, f'{title}.html'))
        # fig.write_image(join(outdir, f'{title}.png'))
        # fig.show()


    def plot_p_value_spatial(self):
        fdir = self.fdir
        var_ = 'Temp'
        # var_ = 'CCI_SM'
        period = 'early'
        f = join(fdir, '2000-2018_partial_correlation_p_value_early_LAI3g.npy')
        dict_ = T.load_npy(f)
        spatial_dict = {}
        for pix in tqdm(dict_):
            dict_i = dict_[pix]
            if not var_ in dict_i:
                continue
            value = dict_i[var_]
            spatial_dict[pix] = value
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict)
        arr = arr[:180]
        arr[arr>0.1] = 1
        plt.imshow(arr, cmap='jet',aspect='auto')
        plt.colorbar()
        DIC_and_TIF().plot_back_ground_arr_north_sphere(global_land_tif,aspect='auto')
        plt.title(f'{var_}_{period}')
        plt.show()

        pass

    def max_contribution_bar(self):
        fdir = '/Volumes/NVME2T/greening_project_redo/results/Main_flow_1/tif/Sankey_plot_max_contribution/plot_max_corr_spatial'
        outdir = join(self.this_class_png, 'max_contribution_bar')
        T.mkdir(outdir)
        T.open_path_and_file(outdir)
        var_value_dict = {'CCI_SM': 0, 'CO2': 1, 'PAR': 2, 'Temp': 3, 'VPD': 4}
        color_dict = {'CCI_SM': '#00E7FF', 'CO2': '#00FF00', 'PAR': '#FFFF00', 'Temp': '#FF0000', 'VPD': '#B531AF'}
        var_value_dict_reverse = T.reverse_dic(var_value_dict)
        all_dict = {}
        product_list = []
        period_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fname = f.split('.')[0]
            product,period = fname.split('_')
            product_list.append(product)
            period_list.append(period)
            arr = ToRaster().raster2array(join(fdir, f))[0]
            arr = T.mask_999999_arr(arr,warning=False)
            dic = DIC_and_TIF().spatial_arr_to_dic(arr)
            key = f'{product}_{period}'
            all_dict[key] = dic
        df = T.spatial_dics_to_df(all_dict)
        df = Dataframe().add_Humid_nonhumid(df)
        humid_var = 'HI_reclass'
        humid_list = T.get_df_unique_val_list(df, humid_var)
        for humid in humid_list:
            df_humid = df[df[humid_var] == humid]
            fig,axs = plt.subplots(3,3,figsize=(10,10))
            product_list = list(set(product_list))
            period_list = ['early','peak','late']
            for m,product in enumerate(product_list):
                for n,period in enumerate(period_list):
                    col = f'{product}_{period}'
                    df_i = df_humid[col]
                    df_i = df_i.dropna()
                    unique_value = T.get_df_unique_val_list(df_humid,col)
                    x_list = []
                    y_list = []
                    color_list = []
                    for val in unique_value:
                        df_i_i = df_i[df_i == val]
                        ratio = len(df_i_i)/len(df_i) * 100
                        x = var_value_dict_reverse[int(val)][0]
                        x_list.append(x)
                        y_list.append(ratio)
                        color_list.append(color_dict[x])
                    # plt.figure()
                    # print(m,n)
                    axs[m][n].bar(x_list,y_list,color=color_list)
                    axs[m][n].set_title(f'{product}_{period}')
                    axs[m][n].set_ylim(0,47)
            plt.suptitle(f'{humid}')
            plt.tight_layout()
            outf = join(outdir, f'{humid}.pdf')
            plt.savefig(outf)
            plt.close()

        pass

class Phenology:

    def __init__(self):

        pass
    def run(self):
        self.check_get_early_peak_late_dormant_period_long_term()


    def check_get_early_peak_late_dormant_period_long_term(self):
        fdir = '/Volumes/NVME2T/wen_proj/greening_contribution/20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/'
        out_png_dir = '/Volumes/NVME2T/wen_proj/greening_contribution/check_phenology/'
        for var in T.listdir(fdir):
            print(var)
            dic = T.load_npy(fdir + var)
            spatial_dic = {}
            for pix in tqdm(dic):
                val = dic[pix]
                if len(val) == 0:
                    continue
                # plt.plot(val)
                # plt.show()
                meanarr = np.nanmean(val)
                spatial_dic[pix] = meanarr
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
            plt.imshow(arr)
            plt.title(var)
            plt.colorbar()
            plt.savefig(out_png_dir + var + '.png')
            plt.close()
            # plt.show()

        pass

def check_vod():
    fdir = '/Volumes/NVME2T/project05_redo/data/VOD/per_pix/'
    dic = T.load_npy_dir(fdir)
    for pix in dic:
        vals = dic[pix]
        mean = np.nanmean(vals)
        if np.isnan(mean):
            continue
        print(len(vals))
        plt.plot(vals)
        plt.show()
        print(vals)
    pass

def main():

    # Make_Dataframe().run()
    # Main_flow_shui_re().run()
    # Greening().run()
    # Multi_colormap_spatial_map().run()
    # Unify_date_range().run()
    # check_NIRV_NDVI().run()
    # Linear_contribution().run()
    # Multi_liner_regression().run()
    Multi_liner_regression_for_Trendy().run()
    # plot_partial_plot().run()
    # Sankey_plot_PLS().run()

    # Window_correlation().run()
    # check_vod()
    pass


if __name__ == '__main__':
    main()



