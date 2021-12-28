# coding=gbk
import matplotlib.pyplot as plt

from __init__ import *

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
        self.this_class_arr = '/Volumes/SSD/wen_proj/result/Make_Dataframe/'
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
        fdir = '/Volumes/SSD/wen_proj/result/0523/'
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
        fdir = '/Volumes/SSD/wen_proj/result/CSIF_par/'
        dic = T.load_npy_dir(fdir,condition='005')
        return dic
        # for pix in dic:
        #     # print(pix,dic[pix])
        #     plt.plot(dic[pix])
        #     plt.show()

        pass

    def origin(self):

        # fdir = '/Users/wenzhang/project/wen_proj/result/Hants_annually_smooth/Hants_annually_smooth_CSIF_par/'
        fdir = '/Volumes/SSD/wen_proj/result/Hants_annually_smooth/Hants_annually_smooth_CSIF_par/'
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

        # fdir = '/Volumes/SSD/wen_proj/result/early_peak_late_dormant_period_annually/20%_transform_early_peak_late_dormant_period_annually_CSIF_par/'
        fdir = '/Volumes/SSD/wen_proj/result/early_peak_late_dormant_period_annually/20%_transform_early_peak_late_dormant_period_annually_CSIF_par 2/'
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
        fdir = '/Volumes/SSD/wen_proj/result/0523/'
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
        fdir = '/Volumes/SSD/wen_proj/result/0523/'
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


class Linear_contribution:  # Yang_helps_for Greening_contribution_beta

    def __init__(self):

        time='1982-2015'
        self.period='early'
        self.result_f = results_root+'contribution/contribution_GIMMS_NDVI.npy'

        self.x_dir = fdir=results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/'.format(time, time, self.period)
        self.y_f =results_root + 'extraction_anomaly_val/{}_anomaly_extraction_all_seasons/{}_extraction_during_{}_growing_season_static/' \
                             'during_{}_GIMMS_NDVI.npy'.format(time, time, self.period, self.period)

        pass


    def run(self):
        self.__init__()

        # step 1 build dataframe
        df = self.build_df(self.x_dir,self.y_f,self.period)
        x_var_list = self.__get_x_var_list(self.x_dir,self.period)
        # # step 2 cal correlation
        self.cal_contribution(df, x_var_list)
        # # check model perfermance
        self.contribution_bar(self.result_f, self.x_dir)
        # self.performance(df)
        # self.plot_spatial_max_contribution()


    def __get_x_var_list(self,x_dir,period):
        # x_dir = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        x_f_list = []
        for x_f in T.listdir(x_dir):

            if x_f == 'during_{}_GIMMS_NDVI.npy'.format(period, ):
                continue
            if x_f == 'during_{}_NIRv.npy'.format(period, ):
                continue
            if x_f == 'during_{}_VOD.npy'.format(period, ):
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
            if x_f =='during_{}_GIMMS_NDVI.npy'.format(period,):
                continue
            if x_f == 'during_{}_NIRv.npy'.format(period,):
                continue
            if x_f == 'during_{}_VOD.npy'.format(period, ):
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
            print(var_name)
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
                if len(vals) != 34:
                    x_val_list.append([])
                    continue
                x_val_list.append(vals)
            # x_val_list = np.array(x_val_list)
            df[var_name] = x_val_list
        T.print_head_n(df)

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
                if len(x_vals) == 0:
                    continue


#////////////////////////////detrend and trend 偏导数
                try:

                    # df_new[x] = x_vals   # 不 detrend
                    x_vals_detrend = signal.detrend(x_vals)

                except:
                    x_vals_detrend=[np.nan]*len(x_vals)

                df_new[x] = x_vals_detrend
                x_var_list_valid.append(x)
            if len(df_new) <= 3:
                y_predicted_list.append(np.nan)
                continue
        #//////////////////////detrend& trend

            y_vals_detrend = signal.detrend(y_vals)  #detrend
            df_new['y'] = y_vals_detrend
            # df_new['y'] = y_vals
            T.print_head_n(df_new)


            df_new = df_new.dropna()
            T.print_head_n(df_new)
            exit()


            linear_model = LinearRegression()
            linear_model.fit(df_new[x_var_list_valid],df_new['y'])
            coef_ = linear_model.coef_
            coef_dic = dict(zip(x_var_list_valid,coef_))

            # 3 calculate contribution
            contribution_dic = {}
            y_predicted_i = []
            for x in x_var_list_valid:
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
        # x_dir = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/'
        result_dic = T.load_npy(result_f)
        x_var_list = self.__get_x_var_list(x_dir)
        y_predicted = []
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
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        DIC_and_TIF().plot_back_ground_arr_north_sphere()
        # cmap = sns.color_palette('hls',as_cmap=True)
        plt.imshow(arr,cmap='jet')
        plt.colorbar()
        plt.show()

        pass


class Unify_date_range:

    def __init__(self):

        pass

    def run(self):
        # self.__data_range_index()
        start = 1982
        end = 2000
        X_dir = '/Volumes/SSD/wen_proj/greening_contribution/new/during_early_X/known_year_range/'
        outdirX = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/X_{}-{}/'.format(start,end)
        self.unify(X_dir,outdirX,start,end)
        # #
        Y_dir = '/Volumes/SSD/wen_proj/greening_contribution/new/during_early_Y/'
        outdirY = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/Y_{}-{}/'.format(start, end)
        self.unify(Y_dir,outdirY,start,end)


        # self.check_data_length()
        pass

    def __data_range_index(self,start=0,end=0,product='NIRv',isplot=False):
        dic = {
            'NIRv':list(range(1982,2019)),
            'CSIF_fpar':list(range(2000,2018)),
            'CSIF':list(range(2001,2017)),
            'GIMMS_NDVI':list(range(1982,2016)),
            'CCI_SM':list(range(1982,2016)),

            'PAR':list(range(1982,2019)),
            'Precip':list(range(1982,2019)),
            'root_soil_moisture':list(range(1982,2019)),
            'surf_soil_moisture':list(range(1982,2019)),
            'Temperature':list(range(1982,2019)),
            'VPD':list(range(1982,2019)),
            'SPEI':list(range(1982,2019)),

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


    def unify(self,fdir,outdir,start=2001,end=2015):
        # start = 2001
        # end = 2015
        # X_dir = '/Volumes/SSD/wen_proj/greening_contribution/new/during_early_X/known_year_range/'
        # outdirX = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/X_{}-{}/'.format(start,end)

        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            f_split = f.split('.')[0]
            f_split1 = f_split.split('_')
            product = '_'.join(f_split1[2:])
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



class check_NIRV_NDVI:

    def __init__(self):
        self.template_file = '/Volumes/SSD/drought_legacy_new/conf/tif_template.tif'
        self.phe_dir = '/Volumes/SSD/wen_proj/greening_contribution/20%_transform_early_peak_late_dormant_period_multiyear_CSIF_par/'
        pass


    def run(self):
        # self.resample()
        # self.per_pix()
        # self.anomaly()
        # self.anomaly_wen()
        # self.spatial_correlation()
        # self.pick_early_nirv()
        # self.Moving_window_correlation()
        self.self_make_data()
        pass


    def __cal_anomaly(self,pix_dic):

        anomaly_pix_dic = {}
        for pix in tqdm(pix_dic,desc='cal anomaly'):
            ####### one pix #######
            vals = pix_dic[pix]
            vals = np.array(vals)
            Tools().mask_999999_arr(vals)
            # 清洗数据
            climatology_means = []
            climatology_std = []
            # vals = signal.detrend(vals)
            for m in range(1, 13):
                one_mon = []
                for i in range(len(pix_dic[pix])):
                    mon = i % 12 + 1
                    if mon == m:
                        one_mon.append(pix_dic[pix][i])
                mean = np.nanmean(one_mon)
                std = np.nanstd(one_mon)
                climatology_means.append(mean)
                climatology_std.append(std)

            # 算法1
            # pix_anomaly = {}
            # for m in range(1, 13):
            #     for i in range(len(pix_dic[pix])):
            #         mon = i % 12 + 1
            #         if mon == m:
            #             this_mon_mean_val = climatology_means[mon - 1]
            #             this_mon_std_val = climatology_std[mon - 1]
            #             if this_mon_std_val == 0:
            #                 anomaly = -999999
            #             else:
            #                 anomaly = (pix_dic[pix][i] - this_mon_mean_val) / float(this_mon_std_val)
            #             key_anomaly = i
            #             pix_anomaly[key_anomaly] = anomaly
            # arr = pandas.Series(pix_anomaly)
            # anomaly_list = arr.to_list()
            # anomaly_pix_dic[pix] = anomaly_list

            # 算法2
            pix_anomaly = []
            for i in range(len(vals)):
                mon = i % 12
                std_ = climatology_std[mon]
                mean_ = climatology_means[mon]
                if std_ == 0:
                    anomaly = 0  ##### 修改gpp
                else:
                    anomaly = (vals[i] - mean_) / std_

                pix_anomaly.append(anomaly)
            # pix_anomaly = Tools().interp_1d_1(pix_anomaly,-100)
            # plt.plot(pix_anomaly)
            # plt.show()
            pix_anomaly = np.array(pix_anomaly)
            anomaly_pix_dic[pix] = pix_anomaly
        return anomaly_pix_dic


    def __cal_anomaly_array(self,vals):
        vals = np.array(vals)
        Tools().mask_999999_arr(vals)
        # 清洗数据
        climatology_means = []
        climatology_std = []
        # vals = signal.detrend(vals)
        for m in range(1, 13):
            one_mon = []
            for i in range(len(vals)):
                mon = i % 12 + 1
                if mon == m:
                    one_mon.append(vals[i])
            mean = np.nanmean(one_mon)
            std = np.nanstd(one_mon)
            climatology_means.append(mean)
            climatology_std.append(std)

        # 算法1
        # pix_anomaly = {}
        # for m in range(1, 13):
        #     for i in range(len(pix_dic[pix])):
        #         mon = i % 12 + 1
        #         if mon == m:
        #             this_mon_mean_val = climatology_means[mon - 1]
        #             this_mon_std_val = climatology_std[mon - 1]
        #             if this_mon_std_val == 0:
        #                 anomaly = -999999
        #             else:
        #                 anomaly = (pix_dic[pix][i] - this_mon_mean_val) / float(this_mon_std_val)
        #             key_anomaly = i
        #             pix_anomaly[key_anomaly] = anomaly
        # arr = pandas.Series(pix_anomaly)
        # anomaly_list = arr.to_list()
        # anomaly_pix_dic[pix] = anomaly_list

        # 算法2
        pix_anomaly = []
        for i in range(len(vals)):
            mon = i % 12
            std_ = climatology_std[mon]
            mean_ = climatology_means[mon]
            if std_ == 0:
                anomaly = 0  ##### 修改gpp
            else:
                anomaly = (vals[i] - mean_) / std_

            pix_anomaly.append(anomaly)
        return pix_anomaly



    def resample(self):

        # fdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/MOD13C1.006_NDVI_MVC/'
        # outdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/MOD13C1.006_NDVI_MVC_05/'

        fdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/tif/'
        outdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/tif_05/'

        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            tif = fdir + f
            outtif = outdir + f
            dataset = gdal.Open(tif)
            gdal.Warp(outtif, dataset, xRes=0.5, yRes=0.5, srcSRS='EPSG:4326', dstSRS='EPSG:4326')


    def per_pix(self):
        # fdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/tif_05/2002-2018/'
        # outdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/per_pix/'
        fdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/MOD13C1.006_NDVI_MVC_05/2002-2018/'
        outdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/per_pix/'

        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)
        pass

    def spatial_correlation(self):
        ndvi_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/per_pix_anomaly/'
        nirv_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/per_pix_anomaly_wen/'
        # nirv_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/per_pix_anomaly/'

        ndvi_dic = T.load_npy_dir(ndvi_dir)
        nirv_dic = T.load_npy_dir(nirv_dir)

        spatial_dic = {}
        for pix in tqdm(ndvi_dic):
            if not pix in nirv_dic:
                continue
            ndvi = ndvi_dic[pix]
            nirv = nirv_dic[pix]
            ndvi = np.array(ndvi,dtype=float)
            nirv = np.array(nirv,dtype=float)

            T.mask_999999_arr(ndvi)
            T.mask_999999_arr(nirv)
            r,p = T.nan_correlation(ndvi,nirv)
            spatial_dic[pix] = r
        arr = DIC_and_TIF(self.template_file).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()
        pass

    def anomaly(self):
        in_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/per_pix/'
        out_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/per_pix_anomaly/'
        Pre_Process().cal_anomaly(in_dir,out_dir)
        in_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/per_pix/'
        out_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/per_pix_anomaly/'
        Pre_Process().cal_anomaly(in_dir,out_dir)

        pass

    def anomaly_wen(self): # 实现monthly anomaly
        fdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/per_pix/'
        outdir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NDVI_MODIS/per_pix_anomaly_wen/'
        Tools().mk_dir(outdir)
        dic={}
        for f in tqdm(sorted(os.listdir(fdir))):

            if f.endswith('.npy'):
                # if not '005' in f:
                #     continue
                dic_i = T.load_npy(fdir+f)
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
            # if len(time_series) != 444:
            #     continue
                # plt.plot(time_series)
                # plt.show()

            time_series[time_series < -999] = np.nan
            if np.isnan(np.nanmean(time_series)):
                # print('error')
                continue
            time_series = time_series.reshape(-1, 12)
            # print(time_series)
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
        for key in tqdm(result_dic, desc='output...'):  # 存数据
            flag = flag + 1
            time_series = result_dic[key]
            time_series = np.array(time_series)
            temp_dic[key] = time_series
            if flag % 10000 == 0:
                # print(flag)
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def pick_early_nirv(self):
        outf = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/2001-2015/Y_2001-2015/during_early_NIRv_new.npy'
        nirv_dir = '/Volumes/SSD/wen_proj/nirv_ndvi_check/NIRv/per_pix/'
        nirv_dic = T.load_npy_dir(nirv_dir)
        start_f = self.phe_dir + 'early_start_mon.npy'
        end_f = self.phe_dir + 'early_end_mon.npy'
        start_dic = T.load_npy(start_f)
        end_dic = T.load_npy(end_f)
        nirv_dic_clean = {}
        for pix in tqdm(nirv_dic):
            vals = nirv_dic[pix]
            vals = np.array(vals)
            T.mask_999999_arr(vals)
            if True in np.isnan(vals):
                continue
            nirv_dic_clean[pix] = vals

        nirv_anomaly = self.__cal_anomaly(nirv_dic_clean)
        early_nirv_anomaly = {}

        for pix in nirv_anomaly:
            vals = nirv_anomaly[pix]
            vals_reshape = np.reshape(vals,(-1,12))
            start = start_dic[pix]
            end = end_dic[pix]
            if len(start) == 0:
                continue
            if len(end) == 0:
                continue
            start = start[0]
            end = end[0]
            if end + 1 >= 12:
                continue
            pick_range = list(range(start,end+1))
            annual_vals = []
            for y_val in vals_reshape:
                y_val_picked = T.pick_vals_from_1darray(y_val,pick_range)
                y_val_picked_mean = np.mean(y_val_picked)
                annual_vals.append(y_val_picked_mean)
            annual_vals = np.array(annual_vals)
            early_nirv_anomaly[pix] = annual_vals
        T.save_npy(early_nirv_anomaly,outf)

        pass


    def Moving_window_correlation(self):
        y_f = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/2001-2015/Y_2001-2015/during_early_NIRv_new.npy'
        x_f = '/Volumes/SSD/wen_proj/greening_contribution/new/unified_date_range/2001-2015/X_2001-2015/during_early_Temperature.npy'

        y_dic = T.load_npy(y_f)
        x_dic = T.load_npy(x_f)
        window = 3
        for pix in y_dic:
            if not pix in x_dic:
                continue
            y = y_dic[pix]
            x = x_dic[pix]
            # print(x)
            # print(y)
            corr_list = []
            for i in range(len(x)):
                # print(i+window)
                if i + window >= len(x):
                    continue
                new_x = x[i:i+window]
                new_y = y[i:i+window]
                r,p = stats.pearsonr(new_x,new_y)
                corr_list.append(r)
            if len(corr_list) == 0:
                continue
            plt.plot(corr_list)
            plt.show()

        pass


    def self_make_data(self):
        m = 10
        n = 12*m

        x_vals = np.linspace(0,m*2*np.pi,n) / 4
        trend1 = np.linspace(0,1,n)
        trend2 = np.linspace(0,1.5,n)
        random1 = np.random.random(n)
        random2 = np.random.random(n)
        y1 = np.sin(x_vals) + trend1 + 100
        y2 = np.sin(x_vals) + trend2 + 100

        y1 = y1 + random1
        y2 = y2 + random2

        y1_anomaly = self.__cal_anomaly_array(y1)
        y2_anomaly = self.__cal_anomaly_array(y2)


        r,p = stats.pearsonr(y1,y1)
        r_anomaly,p_anomaly = stats.pearsonr(y1_anomaly,y2_anomaly)
        print('r,p',r,p)
        print('r_anomaly,p_anomaly',r_anomaly,p_anomaly)

        plt.figure()
        plt.plot(x_vals,y1)
        plt.plot(x_vals,y2)
        plt.title('origin')

        plt.figure()
        plt.plot(x_vals, y1_anomaly)
        plt.plot(x_vals, y2_anomaly)
        plt.title('anomaly')
        plt.show()

        pass


def main():

    # Make_Dataframe().run()
    # Main_flow_shui_re().run()
    # Greening().run()
    # Multi_colormap_spatial_map().run()
    # Unify_date_range().run()
    # check_NIRV_NDVI().run()
    Linear_contribution().run()
    pass


if __name__ == '__main__':
    main()



