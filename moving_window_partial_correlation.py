# coding=utf-8
import pandas as pd
from lytools import *
from sklearn.linear_model import LinearRegression

T = Tools()

# project_root='/Volumes/SSD_sumsang/project_greening/'

# result_root=project_root+'Result/new_result/'



# T.mk_dir(result_root)


class Global_vars:

    def __init__(self):
        self.P_PET_fdir = '/Volumes/NVME2T/wen_proj/20220111/aridity_P_PET_dic'

        pass

    def drop_n_std(self,vals, n=1):
        vals = np.array(vals)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        up = mean + n * std
        down = mean - n * std
        vals[vals > up] = np.nan
        vals[vals < down] = np.nan
        return vals

    def P_PET_reclass(self,):
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
        return dic_reclass

    def P_PET_ratio(self,P_PET_fdir):
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


class Partial_corr:

    def __init__(self,season):
        self.season = season
        self.__config__()
        self.this_class_arr = join(result_root,'Partial_corr_{}'.format(self.season))
        T.mk_dir(self.this_class_arr)
        self.dff = join(self.this_class_arr,f'Dataframe_{season}.df')


    def __config__(self):
        self.n = 15

        self.vars_list = [
            'CO2',
            'VPD',
            'PAR',
            'temperature',
            'CCI_SM',
            'GIMMS_NDVI',
        ]
        self.x_var_list = ['CO2',
                      'VPD',
                      'PAR',
                      'temperature',
                      'CCI_SM', ]
        self.y_var = 'GIMMS_NDVI'

    def run(self):
        # self.cal_p_correlation()
        self.dic_to_df()
        pass

    def __partial_corr(self, df, x, y, cov):
        df = pd.DataFrame(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        stats_result = pg.partial_corr(data=df, x=x, y=y, covar=cov, method='pearson').round(3)
        r = float(stats_result['r'])
        p = float(stats_result['p-val'])
        return r, p

    def __cal_partial_correlation(self,df,x_var_list):

        partial_correlation={}
        partial_correlation_p_value = {}
        for x in x_var_list:
            # print(x)
            x_var_list_valid_new_cov=copy.copy(x_var_list)
            x_var_list_valid_new_cov.remove(x)
            r,p=self.__partial_corr(df,x,self.y_var,x_var_list_valid_new_cov)
            partial_correlation[x]=r
            partial_correlation_p_value[x] = p
        return partial_correlation,partial_correlation_p_value

    def __cal_anomaly(self,vals):
        mean = np.nanmean(vals)
        anomaly_list = []
        for val in vals:
            anomaly = val - mean
            anomaly_list.append(anomaly)
        anomaly_list = np.array(anomaly_list)
        return anomaly_list


    def cal_p_correlation(self):
        outdir = join(self.this_class_arr,self.season)

        T.mk_dir(outdir)
        fdir = join(result_root,'extraction_original_val','1982-2015_original_extraction_all_seasons',f'1982-2015_extraction_during_{self.season}_growing_season_static')
        dic_all_var = {}
        for var_i in self.vars_list:
            fname = f'during_{self.season}_{var_i}.npy'
            fpath = join(fdir,fname)
            dic = T.load_npy(fpath)
            dic_all_var[var_i] = dic
        df = T.spatial_dics_to_df(dic_all_var)
        df = df.dropna()
        val_length = 0
        for i, row in df.iterrows():
            y_vals = row[self.y_var]
            val_length = len(y_vals)
            break
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            outpath = join(outdir,f'window_{w+1:02d}-{self.n}.npy')
            if isfile(outpath):
                continue
            pcorr_results_dic_window_i = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=str(w)):
                pix = row.pix
                r,c = pix
                if r > 120:
                    continue
                val_dic = {}
                for x_var in self.vars_list:
                    xvals = row[x_var]
                    pick_index = list(range(w,w+self.n))
                    picked_vals = T.pick_vals_from_1darray(xvals,pick_index)
                    picked_vals_anomaly = self.__cal_anomaly(picked_vals)
                    val_dic[x_var] = picked_vals_anomaly
                df_i = pd.DataFrame()
                for col in self.vars_list:
                    vals_list = []
                    vals = val_dic[col]
                    for val in vals:
                        vals_list.append(val)
                    df_i[col] = vals_list
                df_i = df_i.dropna(axis=1)
                x_var_valid = []
                for col in self.x_var_list:
                    if col in df_i:
                        x_var_valid.append(col)
                dic_partial_corr,dic_partial_corr_p = self.__cal_partial_correlation(df_i,x_var_valid)
                pcorr_results_dic_window_i[pix] = {'pcorr':dic_partial_corr,'p':dic_partial_corr_p}
            T.save_npy(pcorr_results_dic_window_i,outpath)


    def dic_to_df(self):
        fpath = result_root+'Partial_corr_early'
        df_total = pd.DataFrame()
        for f in T.listdir(fpath):
            window = f.replace(f'-{self.n:02d}.npy','')
            window = window.replace('window_','')
            print(window)
            window = int(window)
            dic = T.load_npy(join(fpath,f))
            dic_new = {}
            for pix in dic:
                dic_new[pix] = dic[pix]['pcorr']
            df = T.dic_to_df(dic_new, 'pix')
            pix_list = df['pix']
            df_total['pix'] = pix_list
            for col in self.x_var_list:
                vals = df[col]
                new_col_name = f'{window}_{col}'
                df_total[new_col_name] = vals

        for x_var in self.x_var_list:
            mean_list = []
            std_list = []
            for w in range(99):
                window = w + 1
                col_name = f'{window}_{x_var}'
                if not col_name in df_total:
                    continue
                vals = df_total[col_name]
                vals = np.array(vals)
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                up = mean + std
                down = mean - std
                vals[vals>up] = np.nan
                vals[vals<down] = np.nan
                mean_list.append(np.nanmean(vals))
                std_list.append(std / 8)
            x = list(range(len(mean_list)))
            y = mean_list
            yerr = std_list
            plt.figure()
            plt.plot(x,y,color='r')
            plt.title(x_var)
            Plot_line().plot_line_with_gradient_error_band(x, y, yerr,max_alpha=0.8,min_alpha=0.1,pow=2,color_gradient_n=500)
        plt.show()


class Multi_reg:

    def __init__(self,season):
        self.__config__()
        self.this_class_arr = join(results_root,'Multi_reg')
        T.mk_dir(self.this_class_arr)
        self.season = season

    def __config__(self):
        self.n = 15

        self.vars_list = [
            'CO2',
            'VPD',
            'PAR',
            'temperature',
            'CCI_SM',
            'GIMMS_NDVI',
        ]
        self.x_var_list = ['CO2',
                      'VPD',
                      'PAR',
                      'temperature',
                      'CCI_SM', ]
        self.y_var = 'GIMMS_NDVI'

    def run(self):
        self.cal_multi_reg()
        # self.dic_to_df()
        pass

    def __cal_anomaly(self,vals):
        mean = np.nanmean(vals)
        anomaly_list = []
        for val in vals:
            anomaly = val - mean
            anomaly_list.append(anomaly)
        anomaly_list = np.array(anomaly_list)
        return anomaly_list

    def __multi_reg_fit(self,df,xlist,y_mean):
        reg = LinearRegression()
        X = df[xlist]
        Y = df[self.y_var]
        reg.fit(X,Y)
        coef = reg.coef_ / y_mean
        dic = dict(zip(xlist,coef))
        return dic

    def cal_multi_reg(self):
        outdir = join(self.this_class_arr,self.season)
        T.mk_dir(outdir)
        fdir = join(data_root,f'1982-2015_extraction_during_{self.season}_growing_season_static')
        dic_all_var = {}
        for var_i in self.vars_list:
            fname = f'during_{self.season}_{var_i}.npy'
            fpath = join(fdir,fname)
            dic = T.load_npy(fpath)
            dic_all_var[var_i] = dic
        df = T.spatial_dics_to_df(dic_all_var)
        df = df.dropna()
        val_length = 0
        for i, row in df.iterrows():
            y_vals = row[self.y_var]
            val_length = len(y_vals)
            break
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            outpath = join(outdir,f'window_{w+1:02d}-{self.n}.npy')
            if isfile(outpath):
                continue
            multi_reg_results_dic_window_i = {}
            for i,row in tqdm(df.iterrows(),total=len(df),desc=str(w)):
                pix = row.pix
                r,c = pix
                if r > 120:
                    continue
                y_vals = row[self.y_var]
                y_vals_pick = T.pick_vals_from_1darray(y_vals,pick_index)
                y_mean = np.nanmean(y_vals_pick)
                val_dic = {}
                for x_var in self.vars_list:
                    xvals = row[x_var]
                    picked_vals = T.pick_vals_from_1darray(xvals,pick_index)
                    picked_vals_anomaly = self.__cal_anomaly(picked_vals)
                    val_dic[x_var] = picked_vals_anomaly
                df_i = pd.DataFrame()
                for col in self.vars_list:
                    vals_list = []
                    vals = val_dic[col]
                    for val in vals:
                        vals_list.append(val)
                    df_i[col] = vals_list
                df_i = df_i.dropna(axis=1)
                x_var_valid = []
                for col in self.x_var_list:
                    if col in df_i:
                        x_var_valid.append(col)
                multi_reg_dic = self.__multi_reg_fit(df_i,x_var_valid,y_mean)
                multi_reg_results_dic_window_i[pix] = multi_reg_dic
            T.save_npy(multi_reg_results_dic_window_i,outpath)


    def dic_to_df(self):
        fpath = self.this_class_arr
        df_total = pd.DataFrame()
        for f in T.listdir(fpath):
            window = f.replace(f'-{self.n:02d}.npy','')
            window = window.replace('window_','')
            window = int(window)
            dic = T.load_npy(join(fpath,f))
            df = T.dic_to_df(dic, 'pix')
            pix_list = df['pix']
            df_total['pix'] = pix_list
            for col in self.x_var_list:
                vals = df[col]
                new_col_name = f'{window}_{col}'
                df_total[new_col_name] = vals
        df_total = df_total.dropna()
        for x_var in self.x_var_list:
            # if not x_var == 'CCI_SM':
            #     continue
            mean_list = []
            std_list = []
            for w in range(99):
                window = w + 1
                # if not window == 7:
                #     continue
                col_name = f'{window}_{x_var}'
                if not col_name in df_total:
                    continue
                vals = df_total[col_name]

                vals = np.array(vals)
                mean = np.nanmean(vals)
                std = np.nanstd(vals)
                up = mean + std
                down = mean - std
                vals[vals>up] = np.nan
                vals[vals<down] = np.nan
                mean_list.append(np.nanmean(vals))
                # std_list.append(np.nanstd(vals) / 8)
                std_list.append(np.nanstd(vals))
            x = list(range(len(mean_list)))
            y = mean_list
            yerr = std_list
            plt.figure()
            plt.plot(x,y,color='r')
            plt.title(x_var)
            Plot_line().plot_line_with_gradient_error_band(x, y, yerr,max_alpha=0.8,min_alpha=0.1,pow=2,color_gradient_n=500)
        plt.show()


class Moving_greening_area_ratio:

    def __init__(self,season):
        self.season = season
        self.datadir = '/Volumes/NVME2T/wen_proj/20220111/origional/1982-2015_original_extraction_all_seasons'
        self.y_var = 'GIMMS_NDVI'
        self.n = 15
        pass

    def run(self):
        # self.foo()
        self.ratio_with_limited_area()
        pass

    def foo(self):
        cls_list = [
            'greening p<0.05',
            'greening p<0.1',
            'non sig',
            'browning p<0.1',
            'browning p<0.05',
                    ]
        color_list = [
            'forestgreen',
            'limegreen',
            'gray',
            'peru',
            'sienna',
        ]
        K = KDE_plot()
        f = f'1982-2015_extraction_during_{self.season}_growing_season_static/during_{self.season}_{self.y_var}.npy'
        dic = T.load_npy(join(self.datadir,f))
        dics = {self.y_var:dic}
        df = T.spatial_dics_to_df(dics)
        df = df.dropna()
        val_length = 0
        for i, row in df.iterrows():
            y_vals = row[self.y_var]
            val_length = len(y_vals)
            break
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            spatial_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=str(w)):
                pix = row.pix
                r, c = pix
                if r > 120:
                    continue
                y_vals = row[self.y_var]
                y_vals_pick = T.pick_vals_from_1darray(y_vals, pick_index)
                x = list(range(len(y_vals_pick)))
                try:
                    k,_,_ = K.linefit(x,y_vals_pick)
                    _,p = stats.pearsonr(x,y_vals_pick)
                    spatial_dic[pix] = {'slope':k,'p':p}
                except:
                    continue
            df_i = T.dic_to_df(spatial_dic,'pix')
            df_i = df_i.dropna()
            ratio_list = []
            for cls in cls_list:
                if cls == 'greening p<0.05':
                    df_select = df_i[df_i['slope']>=0]
                    df_select = df_select[df_select['p']<0.05]
                elif cls == 'greening p<0.1':
                    df_select = df_i[df_i['slope'] >= 0]
                    df_select = df_select[df_select['p'] < 0.1]
                    df_select = df_select[df_select['p'] >= 0.05]
                elif cls == 'non sig':
                    df_select = df_i[df_i['p'] > 0.1]
                elif cls == 'browning p<0.1':
                    df_select = df_i[df_i['slope'] < 0]
                    df_select = df_select[df_select['p'] <= 0.1]
                    df_select = df_select[df_select['p'] >= 0.05]
                elif cls == 'browning p<0.05':
                    df_select = df_i[df_i['slope'] < 0]
                    df_select = df_select[df_select['p'] < 0.05]
                else:
                    raise UserWarning
                ratio = len(df_select) / len(df_i)
                ratio_list.append(ratio)
            bottom = 0
            for i in range(len(ratio_list)):
                ratio = ratio_list[i]
                plt.bar(w,ratio,bottom=bottom,color=color_list[i])
                bottom += ratio
        plt.legend(["Browning p<0.05", "Browning p<0.1", "no trend", "Greening p<0.1", "Greening p<0.05"][::-1])
        plt.title(self.season)
        plt.show()

    def ratio_with_limited_area(self):
        P_PET_fdir = Global_vars().P_PET_fdir
        P_PET_ratio = Global_vars().P_PET_ratio(P_PET_fdir)
        P_PET_reclass_dic = Global_vars().P_PET_reclass(P_PET_ratio)
        cls_list = [
            'greening p<0.05',
            'greening p<0.1',
            'non sig',
            'browning p<0.1',
            'browning p<0.05',
                    ]
        color_list = [
            'forestgreen',
            'limegreen',
            'gray',
            'peru',
            'sienna',
        ]
        K = KDE_plot()
        f = f'1982-2015_extraction_during_{self.season}_growing_season_static/during_{self.season}_{self.y_var}.npy'
        dic = T.load_npy(join(self.datadir,f))
        dics = {self.y_var:dic}
        df = T.spatial_dics_to_df(dics)
        df = df.dropna()
        df = T.add_spatial_dic_to_df(df,P_PET_reclass_dic,'HI_reclass')
        HI_class_list = T.get_df_unique_val_list(df,'HI_reclass')
        df = df[df['HI_reclass'] != 'Humid']
        title = f'{self.season} non Humid'
        # print(HI_class_list)
        # exit()
        val_length = 0
        for i, row in df.iterrows():
            y_vals = row[self.y_var]
            val_length = len(y_vals)
            break
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            spatial_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=str(w)):
                pix = row.pix
                r, c = pix
                if r > 120:
                    continue
                y_vals = row[self.y_var]
                y_vals_pick = T.pick_vals_from_1darray(y_vals, pick_index)
                x = list(range(len(y_vals_pick)))
                try:
                    k,_,_ = K.linefit(x,y_vals_pick)
                    _,p = stats.pearsonr(x,y_vals_pick)
                    spatial_dic[pix] = {'slope':k,'p':p}
                except:
                    continue
            df_i = T.dic_to_df(spatial_dic,'pix')
            df_i = df_i.dropna()
            ratio_list = []
            for cls in cls_list:
                if cls == 'greening p<0.05':
                    df_select = df_i[df_i['slope']>=0]
                    df_select = df_select[df_select['p']<0.05]
                elif cls == 'greening p<0.1':
                    df_select = df_i[df_i['slope'] >= 0]
                    df_select = df_select[df_select['p'] < 0.1]
                    df_select = df_select[df_select['p'] >= 0.05]
                elif cls == 'non sig':
                    df_select = df_i[df_i['p'] > 0.1]
                elif cls == 'browning p<0.1':
                    df_select = df_i[df_i['slope'] < 0]
                    df_select = df_select[df_select['p'] <= 0.1]
                    df_select = df_select[df_select['p'] >= 0.05]
                elif cls == 'browning p<0.05':
                    df_select = df_i[df_i['slope'] < 0]
                    df_select = df_select[df_select['p'] < 0.05]
                else:
                    raise UserWarning
                ratio = len(df_select) / len(df_i)
                ratio_list.append(ratio)
            bottom = 0
            for i in range(len(ratio_list)):
                ratio = ratio_list[i]
                plt.bar(w,ratio,bottom=bottom,color=color_list[i])
                bottom += ratio
        plt.legend(["Browning p<0.05", "Browning p<0.1", "no trend", "Greening p<0.1", "Greening p<0.05"][::-1])
        plt.title(title)
        plt.show()

class Moving_window_limitation:
    # co2_ndvi_vpd
    def __init__(self,season):
        self.season = season
        self.this_class_arr = '/Volumes/NVME2T/wen_proj/Moving_window_limitation'
        T.mk_dir(self.this_class_arr)
        self.n = 15

        self.vars_list = [
            'CO2',
            'VPD',
            'PAR',
            'temperature',
            'CCI_SM',
            'GIMMS_NDVI',
            'Aridity',
        ]
        self.y_var = 'GIMMS_NDVI'
        self.x_var_list = ['CO2',
            'VPD',
            'PAR',
            'temperature',
            'CCI_SM',]
        pass

    def run(self):
        # self.get_window_mean()
        # self.plot_limitation_annual_co2()
        self.star_plot()
        # self.plot_limitation_co2_ndvi_corr()
        pass

    def get_window_mean(self):
        data_dir = '/Volumes/NVME2T/wen_proj/20220111/origional/1982-2015_original_extraction_all_seasons/' \
                   f'1982-2015_extraction_during_{self.season}_growing_season_static'
        outf = join(self.this_class_arr,'Dataframe.df')
        # print(outf)
        # exit()
        dic_all_var = {}
        for var_i in self.vars_list:
            fname = f'during_{self.season}_{var_i}.npy'
            fpath = join(data_dir, fname)
            dic = T.load_npy(fpath)
            dic_all_var[var_i] = dic
        df = T.spatial_dics_to_df(dic_all_var)
        df = df.dropna()
        val_length = 0
        for i, row in df.iterrows():
            y_vals = row[self.y_var]
            val_length = len(y_vals)
            break
        df_all = pd.DataFrame()
        for w in range(val_length):
            if w + self.n >= val_length:
                continue
            pick_index = list(range(w, w + self.n))
            window_mean_val_dic = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc=str(w)):
                pix = row.pix
                r, c = pix
                if r > 120:
                    continue
                val_dic = {}
                for x_var in self.vars_list:
                    xvals = row[x_var]
                    picked_vals = T.pick_vals_from_1darray(xvals, pick_index)
                    picked_vals_mean = np.nanmean(picked_vals)
                    val_dic[x_var] = picked_vals_mean
                window_mean_val_dic[pix] = val_dic
            df_j = T.dic_to_df(window_mean_val_dic,'pix')
            colomns = df_j.columns
            df_new = pd.DataFrame()
            df_new['pix'] = df_j['pix']
            for col in colomns:
                if col == 'pix':
                    continue
                new_col = f'{w}_{col}'
                df_new[new_col] = df_j[col]
            df_all[df_new.columns] = df_new[df_new.columns]
        # T.print_head_n(df_all)
        T.save_df(df_all,outf)
        pass

    def plot_limitation_annual_co2(self):
        f = join(self.this_class_arr,'dataframe.df')
        # variabls_x2 = 'Aridity'
        variabls_x2 = 'CO2'
        variabls_x1 = 'VPD'
        variabls_y = 'GIMMS_NDVI'
        df = T.load_df(f)
        col = df.columns
        window_list = []
        for c in col:
            if not 'CO2' in c:
                continue
            w = c.split('_')[0]
            window_list.append(int(w))
        window_list = T.drop_repeat_val_from_list(window_list)
        print(window_list)
        # exit()
        # co2_vals = []
        variable_vals_dic = {
            variabls_x2:[],
            variabls_y:[],
            variabls_x1:[],
        }
        year_dic = {
            'year':[]
        }
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                vals = df[col_name]
                for val in vals:
                    variable_vals_dic[var_i].append(val)
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                vals = df[col_name]
                for val in vals:
                    year_dic['year'].append(w)
                break

        df_new = pd.DataFrame()
        for key in variable_vals_dic:
            df_new[key] = variable_vals_dic[key]
        df_new['year'] = year_dic['year']
        co2 = df_new[variabls_x2]
        VPD = df_new[variabls_x1]
        VPD_bins = np.linspace(0.2, 2, 20)
        # VPD_bins = np.linspace(0.3, 3, 10)
        cmap = KDE_plot().makeColours(window_list, 'Reds')
        df = df_new
        df = df.dropna()
        # matrix = []
        for j in window_list:
            x_list = []
            y_list = []
            df_co2 = df[df['year'] == j]
            # print(df_co2)
            for i in tqdm(range(len(VPD_bins))):
                if i + 1 >= len(VPD_bins):
                    continue
                df_vpd = df_co2[df_co2[variabls_x1] > VPD_bins[i]]
                df_vpd = df_vpd[df_vpd[variabls_x1] < VPD_bins[i + 1]]
                vpd = df_vpd[variabls_x1]
                NDVI = df_vpd[variabls_y]
                x_list.append(np.nanmean(vpd))
                y_list.append(np.nanmean(NDVI))
            # matrix.append(y_list)
            y_list = SMOOTH().smooth_convolve(y_list,window_len=7)
            plt.plot(x_list, y_list[1:],label=f'year of {j}', color=cmap[j],lw=4,alpha=0.8)
        # plt.imshow(matrix)
        # plt.colorbar()
        plt.xlabel('VPD')
        plt.ylabel(variabls_y)
        plt.legend()
        plt.show()

    def star_plot(self):
        f = join(self.this_class_arr,'dataframe.df')
        # variabls_x2 = 'Aridity'
        variabls_x2 = 'CO2'
        variabls_x1 = 'VPD'
        variabls_y = 'GIMMS_NDVI'
        df = T.load_df(f)
        col = df.columns
        window_list = []
        for c in col:
            if not 'CO2' in c:
                continue
            w = c.split('_')[0]
            window_list.append(int(w))
        window_list = T.drop_repeat_val_from_list(window_list)
        print(window_list)
        # exit()
        # co2_vals = []
        variable_vals_dic = {
            variabls_x2:[],
            variabls_y:[],
            variabls_x1:[],
        }
        year_dic = {
            'year':[]
        }
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                vals = df[col_name]
                for val in vals:
                    variable_vals_dic[var_i].append(val)
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                vals = df[col_name]
                for val in vals:
                    year_dic['year'].append(w)
                break

        df_new = pd.DataFrame()
        for key in variable_vals_dic:
            df_new[key] = variable_vals_dic[key]
        df_new['year'] = year_dic['year']
        co2 = df_new[variabls_x2]
        VPD = df_new[variabls_x1]
        VPD_bins = np.linspace(0.2, 2, 20)
        # VPD_bins = np.linspace(0.3, 3, 10)
        cmap = KDE_plot().makeColours(window_list, 'Reds')
        df = df_new
        df = df.dropna()
        # matrix = []
        x_list = []
        x_std_list = []
        y_list = []
        y_std_list = []
        for j in window_list:
            df_co2 = df[df['year'] == j]
            # print(df_co2)
            # for i in tqdm(range(len(VPD_bins))):
            #     if i + 1 >= len(VPD_bins):
            #         continue
            #     df_vpd = df_co2[df_co2[variabls_x1] > VPD_bins[i]]
            #     df_vpd = df_vpd[df_vpd[variabls_x1] < VPD_bins[i + 1]]
            vpd = df_co2[variabls_x1]
            NDVI = df_co2[variabls_y]
            x_list.append(np.nanmean(vpd))
            y_list.append(np.nanmean(NDVI))
            x_std_list.append(np.nanstd(vpd))
            y_std_list.append(np.nanstd(NDVI))
            # matrix.append(y_list)
            # y_list = SMOOTH().smooth_convolve(y_list,window_len=7)
        plt.errorbar(x_list, y_list,xerr=x_std_list,yerr=y_std_list)
            # plt.plot(x_list, y_list[1:],label=f'year of {j}', color=cmap[j],lw=4,alpha=0.8)
        # plt.imshow(matrix)
        # plt.colorbar()
        plt.xlabel('VPD')
        plt.ylabel(variabls_y)
        plt.legend()
        plt.show()

    def plot_limitation_co2_ndvi_corr(self):
        f = join(self.this_class_arr,'dataframe.df')
        # variabls_x2 = 'Aridity'
        variabls_x2 = 'CO2'
        # variabls_x1 = 'CCI_SM'
        variabls_x1 = 'Aridity'
        variabls_y = 'GIMMS_NDVI'
        df = T.load_df(f)
        HI_class_dic = Global_vars().P_PET_reclass()
        df = T.add_spatial_dic_to_df(df,HI_class_dic,'HI_class')
        # df = df[df['HI_class']=='Humid']
        # title = 'Humid'

        # df = df[df['HI_class']!='Humid']
        # title = 'non Humid'
        title = ' '
        T.print_head_n(df)
        # exit()
        col = df.columns
        window_list = []
        for c in col:
            if not 'CO2' in c:
                continue
            w = c.split('_')[0]
            window_list.append(int(w))
        window_list = T.drop_repeat_val_from_list(window_list)
        # co2_vals = []
        variable_vals_dic = {
            variabls_x2:[],
            variabls_y:[],
            variabls_x1:[],
        }
        for w in window_list:
            for var_i in variable_vals_dic:
                col_name = f'{w}_{var_i}'
                vals = df[col_name]
                for val in vals:
                    variable_vals_dic[var_i].append(val)

        df_new = pd.DataFrame()
        for key in variable_vals_dic:
            df_new[key] = variable_vals_dic[key]
        co2 = df_new[variabls_x2]
        VPD = df_new[variabls_x1]

        co2_min = np.nanmin(co2)
        co2_max = np.nanmax(co2)
        VPD_min = np.nanmin(VPD)
        VPD_max = np.nanmax(VPD)
        # plt.hist(VPD,bins=80)
        # plt.show()
        # co2_bins = np.linspace(330, 380, 50)
        # VPD_bins = np.linspace(0.2, 2.2, 20)  # vpd
        VPD_bins = np.linspace(0., 0.6, 15)  # sm
        VPD_bins = np.linspace(0.2, 2, 25)  # aridity
        VPD_bins = np.linspace(0, 2, 25)  # aridity
        # VPD_bins = np.linspace(0.3, 3, 10)
        # cmap = KDE_plot().makeColours(VPD_bins, 'Reds')
        df = df_new
        df = df.dropna()
        # matrix = []
        x_list = []
        y_list = []
        for i in tqdm(range(len(VPD_bins))):
            if i + 1 >= len(VPD_bins):
                continue
            df_vpd = df[df[variabls_x1] > VPD_bins[i]]
            df_vpd = df_vpd[df_vpd[variabls_x1] < VPD_bins[i + 1]]

            NDVI = df_vpd[variabls_y].tolist()
            co2 = df_vpd[variabls_x2].tolist()
            x_list.append(VPD_bins[i])
            try:
                # r,p = stats.pearsonr(NDVI,co2)
                r,p = T.nan_correlation(NDVI,co2)
            except:
                r = np.nan
            y_list.append(r)
            # try:
            #     k,_,_ = KDE_plot().linefit(co2,NDVI)
            # except:
            #     k = np.nan
            # y_list.append(k)
        # print(x_list,y_list)
        y_list = SMOOTH().smooth_convolve(y_list)
        # plt.scatter(x_list, y_list)
        plt.plot(x_list, y_list[1:])
        plt.xlabel(variabls_x1)
        plt.ylabel(f'correlation {variabls_y} vs {variabls_x2}')
        # plt.ylabel(f'slope of {variabls_y} vs {variabls_x2}')
        plt.title(title)
        plt.tight_layout()
        plt.show()

class VPD_NDVI_CO2:
    def __init__(self,season):
        self.start_year = 1982
        self.this_class_arr = '/Volumes/NVME2T/wen_proj/VPD_NDVI_CO2'
        T.mk_dir(self.this_class_arr)
        self.season = season
        self.vars_list = [
            'CO2',
            'VPD',
            'PAR',
            'temperature',
            'CCI_SM',
            'GIMMS_NDVI',
            'Aridity',
        ]
        self.y_var = 'GIMMS_NDVI'
        self.x_var_list = ['CO2',
                           'VPD',
                           'PAR',
                           'temperature',
                           'CCI_SM', ]
        pass

    def run(self):
        # self.get_origin_vals_df()
        self.plot_star()
        pass
    def get_origin_vals_df(self):
        data_dir = '/Volumes/NVME2T/wen_proj/20220111/origional/1982-2015_original_extraction_all_seasons/' \
                   f'1982-2015_extraction_during_{self.season}_growing_season_static'
        outf = join(self.this_class_arr,'Dataframe.df')
        # print(outf)
        # exit()
        df = pd.DataFrame()
        void_dic = DIC_and_TIF().void_spatial_dic()
        for var_i in self.vars_list:
            fname = f'during_{self.season}_{var_i}.npy'
            fpath = join(data_dir, fname)
            dic = T.load_npy(fpath)
            year_list = []
            vals_list = []
            for pix in tqdm(void_dic,desc=var_i):
                if not pix in dic:
                    for i in range(34):
                        year_list.append(np.nan)
                        vals_list.append(np.nan)
                    continue
                vals = dic[pix]
                for i,val in enumerate(vals):
                    year_list.append(i+self.start_year)
                    vals_list.append(val)
            df['year'] = year_list
            df[var_i] = vals_list
            # dic_all_var[var_i] = dic
        # df = T.spatial_dics_to_df(dic_all_var)
        df = df.dropna(how='all')
        T.print_head_n(df)
        T.save_df(df,outf)
        pass

    def plot_star(self):
        f = join(self.this_class_arr, 'dataframe.df')
        # variabls_x2 = 'Aridity'
        variabls_x2 = 'CO2'
        variabls_x1 = 'VPD'
        variabls_y = 'GIMMS_NDVI'
        df = T.load_df(f)
        col = df.columns
        variable_vals_dic = {
            variabls_x2: [],
            variabls_y: [],
            variabls_x1: [],
            'year': [],
        }
        for var_i in variable_vals_dic:
            col_name = f'{var_i}'
            vals = df[col_name]
            for val in vals:
                variable_vals_dic[var_i].append(val)

        df_new = pd.DataFrame()
        for key in variable_vals_dic:
            df_new[key] = variable_vals_dic[key]
        co2 = df_new[variabls_x2]
        VPD = df_new[variabls_x1]
        VPD_bins = np.linspace(0.2, 2, 20)
        # VPD_bins = np.linspace(0.3, 3, 10)
        # cmap = KDE_plot().makeColours(window_list, 'Reds')
        df = df_new
        df = df.dropna()
        year_list = T.get_df_unique_val_list(df,'year')
        cmap = KDE_plot().makeColours(year_list, 'Reds')
        # matrix = []
        x_list = []
        x_std_list = []
        y_list = []
        y_std_list = []
        z_list = []
        for j in year_list:
            df_co2 = df[df['year'] == j]
            # print(df_co2)
            # for i in tqdm(range(len(VPD_bins))):
            #     if i + 1 >= len(VPD_bins):
            #         continue
            #     df_vpd = df_co2[df_co2[variabls_x1] > VPD_bins[i]]
            #     df_vpd = df_vpd[df_vpd[variabls_x1] < VPD_bins[i + 1]]
            vpd = df_co2[variabls_x1]
            NDVI = df_co2[variabls_y]
            co2 = df_co2[variabls_x2]
            x_list.append(np.nanmean(vpd))
            y_list.append(np.nanmean(NDVI))
            x_std_list.append(np.nanstd(vpd))
            y_std_list.append(np.nanstd(NDVI))
            z_list.append(np.nanmean(co2))
            # matrix.append(y_list)
            # y_list = SMOOTH().smooth_convolve(y_list,window_len=7)
            x = np.nanmean(vpd)
            y = np.nanmean(NDVI)
            xerr = np.nanstd(vpd)
            yerr = np.nanstd(NDVI)
            plt.errorbar(x, y, xerr=xerr, yerr=yerr,zorder=0,alpha=0.3,c='gray')
        plt.scatter(x_list, y_list,c=z_list,cmap='Reds')
        # plt.plot(x_list, y_list[1:],label=f'year of {j}', color=cmap[j],lw=4,alpha=0.8)
        # plt.imshow(matrix)
        plt.colorbar()
        plt.xlabel('VPD')
        plt.ylabel(variabls_y)
        # plt.legend()
        plt.show()
        pass

def main():
    season = 'early'
    # season = 'peak'
    # season = 'late'
    # Partial_corr(season).run()
    # Multi_reg(season).run()
    # Moving_greening_area_ratio(season).run()
    # Moving_window_limitation(season).run()
    VPD_NDVI_CO2(season).run()
    pass


if __name__ == '__main__':

    main()