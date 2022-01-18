# coding=utf-8
from lytools import *
import pingouin as pg
from sklearn.linear_model import LinearRegression

T = Tools()

project_root='/Volumes/SSD_sumsang/project_greening/'

result_root=project_root+'Result/new_result/'



T.mk_dir(result_root)

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
        self.cal_p_correlation()
        # self.dic_to_df()
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
        outdir = self.this_class_arr

        T.mk_dir(outdir)
        fdir = join(result_root,'extraction_original_val','1982-2015_original_extraction_all_seasons',f'1982-2015_extraction_during_{self.season}_growing_season_static')
        dic_all_var = {}
        for var_i in self.vars_list:
            fname = f'during_early_{var_i}.npy'
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
        fpath = result_root+'extraction_original_val/1982-2015_original_extraction_all_seasons/1982-2015_extraction_during_early_growing_season_static/'
        df_total = pd.DataFrame()
        for f in T.listdir(fpath):
            window = f.replace(f'-{self.n:02d}.npy','')
            window = window.replace('window_','')
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
        self.dff = join(self.this_class_arr,f'Dataframe_{season}.df')
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
        # self.cal_multi_reg()
        self.dic_to_df()
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
        outdir = self.this_class_arr
        T.mk_dir(outdir)
        fdir = join(data_root,f'1982-2015_extraction_during_{self.season}_growing_season_static')
        dic_all_var = {}
        for var_i in self.vars_list:
            fname = f'during_early_{var_i}.npy'
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



def main():
    season = 'early'
    Partial_corr(season).run()
    # Multi_reg(season).run()
    pass


if __name__ == '__main__':

    main()