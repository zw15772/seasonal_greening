from __init__ import *


class Build_dataframe:

    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame/'
        T.mk_dir(self.this_class_arr, force=True)

        self.dff = self.this_class_arr + 'data_frame_new.df'


    def run(self):
        df = self.__gen_df_init()
        # df = self.foo(df)
        df=self.call_build_dataframe(df)
        # df=self.add_lat_and_lon_df(df)
        T.save_df(df, self.dff)
        self.__df_to_excel(df, self.dff, random=False)



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

    def call_build_dataframe(self,df):
        period='early'
        fdir = results_root + 'extraction_anomaly_val/extraction_during_{}_growing_season_static/'.format(period)

        for f in tqdm(sorted(os.listdir(fdir))):
            if f=='during_early_NIRv.npy':
                continue
            if f=='during_{}_CCI_SM.npy'.format(period):
                continue
            if f=='during_{}_CSIF.npy'.format(period):
                continue
            if f=='during_{}_CSIF_fpar.npy'.format(period):
                continue
            dic = dict(np.load(fdir+f, allow_pickle=True, ).item())

            f_name = f.split('.')[0]
            print(f_name)
            # exit()
            df=self.add_variables_df(df,dic,f_name)

        return df


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
        f= '/Volumes/SSD_sumsang/project_greening/Result/new_result/extraction_anomaly_val/extraction_during_early_growing_season_static/during_early_NIRv.npy'
        outf=self.dff
        result_dic = {}
        fname=f.split('.')[0].split('/')[-1]
        print(fname)
        # exit()
        dic = dict(np.load( f, allow_pickle=True, ).item())
        # dic.update(dic_i)
        pix_list=[]
        variables_list=[]
        year=[]
        for pix in tqdm(dic):
            time_series=dic[pix]
            y=0
            for val in time_series:
                pix_list.append(pix)
                variables_list.append(val)
                year.append(y+1982)
                y=y+1
        df['pix']=pix_list
        df['year'] = year
        df[fname] = variables_list
        return df

    def add_variables_df(self,df,dic, fname):

        varible_list=[]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            year = row['year']
            # pix = row.pix
            pix = row['pix']
            if not pix in dic:
                varible_list.append(np.nan)
                continue
            vals = dic[pix]
            if len(vals) !=37:
                varible_list.append(np.nan)
                continue
            v1 = vals[year-1982]
            varible_list.append(v1)
        df[fname] = varible_list

        return df

    def add_lat_and_lon_df(self,df):

        r_list=[]
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            r,c=pix
            r_list.append(r)
        df['r_list'] = r_list
        return df

class plot_dataframe:
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


def main():

    Build_dataframe().run()
    # Build_early_dataframe().run()
    # Build_peak_dataframe().run()
    # Build_late_dataframe().run()
    # check_data()
    # check_dataframe()
    # test()
if __name__ == '__main__':
    main()
