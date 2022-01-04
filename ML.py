# coding='utf-8'
from __init__ import *
from dataframe import *
# import xgboost as xgb
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import base
#
# y_mark=['greening','browning','notrend']
y_mark=['greening']
y_mark.sort()


class RF:
    def __init__(self):
        self.this_class_arr = results_root + 'RF/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff =results_root+ 'Data_frame/'

    def run(self):
        f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_2002-2015/Data_frame_2002-2015_df.df'
        df, _ = self.__load_df(f)
        df=df[df['row']<120]
        # df=df[df['landcover']=='Grassland']
        df=df.drop_duplicates(subset=('pix'))

        x_variable_dic=self.x_variable()
        y_variable_dic = self.y_variable()
        for period in x_variable_dic:

            x_list=x_variable_dic[period]
            y_list=y_variable_dic[period]
            df_temp = pd.DataFrame()
            df_temp[x_list]=df[x_list]
            df_temp[y_list] = df[y_list]
            invalid=self.check_non_ratio(df_temp)
            exit()
            df_temp=df_temp.drop(columns=invalid)
            # print(len(df_temp))
            df_temp=df_temp.dropna()
            # T.print_head_n(df_temp)
            # print(len(df_temp))
            # exit()
            x_list_new=[]
            for x in x_list:
                if x in invalid:
                    continue
                x_list_new.append(x)
            X=df_temp[x_list_new]
            Y=df_temp[y_list]
            selected_labels=self.multi_colliner(x_list_new,y_list,X,Y)

            Partial_Dependence_Plots().partial_dependent_plot(X,Y,selected_labels)
            # self.train_classfication_permutation_importance(X,Y,x_list_new)

            # self.train_classfication(X,Y,selected_labels=selected_labels)

        pass

    def run1(self):
        f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_2002-2015/Data_frame_2002-2015_df.df'
        df, _ = self.__load_df(f)

        df = df[df['row'] < 120]
        # df=df[df['landcover']=='BF']


        x_variable_dic = self.x_variable_greeness()

        y_variable_dic = self.y_variable_greeness()

        for period in x_variable_dic:

            x_list = x_variable_dic[period]
            y_list = y_variable_dic[period]

            df_temp = pd.DataFrame()
            df_temp[x_list] = df[x_list]
            df_temp[y_list] = df[y_list]
            invalid = self.check_non_ratio(df_temp)

            df_temp = df_temp.drop(columns=invalid)
            # print(len(df_temp))
            df_temp = df_temp.dropna()
            # T.print_head_n(df_temp)
            # print(len(df_temp))
            # exit()
            x_list_new = []
            for x in x_list:
                if x in invalid:
                    continue
                x_list_new.append(x)
            X = df_temp[x_list_new]
            Y = df_temp[y_list]

            selected_labels = self.multi_colliner(x_list_new, y_list, X, Y)
            # selected_labels=self.select_vif_vars(X,x_list_new,threshold=3)
            # print(selected_labels)
            # exit()

            # Partial_Dependence_Plots().partial_dependent_plot_regression(X,Y,x_list_new)
            self.train_classfication_permutation_importance(X, Y, x_list_new)

            # self.train_classfication(X,Y,selected_labels=selected_labels)
        pass

    def __load_df(self,file):
        # dff = self.dff
        df = T.load_df(file)

        return df,file

    def XG_boost_train(self, X, Y, selected_labels, model_name):
        # outfir_fig = self.this_class_arr + 'peak_results_via_landuse_fig_for_%GPP/'  # 修改
        # Tools().mk_dir(outfir_fig)
        # outfig = outfir_fig + model_name + '.jpg'

        X = X[selected_labels]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        print(type(X_train))
        traindataset=xgb.DMatrix(data=X_train,label=Y_train)
        testdataset = xgb.DMatrix(data=X_test, label=Y_test)
        param={}
        bst=xgb.train(param,traindataset)
        y_pred=bst.predict(testdataset)
        plt.scatter(y_pred, Y_test)
        plt.axis('equal')
        plt.show()
        from sklearn.metrics import _regression
        # score=_regression.r2_score(y_pred,Y_test)
        r=stats.pearsonr(y_pred,Y_test)
        # score=base.RegressorMixin().score(pred,Y_test)
        print('score',r)
        xgb.plot_importance(bst)

        plt.show()



        ##-----------------------------------------储存数据---------------------------------------------------------

        # imp_dic = {}
        # xval_labels = []
        # importance = []
        # for i in range(len(selected_labels_sorted)):
        #     xval_labels.append(selected_labels_sorted[i])
        #     importance.append(bst.feature_importances_[tree_importance_sorted_idx][i])
        # for j in range(len(importance)):
        #     imp_dic[xval_labels[j]] = importance[j]
        # model = {}
        # model['R2'] = score
        # model['MSE'] = mse
        # model['importance'] = imp_dic
        # return model


    def train(self,X,Y,selected_labels,period):
        X = X.loc[:, ~X.columns.duplicated()]
        outfir_fig=self.this_class_arr+'fig_driver/'  #修改
        Tools().mk_dir(outfir_fig)
        # model_name=file.split('.')[0]
        # model_name=model_name.split('/')[-1]
        outfig = outfir_fig +'{}'.format(period)+'.jpg'
        rf = RandomForestRegressor(n_jobs=-1)
        X=X[selected_labels]
        X=pd.DataFrame(X)
        # T.print_head_n(X)
        # print('1:',len(selected_labels))
        # print(len(X.columns))
        # print(X.columns)
        # print(selected_labels)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


        rf.fit(X_train,Y_train)
        score = rf.score(X_test,Y_test)
        mse = sklearn.metrics.mean_squared_error(Y_test, Y_test)

        y_pred = rf.predict(X_test)

        # KDE_plot().plot_scatter(y_pred,Y_test)
        plt.scatter(y_pred, Y_test)
        plt.show()

        importances = rf.feature_importances_

        print(score)
        print(len(importances))
        plt.barh(selected_labels, importances)
        plt.tight_layout()
        plt.show()


        # importances = rf.feature_importances_
        # print(score)
        # plt.barh(selected_labels,importances)
        # plt.tight_layout()
        # plt.show()

        result = permutation_importance(rf, X_train, Y_train, n_repeats=10,
                                        random_state=42)
        correlation_dic={}
        for i in selected_labels:
            xi=X[i]
            r,p=stats.pearsonr(xi,Y)
            correlation_dic[i]=(r,p)


        # print(result)
        # exit()
        perm_sorted_idx = result.importances_mean.argsort()
        selected_labels_sorted = []
        for i in perm_sorted_idx:
            selected_labels_sorted.append(selected_labels[i])
        tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
        tree_indices = np.arange(0, len(rf.feature_importances_)) + 0.5

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.barh(tree_indices,
                 rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
        print(rf.feature_importances_[tree_importance_sorted_idx])
        ax1.set_yticks(tree_indices)
        ax1.set_yticklabels(selected_labels_sorted)
        ax1.set_ylim((0, len(rf.feature_importances_)))
        ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                    labels=selected_labels_sorted)
        print(result.importances_mean)
        plt.title('peak') # 修改
        fig.tight_layout()
        # plt.show()  #如果是存数据就不能有show
        plt.savefig(outfig, dpi=300)
        plt.close()
        np.save(outfig, result.importances_mean)
        pass
    ##-----------------------------------------储存数据---------------------------------------------------------

        imp_dic = {}
        xval_labels = []
        importance = []
        for i in range(len(selected_labels_sorted)):
            xval_labels.append(selected_labels_sorted[i])
            importance.append(rf.feature_importances_[tree_importance_sorted_idx][i])
        for j in range(len(importance)):
            imp_dic[xval_labels[j]]=importance[j]
        model = {}
        model['R2']=score
        model['MSE']=mse
        model['sample_length']=len(X)
        print('len_wriiten',len(X) )
        model['importance']=imp_dic
        return model,correlation_dic


    def train_classfication(self,X,Y,selected_labels):

        rf = RandomForestClassifier(n_jobs=1, n_estimators=100, )
        X = X[selected_labels]
        X = pd.DataFrame(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        Y_train=np.array(Y_train)
        rf.fit(X_train, Y_train)

        return rf

    def train_regression(self,X,Y,selected_labels):

        rf = RandomForestRegressor(n_jobs=1, n_estimators=100, )
        X = X[selected_labels]
        X = pd.DataFrame(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        Y_train=np.array(Y_train)
        rf.fit(X_train, Y_train)

        return rf


    def train_classfication_permutation_importance(self,X,Y,selected_labels):
        # X = X.loc[:, ~X.columns.duplicated()]
        outfir_fig_dir=self.this_class_arr+'train_classfication_fig_driver/'  #修改
        Tools().mk_dir(outfir_fig_dir)

        rf = RandomForestRegressor(n_jobs=4,n_estimators=100,)
        X=X[selected_labels]
        X=pd.DataFrame(X)
        # T.print_head_n(X)
        # print('1:',len(selected_labels))
        # print(len(X.columns))
        # print(X.columns)
        # print(selected_labels)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        print(Y_train)
        # exit()


        rf.fit(X_train,Y_train)
        score = rf.score(X_test,Y_test)

        probability=rf.predict(X)
        print(score)

        #
        # importances = rf.feature_importances_
        #
        # print(score)
        # print(len(importances))
        # plt.barh(selected_labels, importances)
        # plt.tight_layout()
        # plt.show()

        result = permutation_importance(rf, X_train, Y_train, n_repeats=10,
                                        random_state=42)

        perm_sorted_idx = result.importances_mean.argsort()
        selected_labels_sorted = []
        for i in perm_sorted_idx:
            selected_labels_sorted.append(selected_labels[i])
        tree_importance_sorted_idx = np.argsort(rf.feature_importances_)
        tree_indices = np.arange(0, len(rf.feature_importances_)) + 0.5

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.barh(tree_indices,
                 rf.feature_importances_[tree_importance_sorted_idx], height=0.7)
        print(rf.feature_importances_[tree_importance_sorted_idx])
        ax1.set_yticks(tree_indices)
        ax1.set_yticklabels(selected_labels_sorted)
        ax1.set_ylim((0, len(rf.feature_importances_)))
        ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                    labels=selected_labels_sorted)
        print(result.importances_mean)
        plt.title('peak') # 修改
        fig.tight_layout()
        # plt.show()  #如果是存数据就不能有show
        now = datetime.datetime.now()
        a = now.strftime('%Y-%m-%d-%H-%M')
        plt.savefig(outfir_fig_dir + a, dpi=300)
        plt.close()
        np.save(outfir_fig_dir + a, result)
        pass
    ##-----------------------------------------储存数据---------------------------------------------------------

        imp_dic = {}
        xval_labels = []
        importance = []
        for i in range(len(selected_labels_sorted)):
            xval_labels.append(selected_labels_sorted[i])
            importance.append(rf.feature_importances_[tree_importance_sorted_idx][i])
        for j in range(len(importance)):
            imp_dic[xval_labels[j]]=importance[j]
        model = {}
        model['R2']=score

        model['sample_length']=len(X)
        print('len_written',len(X) )
        model['importance']=imp_dic
        return model,

    def select_vif_vars(self, df, x_vars_list, threshold=5):
        vars_list_copy = copy.copy(x_vars_list)
        X = df[vars_list_copy]
        X = X.dropna()
        vif = pd.DataFrame()
        vif["features"] = X.columns
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif.round(1)
        selected_vif_list = []
        for i in range(len(vif)):
            feature = vif['features'][i]
            VIF_val = vif['VIF Factor'][i]
            if VIF_val < threshold:
                selected_vif_list.append(feature)
        return selected_vif_list

    def multi_colliner(self,x_list, y_list,X,Y):

        x_val_label = x_list

        print('len(x_val_label):',len(x_list))  # 修改

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        corr = spearmanr(X).correlation
        corr_linkage = hierarchy.ward(corr)
        dendro = hierarchy.dendrogram(
            corr_linkage, labels=x_list, ax=ax1, leaf_rotation=90
        )
        dendro_idx = np.arange(0, len(dendro['ivl']))

        ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
        ax2.set_yticklabels(dendro['ivl'])
        fig.tight_layout()
        plt.show()

        cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        print(selected_features)
        selected_labels=[]

        for i in selected_features:
            selected_labels.append(x_list[i])  # 修改

        print(selected_labels)
        print('len of selected labels:',len(selected_labels))
        # exit()
        return selected_labels

    def screening_variable(self,file,period):
        df, dff = self.__load_df(file)
        print('original_len:', len(df))
        df = df[df['row'] < 120]  # 修改 需要北纬30度以上

        print('trend_during_{}_CSIF_par'.format(period))
        df = df[df['trend_during_{}_CSIF_par'.format(period)] > 0]  # 修改 并且是significant
        df = df[df['CSIF_par_p_value_{}'.format(period)] < 0.1]  # 修改 并且是significant
        df = df[df['during_{}_CSIF_par'.format(period)] > -3]  # 修改
        df = df[df['during_{}_CSIF_par'.format(period)] < 3]  # 修改
        #
        # df = df[df['koppen'] ==koppen]  # 修改 并且是significant
        # df = df[df['landcover'] == 'NF']  # 修改 并且是significant

        y = df['during_{}_CSIF_par'.format(period)]
        plt.hist(y, bins=80)
        plt.show()

    def x_variable(self):

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_1982-2015/Data_frame_1982-2015_df.df'
        df,_=self.__load_df(f)
        # for i in df:
        #     print(i)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        period_list=['early','peak','late']
        x_variable_dic={}
        for period in period_list:
            x_list=[]
            for i in df:
                if period in i:
                    if not '1982-2015' in i:
                        continue
                    if 'NIRv' in i:
                        continue
                    if 'GIMMS_NDVI' in i:
                        continue
                    if 'CV' in i:
                        x_list.append(i)
                    if 'mean' in i:
                        x_list.append(i)
                    if 'trend' in i:
                        x_list.append(i)

            for i in df:
                if 'anomaly' in i:
                    continue

                if 'winter' in i:
                    x_list.append(i)
            for i in ['BDOD', 'CEC', 'clay', 'Nitrogen', 'OCD', 'PH', 'sand','MAT','MAP',]:
                x_list.append(i)

            x_list.sort()
            x_variable_dic[period]=x_list
        # for period in x_variable_dic:
        #     x_list=x_variable_dic[period]
        #     for i in x_list:
        #         print(i)
        #     print('********************************')

        return x_variable_dic

    def y_variable(self):

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_1982-2015/Data_frame_1982-2015_df.df'
        df,_=self.__load_df(f)
        # for i in df:
        #     print(i)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        period_list=['early','peak','late']
        y_variable_dic={}
        for period in period_list:
            y_list=[]
            for i in df:
                if 'mark' not in i:
                    continue
                if period in i:
                    y_list.append(i)
            y_variable_dic[period]=y_list
        # for period in y_variable_dic:
        #     y_list = y_variable_dic[period]
        #     for i in y_list:
        #         print(i)
        #     print('********************************')

        return y_variable_dic

    def x_variable_trend(self):

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_1982-2015/Data_frame_1982-2015_df.df'
        df,_=self.__load_df(f)
        # for i in df:
        #     print(i)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        period_list=['early','peak','late']
        x_variable_dic={}
        for period in period_list:
            x_list=[]
            for i in df:
                if period in i:
                    if not '1982-2015' in i:
                        continue
                    if 'NIRv' in i:
                        continue
                    if 'GIMMS_NDVI' in i:
                        continue
                    if 'CV' in i:
                        # x_list.append(i)
                        continue
                    if 'mean' in i:
                        # x_list.append(i)
                        continue
                    if 'trend' in i:
                        x_list.append(i)

            for i in df:
                if 'anomaly' in i:
                    continue

                if 'winter' in i:
                    x_list.append(i)
            for i in ['BDOD', 'CEC', 'clay', 'Nitrogen', 'OCD', 'PH', 'sand','MAT','MAP',]:
                # x_list.append(i)
                continue

            x_list.sort()
            x_variable_dic[period]=x_list
        # for period in x_variable_dic:
        #     x_list=x_variable_dic[period]
        #     for i in x_list:
        #         print(i)
        #     print('********************************')


        return x_variable_dic

    def y_variable_trend(self):

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_1982-2015/Data_frame_1982-2015_df.df'
        df,_=self.__load_df(f)
        # for i in df:
        #     print(i)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        period_list=['early','peak','late']
        y_variable_dic={}
        for period in period_list:
            y_list=[]
            for i in df:
                if 'original_during_{}_GIMMS_NDVI_trend_1982-2015'.format(period) in i:
                    y_list.append(i)
            y_variable_dic[period]=y_list
        # for period in y_variable_dic:
        #     y_list = y_variable_dic[period]
        #     for i in y_list:
        #         print(i)
        #     print('********************************')

        return y_variable_dic

    def x_variable_greeness(self,):

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_2002-2015/Data_frame_2002-2015_df.df'
        df,_=self.__load_df(f)

        # for i in df:
        #     print(i)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        period_list = ['early','peak','late']
        x_variable_dic={}
        time_range='2002-2015'
        # x_list=[]
        for period in period_list:
            df = df[df['{}_during_{}_MODIS_NDVI_trend'.format(time_range,period)] >= 0]

            x_list = []
            if period =='peak':
                for i in df:
                    if period in i:
                        if not '{}'.format(time_range) in i:
                            continue
                        # if 'CV' in i:
                        #     x_list.append(i)
                        # if 'mean' in i:
                        #     x_list.append(i)
                        # if 'MODIS_NDVI_trend' in i:
                        #     x_list.append(i)
                        # if 'trend' in i:
                        #     x_list.append(i)

                for i in df:

                    if 'CSIF' in i:
                        continue
                    if 'CSIF_fpar' in i:
                        continue
                    if 'GIMMS_NDVI' in i:
                        continue
                    if 'MODIS_NDVI' in i:
                        continue
                    if 'SPEI3' in i:
                        continue

                    if 'anomaly_{}_during_early_VPD'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_early_temperature'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_early_CCI_SM'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_early_Precip'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_peak_'.format(time_range) in i:
                        x_list.append(i)
                    if 'original' in i:
                        continue
                    if 'winter' in i:
                        x_list.append(i)
                for i in ['BDOD', 'CEC', 'clay', 'Nitrogen', 'OCD', 'PH', 'sand','MAT','MAP',]:
                    x_list.append(i)

                x_list.sort()
                x_variable_dic[period]=x_list
            # for period in x_variable_dic:
            #     x_list=x_variable_dic[period]
            #     for i in x_list:
            #         print(i)
            #     print('********************************')


            if period == 'early':
                df = df[df['{}_during_{}_MODIS_NDVI_trend'.format(time_range, period)] >= 0]
                for i in df:
                    if period in i:
                        if not '{}'.format(time_range) in i:
                            continue

                        # if 'CV' in i:
                        #     x_list.append(i)
                        if 'mean' in i:
                            x_list.append(i)
                        # if 'trend' in i:
                        #     x_list.append(i)


                for i in df:

                    if 'CSIF' in i:
                        continue
                    if 'CSIF_fpar' in i:
                        continue
                    if 'GIMMS_NDVI' in i:
                        continue
                    if 'MODIS_NDVI' in i:
                        continue
                    if 'SPEI3' in i:
                        continue

                    if 'anomaly_{}_during_early'.format(time_range) in i:
                        x_list.append(i)
                    if 'original' in i:
                        continue
                    if 'winter' in i:
                        x_list.append(i)
                for i in ['BDOD', 'CEC', 'clay', 'Nitrogen', 'OCD', 'PH', 'sand', 'MAT', 'MAP', ]:
                    x_list.append(i)

                x_list.sort()
                x_variable_dic[period] = x_list

            # for period in x_variable_dic:
            #     x_list=x_variable_dic[period]
            #     for i in x_list:
            #         print(i)
            #     print('********************************')

            if period == 'late':
                df = df[df['{}_during_{}_MODIS_NDVI_trend'.format(time_range,period)] >= 0]
                for i in df:
                    if period in i:
                        if not '{}'.format(time_range) in i:
                            continue


                        # if 'CV' in i:
                        #     x_list.append(i)
                        if 'mean' in i:
                            x_list.append(i)
                        # if 'trend' in i:
                        #     x_list.append(i)


                for i in df:

                    if 'CSIF' in i:
                        continue
                    if 'CSIF_fpar' in i:
                        continue
                    if 'GIMMS_NDVI' in i:
                        continue
                    if 'MODIS_NDVI' in i:
                        continue
                    if 'SPEI3' in i:
                        continue

                    if 'anomaly_{}_during_late'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_peak_temperature'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_peak_CCI_SM'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_peak_Precip'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_peak_VPD'.format(time_range) in i:
                        x_list.append(i)
                    if 'anomaly_{}_during_early_Precip'.format(time_range) in i:
                        x_list.append(i)
                    if 'original' in i:
                        continue
                    if 'winter' in i:
                        x_list.append(i)
                for i in ['BDOD', 'CEC', 'clay', 'Nitrogen', 'OCD', 'PH', 'sand', 'MAT', 'MAP', ]:
                    x_list.append(i)

                x_list.sort()
                x_variable_dic[period] = x_list
        for period in x_variable_dic:
            x_list=x_variable_dic[period]
            for i in x_list:
                print(i)
            print('********************************')


        return x_variable_dic

    def y_variable_greeness(self):

        f='/Volumes/SSD_sumsang/project_greening/Result/new_result/Data_frame_2002-2015/Data_frame_2002-2015_df.df'

        df,_=self.__load_df(f)

        # for i in df:
        #     print(i)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        time_range='2002-2015'

        period_list=['early','peak','late']

        y_variable_dic={}
        for period in period_list:
            df = df[df['{}_during_{}_MODIS_NDVI_trend'.format(time_range, period)] >= 0]
            y_list=[]
            for i in df:
                if 'anomaly_{}_during_{}_MODIS_NDVI'.format(time_range,period) in i:
                    y_list.append(i)
            y_variable_dic[period]=y_list
        # for period in y_variable_dic:
        #     y_list = y_variable_dic[period]
        #     for i in y_list:
        #         print(i)
        #     print('********************************')
        # exit()

        return y_variable_dic


    def check_non_ratio(self,df):
        invalid_variables=[]
        for i in df:
            val=df[i]
            total=len(val)
            val_valid=val.dropna()
            valid=len(val_valid)
            ratio=valid/total
            # print(i,ratio)
            if ratio<0.8:
                invalid_variables.append(i)

        return invalid_variables

        pass
class Partial_Dependence_Plots:
    '''
    Ref:
    https://towardsdatascience.com/looking-beyond-feature-importance-37d2807aaaa7
    '''
    def __init__(self):
        self.this_class_arr = results_root + 'Data_frame/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'late_df.df'
        pass


    def run(self):


        pass

    def __load_df_wen(self, file):
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


    def partial_dependent_plot_classification(self,X,Y,selected_labels):

        outpngdir = results_root + 'partial_plot/partial_dependent_fig/'
        T.mk_dir(outpngdir, force=True)
        outdir = results_root + 'partial_plot/partial_dependent_plot_file/'
        T.mk_dir(outdir,force=True)

        flag = 0
        xv = X[selected_labels]

        print(selected_labels)
        yv = Y
        model = RF().train_regression(xv, yv,selected_labels)

        # plt.figure(figsize=(12, 8))

        for var in tqdm(xv,total=len(xv.columns)):
            flag += 1

            # ax = plt.subplot(1, len(selected_labels), flag)

            sequence, Y_pdp_mean = self.__get_PDPvalues_classification_1(var, xv, model)


            # ppx_smooth = SMOOTH().smooth_convolve(ppx,window_len=11)
            # ppy_smooth = SMOOTH().smooth_convolve(ppy,window_len=11)
            for y in y_mark:
                xx=[]
                yy=[]
                up=[]
                down=[]
                for i in range(len(sequence)):
                    xx.append(sequence[i])
                    yy.append(Y_pdp_mean[i][y])
                    # up.append(Y_pdp_mean[i][y]+Y_pdp_std[i][y])
                    # down.append(Y_pdp_mean[i][y]- Y_pdp_std[i][y])

                plt.plot(xx,yy,label=y)
                # plt.fill_between(xx,y1=up,y2=down,alpha=0.3)
            plt.legend()
            # plt.show()
            plt.title(var)
            plt.savefig(outpngdir + var + '.png')
            plt.close()
        # plt.show()

    def partial_dependent_plot_regression(self,X,Y,selected_labels):

        outpngdir = results_root + 'partial_plot/partial_dependent_fig/'
        T.mk_dir(outpngdir, force=True)
        outdir = results_root + 'partial_plot/partial_dependent_plot_file/'
        T.mk_dir(outdir,force=True)

        flag = 0
        xv = X[selected_labels]

        print(selected_labels)

        yv = Y
        model = RF().train_regression(xv, yv,selected_labels)

        # plt.figure(figsize=(12, 8))

        for var in tqdm(xv,total=len(xv.columns)):
            flag += 1

            # ax = plt.subplot(1, len(selected_labels), flag)


            # ppx_smooth = SMOOTH().smooth_convolve(ppx,window_len=11)
            # ppy_smooth = SMOOTH().smooth_convolve(ppy,window_len=11)
            self.__plot_PDP(var,xv,model,)


            plt.legend()
            # plt.show()
            plt.title(var)
            plt.savefig(outpngdir + var + '.png')
            plt.close()
        # plt.show()



    def __get_PDPvalues_regression(self, col_name, data, model, grid_resolution=50):
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)
        Y_pdp = []
        for each in sequence:
            Xnew[col_name] = each
            # T.print_head_n(Xnew)
            try:
                Y_temp = model.predict(Xnew)
                Y_pdp.append(np.mean(Y_temp))
            except Exception as e:
                print(e)
                print(col_name, data, model)
                print(123)
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp})

    def __get_PDPvalues_classification(self, col_name, data, model, grid_resolution=50):


        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)
        Y_pdp_std = []
        Y_pdp_mean = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict_proba(Xnew)
            Y_temp_T=Y_temp.T
            probability_mean_list=[]
            probability_std_list=[]
            for i in Y_temp_T:
                probability_mean=np.mean(i)
                probability_std = np.std(i)
                probability_mean_list.append(probability_mean)
                probability_std_list.append(probability_std)
            # print(probability_mean_list)
            # print(probability_std_list)
            # exit()
            probability_mean_dic=dict(zip(y_mark,probability_mean_list))
            probability_std_dic = dict(zip(y_mark, probability_std_list))
            Y_pdp_mean.append(probability_mean_dic)
            Y_pdp_std.append(probability_std_dic)
        return sequence,Y_pdp_mean,Y_pdp_std

    def __get_PDPvalues_classification_1(self, col_name, data, model, grid_resolution=50):


        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution)

        Y_pdp_ratio = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            # print(Y_temp)

            Y_temp=list(Y_temp)
            total=len(Y_temp)
            probability_ratio_dic = {}
            for y in y_mark:
                count=Y_temp.count(y)
                ratio=count/total
                probability_ratio_dic[y]=ratio
            Y_pdp_ratio.append(probability_ratio_dic)

        return sequence,Y_pdp_ratio



    def __plot_PDP(self,col_name, data, model):
        df = self.__get_PDPvalues_regression(col_name, data, model)
        plt.rcParams.update({'font.size': 16})
        plt.rcParams["figure.figsize"] = (6,5)
        fig, ax = plt.subplots()
        # ax.plot(data[col_name], np.zeros(data[col_name].shape)+min(df['PDs'])-1, 'k|', ms=15)  # rug plot
        ax.plot(df[col_name], df['PDs'], lw = 2)
        ax.set_ylabel('anomaly')
        ax.set_xlabel(col_name)
        plt.tight_layout()
        return ax

def main():
    # RF().run()
    RF().run1()
    # Partial_Dependence_Plots().run()
if __name__ == '__main__':
    main()
