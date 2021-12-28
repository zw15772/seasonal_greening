# coding='utf-8'
from __init__ import *
from dataframe import *
import xgboost as xgb
from collections import defaultdict



class RF:
    def __init__(self):
        self.this_class_arr = results_root + 'RF/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # self.dff = self.this_class_arr + 'data_frame.df'
        self.dff =results_root+ 'Data_frame/'


    def run(self):
        outfir = self.this_class_arr + 'detrend_late_for_%CSIF_par_withoutCO2/'
        Tools().mk_dir(outfir)

        for f in tqdm(os.listdir(self.dff)):
            print(f)
            model_name = '''greening'''
            outf = outfir + model_name+''
            if not f.endswith('late_df.df'):
                continue
            # if f not in filelist:
            #     continue
            file=self.dff+f
            x_vals,y_vals=self.gen_data(file)
            selected_labels=self.multi_colliner(x_vals,y_vals)

            model_result=self.train(x_vals, y_vals, selected_labels, model_name)
            # model_result=self.XG_boost_train(x_vals, y_vals, selected_labels, model_name)
            Tools().save_dict_to_txt(model_result,outf)
        pass

    def __load_df(self,file):
        # dff = self.dff
        df = T.load_df(file)

        return df,file

    def XG_boost_train(self, X, Y, selected_labels, model_name):
        outfir_fig = self.this_class_arr + 'peak_results_via_landuse_fig_for_%GPP/'  # 修改
        Tools().mk_dir(outfir_fig)
        # model_name=file.split('.')[0]
        # model_name=model_name.split('/')[-1]
        outfig = outfir_fig + model_name + '.jpg'

        X = X[selected_labels]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
        print(type(X_train))
        Dmatrix=xgb.DMatrix(data=X_train,label=Y_train)
        Dmatrix_test = xgb.DMatrix(data=X_test, label=Y_test)
        param={}
        bst=xgb.train(param,Dmatrix)
        # score = bst.score(X_test, Y_test)
        mse = sklearn.metrics.mean_squared_error(Y_test, Y_test)

        # y_pred = bst.predict(X_test)

        # KDE_plot().plot_scatter(y_pred,Y_test)
        # plt.show()

        # importances = rf.feature_importances_
        # print(score)
        # plt.barh(selected_labels,importances)
        # plt.tight_layout()
        # plt.show()

        result = permutation_importance(xgb, X_train, Y_train, n_repeats=10,
                                        random_state=42)
        # print(result)
        # exit()
        perm_sorted_idx = result.importances_mean.argsort()
        selected_labels_sorted = []
        for i in perm_sorted_idx:
            selected_labels_sorted.append(selected_labels[i])
        tree_importance_sorted_idx = np.argsort(bst.feature_importances_)
        tree_indices = np.arange(0, len(bst.feature_importances_)) + 0.5

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.barh(tree_indices,
                 bst.feature_importances_[tree_importance_sorted_idx], height=0.7)
        print(bst.feature_importances_[tree_importance_sorted_idx])
        ax1.set_yticks(tree_indices)
        ax1.set_yticklabels(selected_labels_sorted)
        ax1.set_ylim((0, len(bst.feature_importances_)))
        ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                    labels=selected_labels_sorted)
        print(result.importances_mean)
        plt.title(model_name)
        fig.tight_layout()
        # plt.show()  #如果是存数据就不能有show
        plt.savefig(outfig, dpi=300)
        plt.close()
        # np.save(outf, result.importances_mean)
        # pass
        ##-----------------------------------------储存数据---------------------------------------------------------

        imp_dic = {}
        xval_labels = []
        importance = []
        for i in range(len(selected_labels_sorted)):
            xval_labels.append(selected_labels_sorted[i])
            importance.append(bst.feature_importances_[tree_importance_sorted_idx][i])
        for j in range(len(importance)):
            imp_dic[xval_labels[j]] = importance[j]
        model = {}
        model['R2'] = score
        model['MSE'] = mse
        model['importance'] = imp_dic
        return model

    def train(self,X,Y,selected_labels,model_name):
        outfir_fig=self.this_class_arr+'detrend_late_fig_for_%CSIF_par_withoutCO2/'  #修改
        Tools().mk_dir(outfir_fig)
        # model_name=file.split('.')[0]
        # model_name=model_name.split('/')[-1]
        # print(Y)
        # exit()
        outfig = outfir_fig + model_name+ '.jpg'
        rf = RandomForestRegressor(n_jobs=-1)
        X=X[selected_labels]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#----------------------------------查看x 和 y 的关系
        # y_var = '%CSIF_par_trend_late'  # 修改
        # outdir = '/Users/wenzhang/Desktop/untitled folder/'
        # df,dff = self.__load_df(f)
        # df = df[df['r_list'] < 120]  # 修改 需要北纬30度以上
        # df = df[df['%CSIF_par_trend_late'] < 0.04]  # 修改
        # df = df[df['%CSIF_par_trend_late'] > -0.04]  # 修改
        # df = df[df['%CSIF_par_trend_late'] > 0]  # 修改  定义为trend >0
        # for x in selected_labels:
        #     print(x)
        #     yval = Y_train
        #     xval = X_train[x]
        #     plt.scatter(xval,yval)
        #     plt.title(x)
        #     plt.savefig('{}{}.png'.format(outdir,x))
        #     plt.close()
        # exit()
        rf.fit(X_train,Y_train)
        score = rf.score(X_test,Y_test)
        y_pred = rf.predict(X_test)

        mse = sklearn.metrics.mean_squared_error(y_pred, Y_test)


        # KDE_plot().plot_scatter(y_pred,Y_test)
        plt.scatter(y_pred,Y_test)
        plt.show()

        importances = rf.feature_importances_
        print(score)
        plt.barh(selected_labels,importances)
        plt.tight_layout()
        plt.show()

        result = permutation_importance(rf, X_train, Y_train, n_repeats=10,
                                        random_state=42)
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
        plt.title(model_name)
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
        return model






    def multi_colliner(self,x_vals, y_vals):


        x_var_driver_label_late_withoutCO2 = [
            'detrend_late_pre_15_PAR', 'detrend_late_pre_30_PAR', 'detrend_late_pre_60_PAR', 'detrend_late_pre_90_PAR',
            'detrend_late_pre_15_GPCP', 'detrend_late_pre_30_GPCP', 'detrend_late_pre_60_GPCP', 'detrend_late_pre_90_GPCP',
            'detrend_late_pre_15_LST', 'detrend_late_pre_30_LST', 'detrend_late_pre_60_LST', 'detrend_late_pre_90_LST',
            'detrend_late_pre_1mo_SPEI3', 'detrend_late_pre_2mo_SPEI3', 'detrend_late_pre_3mo_SPEI3',

            'detrend_during_late_PAR', 'detrend_during_late_GPCP', 'detrend_during_late_LST', 'detrend_during_late_SPEI3',

           ]
        x_var_facilitators_label_late_withoutCO2=['late_mean_pre_15_GPCP_original', 'late_mean_pre_30_GPCP_original',
                    'late_mean_pre_60_GPCP_original', 'late_mean_pre_90_GPCP_original',
                    'late_mean_pre_15_PAR_original', 'late_mean_pre_30_PAR_original', 'late_mean_pre_60_PAR_original',  'late_mean_pre_90_PAR_original',
                    'late_mean_pre_15_LST_original', 'late_mean_pre_30_LST_original', 'late_mean_pre_60_LST_original', 'late_mean_pre_90_LST_original',
                    'late_mean_pre_3mo_SPEI3', 'late_mean_pre_2mo_SPEI3', 'late_mean_pre_1mo_SPEI3',
                    'mean_during_late_PAR', 'mean_during_late_GPCP', 'mean_during_late_LST', 'mean_during_late_SPEI3', 'mean_during_late_CO2',
                    'mean_during_peak_CSIF_par', 'mean_during_early_CSIF_par',
                     'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand',]

        # x_var_driver_label_late_withCO2= [ 'late_pre_15_PAR', 'late_pre_30_PAR', 'late_pre_60_PAR', 'late_pre_90_PAR',
        #             'late_pre_15_GPCP', 'late_pre_30_GPCP', 'late_pre_60_GPCP', 'late_pre_90_GPCP',
        #             'late_pre_15_LST', 'late_pre_30_LST', 'late_pre_60_LST', 'late_pre_90_LST',
        #             'late_pre_1mo_SPEI3', 'late_pre_2mo_SPEI3', 'late_pre_3mo_SPEI3',
        #             'late_mean_pre_1mo_CO2_original', 'late_mean_pre_2mo_CO2_original', 'late_mean_pre_3mo_CO2_original',
        #
        #             'during_late_PAR', 'during_late_GPCP', 'during_late_LST', 'during_late_SPEI3', 'during_late_CO2',
        #
        #             'late_trend_pre_15_PAR', 'late_trend_pre_30_PAR', 'late_trend_pre_60_PAR', 'late_trend_pre_90_PAR',
        #             'late_trend_pre_15_GPCP', 'late_trend_pre_30_GPCP', 'late_trend_pre_60_GPCP', 'late_trend_pre_90_GPCP',
        #             'late_trend_pre_15_LST', 'late_trend_pre_30_LST', 'late_trend_pre_60_LST', 'late_trend_pre_90_LST',
        #             'late_trend_pre_3mo_SPEI3', 'late_trend_pre_2mo_SPEI3', 'late_trend_pre_1mo_SPEI3',
        #             'trend_pre_late_1mo_CO2_original', 'trend_pre_late_3mo_CO2_original', 'trend_pre_late_2mo_CO2_original',
        #
        #             'trend_during_late_PAR', 'trend_during_late_GPCP', 'trend_during_late_LST', 'trend_during_late_CO2',
        #             'trend_during_late_SPEI3',
        #     ]


        # x_var_driver_label_peak_withCO2 =['peak_pre_15_PAR', 'peak_pre_30_PAR', 'peak_pre_60_PAR', 'peak_pre_90_PAR',
        #     'peak_pre_15_GPCP', 'peak_pre_30_GPCP', 'peak_pre_60_GPCP', 'peak_pre_90_GPCP',
        #     'peak_pre_15_LST', 'peak_pre_30_LST', 'peak_pre_60_LST', 'peak_pre_90_LST',
        #     'peak_pre_1mo_SPEI3', 'peak_pre_2mo_SPEI3', 'peak_pre_3mo_SPEI3',
        #     'peak_mean_pre_2mo_CO2_original', 'peak_mean_pre_3mo_CO2_original','peak_mean_pre_1mo_CO2_original',
        #     'during_peak_PAR', 'during_peak_GPCP', 'during_peak_LST', 'during_peak_SPEI3', 'during_peak_CO2',
        #
        #     'peak_trend_pre_15_PAR', 'peak_trend_pre_30_PAR', 'peak_trend_pre_60_PAR', 'peak_trend_pre_90_PAR',
        #     'peak_trend_pre_15_GPCP', 'peak_trend_pre_30_GPCP', 'peak_trend_pre_60_GPCP', 'peak_trend_pre_90_GPCP',
        #     'peak_trend_pre_15_LST', 'peak_trend_pre_30_LST', 'peak_trend_pre_60_LST', 'peak_trend_pre_90_LST',
        #     'peak_trend_pre_3mo_SPEI3', 'peak_trend_pre_2mo_SPEI3', 'peak_trend_pre_1mo_SPEI3',
        #     'trend_pre_peak_1mo_CO2_original', 'trend_pre_peak_3mo_CO2_original', 'trend_pre_peak_2mo_CO2_original',
        #
        #     'trend_during_peak_PAR', 'trend_during_peak_GPCP', 'trend_during_peak_LST', 'trend_during_peak_CO2',
        #     'trend_during_peak_SPEI3', 'trend_during_early_CSIF_par']
        # x_var_driver_label_peak_withCO2_facilitators=[
        #                            'peak_mean_pre_15_GPCP_original', 'peak_mean_pre_30_GPCP_original', 'peak_mean_pre_60_GPCP_original', 'peak_mean_pre_90_GPCP_original',
        #                            'peak_mean_pre_15_PAR_original', 'peak_mean_pre_30_PAR_original','peak_mean_pre_60_PAR_original', 'peak_mean_pre_90_PAR_original',
        #                             'peak_mean_pre_15_LST_original', 'peak_mean_pre_30_LST_original','peak_mean_pre_60_LST_original', 'peak_mean_pre_90_LST_original',
        #                            'peak_mean_pre_3mo_SPEI3', 'peak_mean_pre_2mo_SPEI3', 'peak_mean_pre_1mo_SPEI3',
        #                            'peak_mean_pre_2mo_CO2_original', 'peak_mean_pre_3mo_CO2_original','peak_mean_pre_1mo_CO2_original',
        #                            'mean_during_peak_PAR', 'mean_during_peak_GPCP', 'mean_during_peak_LST',
        #                            'mean_during_peak_SPEI3', 'mean_during_peak_CO2',
        #
        #                            'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand',]

        # x_var_driver_label_early_withCO2 = ['early_pre_15_PAR', 'early_pre_30_PAR', 'early_pre_60_PAR', 'early_pre_90_PAR',
        #                     'early_pre_15_GPCP', 'early_pre_30_GPCP', 'early_pre_60_GPCP', 'early_pre_90_GPCP',
        #                     'early_pre_15_LST', 'early_pre_30_LST', 'early_pre_60_LST', 'early_pre_90_LST',
        #                     'early_pre_1mo_SPEI3', 'early_pre_2mo_SPEI3', 'early_pre_3mo_SPEI3',
        #                     'early_mean_pre_1mo_CO2_original', 'early_mean_pre_2mo_CO2_original', 'early_mean_pre_3mo_CO2_original',
        #                     'during_early_PAR', 'during_early_GPCP', 'during_early_LST', 'during_early_SPEI3', 'during_early_CO2',
        #
        #                  'early_trend_pre_15_PAR', 'early_trend_pre_30_PAR', 'early_trend_pre_60_PAR','early_trend_pre_90_PAR',
        #                  'early_trend_pre_15_GPCP', 'early_trend_pre_30_GPCP', 'early_trend_pre_60_GPCP', 'early_trend_pre_90_GPCP',
        #                  'early_trend_pre_15_LST', 'early_trend_pre_30_LST', 'early_trend_pre_60_LST','early_trend_pre_90_LST',
        #                  'early_trend_pre_3mo_SPEI3', 'early_trend_pre_2mo_SPEI3', 'early_trend_pre_1mo_SPEI3',
        #                  'trend_pre_early_2mo_CO2_original', 'trend_pre_early_1mo_CO2_original','trend_pre_early_3mo_CO2_original',
        #
        #                  'trend_during_early_PAR', 'trend_during_early_GPCP', 'trend_during_early_LST',
        #                  'trend_during_early_SPEI3','trend_during_early_CO2', ]

        print('len(x_val_label):',len(x_var_driver_label_late_withoutCO2))  # 修改
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        # x_vals = x_vals_with_y
        # x_var_driver_label_late_withoutCO2_with_y = copy.copy(x_var_driver_label_late_withoutCO2)
        # x_var_driver_label_late_withoutCO2_with_y.append('%CSIF_par_trend_late')
        corr = spearmanr(x_vals).correlation
        corr_linkage = hierarchy.ward(corr)
        # dendro = hierarchy.dendrogram(
        #     corr_linkage, labels=x_var_driver_label_late_withoutCO2_with_y, ax=ax1, leaf_rotation=90
        # )
        # dendro_idx = np.arange(0, len(dendro['ivl']))
        #
        # ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
        # ax2.set_xticks(dendro_idx)
        # ax2.set_yticks(dendro_idx)
        # ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
        # ax2.set_yticklabels(dendro['ivl'])
        # fig.tight_layout()
        # plt.show()

        cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        print(selected_features)
        selected_labels=[]

        for i in selected_features:
            selected_labels.append(x_var_driver_label_late_withoutCO2[i])  # 修改

        print(selected_labels)
        # exit()
        return selected_labels

    def gen_data(self,file):
        df,dff = self.__load_df(file)
        print('original_len:',len(df))

        df = df[df['r_list'] < 120]  # 修改 需要北纬30度以上
        df = df[df['%CSIF_par_late_detrend'] >-0.5]
        df = df[df['%CSIF_par_late_detrend'] <0.5]  #修改
        # df = df[df['%CSIF_par_trend_late'] <0]  # #修改trend >0
        # df = df[df['%CSIF_par_p_value_late'] <0.01]  # 修改 并且是significant
        # df = df[df['koppen'] =='Cf']  # 修改 并且是significant
        # df = df[df['landcover'] == 'Grassland']  # 修改 并且是significant
        # df = df[df['contribution_index'] ==3]  # 修改 并且是significant

        y = df['%CSIF_par_late']
        plt.hist(y, bins=80)
        plt.show()

        x_val_late_withoutCO2 = [
                                'detrend_late_pre_15_PAR', 'detrend_late_pre_30_PAR', 'detrend_late_pre_60_PAR', 'detrend_late_pre_90_PAR',
                                'detrend_late_pre_15_GPCP', 'detrend_late_pre_30_GPCP', 'detrend_late_pre_60_GPCP',
                                'detrend_late_pre_90_GPCP',
                                'detrend_late_pre_15_LST', 'detrend_late_pre_30_LST', 'detrend_late_pre_60_LST', 'detrend_late_pre_90_LST',
                                'detrend_late_pre_1mo_SPEI3', 'detrend_late_pre_2mo_SPEI3', 'detrend_late_pre_3mo_SPEI3',

                                'detrend_during_late_PAR', 'detrend_during_late_GPCP', 'detrend_during_late_LST',
                                'detrend_during_late_SPEI3',

                                                                                        ]
        x_val_late_withoutCO2_facilitators=['late_mean_pre_15_GPCP_original', 'late_mean_pre_30_GPCP_original',
                                 'late_mean_pre_60_GPCP_original', 'late_mean_pre_90_GPCP_original',
                                 'late_mean_pre_15_PAR_original', 'late_mean_pre_30_PAR_original', 'late_mean_pre_60_PAR_original',
                                 'late_mean_pre_90_PAR_original',
                                 'late_mean_pre_15_LST_original', 'late_mean_pre_30_LST_original', 'late_mean_pre_60_LST_original',
                                 'late_mean_pre_90_LST_original',
                                 'late_mean_pre_3mo_SPEI3', 'late_mean_pre_2mo_SPEI3', 'late_mean_pre_1mo_SPEI3',
                                 'mean_during_late_PAR', 'mean_during_late_GPCP', 'mean_during_late_LST', 'mean_during_late_SPEI3',
                                 'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand',]


        # x_val_late_withCO2_facilitators=['late_mean_pre_15_GPCP_original', 'late_mean_pre_30_GPCP_original',
        #     'late_mean_pre_60_GPCP_original', 'late_mean_pre_90_GPCP_original',
        #     'late_mean_pre_15_PAR_original', 'late_mean_pre_30_PAR_original', 'late_mean_pre_60_PAR_original', 'late_mean_pre_90_PAR_original',
        #     'late_mean_pre_15_LST_original', 'late_mean_pre_30_LST_original', 'late_mean_pre_60_LST_original', 'late_mean_pre_90_LST_original',
        #     'late_mean_pre_3mo_SPEI3', 'late_mean_pre_2mo_SPEI3', 'late_mean_pre_1mo_SPEI3',
        #     'late_mean_pre_2mo_CO2_original', 'late_mean_pre_3mo_CO2_original', 'late_mean_pre_1mo_CO2_original',
        #     'mean_during_late_PAR', 'mean_during_late_GPCP', 'mean_during_late_LST', 'mean_during_late_SPEI3', 'mean_during_late_CO2',
        #     'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand',]


        #
        # x_val_peak_withCO2_facilitators=['peak_mean_pre_15_GPCP_original', 'peak_mean_pre_30_GPCP_original', 'peak_mean_pre_60_GPCP_original', 'peak_mean_pre_90_GPCP_original',
        #                            'peak_mean_pre_15_PAR_original', 'peak_mean_pre_30_PAR_original','peak_mean_pre_60_PAR_original', 'peak_mean_pre_90_PAR_original',
        #                             'peak_mean_pre_15_LST_original', 'peak_mean_pre_30_LST_original','peak_mean_pre_60_LST_original', 'peak_mean_pre_90_LST_original',
        #                            'peak_mean_pre_3mo_SPEI3', 'peak_mean_pre_2mo_SPEI3', 'peak_mean_pre_1mo_SPEI3',
        #                            'peak_mean_pre_2mo_CO2_original', 'peak_mean_pre_3mo_CO2_original','peak_mean_pre_1mo_CO2_original',
        #                            'mean_during_peak_PAR', 'mean_during_peak_GPCP', 'mean_during_peak_LST',
        #                            'mean_during_peak_SPEI3', 'mean_during_peak_CO2',
        #
        #                            'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand',]

        #
        # x_val_peak_withCO2_facilitators=['early_mean_pre_15_GPCP_original', 'early_mean_pre_30_GPCP_original', 'early_mean_pre_60_GPCP_original',
        #              'early_mean_pre_90_GPCP_original',
        #              'early_mean_pre_15_PAR_original', 'early_mean_pre_30_PAR_original', 'early_mean_pre_60_PAR_original',
        #              'early_mean_pre_90_PAR_original',
        #              'early_mean_pre_15_LST_original', 'early_mean_pre_30_LST_original', 'early_mean_pre_60_LST_original', 'early_mean_pre_90_LST_original',
        #              'early_mean_pre_3mo_SPEI3', 'early_mean_pre_2mo_SPEI3', 'early_mean_pre_1mo_SPEI3',
        #              'early_mean_pre_3mo_CO2_original',	'early_mean_pre_2mo_CO2_original',	'early_mean_pre_1mo_CO2_original',
        #              'mean_during_early_PAR', 'mean_during_early_GPCP', 'mean_during_early_LST', 'mean_during_early_SPEI3', 'mean_during_early_CO2',
        #              'BDOD', 'CEC', 'Clay', 'Nitrogen', 'OCD', 'PH', 'Sand', ]

        print('x_val:',len(x_val_late_withoutCO2))

        y_var = '%CSIF_par_late_detrend'  # 修改
        # print(y_var in x_val_late_withoutCO2)

        # df=df.fillna(0)
        # df=df.replace(np.nan, 0)
        df = df.dropna()
        # T.print_head_n(df,10)
        # print(df_head)
        # exit()
        # print(df)
        print('after clean data length:', len(df))
        # print('after drop:', len(df))
        # exit()

        x_vals = df[x_val_late_withoutCO2] # 修改
        y_vals = df[y_var]
        # x_vars_with_y = copy.copy(x_val_late_withoutCO2)
        # x_vars_with_y.append(y_var)
        # x_vals_with_y = df[x_vars_with_y]
        return x_vals,y_vals



def main():
    RF().run()
if __name__ == '__main__':
    main()
