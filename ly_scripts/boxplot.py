import pandas as pd
from __init__ import *

T = Tools()

def plot_box():
    # f = '/Volumes/NVME2T/wen_proj/Matrix/data/2002-2015_partial_correlation_p_value_early_anomaly.npy'
    f = '/Volumes/SSD_sumsang/project_greening/Result/new_result/partial_correlation_anomaly/MODIS_NDVI/0105/2002-2015_partial_correlation_late_anomaly_MODIS_NDVI2.npy'
    dic = T.load_npy(f)
    df_temp= T.dic_to_df_different_columns(dic,'pix')
    df_f= results_root+'Data_frame_2002-2015/Data_frame_2002-2015_df.df'
    df=T.load_df(df_f)
    df=df[df['2002-2015_during_late_MODIS_NDVI_trend']>0]
    pix_list=T.get_df_unique_val_list(df,'pix')
    r_list=[]
    for i,row in df_temp.iterrows():
        pix=row.pix
        if not pix in pix_list:
            r_list.append(np.nan)
            continue
        r,c=row['pix']
        r_list.append(r)
    df_temp['r']=r_list
    df_temp=df_temp[df_temp['r']<120]
    box = []
    labels = []
    for col in df_temp:
        print(col)
        if col=='r':
            continue
        if col == 'pix':
            continue
        vals = df_temp[col]
        box_i = []
        for i in vals:
            if np.isnan(i):
                continue
            box_i.append(i)
        box.append(box_i)
        labels.append(col)
    plt.boxplot(box,labels=labels,showfliers=False)
    plt.show()

def main():
    plot_box()
    pass


if __name__ == '__main__':
    main()