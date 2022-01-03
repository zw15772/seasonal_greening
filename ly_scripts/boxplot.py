import pandas as pd
from lytools import *
T = Tools()

def plot_box():
    # f = '/Volumes/NVME2T/wen_proj/Matrix/data/2002-2015_partial_correlation_p_value_early_anomaly.npy'
    f = '/Volumes/NVME2T/wen_proj/Matrix/data/2002-2015_partial_correlationearly_anomaly.npy'
    dic = T.load_npy(f)
    df = T.dic_to_df_different_columns(dic,'pix')
    box = []
    labels = []
    for col in df:
        print(col)
        if col == 'pix':
            continue
        vals = df[col]
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