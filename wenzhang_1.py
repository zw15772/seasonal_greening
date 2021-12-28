# coding=utf-8
import lib2to3.pgen2.grammar

import numpy as np

from __init__ import *


def main():
    fdir = '/Users/liyang/Desktop/wen_temp/'
    for f in T.list_dir(fdir):
        print(f)
        # if not 'early' in f:
        if not 'late' in f:
        # if not 'peak' in f:
            continue
        dic = T.load_npy(fdir+f)
        data_length = 0
        for pix in dic:
            vals = dic[pix]
            if len(vals) == 0:
                continue
            data_length = len(vals)
            break
        time_series = []
        for i in tqdm(range(data_length)):
            picked_vals = []
            for pix in dic:
                r,c = pix
                if r > 120:
                    continue
                vals = dic[pix]
                if len(vals) != data_length:
                    continue
                # print(i)
                val_i = vals[i]
                if np.isnan(val_i):
                    continue
                picked_vals.append(val_i)
            mean_val = np.mean(picked_vals)
            time_series.append(mean_val)

        plt.plot(time_series,label=f)
        # plt.title(f)
    plt.legend()
    plt.show()


def window_correlation():
    nir_f = '/Users/liyang/Desktop/wen_temp/during_early_NIRv.npy'
    t_f = '/Users/liyang/Desktop/wen_temp/during_early_Temperature.npy'

    nir_dic = T.load_npy(nir_f)
    t_dic = T.load_npy(t_f)

    window = 10

    for pix in nir_dic:
        nir = nir_dic[pix]
        t = t_dic[pix]
        if len(nir) == 0:
            continue
        if len(t) == 0:
            continue
        plt.plot(t)
        plt.plot(nir)
        plt.show()

    pass


if __name__ == '__main__':
    window_correlation()
    # main()