import matplotlib.pyplot as plt
import toolz

from __init__ import *
from lytools import *
T = Tools()
results_root = '/Volumes/NVME2T/wen_proj/drought_events_results/'
spei_data_dir = '/Volumes/Ugreen_4T_25/project05_redo/data/SPEI/compose_spei_n_to_one_file/'

class Pick_drought_events:
    '''
    分月份
    '''
    def __init__(self):
        self.this_class_arr = results_root + 'Pick_drought_events/'
        # self.this_class_tif = results_root_new_2021 + 'tif/Main_flow_Pick/'
        # self.this_class_png = results_root_new_2021 + 'png/Main_flow_Pick/'
        Tools().mk_dir(self.this_class_arr, force=True)
        # Tools().mk_dir(self.this_class_tif, force=True)
        # Tools().mk_dir(self.this_class_png, force=True)
        pass


    def run(self):
        # # # ***********************1 pick******************************
        # # # # ----------------------SPEI 1-12----------------------------
        outdir_single = self.this_class_arr + 'single_events_spei_1_12/'
        outdir_non_single = self.this_class_arr + 'non_single_events_spei_1_12/'
        # T.mk_dir(outdir_single)
        # T.mk_dir(outdir_non_single)
        # ####### single events #########
        # for f in T.listdir(spei_data_dir):
        #     drought_index_f = spei_data_dir+f
        #     outdir_single_i = outdir_single + f.split('.')[0] + '/'
        #     self.pick_single_events(drought_index_f,outdir_single_i)
        # ####### non single events #########
        # for f in T.listdir(spei_data_dir):
        #     drought_index_f = spei_data_dir+f
        #     outdir_non_single_i = outdir_non_single + f.split('.')[0] + '/'
        #     self.pick_non_single_events(drought_index_f,outdir_non_single_i)


        # # # *********************** 2 cal intensity and frequency ******************************
        # # # # ----------------------SPEI 1-12 single ----------------------------
        single_intensity_out_dir = self.this_class_arr + 'single_intensity'
        single_freq_out_dir = self.this_class_arr + 'single_freq'
        T.mk_dir(single_intensity_out_dir)
        T.mk_dir(single_freq_out_dir)
        for scale in range(1,13):
            spei_f = T.path_join(spei_data_dir,f'spei{scale:02d}.npy')
            events_f = T.path_join(outdir_single,f'spei{scale:02d}')
            intensity_outf = T.path_join(single_intensity_out_dir,f'spei{scale:02d}')
            freq_outf = T.path_join(single_freq_out_dir,f'spei{scale:02d}')
            self.drought_intensity(events_f,spei_f,intensity_outf)
            self.drought_freq(events_f,freq_outf)
        # # # # ----------------------SPEI 1-12 non single ----------------------------
        non_single_intensity_out_dir = self.this_class_arr + 'non_single_intensity'
        non_single_freq_out_dir = self.this_class_arr + 'non_single_freq'
        T.mk_dir(non_single_intensity_out_dir)
        T.mk_dir(non_single_freq_out_dir)
        for scale in range(1,13):
            spei_f = T.path_join(spei_data_dir,f'spei{scale:02d}.npy')
            events_f = T.path_join(outdir_non_single,f'spei{scale:02d}')
            intensity_outf = T.path_join(non_single_intensity_out_dir, f'spei{scale:02d}')
            freq_outf = T.path_join(non_single_freq_out_dir, f'spei{scale:02d}')
            self.drought_intensity(events_f,spei_f,intensity_outf)
            self.drought_freq(events_f,freq_outf)



    def compose_spei_n_to_one_file(self):
        fdir = data_root + 'SPEI/per_pix_clean_smooth_detrend/'
        outdir = data_root + 'SPEI/compose_spei_n_to_one_file/'
        T.mk_dir(outdir)
        for spei in tqdm(T.listdir(fdir)):
            dic = {}
            for f in T.listdir(fdir + spei):
                dic_i = T.load_npy(os.path.join(fdir,spei,f))
                dic.update(dic_i)
            np.save(outdir + spei,dic)
        pass

    def pick_non_single_events(self,f, outdir):
        # 前n个月和后n个月无极端干旱事件
        n = 24.
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        dic = T.load_npy(f)
        for pix in tqdm(dic,desc='picking {}'.format(f)):
            vals = dic[pix]
            # print list(vals)
            # f = '{}_{}.txt'.format(pix[0],pix[1])
            # fw = open(f,'w')
            # fw.write(str(list(vals)))
            # fw.close()
            # pause()
            # mean = np.nanmean(vals)
            # std = np.std(vals)
            # threshold = mean - 2 * std
            threshold = -1.5
            # threshold = np.quantile(vals, 0.05)
            event_list,key = self.kernel_find_drought_period([vals,pix,threshold])
            if len(event_list) == 0:
                continue
            events_4 = []
            for i in event_list:
                level,drought_range = i
                events_4.append(drought_range)
            single_event_dic[pix] = events_4
        np.save(outdir + 'non_single_events',single_event_dic)

    def pick_single_events(self,f, outdir):
        # 前n个月和后n个月无极端干旱事件
        n = 24.
        T.mk_dir(outdir,force=True)
        single_event_dic = {}
        dic = T.load_npy(f)
        for pix in tqdm(dic,desc='picking {}'.format(f)):
            vals = dic[pix]
            # print list(vals)
            # f = '{}_{}.txt'.format(pix[0],pix[1])
            # fw = open(f,'w')
            # fw.write(str(list(vals)))
            # fw.close()
            # pause()
            mean = np.nanmean(vals)
            std = np.std(vals)
            threshold = mean - 2 * std
            # threshold = -1.5
            # threshold = np.quantile(vals, 0.05)
            event_list,key = self.kernel_find_drought_period([vals,pix,threshold])
            if len(event_list) == 0:
                continue
            events_4 = []
            for i in event_list:
                level,drought_range = i
                events_4.append(drought_range)

            single_event = []
            for i in range(len(events_4)):
                if i - 1 < 0:  # 首次事件
                    if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(vals):  # 触及两边则忽略
                        continue
                    if len(events_4) == 1:
                        single_event.append(events_4[i])
                    elif events_4[i][-1] + n <= events_4[i + 1][0]:
                        single_event.append(events_4[i])
                    continue

                # 最后一次事件
                if i + 1 >= len(events_4):
                    if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(vals):
                        single_event.append(events_4[i])
                    break

                # 中间事件
                if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                    single_event.append(events_4[i])
            # print single_event
            # sleep(0.1)
            single_event_dic[pix] = single_event
            # for evt in single_event:
            #     picked_vals = T.pick_vals_from_1darray(vals,evt)
            #     plt.scatter(evt,picked_vals,color='r')
            # plt.plot(vals)
            # plt.show()
        np.save(outdir + 'single_events',single_event_dic)
        # plt.show()


    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        key = params[1]
        threshold = params[2]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:# SPEI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []

        flag = 0
        events_list = []
        # 不取两个端点
        for i in events:
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                new_i.append(jj)
            # print(new_i)
            # exit()
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:
            # SPEI
            min_val = min(vals)
            if min_val < -99999:
                continue
            if min_val < threshold:
                level = 4
            # if -1 <= min_val < -.5:
            #     level = 1
            # elif -1.5 <= min_val < -1.:
            #     level = 2
            # elif -2 <= min_val < -1.5:
            #     level = 3
            # elif min_val <= -2.:
            #     level = 4
            else:
                level = 0

            events_list.append([level, new_i])
            # print(min_val)
            # plt.plot(vals)
            # plt.show()
        # for key in events_dic:
        #     # print key,events_dic[key]
        #     if 0 in events_dic[key][1]:
        #         print(events_dic[key])
        # exit()
        return events_list, key


    def drought_intensity(self,events_f,spei_f,outf):
        # events_f = '/Volumes/NVME2T/wen_proj/drought_events_results/Pick_drought_events/non_single_events_spei_1_12/spei03/non_single_events.npy'
        # spei_f = '/Volumes/Ugreen_4T_25/project05_redo/data/SPEI/compose_spei_n_to_one_file/spei03.npy'
        events_dic = T.load_npy_dir(events_f)
        spei_dic = T.load_npy(spei_f)
        spatial_dic = {}
        for pix in tqdm(events_dic):
            events = events_dic[pix]
            spei = spei_dic[pix]
            if len(events)==0:
                continue
            spei_sum = 0
            for event in events:
                # print(event)
                picked_spei = T.pick_vals_from_1darray(spei,event)
                severity_i = np.sum(picked_spei)
                spei_sum += severity_i
            spei_sum = -spei_sum
            spatial_dic[pix] = spei_sum
        T.save_npy(spatial_dic,outf)


    def drought_freq(self,events_f,outf):
        # events_f = '/Volumes/NVME2T/wen_proj/drought_events_results/Pick_drought_events/non_single_events_spei_1_12/spei03/non_single_events.npy'
        events_dic = T.load_npy_dir(events_f)
        spatial_dic = {}
        for pix in tqdm(events_dic):
            events = events_dic[pix]
            if len(events)==0:
                continue
            drought_freq = len(events)
            spatial_dic[pix] = drought_freq
        T.save_npy(spatial_dic,outf)



def main():
    Pick_drought_events().run()
    pass


if __name__ == '__main__':
    main()

