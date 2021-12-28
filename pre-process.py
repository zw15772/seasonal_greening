# coding=gbk

from __init__ import *
from osgeo import gdal
from LY_Tools import *
import h5py
import scipy.io
project_root='/Volumes/SSD_sumsang/project_greening/'
data_root=project_root+'Data/'
result_root=project_root+'Result/'

class process_data:

    # def __init__(self):
    #     self.this_class_arr =data_root+'Terraclimate/tmmx/'
    #     Tools().mk_dir(self.this_class_arr, force=True)


    def run(self):
        # self.resample()
        # self.resample_NIRv()
        self.resample_VOD()
        # self.Mean_LST()
        # self.conversion()
        # self.read_mat()
        # self.MVC()

    def projection(self):
        fdir=self.this_class_arr = data_root + 'Terraclimate/tmmx/'
        outdir=self.this_class_arr =data_root+'Terraclimate/tmmx_resample/'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            f=f[0:3]+f[5:6]+f[8:9]
            gdal.Warp(outdir+'projection_{}.tif'.format(f), f, dstSRS='EPSG:4326')  # ?�Ⱦ�γ��ͶӰ

        #in_f='/Users/admin/Downloads/201504.tif'
        # out_f1='/Users/admin/Downloads/reprojection01.tif'
        # out_f2 = '/Users/admin/Downloads/reprojection02.tif'

        # ���ؽ����һ��list��list�е�ÿ��Ԫ����һ��tuple��ÿ��tuple�а����˶����ݼ���·����Ԫ���ݵȵ�������Ϣ
        # tuple�еĵ�һ��Ԫ���������������Ӽ���ȫ·��
        #datasets=in_f.GetSubDatasets()

        # ȡ����1�������Ӽ���MODIS�����ʲ�Ʒ�ĵ�һ�����Σ�����ת��
        # ��һ��������������ݣ��ڶ����������������ݣ�������Ը������ѡ��
        # gdal.Warp(out_f1,in_f,dstSRS = 'EPSG:32649') # ?UTMͶӰ
        # gdal.Warp(out_f2,in_f,dstSRS = 'EPSG:4326')  # ?�Ⱦ�γ��ͶӰ

    def resample(self):
        fdir = data_root+'NIRv/NIRv_tif/'
        outdir=data_root + '/NIRv/NIRv_resample/'

        T.mk_dir(outdir)
        year=list(range(1982,2020))
        # print(year)
        # exit()
        for f in tqdm(os.listdir(fdir),):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            # year_selection=f.split('.')[1].split('_')[1]
            # print(year_selection)
            # if not int(year_selection) in year:  ##һ��ע���ʽ
            #     continue
            # fcheck=f.split('.')[0]+f.split('.')[1]+f.split('.')[2]+'.'+f.split('.')[3]
            # if os.path.isfile(outdir+'resample_'+fcheck):  # �ļ��Ѿ����ڵ�ʱ������
            #     continue
            # date = f[0:4] + f[5:7] + f[8:10] MODIS
            print(f)
            date=f.split('.')[0].split('_')[1]
            print(date)
            # exit()
            dataset = gdal.Open(fdir+f)
            # band = dataset.GetRasterBand(1)
            # newRows = dataset.YSize * 2
            # newCols = dataset.XSize * 2
            try:
                 gdal.Warp(outdir + 'NIRv_resample_{}.tif'.format(date), dataset, xRes=0.5, yRes=0.5,dstSRS = 'EPSG:4326')
            #�������ʹ��Ĭ�ϵ�������ز�����������ô����Warp������������resampleAlg������ָ��Ҫʹ�õ��ز�����������������һ��ָ�����ز�������Ϊ˫�����ز�����
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                 pass


    def resample_NIRv(self):
        fdir = data_root+'NIRv/NIRv_tif/'
        outdir=data_root + '/NIRv/NIRv_resample/'

        T.mk_dir(outdir)
        year=list(range(1982,2020))
        # print(year)
        # exit()

        for f in tqdm(os.listdir(fdir),):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            print(f)
            date = f.split('.')[0].split('_')[1]
            print(date)

            # dataset = gdal.Open(fdir+f)
            # band1 = dataset.GetRasterBand(1)
            # band1_array = band1.ReadAsArray()
            #
            # # �����ļ�
            # geotransform = dataset.GetGeoTransform()
            # longitude_start = geotransform[0]
            # latitude_start = geotransform[3]
            # pixelWidth = geotransform[1]
            # pixelHeight = geotransform[5]

            arr = to_raster.raster2array(fdir+f)[0]
            arr[arr < 0] = np.nan
            arr = np.array(arr)
            window = 10
            arr_new = []
            for i in tqdm(range(int(len(arr) / window))):
                temp = []
                for j in range(int(len(arr[0]) / window)):
                    row_range = range(i * window, (i + 1) * window)
                    col_range = range(j * window, (j + 1) * window)
                    select_list = []
                    for ii in row_range:
                        for jj in col_range:
                            select_list.append(arr[ii][jj])
                    select_list_mean = np.nanmean(select_list)
                    temp.append(select_list_mean)
                arr_new.append(temp)
            arr_new = np.array(arr_new)
            # plt.figure()
            # plt.imshow(arr_new)
            # plt.show()
            # plt.imshow(arr)
            # plt.colorbar()
            # plt.show()
            # newRasterfn = outdir + '{}'.format(date) + '.tif'
            DIC_and_TIF().arr_to_tif(arr_new,outdir + 'NIRv_resample_'+'{}'.format(date) + '.tif')

    def resample_VOD(self):
        fdir = data_root+'VOD/VOD_tif/'
        outdir=data_root + '/VOD/VOD_resample/'

        T.mk_dir(outdir)
        year=list(range(1982,2020))
        # print(year)
        # exit()

        for f in tqdm(os.listdir(fdir),):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            print(f)
            date = f.split('.')[0].split('_')[0]+f.split('.')[0].split('_')[1]
            # date = f.split('.')[0].split('_')[1]
            print(date)

            # dataset = gdal.Open(fdir+f)
            # band1 = dataset.GetRasterBand(1)
            # band1_array = band1.ReadAsArray()
            #
            # # �����ļ�
            # geotransform = dataset.GetGeoTransform()
            # longitude_start = geotransform[0]
            # latitude_start = geotransform[3]
            # pixelWidth = geotransform[1]
            # pixelHeight = geotransform[5]

            arr = to_raster.raster2array(fdir+f)[0]
            arr[arr < 0] = np.nan
            arr = np.array(arr)
            window = 2
            arr_new = []
            for i in tqdm(range(int(len(arr) / window))):
                temp = []
                for j in range(int(len(arr[0]) / window)):
                    row_range = range(i * window, (i + 1) * window)
                    col_range = range(j * window, (j + 1) * window)
                    select_list = []
                    for ii in row_range:
                        for jj in col_range:
                            select_list.append(arr[ii][jj])
                    select_list_mean = np.nanmean(select_list)
                    temp.append(select_list_mean)
                arr_new.append(temp)
            arr_new = np.array(arr_new)
            # plt.figure()
            # plt.imshow(arr_new)
            # plt.show()
            # plt.imshow(arr)
            # plt.colorbar()
            # plt.show()
            # newRasterfn = outdir + '{}'.format(date) + '.tif'
            DIC_and_TIF().arr_to_tif(arr_new,outdir + 'VOD_resample_'+'{}'.format(date) + '.tif')


    # function ʵ��MODIS LST �¶�/BESS_PAR����
    def conversion(self):
        fdir = data_root + 'Terraclimate/PAR/PAR_resample/'
        outdir = data_root + 'Terraclimate/PAR/scale_PAR/'
        # fdir = data_root + '/BESS/PAR_tif_resample/'
        # outdir = data_root + '/BESS/PAR_tif_resample_scale/'

        T.mk_dir(outdir)

        for f in tqdm(sorted(os.listdir(fdir))):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            # fcheck=f.split('_')[1]
            # # print(fcheck)
            # # print(outdir + 'conversion_' + fcheck)
            # # exit()
            # if os.path.isfile(outdir + 'scale_' + fcheck):  # �ļ��Ѿ����ڵ�ʱ������
            #     continue
            dataset = gdal.Open(fdir + f)
            band1 = dataset.GetRasterBand(1)
            band1_array = band1.ReadAsArray()

            # �����ļ�
            geotransform = dataset.GetGeoTransform()
            longitude_start = geotransform[0]
            latitude_start = geotransform[3]
            pixelWidth = geotransform[1]
            pixelHeight=geotransform[5]
            date = f.split('_')[0]+'_'+f.split('_')[2]
            # date = f.split('_')[0] + '_' + f.split('_')[1]+'_'+f.split('_')[3]
            # print(date)
            # exit()

            for r in range(len(band1_array)):
                for c in range(len(band1_array[0])):
                    # band1_array[r][c]= band1_array[r][c] * 0.02 - 273.15  # LST �¶Ȼ�������϶�
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # BESS scale
                     band1_array[r][c] = band1_array[r][c] *0.01 # Terraclimate VPD
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # Terraclimate Tempre
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # Terraclimate soil
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # Terraclimate par
            newRasterfn=outdir+'scale_{}.tif'.format(date)

            raster2array.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,band1_array,ndv = -999999)
            # plt.imshow(band1_array)
            # plt.show()

        # raster2array.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array,
        #                                   ndv=-999999)

    def read_mat(self):
        fdir = data_root + 'N_Pdep_land/'
        outdir=data_root+'N_Pdep_land/N_Pdep_land/'
        T.mk_dir(outdir,True)

        for f in tqdm(os.listdir(fdir), desc='loading...'):
            if f.startswith('.'):
                continue
            if not f.endswith('.mat'):
                continue
            print(f)
            # year=f.split('.')[0].split('_')[3]
            # print(year)
            # exit()
            mat = h5py.File(fdir+f)
            print(list(mat.keys()))
            NIRv_arr_list = mat['monthly_NIRv_point05']
            print(NIRv_arr_list)

            for i in range(len(NIRv_arr_list)):

                fname = 'NIRv_' + '{}{:02d}.tif'.format(year, i+1)
                print(fname)
                # exit()
                newRasterfn = outdir + fname
                print(newRasterfn)
                array = NIRv_arr_list[i]
                array = np.array(array)

                # array = array[::-1]
                array = array.T
                array[array > 1] = np.nan
                array[array < 0] = np.nan
                # plt.imshow(array, cmap='jet', vmin=0, vmax=0.4)
                # plt.colorbar()
                # plt.show()
                to_raster.array2raster(newRasterfn, -180, 90, 0.05, -0.05, array,
                                   ndv=-999999)

    def NAN_TIF(self):
        f = data_root + 'AIRS_CO2/CO2_tif_resample/resample_2004.07.01.tif'
        outf = data_root + '/AIRS_CO2/CO2_tif_resample/resample_2004.08.01.tif'
        # fdir = data_root + '/BESS/PAR_tif_resample/'
        # outdir = data_root + '/BESS/PAR_tif_resample_scale/'
        dataset = gdal.Open(f)
        band1 = dataset.GetRasterBand(1)
        band1_array = band1.ReadAsArray()

        # �����ļ�
        geotransform = dataset.GetGeoTransform()
        longitude_start = geotransform[0]
        latitude_start = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        for r in range(len(band1_array)):
            for c in range(len(band1_array[0])):
                band1_array[r][c]=-9999
                # band1_array[r][c] = band1_array[r][c] * 0.1  # BESS scale
        newRasterfn=outf

        raster2array.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,band1_array,ndv = -999999)
        # plt.imshow(band1_array)
        # plt.show()



    # function ��LST ƽ���¶�
    def Mean_LST(self):
        fdir1 = data_root + 'Terraclimate/tmmn_resample/'
        fdir2 = data_root + 'Terraclimate/tmmx_resample/'
        outdir = data_root + 'Terraclimate/mean_temperature/'

        T.mk_dir(outdir)

        for f1 in tqdm(sorted(os.listdir(fdir1))):
            for f2 in tqdm(sorted(os.listdir(fdir2))):
                if not f1.endswith('.tif'):
                    continue
                if not f2.endswith('.tif'):
                    continue
                # print(f1.split('_')[2])
                # print(f2.split('_')[2])
                # exit()
                if f1.split('_')[2]==f2.split('_')[2]:
                    dataset1 = gdal.Open(fdir1 + f1)
                    dataset2 = gdal.Open(fdir2 + f2)
                    bandf1 = dataset1.GetRasterBand(1)
                    bandf1_array = bandf1.ReadAsArray()

                    bandf2= dataset2.GetRasterBand(1)
                    bandf2_array = bandf2.ReadAsArray()
                    # mean_array = np.zeros(range(len(band2_array)), range(len(band2_array)[0]))
                    # mean_array = [[0] * len(band2_array[0]) for row in range(len(band2_array))]
                    mean_array = [[0 for col in range(len(bandf1_array[0]))] for row in range(len(bandf1_array))]
                    for r in range(len(bandf1_array)):
                        for c in range(len(bandf1_array[0])):
                            mean_array[r][c] = (bandf1_array[r][c]+bandf2_array[r][c])/2
                    mean_array=np.array(mean_array)
                    # �����ļ�
                    geotransform = dataset1.GetGeoTransform()
                    longitude_start = geotransform[0]
                    latitude_start = geotransform[3]
                    pixelWidth = geotransform[1]
                    pixelHeight = geotransform[5]
                    date = f1.split('_')[2]
                    print(date)

                    newRasterfn=outdir+'mean_temperature_{}'.format(date)

                    raster2array.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,mean_array,ndv = -999999)
            # plt.imshow(band1_array)
            # plt.show()

    def alignment(self): # ��������ȱʧ�ģ�����ȱʧ�ϼ������빦�ܣ�

        File = '/Users/admin/Downloads/reprj_resamp2.tif'
        #dataset = gdal.Open(File)
        array,originX,originY,pixelWidth,pixelHeight=raster2array.raster2array(File)
        print(array,originX,originY,pixelWidth,pixelHeight)
        addrow1=int((90-originY)/pixelWidth)
        #print(addrow1)
        addcol=7200
        array = np.array(array)
        part1=np.ones((addrow1,addcol))*np.nan
        #print(part1)
        #print(part1.shape)
        all_array=np.vstack((part1, array))
        # print(all_array)
        # print(all_array.shape)
        addrow2=(180.-all_array.shape[0]*pixelWidth)/pixelWidth
        print(addrow2)
        part2 = np.ones((int(round(addrow2)), addcol)) *np.nan
        all_array = np.vstack((all_array, part2))
        print(all_array.shape)
        all_array[all_array<-999]=np.nan #array--mask
        raster2array.array2raster('/Users/admin/Downloads/result3.tif',-180,90,pixelWidth,pixelHeight,all_array)
        plt.imshow(all_array)
        plt.show()


    def MVC(self):  # wen 2021-4-4
        fdir = data_root+'CSIF/CSIF-TIFF/'
        outdir = data_root+'CSIF/CSIF_MVC/'
        Tools().mk_dir(outdir)
        for year in range (20001,2017):
            for month in range (1,13):
                date='{}{:02}'.format(year,month)
                one_month_tif=[]
                for f in os.listdir(fdir):
                    if not f.endswith('.tif'):
                        continue
                    date_i=f.split('.')[0]
                    date_i=date_i[:6]
                    # print(date_i)
                    # print(date)
                    # print('------------------------')
                    if date_i==date:
                        one_month_tif.append(f)
                print(len(one_month_tif))

                arrs=[]
                for tif in one_month_tif:
                    arr,originX,originY,pixelWidth,pixelHeight =to_raster.raster2array(fdir + tif)
                    arr = np.array(arr,dtype=np.float64)
                    arr[arr < -999] = np.nan
                    arrs.append(arr)
                MVC_value=np.zeros(np.shape(arrs[0]))
                arrs = np.array(arrs)
                for i in tqdm(range(len(arrs[0]))):
                    for j in range(len(arrs[0][0])):
                        values=[]
                        for arr in arrs:
                            values.append(arr[i][j])
                        # values = np.array([arrs[0][i][j],arrs[1][i][j]])
                        values = np.array(values)
                        maxv = np.nanmax(values)
                        MVC_value[i][j] = maxv
                # plt.imshow(MVC_value)
                # plt.colorbar()
                # plt.show()
                to_raster.array2raster(outdir + '{}{:02}.tif'.format(year,month), originX, originY, pixelWidth, pixelHeight, MVC_value, ndv=-999999)
                #np.save(outdir + '{}{:02}'.format(year,month), MVC_value)





    def bi_weekly_to_monthly(fdir,outdir, template_tif):  # Yang

        # fdir = '/Users/admin/Downloads/MOD13C1.006_NDVI'
        # outdir = '/Users/admin/Downloads/MOD13C1.006_NDVI_MVC'
        # template_tif = this_root + 'conf\\Namibia.tif'

        D = DIC_and_TIF(template_tif)
        Tools().mk_dir(outdir)

        for y in range(2002,2020):
            for m in range(1, 13):
                date = '{}{:02d}'.format(y, m)
                print (date)
                one_month_tif = []
                for f in os.listdir(fdir):
                    if not f.endswith('.tif'):
                        continue
                    date_i = f[:6]
                    if date_i == date:
                        one_month_tif.append(f)
                arrs = []
                for tif in one_month_tif:
                    arr = to_raster.raster2array(fdir + tif)[0]
                    arr = np.array(arr,dtype=np.float)
                    arr[arr < -2999] = np.nan
                    arrs.append(arr)
                # print arrs
                if len(arrs) == 0:
                    continue
                one_month_mean = []
                for i in range(len(arrs[0])):
                    temp = []
                    for j in range(len(arrs[0][0])):
                        sum_ = []
                        for k in range(len(arrs)):
                            val = arrs[k][i][j]
                            sum_.append(val)
                        mean = np.nanmean(sum_)
                        temp.append(mean)
                    one_month_mean.append(temp)
                one_month_mean = np.array(one_month_mean)
                D.arr_to_tif(one_month_mean, outdir + '{}.tif'.format(date))

        pass


    def tif2dict():
        fdir = '/Users/admin/Downloads/MOD13C1.006_NDVI'
        outdir = '/Users/admin/Downloads/MOD13C1.006_NDVI_MVC'
        Tools().mk_dir(outdir)
        datelist=[]
        flist=os.listdir(fdir)
        for f in flist:
            if f.endswith('tif'):
                date=f.split('.')[0]
                datelist.append(date)
        datelist.sort()
        all_array=[]
        for f in flist:
            for datename in datelist:
                temporal_date = f.split('.')[0]
                if f.split('.')[0]==datename:
                    array, originX, originY, pixelWidth, pixelHeight =raster2array(fdir + f)
                    array = np.array(array, dtype=np.float64)
                    all_array.append(array)

        key_list=[]
        dic={}
        for r in range(len(all_array[0])): #�����ֵ�ļ�ֵ�������ֵ�ļ���ֵ��ʼ��
            for c in range(len(all_array[1])):
                dic[(r,c)]=[]
                key_list.append((r,c))
        #print(dic_key_list)

        for r in range(len(all_array[0])): # ����time series
            for c in range(len(all_array[1])):
                for arr in all_array:
                    value=arr[r][c]
                    dic[(r,c)].append(value)
                #print(dic)
        time_series=[]
        flag=0
        temp_dic={}
        for key in key_list: #������
            flag=flag+1
            time_series=dic[key]
            time_series=np.array(time_series)
            temp_dic[key]=time_series
            if flag %10000 == 0:
                np.save(outdir +'per_pix_dic_%03d' % (flag/10000),temp_dic)
                temp_dic={}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)
    def unify_raster(raster1, raster2):
        array1 = to_raster.raster2array(raster1)[0]
        # array1 = np.array(array1)
        shape1 = np.shape(array1)

        array2 = to_raster.raster2array(raster2)[0]
        shape2 = np.shape(array2)

        # print shape1, shape2
        if not shape1 == shape2:
            mincol = min(shape1[0], shape2[0])
            minrow = min(shape1[1], shape2[1])

            temp1 = array1[:mincol]
            temp2 = array2[:mincol]

            array1_new = temp1.T[:minrow].T
            array2_new = temp2.T[:minrow].T

            # print np.shape(array1_new)
            # print np.shape(array2_new)
            return array1_new, array2_new
            # if not shape1[0] == shape2[0] and shape1[1] == shape2[1]:
            #     min_col = min(shape1[0],shape2[0])
            #     array1_new = array1[:min_col]
            #     array2_new = array2[:min_col]
            #     print 'mincol'
            #     return array1_new,array2_new
            #
            # elif not shape1[1] == shape2[1] and shape1[0] == shape2[0]:
            #     min_row = min(shape1[1],shape2[1])
            #     array1_new = array1.T[:min_row].T
            #     array2_new = array2.T[:min_row].T
            #     print 'minrow'
            #     return array1_new,array2_new
            #
            # elif not shape1[0] == shape2[0] and not shape1[1] == shape2[1]:

        else:
            return array1, array2


    def check_data_transform(fdir):
        for f in os.listdir(fdir):
            dic = T.load_npy(fdir+f)
            for pix in dic:
                vals = dic[pix]
                if vals[0]<=-9999:
                    continue
                vals = np.array(vals,dtype=np.float)
                vals[vals<=-9999]=np.nan
                plt.plot(vals)
                plt.show()


class CleanData:

    def __init__(self):

        pass

    def run(self):
        for region in ['CJZXY','Namibia']:
            for x in ['MOD13A3.006', 'MOD17A2H.006']:
                fdir = data_root + r'perpix\{}\{}\\'.format(region,x)
                outdir = data_root + r'perpix\{}\{}_clean\\'.format(region,x)
                self.clean_origin_vals(fdir,outdir)
        pass


    def clean_origin_vals(self,fdir,outdir):
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir+f)
            clean_dic = {}
            for pix in dic:
                val = dic[pix]
                val = np.array(val,dtype=np.float)
                val[val<-9999]=np.nan
                val[val>30000]=np.nan
                new_val = T.interp_nan(val,kind='linear')
                if len(new_val) == 1:
                    continue
                # plt.plot(val)
                # plt.show()
                clean_dic[pix] = new_val
            np.save(outdir+f,clean_dic)

    def clean_origin_vals_SWE(self,x='SWE'):
        fdir = data_root+'{}\\per_pix\\'.format(x)
        outdir = data_root+'{}\\per_pix_clean\\'.format(x)
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir+f)
            clean_dic = {}
            for pix in dic:
                val = dic[pix]
                val = np.array(val,dtype=np.float)
                val_filter = T.interp_nan(val)
                if len(val_filter) == 1:
                    continue
                new_val = []
                for i in val:
                    if np.isnan(i):
                        v = 0
                    else:
                        v = i
                    new_val.append(v)
                # plt.plot(new_val)
                # plt.show()
                clean_dic[pix] = new_val
            np.save(outdir+f,clean_dic)


    def check_clean(self):
        x = 'SWE'
        fdir = data_root + '{}\\per_pix_clean\\'.format(x)
        x_dic = {}
        for f in tqdm(os.listdir(fdir)):
            dic = T.load_npy(fdir+f)
            for key in dic:
                if len(dic[key]) == 0:
                    continue
                x_dic[key] = np.mean(dic[key])
        arr = D.pix_dic_to_spatial_arr(x_dic)
        # plt.imshow(arr,vmin=0,vmax=100) #pre
        # plt.imshow(arr,vmin=-30,vmax=30) # tmp
        # plt.imshow(arr,vmin=0,vmax=0.3) # sm
        plt.imshow(arr)
        plt.colorbar()
        plt.show()

    def clean_SPEI(self):
        for i in range(1,13):
            fdir = data_root + 'SPEI\\per_pix\\spei{:02d}\\'.format(i)
            outdir = data_root + 'SPEI\\per_pix_clean\\spei{:02d}\\'.format(i)
            T.mk_dir(outdir,force=1)
            for f in tqdm(os.listdir(fdir),desc=str(i)):
                dic = T.load_npy(fdir + f)
                clean_dic = {}
                for pix in dic:
                    val = dic[pix]
                    val = np.array(val, dtype=np.float)
                    val[val < -9999] = np.nan
                    # new_val = T.interp_nan(val, kind='linear')
                    new_val = T.interp_nan(val)
                    if len(new_val) == 1:
                        continue
                    # plt.plot(val)
                    # plt.show()
                    clean_dic[pix] = new_val
                np.save(outdir + f, clean_dic)


        pass


def gen_monthly_mean(fdir,outdir,template):
    D = DIC_and_TIF(template)
    T.mk_dir(outdir,force=True)
    for mon in tqdm(range(1, 13)):
        # print x,mon
        arr_sum = []
        flag = 0
        void_dic = D.void_spatial_dic()
        for year in range(2003,2020):
            f = fdir+'{}{:02d}.tif'.format(year,mon)
            if not os.path.isfile(f):
                continue
            flag += 1
            arr = to_raster.raster2array(f)[0]
            T.mask_999999_arr(arr)
            dic = D.spatial_arr_to_dic(arr)
            for pix in dic:
                void_dic[pix].append(dic[pix])
        mon_mean_dic = {}
        for pix in void_dic:
            vals = void_dic[pix]
            mean_vals = np.nanmean(vals)
            mon_mean_dic[pix] = mean_vals
        mon_mean = D.pix_dic_to_spatial_arr(mon_mean_dic)
        D.arr_to_tif(mon_mean,outdir+'{:02d}.tif'.format(mon))


def main():
    # MVC()
    # tif2dict()
    # bi_weekly_to_monthly()
    process_data().run()

    pass



if __name__ == '__main__':
    main()