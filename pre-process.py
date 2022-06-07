# coding=gbk

from __init__ import *
from osgeo import gdal
from lytools import *

# import h5py
import scipy.io as scio
import scipy.io
project_root='/Volumes/SSD_sumsang/project_greening/'
data_root=project_root+'Data/'
result_root=project_root+'Result/'

class process_data:

    # def __init__(self):
    #     self.this_class_arr =data_root+'Terraclimate/tmmx/'
    #     Tools().mk_dir(self.this_class_arr, force=True)


    def run(self):

        # self.resample_NIRv()
        # self.resample_VOD()
        # self.sum_PET()
        # self.aridity()
        # self.mean_aridity()
        # self.Mean_LST()
        # self.conversion()
        # self.read_mat()
        # self.aggregation()
        # self.resample_trendy()
        self.unify_raster()

        # self.re_name()
        # self.re_name_doy_month()
        # self.monthly_composite_ly()
        # self.flip()
        # self.resample()
        # self.MVC()
        # self.days_to_monthly()


    def projection(self):
        fdir=self.this_class_arr = data_root + 'Terraclimate/tmmx/'
        outdir=self.this_class_arr =data_root+'Terraclimate/tmmx_resample/'
        for f in tqdm(os.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            f=f[0:3]+f[5:6]+f[8:9]
            gdal.Warp(outdir+'projection_{}.tif'.format(f), f, dstSRS='EPSG:4326')  # ?等经纬度投影

        #in_f='/Users/admin/Downloads/201504.tif'
        # out_f1='/Users/admin/Downloads/reprojection01.tif'
        # out_f2 = '/Users/admin/Downloads/reprojection02.tif'

        # 返回结果是一个list，list中的每个元素是一个tuple，每个tuple中包含了对数据集的路径，元数据等的描述信息
        # tuple中的第一个元素描述的是数据子集的全路径
        #datasets=in_f.GetSubDatasets()

        # 取出第1个数据子集（MODIS反射率产品的第一个波段）进行转换
        # 第一个参数是输出数据，第二个参数是输入数据，后面可以跟多个可选项
        # gdal.Warp(out_f1,in_f,dstSRS = 'EPSG:32649') # ?UTM投影
         # gdal.Warp(out_f2,in_f,dstSRS = 'EPSG:4326')  # ?等经纬度投影
    def aggregation(self):  # LAI aggregation  不同于gdal resample

        fdir=data_root+'LAI_3g/LAI_3g_TIFF/'
        outdir=data_root+'LAI_3g/LAI_3g_resample_522/'
        T.mk_dir(outdir)
        for f in tqdm(os.listdir(fdir),):
            if f.startswith('.'):
                continue
            fpath=join(fdir,f)
            array, originX, originY, pixelWidth, pixelHeight=ToRaster().raster2array(fpath)
            resample_array=T.resample_nan(array,0.5,pixelWidth)
            outf=join(outdir,f)
            DIC_and_TIF().arr_to_tif(resample_array,outf)

            # plt.imshow(resample_array)
            # plt.show()

    def resample_trendy(self):
        fdir_all = 'C:/Users/pcadmin/Desktop/Trendy_TIFF/'
        outdir_all = 'C:/Users/pcadmin/Desktop/Trendy_TIFF_resample/'

        for fdir in tqdm(os.listdir(fdir_all)):
            print(fdir)
            if 'CLM5' not in fdir:
                continue
            outdir=join(outdir_all,fdir+'/')
            T.mk_dir(outdir, force=True)

            if '0.5' in fdir:
                continue
            for f in os.listdir(fdir_all+fdir+'/'):
                if f.startswith('.'):
                    continue
                if f.endswith('.xml'):
                    continue
                fpath=join(fdir_all+fdir+'/',f)

                dataset = gdal.Open(fpath)
                outf = join(outdir, f)

                try:
                    gdal.Warp(outf+'.tif', dataset, xRes=0.5, yRes=0.5, dstSRS='EPSG:4326',)
                # 如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
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
            # # 配置文件
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


    def resample(self):
        fdir = data_root+'CCI_SM_2020/CCI_SM_montly_composite/'
        outdir=data_root + 'CCI_SM_2020/CCI_SM_resample/'

        T.mk_dir(outdir)
        year=list(range(2018,2021))
        # print(year)
        # exit()
        for f in tqdm(os.listdir(fdir),):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue

            # year_selection=f.split('.')[1].split('_')[1]
            # print(year_selection)
            # if not int(year_selection) in year:  ##一定注意格式
            #     continue
            # fcheck=f.split('.')[0]+f.split('.')[1]+f.split('.')[2]+'.'+f.split('.')[3]
            # if os.path.isfile(outdir+'resample_'+fcheck):  # 文件已经存在的时候跳过
            #     continue
            # date = f[0:4] + f[5:7] + f[8:10] MODIS
            print(f)
            # exit()
            date=f.split('.')[0]
            # print(date)
            # exit()
            dataset = gdal.Open(fdir+f)
            # band = dataset.GetRasterBand(1)
            # newRows = dataset.YSize * 2
            # newCols = dataset.XSize * 2
            try:
                 gdal.Warp(outdir + '{}.tif'.format(date), dataset, xRes=0.5, yRes=0.5,dstSRS = 'EPSG:4326')
            #如果不想使用默认的最近邻重采样方法，那么就在Warp函数里面增加resampleAlg参数，指定要使用的重采样方法，例如下面一行指定了重采样方法为双线性重采样：
            # gdal.Warp("resampletif.tif", dataset, width=newCols, height=newRows, resampleAlg=gdalconst.GRIORA_Bilinear)
            except Exception as e:
                 pass

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
            # # 配置文件
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


    # function 实现MODIS LST 温度/BESS_PAR换算
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
            # if os.path.isfile(outdir + 'scale_' + fcheck):  # 文件已经存在的时候跳过
            #     continue
            dataset = gdal.Open(fdir + f)
            band1 = dataset.GetRasterBand(1)
            band1_array = band1.ReadAsArray()

            # 配置文件
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
                    # band1_array[r][c]= band1_array[r][c] * 0.02 - 273.15  # LST 温度换算成摄氏度
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # BESS scale
                     band1_array[r][c] = band1_array[r][c] *0.01 # Terraclimate VPD
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # Terraclimate Tempre
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # Terraclimate soil
                    # band1_array[r][c] = band1_array[r][c] * 0.1  # Terraclimate par
            newRasterfn=outdir+'scale_{}.tif'.format(date)

            ToRaster.array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,band1_array,ndv = -999999)
            # plt.imshow(band1_array)
            # plt.show()

        # raster2array.array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, array,
        #                                   ndv=-999999)

    def re_name(self):  #////////////////////////////////

        dic={}
        fdir = data_root + 'LAI_4g/LAI_4g_resample/'
        outdir = data_root + 'LAI_4g/LAI_4g_rename/'
        T.mk_dir(outdir)
        mon_dic={'jan':'01','feb':'02','mar':'03','apr':'04',
                 'may':'05','jun':'06','jul':'07','aug':'08','sep':'09',
                 'oct':'10','nov':'11','dec':'12'}
        # day_dic={'a':'01','b':'02'}
        day_dic = {'1': '01', '2': '02'}

        for f in tqdm(sorted(os.listdir(fdir))):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            date=f.split('.')[0].split('_')[-1]
            mon_str=date[4:-1]
            year_str=date[0:4]
            day_str=date[-1]
            print(mon_str)
            print(year_str)
            print(day_str)
            # mon=mon_dic[mon_str]
            day=day_dic[day_str]
            new_date=year_str+mon_str+day
            print(new_date,date)
            src=join(fdir,f)
            destination=join(outdir,new_date+'.tif')
            shutil.copy(src,destination)

    def re_name_doy_month(self):  #//////////////////////////rename MODIS_LAI/

        dic={}
        fdir = data_root + '/MODIS_LAI/MODIS_LAI_resample/'
        T.mk_dir(outdir)
        mon_dic={'jan':'01','feb':'02','mar':'03','apr':'04',
                 'may':'05','jun':'06','jul':'07','aug':'08','sep':'09',
                 'oct':'10','nov':'11','dec':'12'}
        # day_dic={'a':'01','b':'02'}
        day_dic = {'1': '01', '2': '02'}

        for f in tqdm(sorted(os.listdir(fdir))):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            date=f.split('.')[0].split('_')[-3]
            newname=join(fdir,date+'.tif')
            print(newname)
            oldname=join(fdir,f)
            os.rename(oldname,newname)

    def monthly_composite_ly(self):  #//////////////////////////composite ly ===MVC/

        fdir = data_root + 'MODIS_LAI/MODIS_LAI_resample/'
        outdir=data_root +'/MODIS_LAI/MODIS_LAI_composite/'
        # Pre_Process().monthly_compose(fdir,outdir,date_fmt='yyyymmdd',method='max')
        Pre_Process().monthly_compose(fdir, outdir, date_fmt='doy', method='max')


    def flip(self):  # LAI 4g 需要上下翻转
        fdir = data_root+'/LAI_BU_4g/LAI_BU_4g_rename/'
        outdir=data_root + '/LAI_BU_4g/LAI_BU_4g_flip/'

        T.mk_dir(outdir)
        year=list(range(1982,2020))
        # print(year)
        # exit()
        for f in tqdm(os.listdir(fdir),):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            if  int(f[0:4])!= 2019:
                continue


            print(f)
            newRasterfn=outdir+f
            # exit()
            date=f.split('.')[0]
            array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
            array = np.array(array, dtype=np.float)
            array = array[::-1]
            to_raster.array2raster(newRasterfn,originX,originY,pixelWidth,pixelHeight,array,ndv = -999999)
            # plt.imshow(array)
            # plt.show()



    def read_mat(self):
        fdir = data_root + '/MODIS_LAI/BU_MCD_LAI_CMG005_NSustain_00_19/'
        outdir=data_root+'Data/MODIS_LAI/BU_MCD_LAI/'
        T.open_path_and_file(outdir)
        T.mk_dir(outdir,True)

        for f in tqdm(os.listdir(fdir), desc='loading...'):

            if f.startswith('.'):
                continue
            if not f.endswith('.mat'):
                continue
            print(f)
            fname = 'MODIS_LAI_' + f.split('.')[0].split('_')[7]
            print(fname)
            # year=f.split('.')[0].split('_')[3]
            # print(year)
            # exit()
            mat=scio.loadmat(fdir+f)

            # mat = h5py.File(fdir+f)
            # print(list(mat.keys()))

            # NIRv_arr_list = mat['outmat']
            print(mat.items())
            NIRv_arr = mat['outmat'][0][0][0]
            print(NIRv_arr)
            # exit()

            newRasterfn = outdir + fname
            print(newRasterfn)
            array = NIRv_arr
            array = np.array(array)
            # plt.imshow(array)
            # plt.show()

            # array = array[::-1]
            # array = array.T
            # array[array > 1] = np.nan
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

        # 配置文件
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



    # function 求LST 平均温度
    def Mean_LST(self):
        fdir1 = data_root + 'Terraclimate/Temp/tmmn_resample/'
        fdir2 = data_root + 'Terraclimate/Temp/tmmx_resample/'
        outdir = data_root + 'Terraclimate/Temp/mean_temperature/'

        T.mk_dir(outdir)

        for f1 in tqdm(sorted(os.listdir(fdir1))):
            for f2 in tqdm(sorted(os.listdir(fdir2))):
                if not f1.endswith('.tif'):
                    continue
                if not f2.endswith('.tif'):
                    continue
                print(f1.split('_')[2])
                print(f2.split('_')[2])
                exit()
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
                    # 配置文件
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

    def sum_PET(self):
        fdir = data_root + 'Terraclimate/test/'
        outdir = data_root + 'Terraclimate/test/'

        T.mk_dir(outdir)
        arr_sum=0

        for f in tqdm(os.listdir(fdir), ):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            print(f)
            date = f.split('.')[0]
            print(date)

            # dataset = gdal.Open(fdir+f)
            # band1 = dataset.GetRasterBand(1)
            # band1_array = band1.ReadAsArray()
            #
            # # 配置文件
            # geotransform = dataset.GetGeoTransform()
            # longitude_start = geotransform[0]
            # latitude_start = geotransform[3]
            # pixelWidth = geotransform[1]
            # pixelHeight = geotransform[5]

            arr = to_raster.raster2array(fdir + f)[0]
            arr[arr < 0] = np.nan
            arr_sum+=arr


        plt.figure()
        plt.imshow(arr_sum)
        plt.show()
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        # newRasterfn = outdir + '{}'.format(date) + '.tif'
        DIC_and_TIF().arr_to_tif(arr_sum, outdir + 'NIRv_resample_' + '{}'.format(date) + '.tif')

    def mean_aridity(self):
        fdir = data_root + 'Terraclimate/test/'
        outdir = data_root + 'Terraclimate/test/'

        T.mk_dir(outdir)

        arr_list=[]
        for f in tqdm(os.listdir(fdir), ):
            if not f.endswith('.tif'):
                continue
            if f.startswith('._'):
                continue
            print(f)
            date = f.split('.')[0]
            print(date)

            # dataset = gdal.Open(fdir+f)
            # band1 = dataset.GetRasterBand(1)
            # band1_array = band1.ReadAsArray()
            #
            # # 配置文件
            # geotransform = dataset.GetGeoTransform()
            # longitude_start = geotransform[0]
            # latitude_start = geotransform[3]
            # pixelWidth = geotransform[1]
            # pixelHeight = geotransform[5]

            arr = to_raster.raster2array(fdir + f)[0]
            arr[arr < 0] = np.nan
            arr_list.append(arr)

        aridity_array_mean = [[0 for col in range(len(arr[0]))] for row in range(len(arr))]

        for r in tqdm(range(len(arr))):
            for c in range(len(arr[0])):
                aridity_array_list = []
                for arr in arr_list:
                    aridity_array_list.append(arr[r][c])
                val=np.nanmean(aridity_array_list)
                aridity_array_mean[r][c] = np.array(val)


        plt.figure()
        plt.imshow(aridity_array_mean)
        plt.show()
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()
        # newRasterfn = outdir + '{}'.format(date) + '.tif'
        # DIC_and_TIF().arr_to_tif(arr_sum, outdir + 'NIRv_resample_' + '{}'.format(date) + '.tif')


    def aridity(self): # 实现P:PET  #### 今后遍历

        fdir1 = data_root + 'Terraclimate/Precip/precip_resample/'
        fdir2 = data_root + 'Terraclimate/PET/'

        outdir = data_root + 'Terraclimate/aridity_P_PET/'
        T.mk_dir(outdir)
        date_dic={}
        date_dic_2={}
        for year in range(1982,2019):
            for month in range(1,13):
                date_dic[(year,month)]=[]
                date_dic_2[(year,month)]=[]

        for f1 in tqdm(sorted(os.listdir(fdir1))):
            if not f1.endswith('.tif'):
                continue
            if f1.startswith('.'):
                continue
            f1_re=f1.replace('precip_resample_','')
            f1_slpit=f1_re.split('.')[0].split('-')
            year,month,day=f1_slpit
            year=int(year)
            month=int(month)
            if (year,month) not in date_dic:
                continue
            date_dic[(year,month)].append(fdir1+f1)
        # print(date_dic)

        for f2 in tqdm(sorted(os.listdir(fdir2))):
            if not f2.endswith('.tif'):
                continue
            if f2.startswith('.'):
                continue

            f2_slpit=f2.split('.')[0]
            year=f2_slpit[0:4]
            month=f2_slpit[4:7]
            year=int(year)
            month=int(month)

            if (year,month) not in date_dic_2:
                continue
            date_dic_2[(year,month)].append(fdir2+f2)
        # print(date_dic_2)

        for date in tqdm(date_dic):
            fpath1=date_dic[date][0]
            fpath2=date_dic_2[date][0]
            dataset1 = gdal.Open(fpath1)
            dataset2 = gdal.Open(fpath2)
            bandf1 = dataset1.GetRasterBand(1)
            bandf1_array = bandf1.ReadAsArray()
            bandf1_array = bandf1_array[:-1]

            bandf2 = dataset2.GetRasterBand(1)
            bandf2_array = bandf2.ReadAsArray()
            # mean_array = np.zeros(range(len(band2_array)), range(len(band2_array)[0]))
            # mean_array = [[0] * len(band2_array[0]) for row in range(len(band2_array))]
            aridity_array = np.ones_like(bandf2_array)*(-999999)
            for r in range(len(bandf1_array)):
                for c in range(len(bandf1_array[0])):
                    if bandf2_array[r][c]<= 0:
                        continue
                    val1=bandf1_array[r][c]
                    val2 = bandf2_array[r][c]
                    if val1<-9999:
                        continue
                    if val2<-9999:
                        continue
                    # print(r,c)
                    aridity_array[r][c] = (bandf1_array[r][c] / bandf2_array[r][c])
            aridity_array = np.array(aridity_array)
            # plt.imshow(aridity_array)
            # plt.show()
            # 配置文件
            geotransform = dataset1.GetGeoTransform()
            longitude_start = geotransform[0]
            latitude_start = geotransform[3]
            pixelWidth = geotransform[1]
            pixelHeight = geotransform[5]
            year,month=date

            newRasterfn = outdir + 'aridity_{}{:02d}.tif'.format(year,month)

            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, aridity_array, ndv=-999999)

    def alignment(self): # 对于行列缺失的（例如缺失南极，补齐功能）

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
        fdir = data_root+'LAI_BU_4g/LAI_BU_4g_TIFF_resample_0.5/'
        outdir = data_root+'LAI_BU_4g/LAI_BU_4g_MVC/'
        Tools().mk_dir(outdir)
        for year in range (1982,2021):
            for month in range (1,13):
                date='{}{:02}'.format(year,month)
                one_month_tif=[]
                for f in os.listdir(fdir):
                    if not f.endswith('.tif'):
                        continue

                    date_i=f.split('.')[0]
                    date_i=date_i[:6]
                    print(date_i)
                    print(date)
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
                        MVC_value[i][j] = maxv/1000.
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


    def days_to_monthly(self):  # Wen CCI_SM  daily to monthly

        fdir = '/Volumes/SSD_sumsang/project_greening/Data/CCI_SM_2020/CCI_SM_2020_TIFF/'
        outdir = '/Volumes/SSD_sumsang/project_greening/Data/CCI_SM_2020/CCI_SM_montly_composite/'
        T.mk_dir(outdir)


        for y in range(2018,2021):
            for m in range(1, 13):
                date = '{}{:02d}'.format(y, m)
                print (date)
                one_month_tif = []
                for f in os.listdir(fdir):
                    if not f.endswith('.tif'):
                        continue
                    date_i = f.split('.')[0].split('_')[2][0:6]
                    if date_i == date:
                        one_month_tif.append(f)
                arrs = []
                for tif in one_month_tif:
                    arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + tif)
                    # arr = to_raster.raster2array(fdir + tif)[0]
                    arr = np.array(arr,dtype=np.float)
                    arr[arr < 0] = np.nan
                    arrs.append(arr)
                # print arrs
                if len(arrs) == 0:
                    continue
                print(len(arrs))
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
                # arr = DIC_and_TIF().pix_dic_to_spatial_arr(one_month_mean)
                newRasterfn = outdir + 'CCI_SM_'+ date + '.tif'
                to_raster.array2raster(newRasterfn, originX, originY, pixelWidth, pixelHeight, one_month_mean, ndv=-999999)


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
        for r in range(len(all_array[0])): #构造字典的键值，并且字典的键：值初始化
            for c in range(len(all_array[1])):
                dic[(r,c)]=[]
                key_list.append((r,c))
        #print(dic_key_list)

        for r in range(len(all_array[0])): # 构造time series
            for c in range(len(all_array[1])):
                for arr in all_array:
                    value=arr[r][c]
                    dic[(r,c)].append(value)
                #print(dic)
        time_series=[]
        flag=0
        temp_dic={}
        for key in key_list: #存数据
            flag=flag+1
            time_series=dic[key]
            time_series=np.array(time_series)
            temp_dic[key]=time_series
            if flag %10000 == 0:
                np.save(outdir +'per_pix_dic_%03d' % (flag/10000),temp_dic)
                temp_dic={}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)


    def unify_raster(self):

        fdir_all = 'C:/Users/pcadmin/Desktop/Trendy_TIFF_resample/'
        outdir_all = 'C:/Users/pcadmin/Desktop/Trendy_TIFF_resample_unify/'

        for fdir in tqdm(os.listdir(fdir_all)):
            print(fdir)
            outdir = join(outdir_all, fdir + '/')
            T.mk_dir(outdir, force=True)

            for f in os.listdir(fdir_all + fdir + '/'):
                if f.startswith('.'):
                    continue
                if f.endswith('.xml'):
                    continue
                fin = join(fdir_all + fdir + '/', f)
                print(fin)
                outf = join(outdir, f.split('.')[0] + '.tif')
                print(outf)
                ToRaster().unify_raster(fin, outf)



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