# coding=utf-8
from __init__ import *
import ee
variable='soil'
outdir=data_root+'Terraclimate/{}/'.format(variable)
T.mk_dir(outdir,True)


def run():
    year_range=range(1991,2021)
    month_range=range(1,13)
    datelist=[]
    for year in year_range:
        for month in month_range:
            datestr=str(year)+'-'+'{:02d}'.format(month)+'-'+'01'
            datelist.append(datestr)
    # print(datelist)
    for i in tqdm(range(len(datelist))):
        outf=outdir+'{}_{}'.format(variable, datelist[i])+'.zip'
        print(outf)
        if i+1>=len(datelist):
            break
        url=calculate_variables(datelist[i],datelist[i+1])
        download_file(url,outf)


def calculate_variables(start, end):

    # para=0.45
    ee.Initialize()
    dataset_name="IDAHO_EPSCOR/TERRACLIMATE"  # 数据集
    dataset=ee.ImageCollection(dataset_name)

    selected_band=dataset.filterDate(start, end).select(variable)  # 想要的变量
    image=selected_band.mean()
    newname='{}_'.format(variable)+start
    # print(newname)
    # exit()
    # par=image.multiply(0.45).rename(newname)
    # print(newname)
    tmin=image.rename(newname)
    url = tmin.getDownloadUrl({'scale': 40000,
                              'crs': 'EPSG:4326', })

    # url=par.getDownloadUrl({ 'scale':40000,
    #                          'crs':'EPSG:4326',})
    return url


def download_file(url, outf):
    response=requests.get(url)
    fw=open(outf,'wb')
    fw.write(response.content)
    fw.close()


def main():

    run()

if __name__ == '__main__':
    main()





