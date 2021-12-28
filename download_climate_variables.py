# coding=utf-8
import requests
import os
import codecs
import time
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
# import analysis
from __init__ import *

year = []
for y in range(1982,2021):
    year.append(str(y))

def download(y):
    outdir = '/Volumes/SSD_sumsang/project_greening/Data/climate_data/'
    T.mk_dir(outdir,force=True)
    product = 'srad'
    # url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_pet_{}.nc'.format(y)
    # url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_ppt_{}.nc'.format(y)
    # url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_vpd_{}.nc'.format(y)
    # url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{}_{}.nc'.format(product,y)
    url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{}_{}.nc'.format(product,y)
    print(url)
    while 1:
        try:
            req = requests.request('GET', url)
            content = req.content
            fw = open(outdir+'{}_{}.nc'.format(product,y), 'wb')
            fw.write(content)
            return None
        except Exception as e:
            print(url, 'error sleep 5s')
            time.sleep(5)


MULTIPROCESS(download,year).run(process=3,process_or_thread='t')