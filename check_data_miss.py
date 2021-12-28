# coding=utf-8
import os
from tqdm import tqdm
from LY_Tools import *
fdir1 = '/Volumes/1T/wen_prj/Data/MYD11C2/MYD11C2_LST_day/urls/MYD11C2.006/'
fdir2 = '/Volumes/1T/wen_prj/Data/MYD11C2/MYD11C2_LST_day/tif/'
outdir = '/Volumes/1T/wen_prj/Data/MYD11C2/MYD11C2_LST_day/urls/missingdata_check/'
Tools().mk_dir(outdir)
f_invalid = outdir + 'url_sup.txt'
fw_invalid = open(f_invalid,'w')
for f1 in tqdm(sorted(os.listdir(fdir1))):
    success=0
    # f1 = str(f1)
    if f1.startswith('.'):
        continue
    for f2 in sorted(os.listdir(fdir2)):
        if f1[:10]==f2[:10]:
            success=1
            break
    if success == 0:
        # print(f1)
        fr = open(fdir1 + f1,'r').read()
        fw_invalid.write(fr)
fw_invalid.close()






