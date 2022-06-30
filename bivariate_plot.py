# coding=utf-8

from libpysal import examples
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import rasterio as rio
from splot.mapping import vba_choropleth
from shapely.geometry import Point, Polygon

from __init__ import *


def wkt():
    wkt = '''
    PROJCRS["North_Pole_Orthographic",
        BASEGEOGCRS["WGS 84",
            DATUM["World Geodetic System 1984",
                ELLIPSOID["WGS 84",6378137,298.257223563,
                    LENGTHUNIT["metre",1]]],
            PRIMEM["Greenwich",0,
                ANGLEUNIT["Degree",0.0174532925199433]]],
        CONVERSION["North_Pole_Orthographic",
            METHOD["Orthographic",
                ID["EPSG",9840]],
            PARAMETER["Latitude of natural origin",90,
                ANGLEUNIT["Degree",0.0174532925199433],
                ID["EPSG",8801]],
            PARAMETER["Longitude of natural origin",0,
                ANGLEUNIT["Degree",0.0174532925199433],
                ID["EPSG",8802]],
            PARAMETER["False easting",0,
                LENGTHUNIT["metre",1],
                ID["EPSG",8806]],
            PARAMETER["False northing",0,
                LENGTHUNIT["metre",1],
                ID["EPSG",8807]]],
        CS[Cartesian,2],
            AXIS["(E)",east,
                ORDER[1],
                LENGTHUNIT["metre",1]],
            AXIS["(N)",north,
                ORDER[2],
                LENGTHUNIT["metre",1]],
        USAGE[
            SCOPE["unknown"],
            AREA["World - north of 0°N"],
            BBOX[0,-180,90,180]],
        ID["ESRI",102035]]
    Proj4
    +proj=ortho +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs
    Extent
    -180.00, 0.00, 180.00, 90.00'''
    return wkt

def bivariate_plot():
    # todo: bivariate color ramp for classes needs to be fixed
    world_shp = '/Volumes/NVME2T/hotcold_drought/shp/world.shp'
    tif1 = '/Volumes/NVME2T/greening_project_redo/data/moving_window_corr/GIMMS3g_1982-2020_r-trend/CCI_SM_r-trend.tif'
    tif2 = '/Volumes/NVME2T/greening_project_redo/data/moving_window_corr/GIMMS3g_1982-2020_r-trend/CO2_r-trend.tif'
    # gdf1 = raster_to_gdf(tif1,colname='CCI_SM')
    # gdf2 = raster_to_gdf(tif2,colname='CO2')
    gdf = raster_list_to_gdf([tif1,tif2],['CCI_SM','CO2'])
    world = gpd.read_file(world_shp)
    gdf = gdf.to_crs(crs=wkt())
    world = world.to_crs(crs=wkt())
    fig = plt.figure(figsize=(15, 3))
    ax = fig.add_subplot(111)
    world.plot(ax=ax, color='w', edgecolor='black')
    vba_choropleth('CCI_SM', 'CO2', gdf,cmap='RdBu',divergent=False,revert_alpha=True,
                            legend=True,
                            rgb_mapclassify=dict(classifier='std_mean', k=10),
                            alpha_mapclassify=dict(classifier='std_mean', k=10),
                            ax=ax)
    plt.tight_layout()
    plt.show()



def bivariate_tifs_to_shp_and_reclass():
    # todo: change here @Wen bivariate_tifs_to_shp

    ## tif1 and tif2 are the two tifs to be bivariate reclassified
    tif1 = '/Volumes/NVME2T/greening_project_redo/data/moving_window_corr/GIMMS3g_1982-2020_r-trend/CCI_SM_r-trend.tif'
    tif2 = '/Volumes/NVME2T/greening_project_redo/data/moving_window_corr/GIMMS3g_1982-2020_r-trend/Temp_r-trend.tif'
    outshp = 'xxxx.shp'  # needs to be modified


    df = raster_list_to_gdf([tif1, tif2], ['var1', 'var2'])
    var1_std = df['var1'].std()/2
    var2_std = df['var2'].std()/2
    var1_mean = df['var1'].mean()
    var2_mean = df['var2'].mean()
    # reindex
    df = df.reset_index(drop=True)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        val1 = row['var1']
        val2 = row['var2']
        if val1 >= var1_mean + var1_std:
            df.loc[i, 'var1_class'] = '3'
        elif var1_mean - var1_std < val1 < var1_mean + var1_std:
            df.loc[i, 'var1_class'] = '2'
        elif val1 <= var1_mean - var1_std:
            df.loc[i, 'var1_class'] = '1'
        else:
            df.loc[i, 'var1_class'] = '9999'
        if val2 >= var2_mean + var2_std:
            df.loc[i, 'var2_class'] = 'C'
        elif var2_mean - var2_std < val2 < var2_mean + var2_std:
            df.loc[i, 'var2_class'] = 'B'
        elif val2 < var2_mean - var2_std:
            df.loc[i, 'var2_class'] = 'A'
        else:
            df.loc[i, 'var2_class'] = 'ZZZZ'
    df = df.dropna()
    # merge var1_class and var2_class
    df['class'] = df['var1_class'] + df['var2_class']
    T.print_head_n(df, n=10)
    df.to_file(outshp)

def raster_to_gdf(intif,colname='value',unique_key='X_Y'):
    '''
    Read a tif file and return a geopandas dataframe
    '''
    with rio.open(intif) as src:
        crs = src.crs
        meta = src.meta
        xmin, ymin, xmax, ymax = src.bounds
        pix_width = (xmax - xmin) / meta['width']
        pix_height = (ymax - ymin) / meta['height']
        # create 1D coordinate arrays (coordinates of the pixel center)
        x = np.linspace(xmin, xmax-pix_width, src.width)
        y = np.linspace(ymax, ymin+pix_height, src.height)  # max -> min so coords are top -> bottom
        # create 2D arrays
        xs, ys = np.meshgrid(x, y)
        zs = src.read(1)
    mask = zs == src.nodata
    xs = xs[~mask]
    ys = ys[~mask]
    zs = zs[~mask]
    data = {"X": pd.Series(xs.ravel()),
            "Y": pd.Series(ys.ravel()),
            colname: pd.Series(zs.ravel())}
    df = pd.DataFrame(data=data)
    geometry = gpd.points_from_xy(df.X, df.Y)
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    gdf[unique_key] = gdf.X.astype(str) + '_' + gdf.Y.astype(str)
    return gdf

def raster_list_to_gdf(intif_list,colname_list):
    '''
    Read a tif file and return a geopandas dataframe
    '''
    Xs_list = []
    Ys_list = []
    Zs_list = []
    meta_list = []
    for intif, colname in zip(intif_list, colname_list):
        with rio.open(intif) as src:
            crs = src.crs
            meta = src.meta
            xmin, ymin, xmax, ymax = src.bounds
            pix_width = (xmax - xmin) / meta['width']
            pix_height = (ymax - ymin) / meta['height']
            # create 1D coordinate arrays (coordinates of the pixel center)
            x = np.linspace(xmin, xmax-pix_width, src.width)
            y = np.linspace(ymax, ymin+pix_height, src.height)  # max -> min so coords are top -> bottom
            # create 2D arrays
            xs, ys = np.meshgrid(x, y)
            zs = src.read(1)
            Xs_list.append(xs)
            Ys_list.append(ys)
            Zs_list.append(zs)
            meta_list.append(meta)
    data = {"X": pd.Series(Xs_list[0].ravel()),
            "Y": pd.Series(Ys_list[0].ravel())}
    for i in range(len(Xs_list)):
        data[colname_list[i]] = pd.Series(Zs_list[i].ravel())
    df = pd.DataFrame(data=data)
    geometry = [Polygon([(x, y), (x, y+pix_height), (x+pix_width, y+pix_height), (x+pix_width, y)]) for x, y in zip(df.X, df.Y)]
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    # print(gdf)
    for col in colname_list:
        gdf[gdf[col]==src.nodata] = np.nan
    gdf = gdf.dropna(subset=colname_list)
    return gdf



def main():
    bivariate_tifs_to_shp_and_reclass()
    # bivariate_plot()

if __name__ == '__main__':

    main()




