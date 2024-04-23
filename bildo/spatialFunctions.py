# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:33:31 2020

@author: jcintasr-work
"""

import os
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy as np
import bildo
import geopandas as gpd
# import matplotlib.pyplot as plt
# import time

gdal.UseExceptions()

class NoneDsErrorClass(ValueError):
    pass
    #print(ValueError)


def get_rasterExtent(raster, dictionary=False):
    if type(raster) is str:
        r = gdal.Open(raster)
    else:
        r = raster
    ulx, xres, xskew, uly, yskew, yres = r.GetGeoTransform()
    lrx = ulx + (r.RasterXSize * xres)
    rly = uly + (r.RasterYSize * yres)

    # xmin, xmax, ymin and ymax
    extent = [ulx, lrx, rly, uly]
    if dictionary:
        return({raster: extent})
    else:
        return (extent)


def getPolyBoundary(raster):
    """
    Creates polygon from extent
    """

    if type(raster) is str:
        r = gdal.Open(raster)
    else:
        r = raster

    srs = r.GetProjection()
    lx, rx, ly, uy = get_rasterExtent(r)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(lx, uy)  # ul corner
    ring.AddPoint(rx, uy)  # ur corner
    ring.AddPoint(rx, ly)  # lr corner
    ring.AddPoint(lx, ly)  # ll corner
    ring.AddPoint(lx, uy)  # ul corner again to close polygon

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly

def getPolyBoundary_(bildo_):
    """
    Creates polygon from extent
    """

    if type(bildo_) is str:
        r = bildo.openBildo(bildo_)
    else:
        r = bildo_

    srs = r.crs
    lx, rx, ly, uy = r.extent

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(lx, uy)  # ul corner
    ring.AddPoint(rx, uy)  # ur corner
    ring.AddPoint(rx, ly)  # lr corner
    ring.AddPoint(lx, ly)  # ll corner
    ring.AddPoint(lx, uy)  # ul corner again to close polygon

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    return poly


def getCoordinates(pointGeometry):
    x = pointGeometry.GetX()
    y = pointGeometry.GetY()
    return([x, y])


def getPixelIndexFromCoordinates(pointGeometry, GeoTransform):
    x, y = getCoordinates(pointGeometry)

    # Adjusting coordinates to pixel indexes.
    # It only works on porjections with no rotation
    xorigin = GeoTransform[0]
    xscale = GeoTransform[1]
    yorigin = GeoTransform[3]
    yscale = GeoTransform[5]

    pi = int((y - yorigin)/yscale)
    pj = int((x - xorigin)/xscale)

    return([pi, pj])


def ds_readAsArray(ds, index0=False):
    # This images seems to not be 0 indexed code(I don't know how)
    if index0 is True:
        nBands = ds.RasterCount
        rango = range(nBands)
    else:
        nBands = ds.RasterCount + 1
        rango = range(1, nBands)

    arrayList = list()
    for k in rango:
        tmpBand = ds.GetRasterBand(k)
        if tmpBand is not None:
            tmpArray = tmpBand.ReadAsArray()
            arrayList.append(tmpArray)

    return np.array(arrayList)


def normalise(array):
    minimum, maximum = array.min(), array.max()
    normal = (array - minimum)*((255)/(maximum - minimum))+0
    return normal


# def plotRGB(nArray, r=3, g=2, b=1, normalization=False):
#     import matplotlib.pyplot as plt

#     if type(nArray) is gdal.Dataset:
#         nArray = ds_readAsArray(nArray)

#     if type(nArray) is not np.ndarray:
#         print("nArray must be an array or a gdal.Dataset")
#         return None

#     red = nArray[:][:][r-1]
#     green = nArray[:][:][g-1]
#     blue = nArray[:][:][b-1]

#     if normalization:
#         def normalise(array):
#             minimum, maximum = array.min(), array.max()
#             normal = (array - minimum)*((255)/(maximum - minimum))+0
#             return normal

#         red = normalise(red)
#         green = normalise(green)
#         blue = normalise(blue)

#     rgb = np.dstack((red, green, blue)).astype(int)

#     plt.imshow(rgb)


def getLayerExtent(layer_path):
    longitud = len(layer_path.split("."))
    driver_name = layer_path.split(".")[longitud - 1]
    if driver_name == "gpkg":
        driver = ogr.GetDriverByName("GPKG")
    elif driver_name == "shp":
        driver = ogr.GetDriverByName("ESRI Shapefile")

    elif driver_name == "kml":
        driver = ogr.GetDriverByName("KML")

    ds = driver.Open(layer_path)
    xmin, xmax, ymin, ymax = ds.GetLayer().GetExtent()
    extent = [xmin, ymin, xmax, ymax]

    del ds

    return extent


def create_raster(in_ds, fn, data, data_type, nodata=None, driver="GTiff",
                  band_names=None, createOverviews=False, crs=None,
                  GeoTransform=None, rows_cols=None, return_ds=False,
                  compute_statistics=False, **kwargs):
    """
    Based on Geoprocessing with python.
    Create a one-band GeoTiff

    in_ds         - datasource to copy projection and geotransform from
    fn            - path to the file to create
    data          - NumPy array containing data to write
    data_type     - output data type
    nodata        - optional NoData value
    band_names    - optional. It gives a name to each band for easier identification. It has to have same length than data dimensons.
    """

    driver = gdal.GetDriverByName(driver)
    #     print(band_names)
    # Creating out raster

    # Getting columns and rows
    if rows_cols is None:
        # columns = in_ds.RasterXSize
        # rows = in_ds.RasterYSize
        lengthShape = len(data.shape)
        if lengthShape > 2:
            nbands, rows, columns = data.shape
        else:
            nbands = 1
            rows, columns = data.shape
    else:
        if type(rows_cols) is not tuple:
            print("rows_cols has to be a tuple")
            return None
        # rows = rows_cols[0]
        # columns = rows_cols[1]
        rows = rows_cols[1]
        columns = rows_cols[2]

    out_ds = driver.Create(fn, columns, rows, nbands, data_type, **kwargs)
    print(out_ds)
    if(out_ds is None):
        print("output creation failed!. Unable to create output datasource")
        return None

    if GeoTransform is not None:
        out_ds.SetGeoTransform(GeoTransform)
    else:
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    # Assigning out raster projection and geotransform
    if crs is None:
        out_ds.SetProjection(in_ds.GetProjection())
    else:
        srs = osr.SpatialReference()
        if type(crs) is int:
            srs.ImportFromEPSG(crs)
        elif type(crs) is str:
            try:
                srs.ImportFromWkt(crs)
            except:
                srs.ImportFromProj4(crs)

        elif type(crs) is osr.SpatialReference:
            srs = crs

        out_ds.SetProjection(srs.ExportToWkt())

    # Iterate through bands if necessary
    if nbands > 1:
        for k in range(0, nbands):
            out_band = out_ds.GetRasterBand(k + 1)
            if nodata is not None:
                out_band.SetNoDataValue(nodata)
            # out_band.WriteArray(data[:, :, k])
            out_band.WriteArray(data[k, :, :])

            if band_names is not None:
                out_band.SetDescription(band_names[k])
                metadata = out_band.GetMetadata()
                metadata = f"TIFFTAG_IMAGEDESCRIPTION={band_names[k]}"
                out_band.SetMetadata(metadata)

    else:
        out_band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)

        if len(data.shape) > 2: data = data[0]
        out_band.WriteArray(data)
    #         print(out_band.ReadAsArray())

    out_band.FlushCache()
    if compute_statistics:
        out_band.ComputeStatistics(False)
    if createOverviews:
        out_band.BuildOverViews('average', [2, 4, 8, 16, 32, 64])

    if return_ds:
        del out_band
        return out_ds
    else:
        del out_band
        del out_ds
        return "Done!"

def create_raster_(data, fn, data_type=None, template_ds=None, nodata=None, driver="GTiff",
                   band_names=None, meta_vortaroj=None, createOverviews=False, crs=None,
                   GeoTransform=None, rows_cols=None, return_ds=False,
                   compute_statistics=False, **kwargs):
    """
    Based on Geoprocessing with python.
    Create a one-band GeoTiff

    fn            - path to the file to create
    data          - NumPy array containing data to write
    data_type     - output data type
    template_ds         - datasource to copy projection and geotransform from
    nodata        - optional NoData value
    band_names    - optional. It gives a name to each band for easier identification. It has to have same length than data dimensons.
    """

    if template_ds is None:
        if GeoTransform is None or crs is None:
            raise ValueError("If template_ds is None, then GeoTransform, and crs must be defined")
        elif data_type is None:
            raise ValueError("If template_ds is None, then data_type must be defined")
        else:
            pass

    if data_type is None:
        print("data_type is None, assuming all the bands has the same type as the first one.")
        data_type = template_ds.GetRasterBand(1).DataType

    driver = gdal.GetDriverByName(driver)
    #     print(band_names)
    # Creating out raster

    # Getting columns and rows
    if rows_cols is None:
        # columns = template_ds.RasterXSize
        # rows = template_ds.RasterYSize
        lengthShape = len(data.shape)
        if lengthShape > 2:
            nbands, rows, columns = data.shape
        else:
            nbands = 1
            rows, columns = data.shape
    else:
        if type(rows_cols) is not tuple:
            raise ValueError("rows_cols has to be a tuple")
            
        # rows = rows_cols[0]
        # columns = rows_cols[1]
        rows = rows_cols[1]
        columns = rows_cols[2]

    out_ds = driver.Create(fn, columns, rows, nbands, data_type, **kwargs)
    print(out_ds)
    if(out_ds is None):
        raise ValueError("output creation failed!. Unable to create output datasource")

    if GeoTransform is not None:
        out_ds.SetGeoTransform(GeoTransform)
    else:
        out_ds.SetGeoTransform(template_ds.GetGeoTransform())

    # Assigning out raster projection and geotransform
    if crs is None:
        out_ds.SetProjection(template_ds.GetProjection())
    else:
        srs = osr.SpatialReference()
        if type(crs) is int:
            srs.ImportFromEPSG(crs)
        elif type(crs) is str:
            try:
                srs.ImportFromWkt(crs)
            except:
                srs.ImportFromProj4(crs)

        elif type(crs) is osr.SpatialReference:
            srs = crs

        out_ds.SetProjection(srs.ExportToWkt())

    # Iterate through bands if necessary
    # I have to deal with more bands in the array
    # than in the template (linkink to bildo.writeToDisk)
    if nbands > 1:
        for k in range(0, nbands):
            out_band = out_ds.GetRasterBand(k + 1)
            if nodata is not None:
                out_band.SetNoDataValue(nodata)
            # out_band.WriteArray(data[:, :, k])
            # out_band.WriteArray(data[k, :, :])

            if band_names is not None:
                out_band.SetDescription(band_names[k])
                metadata = out_band.GetMetadata()
                metadata = f"TIFFTAG_IMAGEDESCRIPTION={band_names[k]}"
                out_band.SetMetadata(metadata)
                del metadata

            if meta_vortaroj is not None:
                metadata = out_band.GetMetadata()

                # for metavort in meta_vortaroj:
                metavort = meta_vortaroj[k]
                for key, value in metavort.items():
                    # print(k,v)
                    # metadata[k].append(v)
                    # print(metadata)
                    #metadata = f"{key}={value}"
                    metadata[str(key)] = str(value)
                    # print(metadata)
                out_band.SetMetadata(metadata)
                del metadata
                    
            elif meta_vortaroj is None and template_ds is not None:
                if template_ds.RasterCount < nbands:
                    pass
                else:
                    template_band = template_ds.GetRasterBand(k+1)
                    metadata = template_band.GetMetadata()
                    # print(metadata)
                    for key, value in metadata.items():
                        metadata[str(key)] = str(value)
                    out_band.SetMetadata(metadata)
                    del metadata, template_band

            elif meta_vortaroj is None and template_ds is None:
                pass

            out_band.WriteArray(data[k, :, :])

            
            # I change this, shoueld I change it again
            # out_band.FlushCache()
            if compute_statistics:
                out_band.ComputeStatistics(False)
            if createOverviews:
                out_band.BuildOverViews('average', [2, 4, 8, 16, 32, 64])
            
            ## This is new
            out_band.FlushCache()
            out_band = None
            del out_band

    else:
        k = 0
        out_band = out_ds.GetRasterBand(1)
        if nodata is not None:
            out_band.SetNoDataValue(nodata)

        if band_names is not None:
            out_band.SetDescription(band_names[k])
            metadata = out_band.GetMetadata()
            metadata = f"TIFFTAG_IMAGEDESCRIPTION={band_names[k]}"
            out_band.SetMetadata(metadata)
            del metadata

        if meta_vortaroj is not None:
            metadata = out_band.GetMetadata()

            # for metavort in meta_vortaroj:
            metavort = meta_vortaroj[k]
            for key, value in metavort.items():
                # print(k,v)
                # metadata[k].append(v)
                # print(metadata)
                #metadata = f"{key}={value}"
                metadata[str(key)] = str(value)
                # print(metadata)
            out_band.SetMetadata(metadata)
            del metadata

        elif meta_vortaroj is None and template_ds is not None:
            # I don't understand this now
            # If meta_vortaroj is None,
            # I don't want to add anything more.
            ## I added an extra condition. template_ds should be none
            ## then, only metadata is copied if template_ds exists
            template_band = template_ds.GetRasterBand(k+1)
            metadata = template_band.GetMetadata()
            # print(metadata)
            for key, value in metadata.items():
                metadata[str(key)] = str(value)
            out_band.SetMetadata(metadata)
            del metadata, template_band
            
        if len(data.shape) > 2: data = data[0]
        out_band.WriteArray(data)

        # I change this, shoueld I change it again
        # out_band.FlushCache()
        if compute_statistics:
            out_band.ComputeStatistics(False)
        if createOverviews:
            out_band.BuildOverViews('average', [2, 4, 8, 16, 32, 64])
        
        
        out_band.FlushCache()
        out_band = None
        del out_band

        
    if return_ds:
        # del template_band
        template_ds = None
        del template_ds
        return out_ds
    else:
        template_ds = None
        del template_ds
        # del template_band
        out_ds = None
        del out_ds
        return "Done!"


def create_layer(output, feature_list,
                 driver_name="GPKG", crs=25830,
                 geom_type=ogr.wkbPolygon, data_type=ogr.OFTReal):
    """
    output_name         -  Name of the file to create with extension
    feature_dictionary  -  list with two elements, geometry and a list with a dictionary with the name of the field at the keys and its values at the values.
    driver_name         -  driver to use. GPKG by default.
    epsg                -  epsg code to assign projection
    geom_type           -  geom type of the geometry suplied
    data_type           -  data_type of the values
    """

    # Getting name of the output without path and extension
    output_layer_tmp = output.split("/")
    output_layer_tmp2 = output_layer_tmp[len(output_layer_tmp) - 1]
    output_layer_name = output_layer_tmp2.split(".")[0]
    #     print(output_layer_name)

    # Getting srs
    out_srs = osr.SpatialReference()

    if type(crs) is int:
        out_srs.ImportFromEPSG(crs)
    elif type(crs) is str:
        out_srs.ImportFromWkt(crs)
    #     print(out_srs)

    # create output layer
    driver = ogr.GetDriverByName(driver_name)
    if os.path.exists(output):
        driver.DeleteDataSource(output)
    out_ds = driver.CreateDataSource(output)
    if out_ds is None:
        print("output data source is None")
        return 1
    out_layer = out_ds.CreateLayer(
        output_layer_name, geom_type=geom_type, srs=out_srs)

    # very important matter to reset Reading after define out layer
    out_layer.ResetReading()
    #     print(out_layer)

    # Iterate through list to get fields and create them
    count = 0
    for feature in feature_list:
        diccionario_tmp = feature[1]
        diccionario = diccionario_tmp[0]

        fieldNames = []
        for field in diccionario.keys():
            # Checking if the field alerady exists
            if count > 0:
                for f in range(out_layer.GetLayerDefn().GetFieldCount()):
                    fieldNames.append(
                        out_layer.GetLayerDefn().GetFieldDefn(f).name)

            if field not in fieldNames:
                outFieldDefn = ogr.FieldDefn(field, data_type)
                out_layer.CreateField(outFieldDefn)

            count += 1

    # Get Layer Definition
    out_layerDefn = out_layer.GetLayerDefn()
    #     print(out_layerDefn.GetGeomFieldDefn())
    #     print(out_layerDefn)

    # Iterate through list to get geometries, fields and values
    # it = 0
    for feature in feature_list:
        geomwkt = feature[0]
        geom = ogr.CreateGeometryFromWkt(geomwkt)

        diccionario_tmp = feature[1]
        diccionario = diccionario_tmp[0]

        ofeat = ogr.Feature(out_layerDefn)
        ofeat.SetGeometry(geom)
        for field, value in diccionario.items():
            ofeat.SetField(field, value)

        #             print(field, value*1.0)

        out_layer.CreateFeature(ofeat)

    out_layer.SyncToDisk()
    out_ds = None


def layer_within_raster(raster_extent, layer_geom, lesser_lextent=False):
    """
    check if a layer is inside the raster
    :param raster_extent: extent of the raster
    :param layer_geom: layer geom
    :param lesser_lextent: If True a smaller extent is evaluated
    :return:
    """
    rxmin, rxmax, rymin, rymax = raster_extent
    lxmin, lxmax, lymin, lymax = layer_geom.GetEnvelope()

    if lesser_lextent:
        # Getting a smaller bounding box
        lxmin = lxmin + 100
        lxmax = lxmax - 100
        lymin = lymin + 100
        lymax = lymax - 100

    i = 0
    if lxmin >= rxmin:  # 1. upper left corner
        i += 1
    if lymax <= rymax:  # 2. upper right corner
        i += 1
    if lxmax <= rxmax:  # 3. lower right corner
        i += 1
    if lymin >= rymin:  # 4. lower left corner
        i += 1

    if i == 4:
        out = True
    else:
        out = False
    return (out)


def compareSpatialReference(ds1, ds2):
    if type(ds1) is gdal.Dataset:
        tmp1 = ds1.GetProjection()
        proj1 = osr.SpatialReference()
        proj1.ImportFromWkt(tmp1)

    elif type(ds1) is ogr.DataSource:
        proj1 = ds1.GetLayer().GetSpatialRef()

    elif type(ds1) is osr.SpatialReference:
        proj1 = ds1

    if type(ds2) is gdal.Dataset:
        tmp2 = ds2.GetProjection()
        proj2 = osr.SpatialReference()
        proj2.ImportFromWkt(tmp2)

    elif type(ds2) is ogr.DataSource:
        proj2 = ds2.GetLayer().GetSpatialRef()

    elif type(ds2) is osr.SpatialReference:
        proj2 = ds2

    if proj1.IsSame(proj2):
        return True
    else:
        return False


def reproject(image, output_folder=None, crs_to=25830, returnPath=False,
              driver="GTiff", resampling=gdal.GRA_NearestNeighbour):
    """
    This function reprojects a raster image
    :param image: path to raster image
    :param output_folder: output folder where the output image will be saved
    :param epsg_to: coordinate epsg code to reproject into
    :param memDs: If True, it returns the output path
    :return: returns a virtual data source
    """

    if driver == "MEM":
        returnPath = False
        output_folder = None

    if output_folder is None:
        driver = "MEM"
        returnPath = False

    else:
        splitted = image.split("/")
        lenout = len(splitted)
        out_name = splitted[lenout-1]
        output = f"{output_folder}/reprojeted_{out_name}"

    dataset = gdal.Open(image)
    srs = osr.SpatialReference()
    if type(crs_to) is int:
        srs.ImportFromEPSG(crs_to)
    elif type(crs_to) is str:
        srs.ImportFromWkt(crs_to)

    vrt_ds = gdal.AutoCreateWarpedVRT(
        dataset, None, srs.ExportToWkt(), resampling)

    if returnPath:
        # cols = vrt_ds.RasterXSize
        # rows = vrt_ds.RasterYSize
        gdal.GetDriverByName(driver).CreateCopy(output, vrt_ds)
        return(output)

    else:
        return(vrt_ds)

def reproject_(bildo_, output_folder=None, crs_to=25830, returnPath=False,
               driver="GTiff", crs_from=None,  resampling=gdal.GRA_NearestNeighbour):
    """
    This function reprojects a raster image
    :param bildo_: path to raster image
    :param output_folder: output folder where the output image will be saved
    :param epsg_to: coordinate epsg code to reproject into
    :param memDs: If True, it returns the output path
    :return: returns a virtual data source
    """

    if type(bildo_) is str:
        image = bildo.openBildo(bildo_)
    elif type(bildo_) is gdal.Dataset:
        image = bildo.openBildo_(bildo_)
    elif type(bildo_) is bildo.bildo:
        image = bildo_
    else:
        raise ValueError("bildo_ should be a bildo class, a gdal dataset or a path to a raster")

    if driver == "MEM":
        returnPath = False
        output_folder = None

    if output_folder is None:
        driver = "MEM"
        returnPath = False

    else:
        splitted = image.path.split("/")
        lenout = len(splitted)
        out_name = splitted[lenout-1]
        output = f"{output_folder}/reprojeted_{out_name}"

    if crs_from is not None:
        srsfrom = osr.SpatialReference()
        if type(crs_from) is int: srsfrom.ImportFromEPSG(crs_from)
        elif type(crs_from) is str: srsfrom.ImportFromWkt(crs_from)
        else: print("crs_from has not been defined")

        image.dataSource.SetProjection(srsfrom.ExportToWkt())
        del srsfrom

    srs = osr.SpatialReference()
    if type(crs_to) is int:
        srs.ImportFromEPSG(crs_to)
    elif type(crs_to) is str:
        srs.ImportFromWkt(crs_to)

    vrt_ds = gdal.AutoCreateWarpedVRT(
        image.dataSource, None, srs.ExportToWkt(), resampling)

    if returnPath:
        # cols = vrt_ds.RasterXSize
        # rows = vrt_ds.RasterYSize
        gdal.GetDriverByName(driver).CreateCopy(output, vrt_ds)
        del vrt_ds
        return(output)

    else:
        return(bildo.openBildo(vrt_ds))


def naming_convention(raster_path, geometry):
    """
    Creates naming based on the raster name and geometries: date_xmin-ymax_sentineltile_band
    :param raster_path: Path to raster file
    :param geometry: geom
    :return:
    """
    # xmin, xmax, ymin, ymax
    xmin, xmax, ymin, ymax = geometry.GetEnvelope()
    splitted = raster_path.split("/")
    len_splitted = len(splitted)
    name_tmp1 = splitted[len_splitted - 1]
    name = name_tmp1.split(".")[0]
    # name_splitted = name.split("_")
    # if len(name_splitted) < 3:
    outaname = f"{name}_{float(xmin)}-{float(ymax)}"
    # else:
    #     sent_tile = name_splitted[0]
    #     band = name_splitted[2]
    #     date_tmp = name_splitted[1]
    #     date = date_tmp.split("T")[0]

    #     # outaname = f"{date}_{int(xmin)}_{int(ymax)}_{sent_tile}_{band}"
    #     outaname = f"{date}_{float(xmin)}-{float(ymax)}_{sent_tile}_{band}"
    return (outaname)


def masking_tiles(layer_tiles,
                  raster_path,
                  output_folder,
                  field="fid_id",
                  naming=False,
                  extent=False,
                  lesser_lextent=False,
                  reproyectar=False,
                  crs=None
                  ):
    """
    It creates tiles from a raster image based on a grid previously created
    :param layer_tiles: Path to grid
    :param raster_path: Path to raster
    :param output_folder: Path to output folder
    :param field: Field with cut tiles with
    :param naming: Apply naming rule
    :param extent: Cut with extent
    :param lesser_lextent: create an smaller extent
    :param reproyectar: If True, reprojection is applied
    :param epsg: EPSG code of the srs to reproject into
    :return:
    """
    if os.path.exists(output_folder) is False:
        os.mkdir(output_folder)

    if reproyectar:
        raster_path2 = raster_path
        raster_path = reproject(raster_path, "/tmp",
                                crs_to=crs, return_output_path=True)
        print(raster_path)

    driver = ogr.GetDriverByName("GPKG")
    ds = driver.Open(layer_tiles)
    layer = ds.GetLayer()
    for feature in layer:
        geom = feature.geometry()
        fid = feature.GetField(field)
        if naming:
            if reproyectar:
                out_name = naming_convention(raster_path2, geom)
            else:
                out_name = naming_convention(raster_path, geom)
        else:
            out_tmp = raster_path.split("/")
            out_tmp2 = out_tmp[len(out_tmp) - 1]
            out_name = out_tmp2.split(".")[0]

        output = f"{output_folder}/{out_name}.tif"

        if extent:
            raster_extent = get_rasterExtent(raster_path)
            sepuede = layer_within_raster(
                raster_extent, geom, lesser_lextent=lesser_lextent)

            if sepuede:
                xmin, xmax, ymin, ymax = geom.GetEnvelope()
                lextent = [xmin, ymin, xmax, ymax]

                ds2 = gdal.Warp(output,
                                raster_path,
                                format="GTiff",
                                outputBounds=lextent)

                del ds2

        else:
            ds2 = gdal.Warp(output,
                            raster_path,
                            format="GTiff",
                            cutlineDSName=layer_tiles,
                            cutlineWhere=f"{field} = '{fid}'",
                            cropToCutline=True)
            del ds2

    layer.ResetReading()
    ds.FlushCache()

    del ds


def whichMin(to_compare, values, axis=0, nanrm=False, returnMask = False):
    """
    Returns an array with the values minimum to_compares is found. To_compare and values must have the same order. Note that, when values in to_compare are the same, then the minimum value is returned.

    Parameters
    ----------
    to_compare : TYPE
        DESCRIPTION.
    values : TYPE
        DESCRIPTION.

    Returns
    -------
    array with the same dimensions.

    """
    # Checking for arrays in the same dimension
    if to_compare.shape != values.shape:
        print("Error!! Both arrays haven't the same dimensions")
        return None

    # Create boolean mask (True or False)
    if nanrm:
        mask = to_compare == np.nanmin(to_compare, axis=axis)
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
    else:
        ## Before it was like this
        mask = np.ma.make_mask(to_compare == np.amin(to_compare, axis=axis))

    if returnMask:
        return mask
    
    # Applying mask and getting maximum value in case two are selected
    if np.issubdtype(values.dtype, np.datetime64):
        # https://stackoverflow.com/questions/45793044/numpy-where-typeerror-invalid-type-promotion
        arrout = np.where(mask == True, values, np.datetime64("NaT"))
        arrout = np.nanmax(arrout, axis=0)

    else:

        if nanrm:
            arrout = np.nanmin(values * mask, axis=axis)
        else:
            arrout = np.min(values * mask, axis=axis)

    return arrout


def whichMax(to_compare, values, axis=0):
    """
    Returns an array with the maximum values of to compare are found. "to_compare" and values must have the same order. Note that, when values in to_compare ar the same, then the maximum value is returned.

    Parameters
    ----------
    to compare: TYPE
        Description
    values: TYPE
        DESCRIPTION

    Returns
    -------
    array with the same dimensions
    """

    # Checking that arrays are in the same dimension
    if to_compare.shape != values.shape:
        print("Error!! Both arrays haven't the same dimensions")
        return None

    # Create boolean mask (True or False)
    mask = np.ma.make_mask(to_compare == np.nanmax(to_compare, axis=axis))

    # Applying mask and getting maximum value in case two are selected
    if np.issubdtype(values.dtype, np.datetime64):
        # https://stackoverflow.com/questions/45793044/numpy-where-typeerror-invalid-type-promotion
        arrout = np.where(mask == True, values, np.datetime64("NaT"))
        arrout = np.nanmax(arrout, axis=0)
    else:
        arrout = np.max(values * mask, axis=0)

    return arrout


def whichMinAngle_withoutClouds(values, angles):

    # angles + 0.01 in case there is a perfect 0 angle.
    # This way min angles now are the maximum angles. To get rid of the 0 case later on,
    # when merging with values I will want the maximum value (minimum), so the 0 (no cloud),
    # is not on may way
    mask_angles = ((1/(angles+0.01))*100)
    mask_values = np.ma.make_mask(values != 0)

    # Getting the right format (each pixel in a row with the K dimensions in columns)
    maskVal = mask_values * 0b1
    maskVal[maskVal == 0] = 0b0

    coso = mask_angles * mask_values
    coso2 = coso == np.amax(coso, axis=0)
    ole = values * coso2
    # ole2 = np.max(ole, axis = 0)

    return ole


def readArrays_(ds, newColsRows=None):
    """   

    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    newColsRows : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if newColsRows is None:

        nbands = ds.RasterCount + 1
        rango = range(1, nbands)

        arrayList = list()
        for k in rango:
            tmpBand = ds.GetRasterBand(k)
            # maybe here I can get out band name

            if tmpBand is not None:
                arrayList.append(tmpBand.ReadAsArray())

    else:
        new_cols, new_rows = newColsRows

        old_cols = ds.RasterXSize
        old_rows = ds.RasterYSize

        height_factor = old_rows/new_rows
        width_factor = old_cols/new_cols

        if height_factor > 1 or width_factor > 1:
            print("Angles resolution larger than tile. NOT IMPLEMENTED")
            return None

        nbands = ds.RasterCount + 1
        rango = range(1, nbands)

        arrayList = list()
        for k in rango:
            tmpBand = ds.GetRasterBand(k)

            if tmpBand is not None:
                arrayList.append(tmpBand.ReadAsArray(0, 0,
                                                     old_cols, old_rows,
                                                     newColsRows[0], newColsRows[1]))

    return np.stack(arrayList)


def createProjWin(geotransform, ncols, nrows):
    ulX, width, wshift, ulY, hshift, height = geotransform
    lrX = ulX + ncols * width
    lrY = ulY + nrows * height

    return [ulX, ulY, lrX, lrY]


def reprojectProjWin(srcProj, dstProj, projWin):
    from pyproj import transform

    if srcProj == dstProj:
        print("Same projection. Returning original projWin")
        return projWin

    ulX, ulY, lrX, lrY = projWin
    projUlX, projUlY = transform(srcProj, dstProj, ulX, ulY)
    projLrX, projLrY = transform(srcProj, dstProj, lrX, lrY)

    return [projUlX, projUlY, projLrX, projLrY]

def gdalDataTypesDict(datatype):
    from osgeo import gdal

    ## Maybe this is not necessary, since gdal.GDT_Whatever returns integers
    vortaro = {
        "Unknown" : gdal.GDT_Unknown,
        "Byte" : gdal.GDT_Byte,
        "UInt16" : gdal.GDT_UInt16,
        "Int16" : gdal.GDT_Int16,
        "UInt32" : gdal.GDT_UInt32,
        "Int32" : gdal.GDT_Int32,
        "Float32" : gdal.GDT_Float32,
        "Float64" : gdal.GDT_Float64,
        "CInt16" : gdal.GDT_CInt16,
        "CInt32" : gdal.GDT_CInt32,
        "CFloat32" : gdal.GDT_CFloat32,
        "CFloat64" : gdal.GDT_CFloat64
    }
    if type(datatype) is int:
        datatype = gdal.GetDataTypeName(datatype)

    outype = vortaro[datatype]
    return outype
        

def resampleRasters(toDs, fromDs, output, resampleAlg="bilinear", use_warpedbounds=True,
                    warpOptions=[], warpMulti=False, warpVRT=False, **kwargs):

    # Loading libraries
    import bildo
    from osgeo import gdal, osr
    import numpy as np
    ## from osgeo import gdal

    # Getting information needed
    to_projection = toDs.GetProjection()
    to_ncols = toDs.RasterXSize
    to_nrows = toDs.RasterYSize
    to_geotransform = toDs.GetGeoTransform()

    from_projection = fromDs.GetProjection()      
    
    # Getting osr objects
    to_srs = osr.SpatialReference()
    from_srs = osr.SpatialReference()
    to_srs.ImportFromWkt(to_projection)
    from_srs.ImportFromWkt(from_projection)

    # creating projwins
    to_projwin = createProjWin(to_geotransform, to_ncols, to_nrows)
    from_projwin = reprojectProjWin(to_projection, from_projection, to_projwin)

    # Reprojecting
    # First crop fromDs in its srs

    if to_srs.IsSame(from_srs) == 1:
        print("CRS is the same between both rasters. Cropping and resampling")
        try:
            outDs = gdal.Translate(output,
                                   fromDs,
                                   projWin=to_projwin,
                                   outputBounds=to_projwin,
                                   # height=to_nrows,
                                   # width=to_ncols,
                                   xRes=to_geotransform[1],
                                   yRes=abs(to_geotransform[5]),
                                   resampleAlg=resampleAlg,
                                   **kwargs
                                   )

            if outDs is None:
                print("gdal.Translated with xRes and yRes failed. Trying height and widht instead")
                raise NoneDsErrorClass(
                    "gdal.Translate returned None. Trying height, width instead")

        except:
            outDs = gdal.Translate(output,
                                   fromDs,
                                   projWin=to_projwin,
                                   height=to_nrows,
                                   width=to_ncols,
                                   # xRes=to_geotransform[1],
                                   # yRes=abs(to_geotransform[5]),
                                   outputBounds=to_projwin,
                                   resampleAlg=resampleAlg,
                                   **kwargs
                                   )

        if outDs is None:
            raise NoneDsErrorClass(
                "gdal.Translate was unable to create a data source. None returned!")
        
        outDs = None
        return outDs


    #tmp_outputfile = f"{tmp_folder}/from_cropped.tif"
    # gettign name
    output_name = output.split("/")
    output_name = output_name[len(output_name)-1]

    # print("Translating to fromDs projwin")
    # from_cropped = gdal.Translate(f"/tmp/from_cropped_{output_name}",
    #                               fromDs,
    #                               projWin=from_projwin,
    #                               outputBounds = from_projwin,
    #                               resampleAlg=resampleAlg)
    # #print(f"fromDs_proj: {from_projection} \n from_projwin: {from_projwin}")
    
    # if from_cropped is None:
    #     print("Cropped operation in source projection (fromDs one) returned None")
    #     return None
    # elif len(np.unique(from_cropped.GetRasterBand(1).ReadAsArray())) == 1:
    #     raise RuntimeError("from_cropped failed. Just an unique value is returned")
   
    #ERASE AFTER FIXED
    #return from_cropped
    # Second, reproject cropped image to sinusoidal
    #tmp_outputfile2 = f"{tmp_folder}/cropped_to_sinu.tif"
    if resampleAlg == "nearest":
        resampleAlg_warp = "near"
    else:
        resampleAlg_warp = resampleAlg

    print("Resampling toDs CRS and projwin extent")
    if use_warpedbounds:
        # to_projwin_warp = [to_projwin[0], to_projwin[3], to_projwin[2], to_projwin[1]]
        to_projwin_warp = [to_projwin[0], to_projwin[3], to_projwin[2], to_projwin[1]]
        if warpVRT:
            vrtfile = f"/tmp/vrt{output}.vrt"
            fromDs = gdal.BuildVRT(vrtfile, fromDs, outputSRS=from_srs)

        cropped_sinu = gdal.Warp(
            f"/vsimem/cropped_sinu_{output_name}",
            # f"/tmp/cropped_sinu_{output_name}",
            fromDs,
            srcSRS = from_srs,
            dstSRS=to_srs,
            # xRes = to_geotransform[1],
            # yRes = to_geotransform[5],
            height = to_nrows,
            width = to_ncols,
            outputBounds = to_projwin_warp,
            resampleAlg=resampleAlg_warp,
            warpOptions=warpOptions,
            copyMetadata=True,
            multithread=warpMulti
        )
    else:
         cropped_sinu = gdal.Warp(
             # f"/vsimem/cropped_sinu_{output_name}",
             f"/tmp/cropped_sinu_{output_name}",
             fromDs,
             srcSRS = from_srs,
             dstSRS=to_srs,
             xRes = to_geotransform[1],
             yRes = to_geotransform[5],
             # height = to_nrows,
             # width = to_ncols,
             resampleAlg=resampleAlg_warp,
             warpOptions=warpOptions,
             copyMetadata=True
        )

    if cropped_sinu is None:
        raise RuntimeError("Warp operation returned None")
    elif len(np.unique(cropped_sinu.GetRasterBand(1).ReadAsArray())) == 1:
        print("Warping to target SRS failed?. Just one value returned")
    # # It failed when rasters has an unique value

    fromDs = None
    del fromDs
    #return cropped_sinu
    #del from_cropped
    # os.remove(tmp_outputfile)
    # Third, crop with sinusoidal window (to_projwin)
    print("Translating into target Ds")
    outDs = gdal.Translate(output,
                           cropped_sinu,
                           # projWin=to_projwin,
                           height=to_nrows,
                           width=to_ncols,
                           outputBounds=to_projwin,
                           resampleAlg=resampleAlg,
                           **kwargs
                           )
    cropped_sinue = None
    del cropped_sinu
    # os.remove(tmp_outputfile2)
    outDs = None
    return outDs

  
          
def checkIfBildo(bildo_):
    if type(bildo_) is str:
        bildo_ = bildo.openBildo(bildo_)
    elif type(bildo_) is gdal.Dataset:
        bildo_ = bildo.openBildo(bildo_)
    elif type(bildo_) is bildo.bildo:
        pass
    else:
        raise ValueError("bildo_ should be a  bildo image, a gdal dataset or a path")
    return bildo_

def getProjwinWarp(projwin):
    projwin_warp = [projwin[0], projwin[3], projwin[2], projwin[1]]
    return projwin_warp

def resampleRasters_(from_, to_, output, resampleAlg="bilinear", wgs_projwin=True,
                     warpOptions=[], return_bildo=True, **kwargs):
    """
    help
    """
    from_ = checkIfBildo(from_)
    to_ = checkIfBildo(to_)

    output_name = output.split("/")
    output_name = output_name[len(output_name)-1]

    if resampleAlg == "nearest":
        resampleAlg_warp = "near"
    else:
        resampleAlg_warp = resampleAlg


    from_srs = osr.SpatialReference()
    to_srs = osr.SpatialReference()
    from_srs.ImportFromWkt(from_.crs)
    to_srs.ImportFromWkt(to_.crs)

    to_projwin = createProjWin(to_.geotransform, to_.dims[2], to_.dims[1])
    if wgs_projwin:
        # projwin masking happens in wgs84
        wgs_srs = osr.SpatialReference()
        nerror = wgs_srs.ImportFromEPSG(4326)
        if nerror != 0:
            print("Failed defining WGS84 from EPSG, trying another approach")
            nerror = wgs_srs.SetWellKnownGeogCS("WGS84")
            if nerror != 0:
                raise ValueError("Unable to define WGS84 projection")
            
        wgs_projection = wgs_srs.ExportToWkt()
        wgs_projwin = reprojectProjWin(to_.crs, wgs_projection, to_projwin)
        wgs_projwarp = getProjwinWarp(wgs_projwin)
        # wgs_projwarp = np.array(wgs_projwarp) + np.array([0.1, 0.1, -0.1, -0.1])
        # wgs_projwarp = wgs_projwarp.tolist()
        print(wgs_projwarp)
        wgsDs = gdal.Warp(
            f"/vsimem/wgswapr_{output_name}",
            from_.dataSource,
            srcSRS = from_srs,
            dstSRS = wgs_srs,
            xRes=0.01,
            yRes=0.01,
            resampleAlg = resampleAlg,
            copyMetadata=True,
            outputBounds=wgs_projwarp
        )
        del from_
        from_ = bildo.openBildo(wgsDs)
        from_srs = osr.SpatialReference()
        from_srs.ImportFromWkt(from_.crs)
    else:
        pass

    if to_srs.IsSame(from_srs) == 1:
        print("CRS is the same between both rasters. Cropping and resampling")
        outDs = gdal.Translate(
            f"/vsimem/{output_name}",
            from_.dataSource,
            projWin=wgs_projwin, # ??
            outputBounds=to_projwin,
            xRes=to_.geotransform[1],
            yRes=to_.geotransform[5],
            resampleAlg=resampleAlg
        )
        if outDs is None:
            print("gdal.Translated with xRes and yRes failed.")
        else:
            pass
    else:
        outDs = gdal.Warp(
            f"/vsimem/{output_name}",
            from_.dataSource,
            srcSRS=from_srs,
            dstSRS=to_srs,
            xRes=to_.geotransform[1],
            yRes=to_.geotransform[5],
            resampleAlg=resampleAlg_warp,
            warpOptions=warpOptions,
            copyMetadata=True
        )
        if outDs is None:
            raise ValueError("Warp operatio FAILED")

    out = bildo.openBildo(outDs)
    out.writeArray(output, **kwargs)
    del out
    if return_bildo:
        out = bildo.openBildo(output)
        return out

    

def downSample(array, multifactor):
    import numpy as np
    if len(array.shape) > 2:
        axi = 1
        axj = 2
    else:
        axi = 0
        axj = 1
    out = np.repeat(array, multifactor, axis = axi)
    out = np.repeat(out, multifactor, axis = axj)
    return out


def fromERA5toWGS84(dataarray, output=None, xaxis = "longitude", name = None):
    import rioxarray
    import xarray as xr
    if type(dataarray) is str:
        if dataarray.split(".")[-1] == "nc":
            dataarray = xr.open_dataset(dataarray, decode_coords="all")
        else:
            dataarray = xr.open_rasterio(dataarray, decode_coords="all")

    else:
        pass
    if type(dataarray) is xr.core.dataset.Dataset:
        if name is None: raise ValueError("I need a variable name")
        dataarray = dataarray[name]

    longitud = dataarray[xaxis].values
    longitud[longitud >= 180] -= 360
    indexes = np.where(longitud >= 0)[0].tolist()
    outindexes = np.where(longitud < 0)[0].tolist()
    array = dataarray.values
    zeros = np.zeros(array.shape)
    zeros[zeros==0] = np.nan
    flipped1 = array[:,:,indexes]
    flipped2 = array[:,:,outindexes]
    zeros[:,:,indexes] = flipped2
    zeros[:,:,outindexes] = flipped1
    longitud.sort()
    dataarray[xaxis] = longitud
    dataarray.values = zeros
    dataarray = dataarray.rio.write_crs(4326)
    if output is not None:
        if output.split(".")[-1] == "nc":
            dataarray.to_netcdf(output)
            del dataarray
            return "Done!"
        elif output.split(".")[-1] == "tif":
            dataarray.rio.to_raster(output, noData=0)
            del dataarray
            return "Done!"
        else:
            raise ValueError("format not implemented")
    return dataarray


def extractPointsFromRaster(gdf_points, bildoimg):
    """ Based on Geoprocessing with Python Pag 193-194
            gdf_points - a geodataframe of points
            bildoimg - a raster file in bildo format
        Notes:
            Right now only extract the first band. It can be further improved.
    """
    from osgeo import gdal, ogr, osr
    points_crs = gdf_points.crs
    bildosrs = osr.SpatialReference()
    bildosrs.ImportFromWkt(bildoimg.crs)
    if type(points_crs) is dict:
        points_srs = osr.SpatialReference()
        points_epsg = int(points_crs["init"].split(":")[1])
        points_srs.ImportFromEPSG(points_epsg)
    elif type(points_crs) is int:
        points_srs = osr.SpatialReference()
        points_srs.ImportFromEPSG(points_crs)
    else:
        points_srs = osr.SpatialReference()
        points_srs.ImportFromWkt(points_crs.to_wkt())
    if bildosrs.IsSame(points_srs) != 1:
        raise ValueError("gdf_points and bildoimg must have the same CRS")
    else:
        if len(bildoimg.dims) > 2 and bildoimg.dims[0] == 1:
            invgt = gdal.InvGeoTransform(bildoimg.geotransform)
            values = list()
            for i, row in gdf_points.iterrows():
                x,y = row.geometry.x, row.geometry.y
                offsets = gdal.ApplyGeoTransform(invgt, x, y)
                offsets = gdal.ApplyGeoTransform(invgt, x, y)
                xoff, yoff = map(int, offsets)
                try:
                    tmp = bildoimg.arrays.values[0, yoff, xoff]
                except:
                    tmp = np.nan
                values.append(tmp)
        else:
            invgt = gdal.InvGeoTransform(bildoimg.geotransform)
            listarr = []
            for i, row in gdf_points.iterrows():
                values = list()
                x,y = row.geometry.x, row.geometry.y
                offsets = gdal.ApplyGeoTransform(invgt, x, y)
                offsets = gdal.ApplyGeoTransform(invgt, x, y)
                xoff, yoff = map(int, offsets)
                for k in range(bildoimg.dims[0]):
                    try:
                        tmp = bildoimg.arrays.values[k, yoff, xoff]
                    except:
                        # print("Outside of the image, assigning nan")
                        tmp = np.nan
                    values.append(tmp)
                listarr.append(np.array(values))
            values = np.stack(listarr, axis=0)
    return values


def rasterigi(bildo_, layer_ds, output="/tmp/rasterigitatmp.tif",
                attribute="wwsos_min", format = "GTiff", allTouched = False,
                outputType=gdal.GDT_Int16, noData=0,**kwargs):

    import bildo
    from osgeo import gdal, ogr
    import geopandas as gpd

    ## Checking bildo_
    if type(bildo_) is str or type(bildo_) is gdal.Dataset:
        bildo_ = bildo.openBildo(bildo_)
    elif type(bildo_) is bildo.bildo:
        pass
    else:
        raise ValueError("bildo_ should an input dataset or a path to a file")

    ## Checking layer_ds input
    if type(layer_ds) is gpd.GeoDataFrame:
        layer_ds.to_file("/tmp/tmptorasterigi.gpkg", driver="GPKG")
        layer_ds = gdal.OpenEx("/tmp/tmptorasterigi.gpkg", gdal.OF_VECTOR)
    elif type(layer_ds) is str:
        layer_ds = gdal.OpenEx(layer_ds, gdal.OF_VECTOR)
    elif type(layer_ds) is gdal.Dataset:
        pass
    else:
        raise ValueError("layer_ds should be a geopdandas.GeoDataFrame, a gdal.Dataset, or a str")

    xres = bildo_.geotransform[1]
    yres = bildo_.geotransform[5]
    xmin, xmax, ymin, ymax = bildo_.extent
    outbounds = [xmin, ymin, xmax, ymax]
    srs = bildo_.crs

    ## I should put here more options I think, but for now it is working well
    burned = gdal.Rasterize(destNameOrDestDS=output,
                            srcDS=layer_ds,
                            attribute=attribute,
                            outputBounds=outbounds,
                            xRes=xres, yRes=yres,
                            format=format,
                            allTouched=allTouched,
                            outputType=outputType,
                            noData=noData,
                            outputSRS=srs,
                            **kwargs)
    del burned
    burned = bildo.openBildo(output)

    return burned


def poligonigi(bildoimg, fieldname, output = "/tmp/tmp.gpkg", mask=None, layername=None,
               driver="GPKG", srs=None):
    """
    bildoimg -- Raster image of bildo class
    fieldname -- Name of the fireld to create
    output -- output geopackage
    mask -- Mask should have the same dimensions than bildoimg and 0 and 1s.
    layername -- name of the layer to create
    driver -- driver to Create layer. GPKG by default
    srs -- Spatial Reference of the output. If None, it tries to generate it from bildoimg. None by default.

    Also, it only takes into account the first raster band (to implement more)
    """
    from osgeo import gdal, ogr

    if layername is None:
        tmpname = output.split("/")[1]
        layername = tmpname.split(".")[0]

    if srs is None:
        from osgeo import osr
        srs = osr.SpatialReference()
        srs.ImportFromWkt(bildoimg.crs)

    else:
        from osgeo import osr
        srs = osr.SpatialReference()
        if type(srs) is int:
            srs.ImportFromEPSG(srs)
        elif type(srs) is str:
            try:
                srs.ImportFromWkt(srs)
            except:
                srs.ImportFromProj4(srs)

    if mask is not None:
        print("Parallel is not implemented right using the mask")
        maskimg = bildoimg.copy()
        maskimg.arrays = mask
        maskimg.writeToDisk("/tmp/tmpmask.tif")
        del maskimg

        maskimg = bildo.openBildo("/tmp/tmpmask.tif")

    drvr = ogr.GetDriverByName(driver)
    if os.path.exists(output): drvr.DeleteDataSource(output)
    outds = drvr.CreateDataSource(output)
    outlayer = outds.CreateLayer(layername, srs=srs)
    newfield = ogr.FieldDefn(fieldname, ogr.OFTInteger)
    outlayer.CreateField(newfield)
    if mask is None:
        gdal.Polygonize(bildoimg.dataSource.GetRasterBand(1), None, outlayer, 0, [], callback=None)
    else:
        gdal.Polygonize(bildoimg.dataSource.GetRasterBand(1), maskimg.dataSource.GetRasterBand(1),
                        outlayer, 0, [], callback=None)
    outds.Destroy()

    del outds, outlayer, newfield, drvr, srs
    return 0

def maskoBildoKunTavolo(bildo_, tavolo_=None, output="/tmp/maskita.tif",
                        crop=True, bounds=None, **kwargs):
    """
    Description: 
    Mask an image with a layer
    ==============================
    bildo_:   bildo image to be cut (Bildo, Path)
    tavolo_:  layer file to cut with (GeoDataFrame, gdal.Dataset, Path)
    output:   where to save everyting
    crop:     If crop the image to the extent of the layer. Default True.
    bounds:   To cut bildo_ without tavolo_. [xmin, ymin, xmax, ymax].
                It is assumed to be in the same crs than bildo.
    **kwargs: Arguments to pass to gdal.Warp
    """
    ## Checking bildo_
    if type(bildo_) is str or type(bildo_) is gdal.Dataset:
        bildo_ = bildo.openBildo(bildo_)
    elif type(bildo_) is bildo.bildo:
        pass
    else:
        raise ValueError("bildo_ should an input dataset or a path to a file")

    if bounds is not None:
        print("Assuming extent is in the same coordenates than bildo_")
        maskita = gdal.Warp(srcDSOrSrcDSTab=bildo_.dataSource,
                            destNameOrDestDS=output,
                            outputBounds = bounds,
                            **kwargs)
        del maskita, bildo_
        out = bildo.openBildo(output)


    else:
        ## Checking layer_ds input
        if type(tavolo_) is gpd.GeoDataFrame:
            tavolo_.to_file("/tmp/tmptomasko.gpkg", driver="GPKG")
            tavolo_ = gdal.OpenEx("/tmp/tmptomasko.gpkg", gdal.OF_VECTOR)
        elif type(tavolo_) is str:
            tavolo_ = gdal.OpenEx(tavolo_, gdal.OF_VECTOR)
        elif type(tavolo_) is gdal.Dataset:
            pass
        elif tavolo_ is None:
            raise ValueError("tavolo_ is None. Either define bounds to cut with or a tavolo_.")
        else:
            raise ValueError("tavolo_ should be a geopdandas.GeoDataFrame, a gdal.Dataset, a str" \
                             "or bounds argument be defined")

        ## Cheking they have the same projection
        bildo_srs = osr.SpatialReference()
        tavolo_srs = osr.SpatialReference()
        bildo_srs.ImportFromWkt(bildo_.crs)
        tavolo_srs.ImportFromWkt(tavolo_.GetLayer().GetSpatialRef().ExportToWkt())
        if bildo_srs.IsSame(tavolo_srs) == False:
            raise ValueError("Both datasets have not the same projection" \
                             "Consider to reproject before masking.")


        tavolo_nomo = tavolo_.GetLayer().GetName()
        tavolo_description = tavolo_.GetDescription()
    
        maskita = gdal.Warp(srcDSOrSrcDSTab=bildo_.dataSource,
                            destNameOrDestDS=output,
                            cutlineDSName=tavolo_description,
                            cropToCutline=crop,
                            cutlineLayer=tavolo_nomo,
                            **kwargs)
        del bildo_srs, tavolo_srs, bildo_, tavolo_, maskita

        out = bildo.openBildo(output)
    return out
 
def getXArray3D(array, geotransform, third_dimension, labels=["time", "Y", "X"], sort=True):
    """
    Generate a three dimensional array with labeled dimensions
    """
    import xarray as xr
    import numpy as np

    gt = geotransform
    ndeep, nrows, ncols = array.shape

    X = np.arange(gt[0], gt[0]+(gt[1]*ncols), gt[1])[:ncols]
    Y = np.arange(gt[3], gt[3]+(gt[5]*nrows), gt[5])[:nrows]

    print(f"array: {array.shape} \nXshape: {X.shape}  Yshape: {Y.shape}")

    if len(third_dimension) != ndeep:
        raise ValueError("third dimension with the wrong length")


    xarr = xr.DataArray(array,
                          coords=[third_dimension, Y, X],
                          dims=labels)

    if sort: xarr = xarr.sortby(labels[0])

    return xarr


def createMosaicVrt(list_of_images, output="/tmp/tmp.vrt"):
    from osgeo import gdal
    vrt = gdal.BuildVRT(output, list_of_images)
    b = bildo.openBildo(vrt)
    return b


def getCrsAsSrs(layer):
    from osgeo import osr
    srs = osr.SpatialReference()
    initcrs = layer.crs
    if type(initcrs) is str:
        srs.ImportFromWkt(initcrs)
    elif type(initcrs) is int:
        srs.ImportFromEPSG(initcrs)
    elif type(initcrs) is dict:
        crs = initcrs["init"]
        crs = crs.split("epsg:")[1]
        srs.ImportFromEPSG(int(crs))
    return srs


def checkSameCRS(listlayers):
    from osgeo import osr
    listcomparisons = []
    for l in range(1, len(listlayers)):
        layer1 = listlayers[l-1]
        layer2 = listlayers[l]
        layer1crs = getCrsAsSrs(layer1)
        layer2crs = getCrsAsSrs(layer2)

        thesame = layer1crs.IsSame(layer2crs)
        listcomparisons.append(thesame)
    if np.sum(listcomparisons) == len(listlayers)-1:
        out = True
    else:
        out = False
    return out


def intersects(multilayer, withlayer):
    """
    Implementation of intersect using GDAL, since the one from
    geopandas (i.e. shapely) its giving me problems
    ------------------------------------------------
    Params:
    - multilayer (geodataframe): the layer that is checking if its components intersect with withlayer
    - withlayer (geodataframe): the layer to check which 'thelayer' components intersects
    Returns:
    - boolean list of intersections
    """
    from osgeo import ogr, osr

    if checkSameCRS([multilayer, withlayer]) == False:
        raise ValueError("CRS is not the same among the layers")

    listintersects = []
    for idx in range(withlayer.shape[0]):
        withwkt = withlayer.iloc[idx].geometry.to_wkt()
        withgeom = ogr.CreateGeometryFromWkt(withwkt)
        listintersect = []
        for i in range(multilayer.shape[0]):
            wkt = multilayer.iloc[i].geometry.to_wkt()
            multigeom = ogr.CreateGeometryFromWkt(wkt)
            listintersect.append(multigeom.Intersect(withgeom))
        listintersects.append(listintersect)
    return listintersects

def getIntersectingMultiGeoms(multilayer, withlayer):
    """
    Implementation of intersection using GDAL, since the one from
    geopandas (i.e. shapely) its giving me problems
    ------------------------------------------------
    Params:
    - multilayer (geodataframe): the layer that is checking if its components intersect with withlayer
    - withlayer (geodataframe): the layer to check which 'thelayer' components intersects
    Returns:
    - Intersection GeoDataFrame based on multilayer
    """

    interidxs = intersects(multilayer, withlayer)
    listintersection = []
    for idx in interidxs:
        listintersection.append(multilayer[idx])
    intersection = pd.concat(listintersection)
    return intersection
        


def bandsToColumns(array):
    """
    Transform numpy.array with tree dimensions into 2,
    where col and rows are in only one column and bands in
    different columns
    """

    if len(array.shape) > 2:
        nbands, nrows, ncols = array.shape
    else:
        nrows, ncols = array.shape
        nbands = 1
    out_array = np.reshape(array, (-1, nbands, nrows*ncols))[0].transpose()

    return out_array

def epsgToCrsWkt(epsg):
    from osgeo import osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs.ExportToWkt()


def rasterToPoints(bildo_, column_names, epsg, output=None, driver="GPKG", navalue=0):
    import pandas as pd
    import geopandas as gpd
    xtl, xres, xtr, ytl, ytr, yres = bildo_.geotransform
    ndim, nrows, ncols = bildo_.dims
    total = ndim*nrows*ncols

    xcoords = np.arange(xtl, xtl+(ncols*xres), xres)+np.abs(xres/2)
    ycoords = np.arange(ytl, ytl+(nrows*yres), yres)+np.abs(yres/2)
    mesh = np.meshgrid(xcoords, ycoords)
    coordinates = np.concatenate([mesh[i].reshape(-1,1) for i in [0,1]], axis=1)

    bandinfo = bandsToColumns(bildo_.arrays.values).astype(np.float64)
    bandinfo[bandinfo == navalue] = np.nan

    xyband = np.concatenate([coordinates, bandinfo], axis=1)
    nonanidx = ~np.isnan(xyband[:,2:])
    nonanidx = np.sum(nonanidx, axis=1) > 0
    xyband = xyband[nonanidx,:]


    columns = ["X", "Y"] + column_names
    df = pd.DataFrame(xyband, columns=columns)
    geodf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs=epsgToCrsWkt(epsg))
    if output is not None:
        geodf.to_file(output, driver=driver)
    return geodf


if __name__ == "__main__":
    a1 = np.array([[0, 2], [3, 4]])
    a2 = np.array([[100, 150], [50, 200]])
    b1 = np.array([[5, 6], [0, 8]])
    b2 = np.array([[200, 100], [500, 300]])
    c1 = np.array([[1, 0], [10, 0]])
    c2 = np.array([[100, 230], [60, 10]])
    d1 = np.array([[10, 300], [0, 5]])
    d2 = np.array([[150, 300], [40, 170]])
    a = np.stack([a1, a2])
    b = np.stack([b1, b2])
    c = np.stack([c1, c2])
    d = np.stack([d1, d2])
    values = np.stack([a1, b1, c1, d1])
    tocompare = np.stack([a2, b2, c2, d2])

    # mask_angles = np.ma.make_mask(tocompare == np.amin(tocompare, axis = 0))
    mask_angles = ((1/tocompare+0.01)*1000).astype(int)
    mask_values = np.ma.make_mask(values != 0)
    masks = np.stack([mask_values, mask_angles])

    # Getting the right format (each pixel in a row with the K dimensions in columns)
    maskVal = mask_values * 0b1
    maskVal[maskVal == 0] = 0b0

    # maskAng = mask_angles * 0b0010
    # maskAng[maskAng == 0] = 0b01

    # Useful later on
    # maskAng = np.reshape(maskAng, (-1,8,1))[0]
    # maskVal = np.reshape(maskVal, (-1,8,1))[0]
    # coso = np.concatenate([maskAng, maskVal], axis = 1)

    # getting values
    # coso = np.bitwise_or(maskAng, maskVal)
    coso = mask_angles * maskVal

    # number 6 is the bit with the best situation, so let's filter cases closer to it
    # coso2 = coso & 6
    coso2 = coso == np.amax(coso, axis=0)

    # Now we have two acceptable values 6 and 4.
    # 6 means the pixel is in the best situation possible. No clouds best view angle
    # 4 means the pixel is not the best option, but is the one we choose. No clouds band view angle.

    # The problem is a pixel can have both bit values. So the best way to deal with it is keeping the maximum value, which is the best situation possible.
    # final_filter = coso2 == np.amax(coso2, axis  = 0)

    # Now we apply the filter, getting the desire value
    # ole = values * final_filter
    ole = values * coso2

    # Reducing to 2d. I am summing them since all the not desired values are 0
    ole2 = np.max(ole, axis=0)

    # All right values
    # coso = coso & 4 # (0b0100 is ArCn & AnrCn. All clear conditions)


# =============================================================================
#     # k dimension into columns
# =============================================================================
    test = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    test = np.reshape(test, (2, 2, 2))
    test = np.reshape(test, (-1, 2, 4))[0].transpose()

    test2 = np.reshape(
        np.array([[9, 10], [11, 12], [13, 14], [15, 16]]), (2, 2, 2))
    test2 = np.reshape(test2, (-1, 2, 4))[0].transpose()

    test3 = np.concatenate([test, test2], axis=1)
