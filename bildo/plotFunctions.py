import matplotlib.pyplot as plt
import matplotlib.colors as clr
import xarray as xr
import bildo


def plotBildo(arr, band=0, figsize=(10,6), output=None, legend=True, extent=None,
              nodata=None, figax=None, scale_factor=1, axlabels = ["x", "y"],
              show=True, nticks = 4, drawlabels=True, drawaxis=True,
              drawaxisticks=True, **kwargs):
    import matplotlib.ticker as plticker


    if type(arr) is xr.DataArray:
        xmin = float(arr[axlabels[0]].min().values)
        xmax = float(arr[axlabels[0]].max().values)
        ymin = float(arr[axlabels[1]].min().values)
        ymax = float(arr[axlabels[1]].max().values)
        extent = [xmin, xmax, ymin, ymax]
    elif type(arr) is bildo.bildo:
        extent = arr.extent
        arr = arr.arrays[band]
    else:
        pass

    if len(arr.shape) > 2:
        arr = arr[band]
    else:
        pass

    ## it seems gdal read now as int by default
    arr = arr/scale_factor
    
    if nodata is not None:
        if "numpy" in str(type(arr)):
            arr[arr == nodata] = np.nan
        elif "xarray" in str(type(arr)):
            arr.values[arr.values == nodata] = np.nan
        else:
            pass
    else:
        pass

    if figax is None:
        fig, ax1 = plt.subplots(1, figsize = figsize)
        fig.patch.set_facecolor('xkcd:white')
    else:
        if type(figax) is not list: raise ValueError("figax has to be a list with fig and ax")
        fig, ax1 = figax
    tmp = ax1.imshow(arr, extent=extent, **kwargs)
    # ax1.set_yticklabels(ax1.get_yticklabels(), rotation=90, ha="right")
    for ylabel in ax1.get_yticklabels():
        ylabel.set_rotation(90)
        ylabel.set_size(8)
        ylabel.set_ha("right")
        ylabel.set_verticalalignment("center")
    for xlabel in ax1.get_xticklabels():
        xlabel.set_size(8)
        xlabel.set_ha("center")

    if drawlabels:
      ax1.set_xlabel(axlabels[0])
      ax1.set_ylabel(axlabels[1])
    else:
      ax1.axes.get_xaxis().set_visible(False)
      ax1.axes.get_yaxis().set_visible(False)

    if drawaxis is False or drawaxisticks is False:
      if drawaxis is False: labellength=0
      else: labellength = 10
      if drawaxisticks is False: longitud = 0
      else: longtiude=10
      ax1.tick_params(axis="both", labelsize=labellength, length=longitud)
    ax1.ticklabel_format(useOffset=False, style="plain")
    if nticks is not None and extent is not None:
        ysteps = abs(int((extent[3]-extent[2])/nticks))
        xsteps = abs(int((extent[1]-extent[0])/nticks))
        locy = plticker.MultipleLocator(base=ysteps) # this locator puts ticks at regular intervals
        locx = plticker.MultipleLocator(base=xsteps)
        ax1.yaxis.set_major_locator(locy)
        ax1.xaxis.set_major_locator(locx)

    if legend:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(tmp, cax=cax)
    if output is not None:
        fig.savefig(output)
        plt.close()
    if show:
        plt.show()
    return [fig, ax1, tmp]


def scaleMinMax(x):
    import numpy as np
    out = (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
    return out

def scaleCCC(x, minp=2, maxp=98):
    import numpy as np
    out = (x-np.nanpercentile(x,minp))/(np.nanpercentile(x,maxp)-np.nanpercentile(x,minp))
    return out

def scaleStd(x, w=2):
    import numpy as np
    std = np.nanstd(x)
    avg = np.nanmean(x)
    minimuma = avg-std*w
    maksimumo = avg+std*w
    out = (x - minimuma)/(maksimumo-minimuma)
    return out

def plotRGB(arr, orderRGB=[0,1,2], scale="std", figsize=(10,10), figax=None, show=True,**kwargs):
    import numpy as np
    import xarray as xr
    if type(arr) is xr.DataArray:
        xmin = float(arr[axlabels[0]].min().values)
        xmax = float(arr[axlabels[0]].max().values)
        ymin = float(arr[axlabels[1]].min().values)
        ymax = float(arr[axlabels[1]].max().values)
        extent = [xmin, xmax, ymin, ymax]
        r,g,b = [arr[orderRGB[0]], arr[orderRGB[1]], arr[orderRGB[2]]]
    elif type(arr) is np.ndarray:
        if arr.shape[0] >= 3:
            r,g,b = [arr[orderRGB[0]], arr[orderRGB[1]], arr[orderRGB[2]]]
        else:
            raise ValueError("not enough values")
    elif type(arr) is bildo.bildo:
        extent = arr.extent
        arr = arr.arrays
        r,g,b = [arr[orderRGB[0]], arr[orderRGB[1]], arr[orderRGB[2]]]
    elif type(arr) is list:
        if len(arr) == 3:
            r,g,b = [arr[orderRGB[0]], arr[orderRGB[1]], arr[orderRGB[2]]]
        else:
            raise ValueError("not enough values")
    else:
        pass

    if scale=="Std" or scale=="std":
        funktion = scaleStd
    elif scale=="ccc" or scale=="cumulative cut":
        funktion = scaleCCC
    elif scale=="minmax" or scale=="MinMax":
        funktion = scaleMinMax
    else:
        raise ValueError("scale should be std, ccc or minmax")
      
    r = funktion(r, **kwargs)
    g = funktion(g, **kwargs)
    b = funktion(b, **kwargs)
    stack = np.dstack((r,g,b))

    if figax is None:
      fig, ax = plt.subplots(1, figsize=figsize)
    else:
      fig, ax = figax

    ax.imshow(stack)
    if show:
      plt.show()
    return [fig,ax]
