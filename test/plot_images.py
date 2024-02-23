import bildo
from bildo import plotFunctions as pf

if __name__ == "__main__":

    path = "../res/test-img.tif"

    img = bildo.openBildo(path)
    img.plot(1)

    img.plotRGB([3, 1, 4], scale="std")
    img.plotRGB([3, 1, 4], scale="minmax")
    img.plotRGB([3, 1, 4], scale="ccc")

    del img

    ####################################
