from astropy.io import fits
from pprint import pprint as pp
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import requests
from urllib.request import urlretrieve
import wget
plt.style.use(astropy_mpl_style)


if __name__ == '__main__':
    pass

    fnpath = 'inputs/cutout_154.7709_46.4537.fits'
    print(fits.info(fnpath))

    with fits.open(fnpath) as hdul: # use memmap=True for extra large files

        # metadata
        print(hdul.info())
        # for header in hdul[0].header[:]:
        #     print(header)
        # hdr = hdul[0].header
        # # print(repr(hdr))  
        # # pp(hdr[:2])
        # print(list(hdr.keys()))

        # # data object
        # data = hdul[0].data
        # print(data.shape) # 'SCI'
        # print(data[10:15, 20:25])

        # cols = hdul[1].columns
        # print(cols.info())
        # print(cols.names)

    # get image data
    image_data = fits.getdata(fnpath, ext=0)
    print(image_data.shape)
    plt.figure()
    plt.imshow(image_data[1], cmap=plt.cm.viridis)
    plt.colorbar()
    plt.show()


