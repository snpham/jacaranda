from astropy.io import fits
from pprint import pprint as pp
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import requests
from urllib.request import urlretrieve
import wget
plt.style.use(astropy_mpl_style)
import numpy as np
import pandas as pd


def get_image(fnpath):
    # get image data
    image_data = fits.getdata(fnpath, ext=0)
    print(image_data.shape)
    plt.figure()
    plt.imshow(image_data[1], cmap=plt.cm.viridis)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    pass

    fnpath = 'inputs/SDSS/ssppOut-dr12.fits'

    with fits.open(fnpath) as hdul: # use memmap=True for extra large files

        headers = hdul[1].header
        # pp(headers)
        # print(list(headers.keys()))
        # print(repr(headers))  

        # metadata
        print(hdul.info())

        # data object
        # data = hdul[0].data
        # print(data.shape) # 'SCI'
        # print(data[10:15, 20:25])

        cols = hdul[1].columns
        # print(cols.info())
        print(cols.names)

    data = np.load('inputs/SDSS/sdss_galaxy_colors.npy')
    print(data)
    # np.savetxt('inputs/SDSS/sdss_galaxy_colors.csv', data)
    df = pd.DataFrame(data)
    df.to_csv('inputs/SDSS/sdss_galaxy_colors.csv',index=False)