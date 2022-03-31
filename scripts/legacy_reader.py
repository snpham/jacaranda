import requests
import shutil
from astropy.io import fits
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import product


def download_legacy(coordinates):

    # coordinates
    ra = coordinates[0]
    dec = coordinates[1]
    
    # parameters
    data_url = 'https://www.legacysurvey.org/viewer/fits-cutout'
    params = {'ra': ra,
              'dec': dec,
              'layer': 'ls-dr9-north',
              'pixscale': '0.04',
              'bands': 'grz',
              }
    
    # read and write
    fn = f'inputs/legacy_survey/ls_{ra}_{dec}.fits'
    r = requests.get(data_url, params=params, allow_redirects=True)
    with open(fn, 'wb') as f:
        f.write(r.content)
    view_fits_img(fn)


def view_fits_img(fn):
    # check the plot
    with fits.open(fn) as hdul: # use memmap=True for extra large files
        # metadata
        print(hdul.info())

    # get image data
    image_data = fits.getdata(fn, ext=0)
    print(image_data.shape)
    print(image_data[2])

    f, axarr = plt.subplots(1,3, figsize=(10,8)) 
    one = axarr[0].imshow(image_data[0], cmap=plt.cm.viridis)
    two = axarr[1].imshow(image_data[1], cmap=plt.cm.viridis)
    thr = axarr[2].imshow(image_data[2], cmap=plt.cm.viridis)
    f.colorbar(one, ax=axarr[0], orientation="horizontal", pad=0.05)
    f.colorbar(two, ax=axarr[1], orientation="horizontal", pad=0.05)
    f.colorbar(thr, ax=axarr[2], orientation="horizontal", pad=0.05)
    # plt.figure()
    # plt.imshow(image_data[2], cmap=plt.cm.viridis)
    plt.show()


if __name__ == '__main__':
    pass

    # ra = '134.7709'
    # dec = '46.4537'
    # coordinates = [ra, dec]
    # download_legacy(coordinates=coordinates)

    coordinates = list(product(
        [[183.5918731689453,56.011436462402344], ]))


    pool = mp.Pool(mp.cpu_count() - 2)
    check = pool.starmap(download_legacy, coordinates)
    pool.close()
    pool.join()
