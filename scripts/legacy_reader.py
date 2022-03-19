import requests
import shutil
from astropy.io import fits
import matplotlib.pyplot as plt





if __name__ == '__main__':
    pass





    # looks like a ton of good parameters for environmental data
    data_url = 'https://www.legacysurvey.org/viewer/fits-cutout'

    params = {'ra': '154.7709',
              'dec': '46.4537',
              'layer': 'ls-dr9-north',
              'pixscale': '0.27',
              'bands': 'grz',
              }
    r = requests.get(data_url, params=params, allow_redirects=True)

    with open('inputs/file.fits', 'wb') as f:
        f.write(r.content)


    with fits.open('inputs/file.fits') as hdul: # use memmap=True for extra large files
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
    image_data = fits.getdata('inputs/file.fits', ext=0)
    print(image_data.shape)
    plt.figure()
    plt.imshow(image_data[1], cmap=plt.cm.viridis)
    plt.colorbar()
    plt.show()