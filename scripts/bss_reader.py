import numpy as np
from astropy.io import fits
import exploratory as astro


def bss_import(fn):
    output = []
    with fits.open(fn) as hdul:
        # info = hdul.info()
        # hdr = hdul[1].header
        # cols = hdul[1].columns
        # cols_names = cols.names
        data = hdul[1].data
        for ii, row in enumerate(data, 1):
            if row[5] == '-':
                output.append((ii, astro.hms2dec(row[2], row[3], row[4]),
                                  -astro.dms2dec(row[6], row[7], row[8])))
            else:
                output.append((ii, astro.hms2dec(row[2], row[3], row[4]),
                                   astro.dms2dec(row[6], row[7], row[8])))
    return np.array(output)



if __name__ == '__main__':
    pass

    fn = 'inputs/J_MNRAS_384_775_table2.fits'
    # cat = np.loadtxt('inputs/J_MNRAS_384_775_table2.dat', usecols=range(1, 7))
    # print(cat[0])
    
    bss_out = bss_import(fn)
