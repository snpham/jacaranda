import numpy as np
from astropy.io import fits
import astronomy as astro
import pandas as pd

"""SuperCOSMOS: catalogue of galaxies generated from several visible light surveys
"""


def supercosmos_import(fn):
    # np_data = np.loadtxt(fn, delimiter=',', skiprows=1, usecols=[0,1], max_rows = 5)
    # print(np_data)

    pd_data = pd.read_csv(fn, sep = ',', nrows=5)
    return pd_data.iloc[:, :3].to_numpy()



if __name__ == '__main__':
    pass
    fn = 'inputs/SCOS_XSC_mCl1_B21.5_R20_noStepWedges.csv'
    sc_out = supercosmos_import(fn)
    print(sc_out)
