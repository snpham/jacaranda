import numpy as np
from astropy.io import fits
import exploratory as astro
import pandas as pd
import time
from dask import dataframe as dd
from pyspark.sql import SparkSession


"""SuperCOSMOS: catalogue of galaxies generated from several visible light surveys
"""


def supercosmos_import(fn):
    # np_data = np.loadtxt(fn, delimiter=',', skiprows=1, usecols=[0,1], max_rows = 5)
    # print(np_data)
    start = time.time()
    pd_data = pd.read_csv(fn, sep = ',', nrows=500000) # use nrows to limit
    print(time.time() - start)

    start = time.time()
    pd_data = pd.read_csv(fn, sep = ',', nrows=500000, chunksize=1000) # use nrows to limit
    pd_data = pd.concat(pd_data)
    print(time.time() - start)

    start = time.time()
    dask_df = dd.read_csv(fn)
    print(time.time() - start)
    # print(len(dask_df))
    # df = dask_df.head(10)
    # df2 = dask_df.head(10)
    # df = dd.concat([df, df2])
    # print(df)
    # print(dask_df.head(10))

    return None
    # return pd_data.iloc[:, :3].to_numpy()



if __name__ == '__main__':
    pass
    # fn = 'inputs/supercosmos/SCOS_XSC_mCl1_B21.5_R20_noStepWedges.csv'
    fn = 'inputs/supercosmos/supercosmos_test.csv'
    sc_out = supercosmos_import(fn)
