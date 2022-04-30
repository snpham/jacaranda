import numpy as np
from astropy.io import fits
import pandas as pd
import crossmatch
import exploratory
import time
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore


def get_galaxies():
    cat_sdss = pd.read_csv('inputs/SDSS/sdss_full.csv', delimiter=',')
    cat_sdss = cat_sdss[cat_sdss['class'] == 'GALAXY'].reset_index(drop=True)
    cat_sdss.to_feather('inputs/SDSS/sdss_galaxies.feather')

    cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz.csv', delimiter=',', header=0)
    cat_gzoo2.to_feather('inputs/galaxyzoo/zoo2MainSpecz.feather')


def get_matches():
    cat_sdss = pd.read_feather('inputs/SDSS/sdss_galaxies.feather')
    # print(cat_sdss)
    cat_sdss_np = cat_sdss.to_numpy()
    cat_sdss_radec_col = [1,2]
    cat_gzoo2 = pd.read_feather('inputs/galaxyzoo/zoo2MainSpecz.feather')
    cat_gzoo2_np = cat_gzoo2.to_numpy()
    cat_gzoo2_radec_col = [3,4]

    max_angle = 1/3600
    matches, non_matches = crossmatch.catalog_crossmatch_kdtree(cat_sdss_np, cat_sdss_radec_col, 
                                                                cat_gzoo2_np, cat_gzoo2_radec_col, 
                                                                max_angle)
    matches = np.vstack(np.array(matches))
    non_matches = np.vstack(np.array(non_matches))
    with open('outputs/matches.csv', 'w') as f:
        for m in matches:
            f.write(f'{int(m[0]-1)},{int(m[1])},{m[2]}\n')


def speed_test():
    # compare reading speeds full size
    start = time.perf_counter()
    cat_sdss = pd.read_csv('inputs/SDSS/sdss_full.csv', delimiter=',')
    cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz.csv', delimiter=',', header=0)
    print('reading time full, csv (s):', time.perf_counter() - start)

    cat_sdss.to_feather('inputs/SDSS/sdss_full.feather')
    cat_gzoo2.to_feather('inputs/galaxyzoo/zoo2MainSpecz.feather')

    start = time.perf_counter()
    cat_sdss = pd.read_feather('inputs/SDSS/sdss_full.feather')
    cat_gzoo2 = pd.read_feather('inputs/galaxyzoo/zoo2MainSpecz.feather')
    print('reading time full, feather (s):', time.perf_counter() - start)


if __name__ == '__main__':
    pass

    get_galaxies()
    get_matches()
    # speed_test()

    # compare reading speeds
    start = time.perf_counter()
    cat_sdss = pd.read_csv('inputs/SDSS/sdss_galaxies.csv', delimiter=',')
    cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz.csv', delimiter=',', header=0)
    print('reading time, csv (s):', time.perf_counter() - start)
    start = time.perf_counter()
    cat_sdss = pd.read_feather('inputs/SDSS/sdss_galaxies.feather')
    cat_gzoo2 = pd.read_feather('inputs/galaxyzoo/zoo2MainSpecz.feather')
    print('reading time, feather (s):', time.perf_counter() - start)
    # reading time, csv (s): 5.792976083
    # reading time, feather (s): 0.3754684580000003

    # read data
    # cat_sdss = pd.read_csv('inputs/SDSS/sdss_galaxies_test.csv', delimiter=',', header=0)
    cat_sdss = pd.read_feather('inputs/SDSS/sdss_galaxies.feather')
    cat_sdss_np = cat_sdss.to_numpy()
    # cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz_test.csv', delimiter=',', header=0)
    cat_gzoo2 = pd.read_feather('inputs/galaxyzoo/zoo2MainSpecz.feather')
    cat_gzoo2_np = cat_gzoo2.to_numpy()

    # get rows of coordinates
    cat_sdss_radec_col = [1,2]
    cat_gzoo2_radec_col = [3,4]

    # crossmatching within max angle
    max_angle = 1/3600
    matches, non_matches = crossmatch.catalog_crossmatch_kdtree(cat_sdss_np, cat_sdss_radec_col, 
                                                                cat_gzoo2_np, cat_gzoo2_radec_col, 
                                                                max_angle)

    # cleaning and saving matches                                                            
    matches = np.vstack(np.array(matches))
    matches[:,0] -= 1
    non_matches = np.vstack(np.array(non_matches))
    with open('outputs/matches.csv', 'w') as f:
        for m in matches:
            f.write(f'{int(m[0])},{int(m[1])},{m[2]}\n')

    # combining dataframes
    cat_gzoo2.rename(columns = {'ra':'ra2', 'dec':'dec2'}, inplace = True)
    cat_sdss_out = cat_sdss.iloc[matches[:,0].astype(int)]
    cat_gzoo2_out = cat_gzoo2.iloc[matches[:,1].astype(int), [3,4,8]]
    df = cat_sdss_out.join(cat_gzoo2_out.set_index(cat_sdss_out.index))
    # print(df)

    # converting spirals to 0, elliptical to 1
    df['gz2class'] = df['gz2class'].str.replace(r'^[S][0-9a-zA-Z:,\D]+', 'spiral')
    df['gz2class'] = df['gz2class'].str.replace(r'^[E][0-9a-zA-Z:,\D]+', 'elliptical')
    df = df[~df['gz2class'].str.contains('A')]
    # print(df['gz2class'].unique())

    df['u-g'] = df['u'] - df['g']
    df['g-r'] = df['g'] - df['r']
    df['r-i'] = df['r'] - df['i']
    df['i-z'] = df['i'] - df['z']

    df['petro_cidx_u'] = df['petroR90_u']/df['petroR50_u']
    df['petro_cidx_g'] = df['petroR90_g']/df['petroR50_g']
    df['petro_cidx_r'] = df['petroR90_r']/df['petroR50_r']
    df['petro_cidx_i'] = df['petroR90_i']/df['petroR50_i']
    df['petro_cidx_z'] = df['petroR90_z']/df['petroR50_z']

    # moving class labels to last column
    df = df.drop(columns=['class', 'subclass', 'ra2', 'dec2'])
    new_cols = [col for col in df.columns if col != 'gz2class'] + ['gz2class']
    df = df[new_cols]
    # print(df.columns)

    # saving file
    df.to_csv('outputs/data_processed.csv')
    df.reset_index().to_feather('outputs/data_processed.feather')