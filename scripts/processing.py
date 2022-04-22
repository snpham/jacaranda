import numpy as np
from astropy.io import fits
import pandas as pd
import crossmatch
import exploratory


def get_galaxies():
    cat_sdss = pd.read_csv('inputs/SDSS/sdss_full_hyp.csv', delimiter=',')
    cat_sdss = cat_sdss[cat_sdss['class'] == 'GALAXY'].reset_index(drop=True)
    cat_sdss.to_csv('inputs/SDSS/sdss_galaxies.csv', index=None)


def get_matches():
    cat_sdss = pd.read_csv('inputs/SDSS/sdss_galaxies.csv', delimiter=',')
    # print(cat_sdss)
    cat_sdss_np = cat_sdss.to_numpy()
    cat_sdss_radec_col = [1,2]
    cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz.csv', delimiter=',')
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


if __name__ == '__main__':
    pass

    # get_galaxies()

    # get_matches()

    # read data
    # cat_sdss = pd.read_csv('inputs/SDSS/sdss_galaxies_test.csv', delimiter=',', header=0)
    cat_sdss = pd.read_csv('inputs/SDSS/sdss_galaxies.csv', delimiter=',', header=0)
    cat_sdss_np = cat_sdss.to_numpy()
    # cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz_test.csv', delimiter=',', header=0)
    cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz.csv', delimiter=',', header=0)
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
    print(df)

    # converting spirals to 0, elliptical to 1
    df['gz2class'] = df['gz2class'].str.replace(r'^[S][0-9a-zA-Z:,\D]+', 'spiral')
    df['gz2class'] = df['gz2class'].str.replace(r'^[E][0-9a-zA-Z:,\D]+', 'elliptical')
    print(df)

    df['u-g'] = df['u'] - df['g']
    df['g-r'] = df['g'] - df['r']
    df['r-i'] = df['r'] - df['i']
    df['r-z'] = df['r'] - df['z']

    df['petro_cidx_u'] = df['petroR90_u']/df['petroR50_u']
    df['petro_cidx_g'] = df['petroR90_g']/df['petroR50_g']
    df['petro_cidx_r'] = df['petroR90_r']/df['petroR50_r']
    df['petro_cidx_i'] = df['petroR90_i']/df['petroR50_i']
    df['petro_cidx_z'] = df['petroR90_z']/df['petroR50_z']

    print(df.columns)
    # saving file
    df.to_csv('outputs/data_processed.csv')