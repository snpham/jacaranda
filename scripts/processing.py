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

    cat_sdss = pd.read_csv('inputs/SDSS/sdss_galaxies.csv', delimiter=',')
    cat_gzoo2 = pd.read_csv('inputs/galaxyzoo/zoo2MainSpecz.csv', delimiter=',')
    cat_gzoo2.rename(columns = {'ra':'ra2', 'dec':'dec2'}, inplace = True)

    matches = pd.read_csv('outputs/matches.csv', delimiter=',', header=None)
    # print(matches)

    cat_sdss = cat_sdss.iloc[matches[0].to_list()]
    # print(cat_sdss)
    cat_gzoo2 = cat_gzoo2.iloc[matches[1].to_list(), [3,4,8]]
    # print(cat_gzoo2)

    df = cat_sdss.join(cat_gzoo2.set_index(cat_sdss.index))
    print(df)
