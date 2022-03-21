from platform import machine
import numpy as np
import astronomy
import bss_reader
import supercosmos
import pandas as pd
import time


def find_closest_obj(catalog, target_coord):
    """reads a catalog file with args (objid, ra, dec) and the
    position of a target object (target_ra, target_dec) and finds
    the closest match for the target object in the catalog.
    :param catalog: (objectid, right ascenion, declination) [degrees]
    :param target_coord: target celestial right ascenion and 
    declination [RA, DEC] [degrees]
    :return min_id: object id of closest match in the catalog
    :return min_dist: angular distance of closest match [degrees]
    """
    min_dist = np.inf
    min_id = None
    cat_copy = np.array(catalog)
    cat_copy[:,1:3] = np.deg2rad(cat_copy[:,1:3])
    target_ra = np.deg2rad(target_coord[0])
    target_dec = np.deg2rad(target_coord[1])

    for id1, ra1, dec1 in cat_copy:
        # get angular distance b/t target and current catalog object
        dist = astronomy.greatcirc_dist([ra1, dec1], 
                                        [target_ra, target_dec])
        # replace if distance is smaller than previous
        if dist < min_dist:
            min_id = id1
            min_dist = dist
    
    min_dist = np.rad2deg(min_dist)
    return min_id, min_dist


def catalog_crossmatch(catalog1, catalog2, max_angle):
    """crossmatches 2 catalogs within a maximum angular distance.
    :param catalog1: array for first catalog (objid, ra, dec) [deg]
    :param catalog2: array for second catalog (objid, ra, dec) [deg]
    :param max_angle: maximum angular distance threshold (degrees)
    :return matches: matches found in both catalog (objid1, objid2,
    ang. dist [degrees])
    :return non_matches: unmatched objids from the first catalog
    """
    start = time.perf_counter()

    cat1_copy = np.array(catalog1)
    cat2_copy = np.array(catalog2)
    cat1_copy[:,1:3] = np.deg2rad(cat1_copy[:,1:3])
    cat2_copy[:,1:3] = np.deg2rad(cat2_copy[:,1:3])
    max_radius = np.deg2rad(max_angle)

    matches = []
    non_matches = []
    for id1, ra1, dec1 in cat1_copy:
        dists = astronomy.greatcirc_dist([ra1, dec1], 
                                         [cat2_copy[:,1], cat2_copy[:,2]])
        closest_objid = np.argmin(dists)
        closest_dist = dists[closest_objid]
        
        # ignore match if it's outside the maximum radius
        if closest_dist > max_radius:
            non_matches.append(id1)
        else:
            matches.append((id1, closest_objid, closest_dist))

    matches = np.array(matches)
    matches[:,2] = np.rad2deg(matches[:,2])

    time_taken = time.perf_counter() - start
    print('processing time (s):', time_taken)
    return matches, non_matches


def catalog_crossmatch_box(catalog1, catalog2, max_angle):
    """crossmatches 2 catalogs within a maximum angular distance.
    :param catalog1: array for first catalog (objid, ra, dec) [deg]
    :param catalog2: array for second catalog (objid, ra, dec) [deg]
    :param max_angle: maximum angular distance threshold (degrees)
    :return matches: matches found in both catalog (objid1, objid2,
    ang. dist [degrees])
    :return non_matches: unmatched objids from the first catalog
    """
    start = time.perf_counter()

    cat1_copy = np.array(catalog1)
    cat2_copy = np.array(catalog2)
    cat1_copy[:,1:3] = np.deg2rad(cat1_copy[:,1:3])
    cat2_copy[:,1:3] = np.deg2rad(cat2_copy[:,1:3])
    max_radius = np.deg2rad(max_angle)

    # Find ascending declination order of second catalogue
    asc_dec = np.argsort(cat2_copy[:,2])
    coords2_sorted = cat2_copy[asc_dec]
    dec2_sorted = coords2_sorted[:,2]

    # order = np.argsort(cat2_copy[:,2])
    # cat2_ordered = cat2_copy[order]

    matches = []
    non_matches = []
    for id1, ra1, dec1 in cat1_copy:
        closest_dist = np.inf
        closest_id2 = None
        # Declination search box
        min_dec = dec1 - max_radius
        max_dec = dec1 + max_radius

        start = dec2_sorted.searchsorted(min_dec, side='left')
        end = dec2_sorted.searchsorted(max_dec, side='right')

        for ii, (id2, ra2, dec2) in enumerate(coords2_sorted[start:end+1], start):
            dist = astronomy.greatcirc_dist([ra1, dec1], 
                                            [ra2, dec2])
            if dist < closest_dist:
                closest_sorted_id2 = ii
                closest_dist = dist
        
        # ignore match if it's outside the maximum radius
        if closest_dist > max_radius:
            non_matches.append(id1)
        else:
            closest_id2 = asc_dec[closest_sorted_id2]
            matches.append((id1, closest_id2, closest_dist))

    matches = np.array(matches)
    matches[:,2] = np.rad2deg(matches[:,2])

    time_taken = time.perf_counter() - start
    print('processing time (s):', time_taken)
    return matches, non_matches


if __name__ == '__main__':
    pass

    bss_fn = 'inputs/J_MNRAS_384_775_table2.fits'
    bss_out = bss_reader.bss_import(bss_fn)
    # print(bss_out)
    fn = 'inputs/SCOS_XSC_mCl1_B21.5_R20_noStepWedges.csv'
    fn = 'inputs/supercosmos_test.csv'
    sc_out = supercosmos.supercosmos_import(fn)
    # print(sc_out)

    min_id, min_dist = find_closest_obj(bss_out, [175.3, -32.5])
    assert np.allclose([min_id, min_dist], [156, 3.76705802264])
    min_id, min_dist = find_closest_obj(bss_out, [32.2, 40.7])
    assert np.allclose([min_id, min_dist], [26, 57.7291357756212])

    max_angle = 40/3600
    matches, non_matches = catalog_crossmatch(catalog1=bss_out, 
                                              catalog2=sc_out, 
                                              max_angle=max_angle)
    assert np.allclose(matches[:2], [(1, 1, 0.00010988610939332616), 
                                     (2, 3, 0.0007649845967242494)])
    assert np.allclose(non_matches[:5], [5, 6, 11, 29, 45])
    assert np.allclose(len(non_matches), 151)
    matches, non_matches = catalog_crossmatch_box(catalog1=bss_out, 
                                              catalog2=sc_out, 
                                              max_angle=max_angle)
    assert np.allclose(matches[:2], [(1, 1, 0.00010988610939332616), 
                                     (2, 3, 0.0007649845967242494)])
    assert np.allclose(non_matches[:5], [5, 6, 11, 29, 45])
    assert np.allclose(len(non_matches), 151)


    # data = pd.read_csv('inputs/supercosmos_test.csv', delimiter=',')
    # data.to_csv('inputs/supercosmos_test.csv', index=True)


