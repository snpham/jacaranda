import requests
from urllib.request import urlopen
from urllib.parse import urlencode
import numpy as np
import json
from bs4 import BeautifulSoup
import urllib.request, urllib.parse, urllib.error
import pandas as pd
import re
import os
import psycopg2


def remove_sql_comments(sql):
    """Strip SQL comments starting with --"""
    return ' \n'.join(map(lambda x: x.split('--')[0], sql.split('\n')))



if __name__ == '__main__':
    pass
 
    NOBJECTS = 20000
 
    # CASjobs
    url = 'https://skyserver.sdss.org/dr16/en/tools/chart/f_sql.aspx'
    
    range_i = [0, 14]
    range2 = list(np.arange(15, 100, 0.01))
    # ranges2_test = list(np.linspace(15.1, 16, 10))
    ranges = np.hstack([range_i, range2])

    range2 = list(np.arange(24.634, 100, 0.001))
    ranges = range2

    count = 1
    for ii, range in enumerate(ranges, 1):
        query_text = ('\n'.join((f"SELECT TOP {NOBJECTS}",
            " p.objid,      p.ra,         p.dec,",
            " p.u,          p.g,          p.r,          p.i,          p.z,",
            " p.mCr4_u,     p.mCr4_g,     p.mCr4_r,     p.mCr4_i,     p.mCr4_z,",
            " p.petroR50_u, p.petroR50_g, p.petroR50_r, p.petroR50_i, p.petroR50_z,",
            " p.petroR90_u, p.petroR90_g, p.petroR90_r, p.petroR90_i, p.petroR90_z,",
            " s.class,      s.subclass,   s.z,          s.zerr",
            " FROM PhotoObj AS p",
            " JOIN SpecObj AS s ON s.bestobjid = p.objid",
            " WHERE ",
            f"  (p.u BETWEEN {ranges[ii-1]} AND {ranges[ii]}",
            "   AND p.g BETWEEN 0 AND 1000)",
            "")))
 
        query_text = remove_sql_comments(query_text)
        params = urlencode(dict(cmd=query_text, format='html'))
    
        # using bs4
        html = urllib.request.urlopen(url + '?%s' % params).read()
        soup = BeautifulSoup(html, 'html.parser')
    
        # getting data tables
        id_coords = soup('textarea')[0].contents[0][:-1]
        attributes = soup('textarea')[1].contents[0][:-1]

        # saving
        if count == 0:
            count += 1
            attributes = np.array(re.split('\n', attributes))
        else:
            attributes = np.array(re.split('\n', attributes))[1:]

        print(f'ranges {ranges[ii-1]}-{ranges[ii]}', len(attributes))
        full_csv = 'inputs/SDSS/sdss_full.csv'
        with open(full_csv, 'a') as f:
            for row in attributes:
                print(row, file=f)
