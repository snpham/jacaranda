import asyncio
import aiohttp
from urllib.parse import urlencode
import numpy as np
from bs4 import BeautifulSoup
import urllib.request, urllib.parse, urllib.error
import re


def remove_sql_comments(sql):
    """Strip SQL comments starting with --"""
    return ' \n'.join(map(lambda x: x.split('--')[0], sql.split('\n')))



if __name__ == '__main__':
    pass


async def fetch(session, url):
    async with session.get(url) as resp:
        return await resp.text()
        # Catch HTTP errors/exceptions here

async def fetch_concurrent(urls):
    loop = asyncio.get_event_loop()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for u in urls:
            tasks.append(loop.create_task(fetch(session, u)))

        for result in asyncio.as_completed(tasks):
            page = await result
            #Do whatever you want with results
            soup = BeautifulSoup(page)

            # print(soup('textarea')[1].contents[0][:-1])
            attributes = soup('textarea')[1].contents[0][:-1]

            # saving
            attributes = np.array(re.split('\n', attributes))[1:]
            full_csv = 'inputs/SDSS/sdss_full_hyp.csv'

            script = str(soup.find_all('script')[1])
            link = re.findall(r'\(p[^()]*\)', script)[0]
            print(link, len(attributes))
            # print(soup.find_all('script')[1])

            with open(full_csv, 'a') as f:
                for row in attributes:
                    print(row, file=f)

NOBJECTS = 10000

# CASjobs
url = 'https://skyserver.sdss.org/dr16/en/tools/chart/f_sql.aspx'

# range_i = [0, 14, 14.5, 15]
# range1 = list(np.arange(range_i[-1]+0.1, 16, 0.1))
# ranges = np.hstack([range_i, range1])
# range2 = list(np.arange(ranges[-1]+0.01, 17.5, 0.01))
# ranges = np.hstack([ranges, range2])
# range3 = list(np.arange(ranges[-1]+0.001, 100, 0.001))
# ranges = np.hstack([ranges, range3])

ranges = list(np.arange(27.911+0.1, 100, 1))

chunks = [ranges[x:x+20] for x in range(0, len(ranges), 20)]

for chunk in chunks:
    print(chunk)
    urls = []
    count = 0
    for ii, range in enumerate(chunk, 0):
        if ii == 0:
            pass
        else:
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
                f"  (p.u BETWEEN {chunk[ii-1]} AND {chunk[ii]})",
                "")))

            query_text = remove_sql_comments(query_text)
            params = urlencode(dict(cmd=query_text, format='html'))
            urls.append(url + '?%s' % params)

    # print(urls)
    asyncio.run(fetch_concurrent(urls))
    