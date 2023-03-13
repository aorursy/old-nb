#!/usr/bin/env python
# coding: utf-8



# crawling data from 2017-09-01 to 2017-09-07
import urllib
import pandas as pd
import numpy as np
import multiprocessing
import warnings
import json

warnings.filterwarnings("ignore")


def get_views(web_info):
    global date
    purl = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/'           '{0}/{1}/{2}/{3}/daily/{4}/{5}'         .format(web_info[0], web_info[1], web_info[2], urllib.parse.quote(web_info[3]).replace("/", "%2F"), date[0], date[-1])
    #print(url)
    #print(qurl)
    res = np.array([np.nan for i in date])

    ok = True


    for tries in range(5):

        try:
            url = urllib.request.urlopen(purl)
            ret =url.read().decode()
            api_res = json.loads(ret)['items']
        except:
            ok = False

        if ok:
            break
    if not ok:
        print(purl, 'erro')
        return res

    for i in api_res:
        time = i['timestamp'][0:-2]
        res[date.index(time)] = i['views']
    return res


def get_views_main(input_page):
    pool_size = 4 #multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size)
    res = pool.map(get_views, input_page)
    pool.close()
    pool.join()
    return res


date = [
    '20170901',
    '20170902',
    '20170903',
    '20170904',
    '20170905',
    '20170906',
    '20170907',
    '20170908',
    '20170909'
]

import time

print("Reading...")

pages = pd.read_csv("/data02/data/WTF/train_2.csv", usecols=['Page'])
page_details = pd.DataFrame([i.split("_")[-3:] for i in pages["Page"]],columns=["project", "access", "agent"])
page_details['PageFull'] = pages

def name_split(row):
    return row.PageFull.split('_'+row.project+'_')[0]

page_details['Page'] = page_details.apply(name_split, axis=1)
del page_details['PageFull']

print("Crawling...")
start = time.time()
page_web_traffic = np.array(get_views_main(page_details.values))

print("Time:", time.time()-start)

print("total:", len(page_web_traffic))

