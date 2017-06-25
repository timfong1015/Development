#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 19:18:19 2017

@author: Tim
Uses craigslist module from:
https://wgagne-maynard.github.io/2017/02/01/Craigslist-Housing-Scraping.html
"""
#Get Craigslist Data and Process
import pandas as pd
import os
import time
from craigslist import CraigslistHousing
import timeit

directory = 'ACTIVE_DIRECTORY'
os.chdir(directory)

#wrap all code above in a function
def ScrapeNYC(area,limit,directory):
    os.chdir(directory)
    cl = CraigslistHousing(site='newyork', area=area)
    gen = cl.get_results(sort_by='newest', geotagged=True, limit=limit)
    
    t = []
    
    while True:
        try:
            result = next(gen)
        except StopIteration:
            break
        except Exception:
            continue
        t.append(result)
    df = pd.DataFrame(t)
    df_NAN = df.dropna(how='any')
    col_list = ['id','datetime','geotag','price']
    df_NAN = df_NAN[col_list]
    
    date = time.strftime("%m_%d_%Y")
    
    if os.path.isfile(date+'.csv'):
        df_NAN.to_csv(date+'.csv', mode='a', header=False, index=False)
    else: 
        df_NAN.to_csv(date+'.csv', index=False)

#Test this for Queens
queens='que'
limit = 100
directory='YOUR_DIRECTORY_HERE'
    
ScrapeNYC(queens,limit,directory)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, *kwargs)
    return wrapped

#wrapped = wrapper(ScrapeNYC, queens, limit, directory)

#timeit.timeit(wrapped, number=1)
#1600 seems to be the limit
#test bronx
bronx = 'brx'
limit = 842
wrapped = wrapper(ScrapeNYC, bronx, limit, directory)
timeit.timeit(wrapped, number=1)

#brooklyn
brk = 'brk'
limit = 1600
wrapped = wrapper(ScrapeNYC, brk, limit, directory)
timeit.timeit(wrapped, number=1)

#staten island
stn = 'stn'
limit = 383
wrapped = wrapper(ScrapeNYC, stn, limit, directory)
timeit.timeit(wrapped, number=1)

#test manhattan, estimated time 5 minutes for 800
mnh = 'mnh'
limit= 1600
wrapped = wrapper(ScrapeNYC, mnh, limit,directory)
timeit.timeit(wrapped, number=1)

#test queens, estimated time 10 minutes for 1600
#
que = 'que'
limit = 1600
wrapped = wrapper(ScrapeNYC, que, limit, directory)
timeit.timeit(wrapped, number=1)

