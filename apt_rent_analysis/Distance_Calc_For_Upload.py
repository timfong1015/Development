#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:57:48 2017

@author: Tim

Using data from CL, calculate distances from
Subways
Starbucks
WholeFoods
If you use this code you will have to get your own secret tokens for 
API access
"""
import pandas as pd
import numpy as np
import os 
from ast import literal_eval as le
import geopy 
import time
from geopy.distance import vincenty

OpenDir ='DIRECTORY_FOR_GITHUB' 
CloseDir = 'DIRECTORY_FOR_DATA'


#Read csv
dt = np.dtype('float,float')


#Get private directory data
os.chdir(CloseDir)
mta_csv = 'MTA_Locations.csv'
mta_stat = pd.read_csv(mta_csv)
mta_col = ['Line', 'Station Name', 'Entrance Latitude','Entrance Longitude']
mta_loc = mta_stat[mta_col]

mta_loc.insert(4, 'geotag',list(zip(mta_stat['Entrance Latitude'],
                                    mta_stat['Entrance Longitude'])))
mta_loc_ar = np.array([list(item) for item in mta_loc['geotag']])

wh_csv = 'WHOLE_FOODS_DATA'
wh_loc = pd.read_csv(wh_csv)
wh_col =['Latitude','Longitude']
wh_loc = wh_loc[wh_col]
#This line combines the Latitude and Longitude columns and converts to tuple
wh_loc['geotag'] = wh_loc[wh_col].apply(tuple, axis=1)
wh_loc_ar = np.array([list(item) for item in wh_loc['geotag']])

sbx_csv = 'STARBUCKS_DATA'
sbx_loc = pd.read_csv(sbx_csv)
sbx_loc = sbx_loc[wh_col]
sbx_loc['geotag'] = sbx_loc[wh_col].apply(tuple, axis=1)
#creates properly sized array by using list comprehension
sbx_loc_ar = np.array([list(item) for item in sbx_loc['geotag']])

#note could write a function to do this

#Choose apartment data
os.chdir(CloseDir)
def LoadApt(file):
    apt_data = pd.read_csv(file)
    apt_data = apt_data.drop_duplicates(['id'])
    apt_data['geotag'] = apt_data['geotag'].apply(func=lambda x:le(x))
    return apt_data

apt_data = LoadApt('06_05_2017.csv')
#convert strings of coordinates to float
#drop duplicate ad IDs
apt_data=apt_data.drop_duplicates(['id'])

#test to make sure all are float
for k in apt_data['geotag']:
   for j in k: 
       print(type(j))

#creates list    
apt_loc_ar = np.array([list(item) for item in apt_data['geotag']])

#get minimum distances
def distance(target, origin):
    dist = vincenty(target,origin)
    #closest = np.amin(dist, axis=0)
    #return closest
    return dist


#Geocoding
from geopy.geocoders import GoogleV3
#test of one point from dataframe
point = apt_data.get_value(2667,'geotag')
GoogAPIKey = 'API_SECRET_KEY'
geolocator = GoogleV3(api_key=GoogAPIKey)
address=geolocator.reverse(point)


#try to make a new column with complete borough name
#this times out so i need to make a new function or use a longer timeout

#apt_data['Borough'] = apt_data['geotag'].apply(func=lambda x:geolocator.reverse(x, timeout=30)[5][0])

def georev(addy, recursion=0):
    try: 
        return geolocator.reverse(addy)
    except GeocoderTimedOut as e:
        if recursion >10: #max recusions 
            raise e
        
        time.sleep(1) #wait
        #try again
        return georev(addy, recursion=recursion + 1)
    
#saving geoencoded data
apt_data.loc[apt_data['Borough'].str.contains('New', case=False),'Borough'] = 'Manhattan'

nbhood = 'NYC_nabe'
nbdf=pd.DataFrame(apt_data['Borough'].value_counts())
nbdf.to_csv(nbhood+'.csv', mode='a', header=True, index=True)

apt_data.to_csv('06_05_2017_geotagged' + '.csv', mode = 'a', header=True,
                index=True)

#Load existing geoencoded data
apt_data = pd.read_csv('06_05_2017_geotagged.csv')

#Measuring Distance test
target = sbx_loc_ar[0]
origin = apt_data.geotag[0]
print(vincenty(target,origin).miles)

#iterating over dataframe to get a scalar 
for idx in apt_data.index:
    print(apt_data.iat[idx,3])

for idx in apt_data.index:
    print(apt_data.get_value(idx,'geotag'))
#verify you can use .iat to get scalar for vicenty    
test = vincenty(apt_data.iat[0,3], sbx_loc_ar[0]).miles
print(test)

#gives minimum distance to a store from target list
def RawDist(point,target):
    distances=[]
    for idx in target.index:
        distances.append(vincenty(point, target.get_value(idx,'geotag')).miles)
    final = min(distances)   
    return final

        
apt_data['sbx_dist'] = apt_data['geotag'].apply(func=lambda x: RawDist(x,sbx_loc))
apt_data['wh_dist'] = apt_data['geotag'].apply(func=lambda x: RawDist(x,wh_loc))
apt_data['mta_dist'] = apt_data['geotag'].apply(func=lambda x: RawDist(x,mta_loc))

