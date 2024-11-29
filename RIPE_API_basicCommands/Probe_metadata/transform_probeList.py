# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 16:09:03 2023

@author: antonio

This file receives a list of probe metadata obtained from RIPE Magellan tool, from a specific country
and edits and transform it to clear and add some metadata
"""

import os, sys, time, requests
import math as m
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


def addPercentageTimeOn(probe_row):
    id_probe = probe_row['id']
    isPublic = probe_row['is_public']

    if isPublic == 'Yes':
        # Specify the URL you want to scrape
        url = 'https://atlas.ripe.net/frames/probes/'+ str(id_probe)+'/#tab-network'

        # Make a request to the URL and get the content
        response = requests.get(url)
        content = response.content

        # Parse the HTML content using BeautifulSoup and lxml parser
        soup = BeautifulSoup(content, "html.parser")

        # soup.find("table", class_ = "resolutions table table-condensed")

        info_section = soup.find(id= "connection-information")
        info_table = info_section.find("table", class_ = "resolutions table table-condensed")


        perc_time_connected_last_week  = info_table.find_all('tr')[1].find_all('td')[1].text
        perc_time_connected_last_month = info_table.find_all('tr')[2].find_all('td')[1].text
        perc_time_connected_last_all   = info_table.find_all('tr')[3].find_all('td')[1].text


        time_connected_last_week  = float(perc_time_connected_last_week[:-1])/100
        time_connected_last_month = float(perc_time_connected_last_month[:-1])/100
        time_connected_last_all   = float(perc_time_connected_last_all[:-1])/100
        print(f"ID: {id_probe} -- Time last week (%): {time_connected_last_week:.4f}\n")

        return pd.Series([time_connected_last_week,time_connected_last_month,time_connected_last_all])
    else:
        print(f"ID: {id_probe} -- Probe not public. Information not available\n")

        return pd.Series([None, None, None])


def format_binaryField(wrong_string):
    finalString ='WWW'
    if wrong_string == 'âœ˜':
        finalString = 'No'
    elif wrong_string == 'âœ”':
        finalString = 'Yes'
    else:
        finalString = 'mistake'
        print(wrong_string)
        sys.err('Not the expected string')
    return finalString

def timeDisconnected(probe_row):
    if probe_row['status'] == 'Connected':
        time_s_disconnected = 0
    else:
        probe_id  = probe_row['id']
        api_url   = "https://atlas.ripe.net/api/v2/probes/"+str(probe_id)
        response  = requests.get(api_url)
        resp_json = response.json()
        lastCon_timestamp = resp_json['last_connected']
        now_timestamp     = int( time.time() )
        time_s_disconnected = now_timestamp - lastCon_timestamp
        print(f"probe_id: {probe_id} - time_s_disconnected: {time_s_disconnected}")

    return time_s_disconnected

#%%

country_code = input("Please, write the country code (e.g.: ES for Spain): ")

## Read file with metadata
filename = "probeList"+country_code.upper()+".txt"
filepath = os.path.join('.',filename)

## open the data file, read the file as a list, and close the file
file = open(filepath)
fileLines = file.readlines()
file.close()

## Read file with IDs and save it as a pandas dataframe
filepathIDs = os.path.join('.',"idsonly"+country_code.upper()+".txt")
df_ids      = pd.read_csv(filepathIDs, header=None)

## number of fields in file
n_fields = 12

### get title of fields
fieldNamesPre = fileLines[4].split(maxsplit=n_fields)
fieldNames = []
for fields in fieldNamesPre:
    if fields == 'coordinates':
        fieldNames.append('latitude')
        fieldNames.append('longitude')
    else:
        fieldNames.append(fields)

## Obtain the useful lines (data) and correct the issue with "never connected" being split in two columns
rows = []
for line in fileLines[6:-2]:
    row = line.split(maxsplit=n_fields+1) # in case status == "never connected"
    if row[8] == "Never":
        row[8] = row[8] + ' ' + row[9]
        del row[9]
    else:
        if len(row) == n_fields+2:
            row[-2] = row[-2] + ' ' + row[-1]
            del row[-1]

    row[-1] = row[-1].strip()
    rows.append(row) # +1 because coordinates is 2 values

## Create dataframe with the previous data and field names
data = pd.DataFrame(rows, columns = fieldNames)

## add the list of IDs obtained from a different file
## ! Note: correct when the pull request is accepted:https://github.com/RIPE-NCC/ripe-atlas-tools/pull/243
data['id'] = df_ids

## Correct formatting of binary field
data['is_public'] = data['is_public'].transform(format_binaryField)

## Add information on portion of time connected from webpage
data_timeOn = data.apply(addPercentageTimeOn, axis=1)
data_timeOn.columns = ['lastWeekOn','lastMonthOn','allTimeOn']
data = data.join(data_timeOn)

## Just checking number of probes on
probesOnNow = data[data['status']=='Connected'].count()[0]
print(f"There are {probesOnNow} connected over {data.id.size} total probes")


#%% saving the dat into a new file

## Saving all the info
data.to_csv('FullProbeList'+country_code.upper()+'.csv')

## Obtaining only the probes that are connected or recently disconnected and are public
data_clean = data[data['status'].isin(['Connected', 'Disconnected']) & (data['is_public']=='Yes')]

## Obtaining the time taht a disconnected probe has been disconnected
row_time_sec  = data_clean.apply(timeDisconnected, axis =1)
row_time_days = [m.floor(rows/(60*60*24)) for rows in row_time_sec]

## Adding disconnected time both in seconds and days
data_clean = data_clean.assign(Seconds_disconnected=row_time_sec.values)
data_clean = data_clean.assign(Days_disconnected   =row_time_days)

# Saving the data into a CSV file
data_clean.to_csv('cleanProbeList'+ country_code.upper() + '.csv')


#%% Filtering Telefonica ASN

## Select the useful fields
data_usefulFields = data_clean.loc[:,['id','address_v4','asn_v4', 'prefix_v4','status','latitude', 'longitude',
'description', 'Seconds_disconnected', 'Days_disconnected', 'lastWeekOn','lastMonthOn','allTimeOn']].reset_index(drop=True)

min_time_thrhld_now = 0.99 # Threshold of uptime in last week
min_time_thrhld_all = 0.8  # Threshold of uptime in last month

asn_tlfnca = '3352' # AS number of TELEFONICA

## Keeping only telefonica probes
dd_tfn = data_usefulFields[data_usefulFields['asn_v4']==asn_tlfnca]

## Keeping only probes with enough uptime
dd_tfn['time_threshold'] = dd_tfn[['lastWeekOn','lastMonthOn']].min(axis=1)
dd_tfn_active = dd_tfn.loc[(dd_tfn['time_threshold']>min_time_thrhld_now) & (dd_tfn['allTimeOn']>min_time_thrhld_all)]

## Sorting probes by uptime (all time)
dd_tfn_active.sort_values(by=['allTimeOn'], inplace = True, ascending=False)

## Getting the first 40
ids_tfn_on = np.sort(dd_tfn_active.id.values[:40])

# Saving the data into a CSV file
dd_tfn_active.to_csv('finalProbeList'+ country_code.upper() + '.csv')


list_id1 = ""
for id_i in ids_tfn_on:
    list_id1 = list_id1 + str(id_i) + ","

list_id2 = ""
for id_i in ids_tfn_on:
    list_id2 = list_id2 + str(id_i) + "  "

print(list_id1)
print(list_id2)

dd_tfn_active.to_csv('finalProbeList'+ country_code.upper() + '.csv')

## saving only ids
np.savetxt("finalIDLists.csv", ids_tfn_on, delimiter=",")


# 334,341,3712,3726,3785,5010,11124,13881,14842,14866,15118,15618,15632,18844,21537,26072,30039,30356,30381,30392,33627,33818,33948,33971,34344,51265,51352,52511,52963,55661,60494,60561,60900,61230,61357,61547,62363,62588,62799,62939,1003090,1003970,1004021,1004200,1005149,1005622,1005642,1005907,1006026,1006085,1006162,1007387,1007401,

# 334  341  3712  3726  3785  5010  11124  13881  14842  14866  15118  15618  15632  18844  21537  26072  30039  30356  30381  30392  33627  33818  33948  33971  34344  51265  51352  52511  52963  55661  60494  60561  60900  61230  61357  61547  62363  62588  62799  62939  1003090  1003970  1004021  1004200  1005149  1005622  1005642  1005907  1006026  1006085  1006162  1007387  1007401



# 341,3712,3726,13881,14866,15618,15632,21537,26072,30392,33627,33818,33948,33971,34344,51265,51352,52511,55661,60494,60561,60900,61230,61357,61547,62363,62588,62799,62939,1003090,1003970,1004021,1004200,1005149,1005642,1005907,1006085,1006162,1007387,1007401,
# 341  3712  3726  13881  14866  15618  15632  21537  26072  30392  33627  33818  33948  33971  34344  51265  51352  52511  55661  60494  60561  60900  61230  61357  61547  62363  62588  62799  62939  1003090  1003970  1004021  1004200  1005149  1005642  1005907  1006085  1006162  1007387  1007401


