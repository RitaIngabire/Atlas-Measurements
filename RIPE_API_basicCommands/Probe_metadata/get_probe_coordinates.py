# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:10:35 2023

@author: antonio
"""
### see documentation at: https://atlas.ripe.net/docs/apis/rest-api-reference/#probes
### and https://ripe-atlas-cousteau.readthedocs.io/en/latest/use.html#id1

from ripe.atlas.cousteau import Probe  # To get the probe information
from geopy.distance import geodesic as geoDistance # To measure distance between coordinates

### Example list of probes id (sae as first experiment)
probe_id_list = [1004991,53229,1004997,1004102,12353,20757,1003454,1003158,1003747,1004200,51381,54470,1002914]

### Just saving the probes coordinates in a list
probe_coordinates = []
for id_i in probe_id_list:
    probe = Probe(id=id_i) # Obtains all metadata of probe id_i
    print(probe.geometry) #probe.geometry is a GeoJSON https://en.wikipedia.org/wiki/GeoJSON
    probe_coordinates.append(probe.geometry['coordinates']) # saving to the list

### How to obtain the distance in KM between 2 probes:
distt = geoDistance(probe_coordinates[0],probe_coordinates[1]).kilometers
print(distt)




