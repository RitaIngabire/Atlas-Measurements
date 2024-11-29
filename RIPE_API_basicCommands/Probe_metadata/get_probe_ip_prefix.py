# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:48:19 2023

@author: antonio
"""

### see documentation at: https://atlas.ripe.net/docs/apis/rest-api-reference/#probes
### and https://ripe-atlas-cousteau.readthedocs.io/en/latest/use.html#id1

from ripe.atlas.cousteau import Probe  # To get the probe information

### Example list of probes id (sae as first experiment)
probe_id_list = [1004991,53229,1004997,1004102,12353,20757,1003454,1003158,1003747,1004200,51381,54470,1002914]

### Just saving the probes coordinates in a list
probe_prefix = []
for id_i in probe_id_list:
    probe = Probe(id=id_i) # Obtains all metadata of probe id_i
    print(probe.prefix_v4)
    probe_prefix.append(probe.prefix_v4) # saving to the list

