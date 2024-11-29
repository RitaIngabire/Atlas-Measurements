#!/usr/bin/env python

#
# This script depends on the external module `cymruwhois`.  You can install it
# with pip:
#
#   $ pip install cymruwhois
# It takes a json file and transform it into a traceroute read friendly output

import sys
import ujson as json
from cymruwhois import Client
import pickle as pickle
import time

# Usage: json2yaml.py file

if len(sys.argv) != 2:
    print("Usage: json2traceroute.py file", file=sys.stderr)
    sys.exit(1)


def whoisrecord(ip):
    currenttime = time.time()
    timestamp = 0
    asn = None
    if ip in whois:
        asn, timestamp = whois[ip]
    if (currenttime - timestamp) > 36000:
        client = Client()
        asn = client.lookup(ip)
        whois[ip] = (asn, currenttime)
    return asn

try:
    pkl_file = open('whois.pkl', 'rb')
    whois = pickle.load(pkl_file)
except IOError:
    whois = {}

# Create traceroute output
with open(sys.argv[1]) as f:

    try:
        for probe in json.load(f):
            probefrom = probe["from"]
            if probefrom:
                ASN = whoisrecord(probefrom)
                print("From: ", probefrom, "  ", ASN.asn, "  ", ASN.owner)
            print("Source address: ", probe["src_addr"])
            print("Probe ID: ", probe["prb_id"])
            result = probe["result"]
            for proberesult in result:
                ASN = {}
                if "result" in proberesult:
                    print(proberesult["hop"], "  ", end=' ')
                    hopresult = proberesult["result"]
                    rtt = []
                    hopfrom = ""
                    for hr in hopresult:
                        if "error" in hr:
                            rtt.append(hr["error"])
                        elif "x" in hr:
                            rtt.append(hr["x"])
                        else:
                            rtt.append(hr["rtt"])
                            hopfrom = hr["from"]
                            ASN = whoisrecord(hopfrom)
                    if hopfrom:
                        print(hopfrom, "  ", ASN.asn, "  ", ASN.owner, "  ", end=' ')
                    print(rtt)
                else:
                    print("Error: ", proberesult["error"])
            print("")

    finally:
        pkl_file = open('whois.pkl', 'wb')
        pickle.dump(whois, pkl_file)
