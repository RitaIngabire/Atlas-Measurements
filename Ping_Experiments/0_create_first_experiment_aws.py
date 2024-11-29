import csv
from datetime import datetime,timedelta
from ripe.atlas.cousteau import (
    Ping,
    Traceroute,
    AtlasSource,
    AtlasCreateRequest
)

ATLAS_API_KEY = "e27c24f4-f226-44f4-9a36-c5b05bd33f55"

#building the probes list
with open("./data/id_probes.csv","r") as probes:
    probesReader= csv.reader(probes,delimiter=";")
    next(probes)
    probesList= []
    for i in probesReader:
        probesList.append(i[0])
    probesStr = ",".join(probesList)

#building the urlList
with open("./data/url_aws.csv","r") as urls:
    urlsReader = csv.reader(urls,delimiter="\n")
    urlsList = []
    for i in urlsReader:
        urlsList.append(i[0])


#generating the measurements
measurement_dict = {}
for i in urlsList[0:5]:
    ping = Ping(
                    af=4,
                    target=i,
                    interval=300, #run ping every 5 minutes
                    description="Ping Test"
                )
    traceroute = Traceroute(
                                af=4,
                                target=i,
                                interval=7200,#run traceroute every 2 hours
                                description="Traceroute Test",
                                protocol="ICMP",
                            )
    source = AtlasSource(
                                type="probes",
                                requested=len(probesList),
                                value= probesStr,
                                tags={"exclude": ["system-anchor"]}
                            )

    atlas_request = AtlasCreateRequest(
                                                start_time=datetime.utcnow()+ timedelta(seconds=300),
                                                stop_time= datetime.utcnow()+ timedelta(seconds=300)+timedelta(days=2),
                                                key=ATLAS_API_KEY,
                                                measurements=[ping, traceroute],
                                                sources=[source, source]
                                        )

    (is_success, response) = atlas_request.create()
    if is_success:
        measurement_dict.update(response)
    else:
        print(response)
    print(measurement_dict)

