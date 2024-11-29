import csv
from datetime import datetime
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

print(type(probesStr))


#building the urlList
with open("./data/url_aws.csv","r") as urls:
    urlsReader = csv.reader(urls,delimiter="\n")
    urlsList = []
    for i in urlsReader:
        urlsList.append(i[0])

#extracting the probes list

measurement_dict = {}
for i in urlsList:
    print(i)
    ping = Ping(
                    af=4,
                    target=i,
                    description="Ping Test"
                )
    traceroute = Traceroute(
                                af=4,
                                target=i,
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
                                            start_time=datetime.utcnow(),
                                            key=ATLAS_API_KEY,
                                            measurements=[ping, traceroute],
                                            sources=[source, source],
                                            is_oneoff=True
                                        )

    (is_success, response) = atlas_request.create()
    if is_success:
        measurement_dict.update(response)
    else:
        print(response)
    print(measurement_dict)

