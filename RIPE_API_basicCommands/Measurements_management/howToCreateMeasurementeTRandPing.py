from datetime import datetime
from ripe.atlas.cousteau import (
    Ping,
    Traceroute,
    AtlasSource,
    AtlasCreateRequest
)

ATLAS_API_KEY = "e27c24f4-f226-44f4-9a36-c5b05bd33f55"

ping = Ping(
    af=4,
    target="ec2.eu-west-3.amazonaws.com",
    description="Ping Test"
)

traceroute = Traceroute(
    af=4,
    target="ec2.eu-west-3.amazonaws.com",
    description="Traceroute Test",
    protocol="ICMP",
)

source = AtlasSource(
    type="probes",
    requested=5,
    value='1004991,1004200,53229,20757,1003454',
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
    print(response,type(response))



