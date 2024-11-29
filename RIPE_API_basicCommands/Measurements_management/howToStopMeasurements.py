msm = [47537181,47537182,47537183,47537184,47537185,47537186,47537187,47537188,47537189,47537190]

msm_response= {}

from ripe.atlas.cousteau import AtlasStopRequest

for i in msm: 

    ATLAS_STOP_API_KEY = "e27c24f4-f226-44f4-9a36-c5b05bd33f55"
    atlas_request = AtlasStopRequest(msm_id=i, key=ATLAS_STOP_API_KEY)
    (is_success, response) = atlas_request.create()
    
    if is_success:
        msm_response.update(response) 
    else:
        print(response)
    