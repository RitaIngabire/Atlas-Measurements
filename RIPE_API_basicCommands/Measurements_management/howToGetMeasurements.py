from ripe.atlas.sagan import PingResult
import json,requests

print("**************************************")
print("                                     ")
print("This is from the local file.\n")

measurement = 46932386
inFile = 1

# from file
if inFile:
    filename = "./data/RIPE-Atlas-measurement-46932386.json"
    with open(filename) as user_file:
        results = json.load(user_file)
        print(results)

print("\n**************************************")
print("                                      ")
print("This is from the api                 ")


# from REST API
source   = "https://atlas.ripe.net/api/v2/measurements/"+str(measurement)+"/results/?format=json"
results = requests.get(source).json()
print(results)

for aa in results:
    my_result = PingResult(aa)
    print("\n", my_result)


