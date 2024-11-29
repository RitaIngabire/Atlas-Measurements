import csv

#building the probes list
with open("./data/id_probes.csv","r") as probes:
    probesReader= csv.reader(probes,delimiter=";")
    next(probes)
    probesList = []
    for i in probesReader:
        probesList.append(i[0])

#building the urlList
with open("./data/url_aws.csv","r") as urls:
    urlsReader = csv.reader(urls,delimiter="\n")
    urlsList = []
    for i in urlsReader:
        urlsList.append(i[0])

#print("\n The probes list = ", probesList, "\n")
#print("\n The servers we are pinging are = ", urlsList , "\n")


probesstr = ",".join(str(j) for j in probesList)
print(probesstr)

for i in probesList:
    print(len(probesList))
