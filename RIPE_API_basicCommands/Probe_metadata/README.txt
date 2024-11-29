The documents in this folder are obtained from command line functions or are using the data obtained from command line.

----------------------------------
DESCRIPTION OF FILES:

	- command_line_query.txt: .txt containing the command line query required to obtain the complete list of probes from a given country.
                        To obtain the probes from a country with code XX, just substitute the "es" after "--country" with the corresponding XX
	- probeListXX.csv/txt:  Output file from running the previous command line query for country XX
	- probesCountry.sh: Shell script that does the same as command_line_query.txt but automatically. It accepts two arguments:
						- first: country_code: ES i Spain, BR if brasil, DE if germany, etc.
						- Second: output file format. Either csv or txt
	- transform_probeList: Python script that takes the data in probeListXX.txt and cleans it. It produces two output files:
						- FullProbeListXX: all probes.
						- cleanProeListXX.csv: Only probes that are connected or disconnected recently.
	- get_probe_metadata.py: Python script to obtain the coordinates of the probes from RIPE Atlas website and obtain the distance between them

