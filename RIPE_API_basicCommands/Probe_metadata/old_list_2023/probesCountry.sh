#! /bin/bash
checkOutputFormat(){
	format=$"${1,,}"
	if [ "$format" = "" ]; then
	    finalformat='txt'
	    echo "Format selected: txt"
	else
	    if [ "$format" != "txt" ]; then
		if [ "$format" != "csv" ]; then
		    echo "Please, select either 'txt' or 'csv'"
		    exit 1
		else
		    finalformat='csv'
		    echo "Format selected: csv"
		fi
	    else 
		finalformat='txt'
		echo "Format selected: txt"    
	    fi
	fi

}

if [ $# -eq 2 ]; then
	country_code=$1
	checkOutputFormat $2
fi
if [ $# -eq 0 ]; then
	echo "Enter the country code (two letters):"
	read country_code
elif [ $# -eq 1 ]; then
	country_code=$1
fi
if { [ $# -eq 1 ] || [ $# -eq 0 ]; }; then
	echo "Enter the format of the output file ('csv' or 'txt'). Enter empty for 'txt':"
	read outputformat
	checkOutputFormat $outputformat
fi

name_file="probeList${country_code^^}.${finalformat,,}"
echo "Saving all the probes from country: ${country_code} in file $name_file"

ripe-atlas probe-search --country ${country_code,,} --limit 1000 --field id --field address_v4 --field asn_v4 --field prefix_v4 --field address_v6 --field asn_v6 --field prefix_v6 --field country --field status --field is_public --field coordinates --field description > $name_file
