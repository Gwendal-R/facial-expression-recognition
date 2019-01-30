#!/bin/bash

trap "kill $$" 2

declare -A personnes
personnes['Thomas']=0
personnes['Raphael']=0
personnes['Gwendal']=0
personnes['Julien']=0
parameterChosen=0

while getopts ":t :r :g :j :c" option; do
    case "${option}" in
        t)
            personnes['Thomas']=1
	    parameterChosen=1
            ;;
        r)
            personnes['Raphael']=1
	    parameterChosen=1
            ;;
        g)
            personnes['Gwendal']=1
	    parameterChosen=1
            ;;
	j)
	    personnes['Julien']=1
	    parameterChosen=1
	    ;;
	*)
	    echo -e "Wrong parameter(s) :\n-t for Thomas\n-r for Raphaël\n-g for Gwendal\n-j for Julien"
    esac
done

if [ "$parameterChosen" -eq 0 ]; then
	echo "You have to use a parameter to chose a person [-t | -r | -g | -j]"
fi

for j in ${!personnes[@]}
do
	if [ "${personnes[$j]}" -eq 1 ]; then
		n=$(ls datasets/$j/grimace_records | wc -l)

		for i in $(ls datasets/$j/grimace_records)
		do
			echo "VIDEOS ARE BEING TREATED, THERE ARE $n LEFT"
			rm -r datasets/$j/record
			python3 app.py extract video --grimaces shapefaciallandmarks datasets/$j/grimace_records/$i datasets/$j/record -v
			mv datasets/$j/record/facial_landmarks_grimaces.csv datasets/$j/csv/grimace_record_$n.csv

            #Adding a Grimace column at the end of the lines
			sed -i '1 s/.*/&,Grimace/' datasets/$j/csv/grimace_record_$n.csv
			sed -i 's/^[0-9].*$/&,1/g' datasets/$j/csv/grimace_record_$n.csv

			let n=n-1
		done
	fi
done
