#!bin/bash

trap "kill $$" 2

declare -A personnes
personnes['Thomas']=0
personnes['Raphael']=0
personnes['Gwendal']=0
personnes['Julien']=0

while getopts ":t :r :g :j :c" option; do
    case "${option}" in
        t)
            personnes['Thomas']=1
            ;;
        r)
            personnes['Raphael']=1
            ;;
        g)
            personnes['Gwendal']=1
            ;;
	j)
	    personnes['Julien']=1
	    ;;
	*)
	    echo -e "Wrong parameter(s) :\n-t for Thomas\n-r for Raphaël\n-g for Gwendal\n-j for Julien"
    esac
done

for j in ${!personnes[@]}
do
	if [ "${personnes[$j]}" -eq 1 ]; then
		n=$(ls datasets/$j/normal_records | wc -l)

		for i in $(ls datasets/$j/normal_records)
		do
			echo "VIDEOS ARE BEING TREATED, THERE ARE $n LEFT"
			python3 app.py extract video --normaux shapefaciallandmarks datasets/$j/normal_records/$i datasets/$j/record -v
			mv datasets/$j/record/stream/facial_landmarks.csv datasets/$j/csv/record_$n.csv
			let n=n-1
		done
	fi
done
