#!/bin/bash

#esta es la ruta del archivo, donde esta el rac
rarcPath="${1}/rarc"
#y este es el archivo firectamente
filesPath="${2}"



generateOutputArgusFiles () {
    for filename in $filesPath/*.argus; do
	echo $filename
    	ra -F $rarcPath -r $filename > ${filename/argus/txt}
    done
}

generateOutputArgusFiles


