#!/opt/homebrew/bin/bash

for file in `ls | grep _c`
do
	newfile=`echo $file | sed 's/_c//'`
	mv $file $newfile
done