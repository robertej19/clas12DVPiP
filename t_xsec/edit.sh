#!/bin/bash

for file in *.png
do
    echo "Processing $file"
    convert "$file" -resize 50% "resized_$file"
done
