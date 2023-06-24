#!/bin/bash

# Loop through each directory
for dir in */
do
    # Extract the directory name
    dir_name=$(basename "$dir")
    
    # Check if the directory name starts with a number
    if [[ $dir_name =~ ^[0-9]+_ ]]
    then
        # Extract the number from the directory name
        number=$(echo "$dir_name" | grep -o '^[0-9]\+')
        
        # Decrease the number by 1
        new_number=$((number - 1))
        
        # Create the new directory name by replacing the number
        new_dir_name="${dir_name/$number/$new_number}"
        
        # Rename the directory
        mv "$dir" "$new_dir_name"
    fi
done
