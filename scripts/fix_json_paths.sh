#!/bin/bash

# Usage: bash fix_json_paths.sh

# This script fixes JSON file paths by replacing everything before "/images/" with "./images/"
# Works with any path structure that contains "/images/" folder

find . -type f -name 'transforms*.json' | while read -r file; do
    echo "Fixing $file"
    # Replace any path ending with /images/ with ./images/
    sed -i 's|"[^"]*\/images\/|"./images/|g' "$file"
done
