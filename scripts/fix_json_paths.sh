#!/bin/bash

# Usage: bash fix_json_paths.sh

# Get the absolute path of the current directory, ensure trailing slash
TARGET_PATH="$(pwd)/"
REPLACEMENT="./"

find . -type f -name 'transforms*.json' | while read -r file; do
    echo "Fixing $file"
    sed -i "s|$TARGET_PATH|$REPLACEMENT|g" "$file"
done
