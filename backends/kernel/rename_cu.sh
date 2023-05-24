for file in "$1"/*; do
    if [ -d "$file" ]; then
        # Recursive call for subdirectories
        rename_files "$file"
    elif [ -f "$file" ] && [[ "$file" == *.cu ]]; then
        # Rename .cu file to .cpp
        new_name="${file%.cu}.cpp"
        mv "$file" "$new_name"
        echo "Renamed: $file -> $new_name"
    fi
done