#!/bin/bash

# check if a directory is given
if [[ $# = 0 ]]; then
    target_directory="."
else
    target_directory=$1
fi

# validate target_directory
# use double quotes around $target_directory to handle any blank spaces
if [[ ! -e "$target_directory" ]]; then
    echo "Error: The given directory does not exist"
    exit 1
fi

if [[ ! -d "$target_directory" ]]; then
    echo "Error: The given path is not a directory"
    exit 1
fi

# check in case the directory is not readable
if [[ ! -r "$target_directory" ]]; then
    echo "Error: The given directory is not readable."
    exit 1
fi

# counters to keep track of various extensions
total_files=0
images_count=0
audio_count=0
videos_count=0
documents_count=0
scripts_count=0
others_count=0

# various extensions
image_ext="jpg jpeg png gif bmp"
audio_ext="mp3 wav"
video_ext="mp4 wmv mov mvi"
doc_ext="pdf doc docx txt md"
script_ext="sh py js php"

for file in "$target_directory"/*; do
    # skip if it is not a regular file
    if [[ ! -f "$file" ]]; then
        continue
    fi

    # skip if it is a hidden file
    if [[ "$(basename "$file")" == .* ]]; then
        continue
    fi

    total_files=$((total_files + 1))

    # check file type
    file_type=""

    for ext in $image_ext; do
        if [[ "$file" == *.$ext ]]; then
            file_type="image"
            break
        fi
    done

    for ext in $audio_ext; do
        if [[ "$file" == *.$ext ]]; then
            file_type="audio"
            break
        fi
    done

    for ext in $video_ext; do
        if [[ "$file" == *.$ext ]]; then
            file_type="video"
            break
        fi
    done

    for ext in $doc_ext; do
        if [[ "$file" == *.$ext ]]; then
            file_type="doc"
            break
        fi
    done

    for ext in $script_ext; do
        if [[ "$file" == *.$ext ]]; then
            file_type="script"
            break
        fi
    done

    if [[ -z "$file_type" ]]; then
        file_type="other"
    fi

    echo "$file_type"

    # moving files depending on their extensions
    # it is much better to use case statements here, but since I haven't covered it, oh well
    if [[ "$file_type" == "image" ]]; then
        mkdir -p "$target_directory"/Images
        mv "$file" "$target_directory"/Images
        images_count=$((images_count + 1))
    fi

    if [[ "$file_type" == "audio" ]]; then
        mkdir -p "$target_directory"/Audio
        mv "$file" "$target_directory"/Audio
        audio_count=$((audio_count + 1))
    fi

    if [[ "$file_type" == "video" ]]; then
        mkdir -p "$target_directory"/Videos
        mv "$file" "$target_directory"/Videos
        videos_count=$((videos_count + 1))
    fi

    if [[ "$file_type" == "doc" ]]; then
        mkdir -p "$target_directory"/Documents
        mv "$file" "$target_directory"/Documents
        documents_count=$((documents_count + 1))
    fi

    if [[ "$file_type" == "script" ]]; then
        mkdir -p "$target_directory"/Scripts
        mv "$file" "$target_directory"/Scripts
        scripts_count=$((scripts_count + 1))
    fi

    if [[ "$file_type" == "other" ]]; then
        mkdir -p "$target_directory"/Others
        mv "$file" "$target_directory"/Others
        others_count=$((others_count + 1))
    fi
done

# print out summary report
echo "Total number of files processed: $total_files"
echo "Number of image files moved: $images_count"
echo "Number of audio files moved: $audio_count"
echo "Number of video files moved: $videos_count"
echo "Number of document files moved: $documents_count"
echo "Number of script files moved: $scripts_count"
echo "Number of other files moved: $others_count"
