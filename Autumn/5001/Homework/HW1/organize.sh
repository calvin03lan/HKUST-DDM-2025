#!/usr/bin/env bash
# organize.sh

# 1. Diretory validity check
target_dir="${1:-.}"
[[ -d $target_dir ]] || mkdir -p "$target_dir"

# 2. Initialze
cd "$target_dir" || exit 1
declare -A cnt
for cat in Images Audio Videos Documents Scripts Others; do
    cnt[$cat]=0
done
total=0
shopt -s nullglob

# 3. Loop
for file in *; do
    [[ -f $file && $file != .* ]] || continue
    ext=${file##*.}
    ext=${ext,,}
    case $ext in
        jpg|jpeg|png|gif|bmp)     category=Images ;;
        mp3|wav)                  category=Audio ;;
        mp4|wmv|mov|mvi)          category=Videos ;;
        pdf|doc|docx|txt|md)      category=Documents ;;
        sh|py|js|php)             category=Scripts ;;
        *)                        category=Others ;;
    esac
    mkdir -p "$category"
    mv -- "$file" "$category/"
    cnt[$category]=$((cnt[$category]+1))
    total=$((total+1))
done

# 4. Report
echo "Total files processed: $total"
for cat in Images Audio Videos Documents Scripts Others; do
    [[ ${cnt[$cat]} -gt 0 ]] && echo "$cat: ${cnt[$cat]}"
done