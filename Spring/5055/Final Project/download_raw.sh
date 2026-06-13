#!/bin/bash
while read category; do
    echo "Downloading $category..."
    gsutil cp "gs://quickdraw_dataset/full/simplified/$category.ndjson" ./raw_data/
done < selected_categories.txt
