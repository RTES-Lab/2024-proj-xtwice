#!/bin/bash

# Define the base path and output base path
base_path="/home/ktw/Downloads"
output_base="./input"

# Define the files to be converted in order
files=("1105_6204_1200_H_F" "1105_6204_1201_H_F")

# Loop through each file and convert both the regular and Start files
for file in "${files[@]}"; do
  # Extract the year as the prefix of the file (e.g., 1104, 1105, etc.)
  year="${file:0:4}"

  # Create the output directory if it does not exist
  mkdir -p "${output_base}/${year}/${file}"

  # Convert the regular file
  ffmpeg -i "${base_path}/${file}.mov" -vf "crop=1920:540:0:540" -c:v libx264 -crf 18 -preset fast -c:a copy "${output_base}/${year}/${file}/${file}.mp4"

  # Convert the Start file
  ffmpeg -i "${base_path}/${file}_Start.mov" -vf "crop=1920:540:0:540" -c:v libx264 -crf 18 -preset fast -c:a copy "${output_base}/${year}/${file}/${file}_Start.mp4"
done