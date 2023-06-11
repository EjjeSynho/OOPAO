#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. Provide the directory containing samples."
    exit 1
fi

# Directory containing sample files
directory="$1"

# List of machines to connect to
machines=("machine1" "machine2" "machine3" "machine4" "machine5")

# Number of sessions per machine
sessions_per_machine=3

# Get the files for processing
files=($(find "$directory" -type f | sort))

# Number of samples to process
total_samples=${#files[@]}

# Calculate the number of samples per session
samples_per_session=$((total_samples / (${#machines[@]} * sessions_per_machine)))

# Check if there are any remaining samples after division
remaining_samples=$((total_samples % (${#machines[@]} * sessions_per_machine)))

# Loop through each machine
for machine in "${machines[@]}"
do
    echo "Connecting to $machine..."

    # Loop through each session
    for ((session_num=1; session_num<=sessions_per_machine; session_num++))
    do
        session="${machine}_$session_num"
        echo "Running session $session..."

        # Create a new tmux session
        tmux new-session -d -s "$session"

        # Calculate the starting and ending index for the current session's files
        start_index=$(((session_num - 1) * samples_per_session))
        end_index=$((start_index + samples_per_session - 1))

        # If we have remaining samples, distribute them across sessions
        if (( remaining_samples > 0 )); then
            end_index=$((end_index + 1))
            remaining_samples=$((remaining_samples - 1))
        fi

        # Loop through the files and execute the Python script
        for ((i=start_index; i<=end_index; i++))
        do
            file="${files[i]}"
            echo "Processing file: $file"

            # SSH into the machine and execute the Python script with the file as an argument
            tmux send-keys "ssh $machine /NFS/anaconda/python3.7/bin/python /path/to/your/python/script.py \"$file\"; exit" C-m
        done

        echo "Finished session $session"
    done

    echo "Disconnected from $machine"
done
