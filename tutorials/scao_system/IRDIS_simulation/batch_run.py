#%%
import numpy as np
import subprocess
import pexpect
import json
import os

# Directory containing sample files

current_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.normpath(os.path.join(current_dir, "settings.json")), "r") as f:
    folder_data = json.load(f)

py_dir     = '/NFS/anaconda/python3.7/bin/python'
user       = folder_data["user"]
password   = folder_data["password"]
directory  = folder_data["path_configs"]
machines   = folder_data["machines"].split(", ")
run_script = folder_data["path_output"] + "run_simulation.py"

sessions_per_machine = 3

# Get the files for processing
files = sorted([int(f.split('.')[0]) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

total_samples = len(files)

file_args = [' '.join(map(str, a.tolist())) for a in np.array_split(files, len(machines)*sessions_per_machine)]


#%%
c = 0
commands = []

for machine in machines:
    print(f"Connecting to {machine}...")
    for session_num in range(1, sessions_per_machine + 1):
        session_name = f"{machine}_{session_num}"
        print(f"Running session {session_name}...")

        subprocess.Popen(["tmux", "new-session", "-d", "-s", session_name])
        
        ssh_command = f"ssh {user}@{machine} {py_dir} {run_script} {file_args[c]}"
        c += 1
        child = pexpect.spawn(f"tmux send-keys -t {session_name} \"{ssh_command}; exit\" Enter", timeout=30)
        child.expect_exact("password:")
        child.sendline(password)

        print(f"Created session {session_name}")

    print(f"Disconnected from {machine}")
# %%
