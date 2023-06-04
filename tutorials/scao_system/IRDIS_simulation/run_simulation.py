#%%
import os
import json
import subprocess

script_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.normpath(os.path.join(script_dir, "settings.json")), "r") as f:
    PATH_CONFIG = json.load(f)["path_configs"]
    
files = os.listdir(PATH_CONFIG)
ids = [int(file.split("_")[0].split(".")[0]) for file in files]

for id in ids:
    try:
        script_path = os.path.normpath(os.path.join(script_dir, "IRDIS_simulation.py"))
        command = ['python', script_path, str(id)]
        subprocess.run(command)
    except:
        print(f"Error in simulation {id}")
