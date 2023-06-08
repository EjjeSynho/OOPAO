import argparse
import subprocess
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

# create parser object
parser = argparse.ArgumentParser(description='Give config IDs for simulation.')

# add arguments
# parser.add_argument('python_path', type=str, help='a string input')
# parser.add_argument('script', type=str, help='another string input')
parser.add_argument('ids', metavar='N', type=int, nargs='+', help='an ID to simulate')

ids = parser.parse_args().ids

for id in ids:
    try:
        for id in ids:
            script_path = os.path.normpath(os.path.join(script_dir, "IRDIS_simulation.py"))
            command = ['/NFS/anaconda/python3.7/bin/python', script_path, str(id)]
            subprocess.run(command)
    except:
        print(f"Error in simulation {id}")
