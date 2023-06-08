#%%
import concurrent.futures
import pexpect
import json
import os
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.normpath(os.path.join(script_dir, "settings.json")), "r") as f:
    f_data = json.load(f)
    PATH_CONFIG = f_data["path_configs"]
    hosts = f_data['machines'].split(', ')

files = os.listdir(PATH_CONFIG)
# Splitting the IDs across the machines
ids = [int(file.split("_")[0].split(".")[0]) for file in files]

command_base = '/NFS/anaconda/python3.7/bin/python' + ' ' + script_dir + '/run_simulation.py' + ' '
ids = np.array_split(ids, len(hosts))

gen_id_str = lambda x: ' '.join([str(i) for i in x])

commands = []
for i in range(len(hosts)):
    command = command_base + gen_id_str(ids[i].tolist())
    commands.append(command)

def ssh_command(user, host, password, command):
    child = pexpect.spawn('ssh %s@%s %s' % (user, host, command))
    child.expect([pexpect.TIMEOUT, '[P|p]assword:'])
    child.sendline(password)
    child.expect(pexpect.EOF)  # Wait for the end of the command output
    return child.before  # This contains the output of your command

# Example usage
user = 'akuznets'
password = '123QwE!@#'

# host = 'mcao153'
# command = 'hostnamectl'
# result = ssh_command(user, host, password, command)
# print(result)

#%%
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(ssh_command, user, host, password, command): host for host, command in zip(hosts, commands)}

for future in concurrent.futures.as_completed(futures):
    host = futures[future]
    try:
        data = future.result()
    except Exception as exc:
        print('%r generated an exception: %s' % (host, exc))
    else:
        print('Host %r has data: \n%s' % (host, data))
