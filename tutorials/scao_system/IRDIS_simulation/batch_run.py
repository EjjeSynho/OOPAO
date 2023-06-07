#%%
import pexpect

def ssh_command(user, host, password, command):
    child = pexpect.spawn('ssh %s@%s %s' % (user, host, command))
    child.expect([pexpect.TIMEOUT, '[P|p]assword:'])
    child.sendline(password)
    child.expect(pexpect.EOF)  # Wait for the end of the command output
    return child.before  # This contains the output of your command


# Example usage
user = 'ghost'
host = 'ghost.ads.eso.org'
password = 'Bnice2me!'
command = 'hostnamectl'
output = ssh_command(user, host, password, command)
print(output.decode('utf-8'))
