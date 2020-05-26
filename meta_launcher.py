import argparse
from collections import OrderedDict
from functools import partial
import sys

# python meta_launcher.py --batch-size 2048 --seed 1 2 3 --lr 0.01 0.003 --enable_cuda --steps 1000 15000 --render --command "python3 main.py"
# sys.argv = ['meta_launcher.py', 'foo', '--batch-size', '2048', '--seed', '1', '2', '3', '--lr', '0.01', '0.003', '--enable_cuda', '--steps', '1000', '15000', '--render', '--command', 'python3 main.py']

args = sys.argv[1:]

def usage_msg():
    return '''meta_launcher.py
        [ordered args]
        --argA val1
        --argB
        --argC val1 val2 val3
        --command [command]
        '''

parser = argparse.ArgumentParser(description='Meta Launcher: ', usage=usage_msg())
parser.add_argument('--command', type=str, required=True)
command_arg, other_args = parser.parse_known_args(args)

base_cmd = command_arg.command

variables = OrderedDict({None: []})
current_var = None
for arg in other_args:
    if '--' == arg[:2]:
        # variable
        if current_var is not None and not variables[current_var]:
            variables[current_var] = ['']
        current_var = arg[2:]
        variables[current_var] = []
    else:
        variables[current_var].append(arg)
if current_var is not None and not variables[current_var]:
    variables[current_var] = ['']
if variables[None]:
    base_cmd = base_cmd + ' ' + ' '.join(variables[None])
    del variables[None]

cmd_list = [base_cmd]
for key, value_list in variables.items():
    cmd_list = [cmd+' --'+key+' {}' for cmd in cmd_list]
    cmd_list = [cmd.format(v) for v in value_list for cmd in cmd_list]

for cmd in cmd_list:
    print(cmd)
