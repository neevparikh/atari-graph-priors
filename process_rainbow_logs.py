import glob
import os
from shutil import copyfile
import re
import json
import sys


parent_name = "processed_logs/"
os.mkdir(parent_name)


for runscript in glob.glob(sys.argv[1]+"/*"):
   text = os.path.splitext(runscript)
   if text[-1] == ".o":
        filename = os.path.split(runscript)[-1]
        print(filename)
        os.mkdir(os.path.join(parent_name,filename))
        copyfile(runscript, os.path.join(parent_name,filename,"raw.txt"))

for raw_output in glob.glob(parent_name+"/*/raw.txt"):
	data_points = []
	with open(raw_output,"r") as raw_file:
		lines = [l for l in raw_file.readlines() if "Avg. reward:" in l]
		for l in lines:
			reward = 	re.search(r'Avg\. reward:(.+)\|', l).group(0).split()[2]
			step = re.search(r'T = .+ /',l).group(0).split()[2]

			data_points.append([step,'-1',reward])

	reward_file = os.path.join(*os.path.split(raw_output)[:-1])+"/reward.csv"
	with open(reward_file, 'w+') as reward_file:
		reward_file.write("steps,episode,reward\n")
		for d in data_points:
		   # print(d)
		   reward_file.write(",".join(d)+"\n")


for raw_output in glob.glob(parent_name+"/*/raw.txt"):
	data_points = []
	with open(raw_output,"r") as raw_file:
		json_text = raw_file.read()
		json_text = json_text[:json_text.find('Namespace')]

	param_dict = {}
	pairs = [l.strip().replace(":","").split() for l in json_text.split("\n") if ': ' in l]
	for key_value in pairs:
		param_dict[key_value[0]] = key_value[1]
	# print(param_dict)

	param_file = os.path.join(*os.path.split(raw_output)[:-1])+"/params.json"

	with open(param_file,"w+") as json_file:
		json.dump(param_dict,json_file)




