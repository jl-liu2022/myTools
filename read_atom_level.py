import numpy as np

def read_atom_level(filename):
	atom_data = {}
	with open(filename, 'r') as f:
		line = f.readline()
		line = f.readline()
		while(line):
			line_split_ = line.split()
			len_split = len(line_split_)
			if len_split == 5:
				J_ = line_split_[2]
				if '/' in J_:
					J_split = J_.split('/')
					J_ = str(int(J_split[0])/int(J_split[1]))
				level_ = line_split_[0] + '_' + line_split_[1]
				energy_ = float(line_split_[3])*1.9864458571489288e-23
				atom_data[level_ + '_' + J_] = energy_
			elif len_split == 3:
				J_ = line_split_[0]
				if '/' in J_:
					J_split = J_.split('/')
					J_ = str(int(J_split[0])/int(J_split[1]))
				energy_ = float(line_split_[1])*1.9864458571489288e-23
				atom_data[level_ + '_' + J_] = energy_
			line = f.readline()
	return atom_data

def cal_Z(atom_data, T):
	Z = 0
	for k,v in atom_data.items():
		J_ = float(k.split('_')[-1])
		Z += (2*J_ + 1)*np.exp(-v/1.380649e-23/T)
	return Z

def cal_level_fration(atom_data, level, T):
	J = float(level.split('_')[-1])
	return (2*J + 1)*np.exp(-atom_data[level]/1.380649e-23/T)/cal_Z(atom_data, T)