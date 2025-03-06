from sys import argv

#file_input = argv[1]


def syn_each_ion(file_input, file_out=None):
	with open(file_input, 'r') as f:
		lines =  f.readlines()

	if file_out is None:
		file_out = file_input.replace('.','_sep.')
	with open(file_out, 'w') as f:
		f.writelines(lines)
	for line_i, line_ in  enumerate(lines):
		if 'setups' in line_:
			start_i = line_i + 1
		elif 'active' in line_:
			ion_n = line_.count('Yes')
			all_no = line_.replace('Yes','No')
			len_all_no = len(all_no)
			lines_ = lines.copy()
			for all_no_i in range(len_all_no):
				if all_no[all_no_i] == 'N':
					new_line_ = all_no[:all_no_i] + all_no[all_no_i:].replace('No','Yes',1)
					lines_[line_i] = new_line_
					with open(file_out, 'a') as f:
						f.writelines(lines_[start_i:])
			break
