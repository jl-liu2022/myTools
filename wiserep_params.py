# require python > 3.8
from sys import argv
from wiserep_api import get_target_property



def properties(name):
	if type(name) == type('haha'):
		name = [name]
	z = []
	ra = []
	dec = []
	for name_ in name:
		z_ =  str(get_target_property(name_,'redshift'))
		ra_, dec_ = get_target_property(name_,'coords_deg').split()
		z.append(z_)
		ra.append(ra_)
		dec.append(dec_)
	print(', '.join(z))
	print(', '.join(ra))
	print(', '.join(dec))

def __main__():
	name = argv[1]
	properties(name)