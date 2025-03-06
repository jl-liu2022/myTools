from .config import filterDirectory
import numpy as np

zp_Vega = {'SLOAN_SDSS.u':3.75079e-9, 'SLOAN_SDSS.g':5.45476e-9, 'SLOAN_SDSS.r':2.49767e-9, 'SLOAN_SDSS.i':1.38589e-9, 'SLOAN_SDSS.z':8.38585e-10, 
		   'Generic_Bessell.U':3.96526e-9, 'Generic_Bessell.B':6.13268e-9, 'Generic_Bessell.V':3.62708e-9, 'Generic_Bessell.R':2.17037e-9, 'Generic_Bessell.I':1.12588e-9,
		   'Palomar_ZTF.g':5.2673e-9, 'Palomar_ZTF.r':2.23049e-9, 'Palomar_ZTF.i':1.1885e-9}

zp_AB = {'SLOAN_SDSS.u':8.60588e-9, 'SLOAN_SDSS.g':4.92255e-9, 'SLOAN_SDSS.r':2.85425e-9, 'SLOAN_SDSS.i':1.94038e-9, 'SLOAN_SDSS.z':1.35994e-9,
		   'Generic_Bessell.U':8.47077e-9, 'Generic_Bessell.B':5.69733e-9, 'Generic_Bessell.V':3.62786e-9, 'Generic_Bessell.R':2.57796e-9, 'Generic_Bessell.I':1.69232e-9,
		   'Palomar_ZTF.g':4.75724e-9, 'Palomar_ZTF.r':2.64344e-9, 'Palomar_ZTF.i':1.75867e-9}

def get_Vega(dir_name=filterDirectory):
	if dir_name[-1] != '/':
		dir_name = dir_name + '/'
	Vega_data = np.loadtxt(dir_name+'Vega_spec.txt').T
	return Vega_data[0], Vega_data[1]

def get_response(filter_name, dir_name=filterDirectory):
	if dir_name[-1] != '/':
		dir_name = dir_name + '/'
	data = np.loadtxt(dir_name+filter_name).T
	return data[0], data[1]

def get_zp(filter_name, filter_sys):
	if filter_sys == 'Vega':
		return zp_Vega[filter_name]
	elif filter_sys == 'AB':
		return zp_AB[filter_name]
	else:
		raise Exception('Unknown magnitude system')


