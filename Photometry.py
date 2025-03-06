import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
import math
import scipy.integrate as itg
import scipy.optimize as optimize
import sncosmo
import emcee
import corner

from astropy.coordinates import Angle
from scipy.interpolate import splrep, BSpline
from speclite import filters
from copy import deepcopy
from astropy.io import fits
from .readPhotometry import *
from .smoothTools import non_uniform_savgol
from .config import workDirectory
from .GalDust import coordsToEbv
from .constants import *
from .timeTransform import *
from .quick_mcmc import quick_mcmc
from sdapy import gaussian_process as GP
from scipy.interpolate import interp1d
from astropy.coordinates import Angle
from snpy.sn import *
from snpy.utils import fit1dcurve
from lightcurve_fitting.filters import all_filters, filtdict, extinction_law
from lightcurve_fitting.lightcurve import LC
from lightcurve_fitting.bolometric import calculate_bolometric, plot_bolometric_results, plot_color_curves

Rf = {'u': 4.786,  'g': 3.587, 'r': 2.471, 'i': 1.798,  'z': 1.403, 'y': 1.228, 'w':2.762,
		      'U': 4.744, 'UVOT.U': 4.94, 'B': 4.016, 'V': 3.011,'UVOT.B': 4.016, 'UVOT.V': 3.011, 'R': 2.386, 'G': 2.216, 'I': 1.684, 'J': 0.813, 'H': 0.516,
		      'K': 0.337, 'S': 8.795, 'D': 9.270, 'A': 6.432, 'UVW2': 8.795, 'UVM2': 9.270, 'UVW1': 6.432,  'F': 8.054,  'N': 8.969, 'o': 2.185, 'c': 3.111,
		      'W': 0.190, 'Q': 0.127, 'C': 0.237, 'E': 0.137, 'U_S': 4.94, 'B_S': 4.016, 'V_S': 3.011}

alpha = 0.25

markersize = 8

class PhotometricData:
	'''
	Instruction:
	Contains photometric data and some methods to deal with the data.

	Parameter:
	SpectraDir: The absolute directory containing the photometry data
	'''

	def __init__(self, PhotometryDir, minDate=None, maxDate=None, name=None, correction=[False, True], clipped=[True, True], Tmax=None, AB=False, 
		swift_galaxy=False, tnt_galaxy=False, database=None, autophot_tel=None, grandma_tel=None, other_exclude=None, forced_ztf=False, sigma=None):
		if name:
			self.name = name
			if not os.path.exists(workDirectory+name+'/'+name+'_results/para.txt'):
				self.z = float(input('refshift: '))
				self.ra = float(input('ra: '))
				self.dec = float(input('dec: '))
				self.EBVmilky = coordsToEbv(self.ra, self.dec)
				with open(workDirectory+name+'/'+name+'_results/para.txt','w') as f:
					f.writelines('redshift %s\n'%self.z)
					f.writelines('RA %s\n'%self.ra)
					f.writelines('DEC %s\n'%self.dec)
					f.writelines('ebv %s\n'%self.ebv)
			else:
				with open(workDirectory+name+'/'+name+'_results/para.txt','r') as f: 
					line     = f.readline().split()
					self.z   = float(line[1])
					line     = f.readline().split()
					self.ra  = float(line[1])
					line     = f.readline().split()
					self.dec = float(line[1])
					self.EBVmilky = coordsToEbv(self.ra, self.dec)
		if database is None:
			self.database = ['ztf', 'atlas','lick','tnt','swift','asassn','nowt','snova','bootes','yahpt','ast3-3','autophot','grandma'] # Database to read
		else:
			self.database = database
		self.Tmax = Tmax
		self.AB = AB

		allData_ = []
		bootes_done = 0
		for curdir in os.listdir(PhotometryDir):
			if curdir == 'ztf' and 'ztf' in self.database:
				curDir = PhotometryDir + '/' + curdir
				ztf_csv = False
				for files in os.listdir(curDir):
					if '.csv' == files[-4:]:
						allData_.append(readZtfData(curDir+ '/' + files, from_finished=True, forced=forced_ztf))
						ztf_csv = True
						#print(curDir+ '/' + files)
						break
			if curdir == 'swift' and 'swift' in self.database:
				curDir = PhotometryDir + '/' + curdir
				swift_json = False
				for files in os.listdir(curDir):
					if 'json' == files[-4:]:
						allData_.append(readSwiftData(curDir+ '/' + files, galaxy=swift_galaxy, json=True))
						swift_json = True
						#print(curDir+ '/' + files)
						break
			if curdir in self.database:
				curDir = PhotometryDir + '/' + curdir
				for files in os.listdir(curDir):
					if files.split('.')[-1] == 'dat':
						if curdir == 'autophot':
							autophot_data = pd.read_csv(curDir+ '/' + files)
							if autophot_tel:
								len_autophot = len(autophot_data)
								autophot_tel_mark = np.array([False for i_autophot in range(len_autophot)])
								for autophot_tel_ in autophot_tel:
									autophot_tel_mark = autophot_tel_mark + (autophot_data['tel'] == autophot_tel_)
								autophot_data = autophot_data[autophot_tel_mark].reset_index(drop=True)
							autophot_data = autophot_data[['MJD','filter','m','dm','detect']]
							allData_.append(autophot_data)
						if curdir == 'grandma':
							grandma_data = pd.read_csv(curDir+ '/' + files)
							if grandma_tel is not None:
								len_grandma = len(grandma_data)
								grandma_tel_mark = np.array([False for i_grandma in range(len_grandma)])
								for grandma_tel_ in grandma_tel:
									grandma_tel_mark = grandma_tel_mark + (grandma_data['tel'] == grandma_tel_)
								grandma_data = grandma_data[grandma_tel_mark].reset_index(drop=True)
							grandma_data = grandma_data[['MJD','filter','m','dm','detect']]
							allData_.append(grandma_data)
						if curdir == 'other':
							other_data = pd.read_csv(curDir+ '/' + files)
							other_data = other_data[['MJD','filter','m','dm','detect']]
							if other_exclude is not None:
								other_len = len(other_data)
								other_mark = [True for other_i in range(other_len)]
								for other_i in range(other_len):
									if other_data.loc[other_i, 'filter'].split('_')[0] in other_exclude:
										other_mark[other_i] = False
								other_data = other_data[other_mark].reset_index(drop=True)
							other_data = other_data.sort_values(by='MJD').reset_index(drop=True)
							allData_.append(other_data)
						if curdir == 'atlas':
							atlasData = readAtlasData(curDir+ '/' + files, minDate, maxDate, correction[0], clipped[0])
							for i in range(len(atlasData)):
								atlasData.loc[i, 'filter'] = 'ATLAS_' + atlasData.loc[i, 'filter']	
							allData_.append(atlasData)
						elif curdir == 'ztf':
							if ztf_csv == False:
								allData_.append(readZtfData(curDir+ '/' + files, minDate, maxDate, correction[1], clipped[1], from_finished=False))
						elif curdir == 'lick':
							allData_.append(readLickData(curDir+ '/' + files))
						elif curdir == 'tnt':
							if tnt_galaxy == True:
								if '_subpsf_' in files:
									allData_.append(readTntData(curDir+ '/' + files))
							else:
								if '_psf_' in files:
									allData_.append(readTntData(curDir+ '/' + files))
						elif curdir == 'swift' and swift_json == False:
							allData_.append(readSwiftData(curDir, galaxy=swift_galaxy))
						elif curdir == 'asassn':
							allData_.append(readASASSN(curDir+'/'+files))
						elif curdir == 'nowt':
							allData_.append(readNOWT(curDir+'/'+files))
						elif curdir == 'snova':
							allData_.append(readSnova(curDir+'/'+files))
						elif curdir == 'bootes' and bootes_done == 0:
							allData_.append(readBootes(curDir))
							bootes_done = 1
						elif curdir == 'yahpt':
							allData_.append(readYahpt(curDir))
						elif curdir == 'ast3-3':
							allData_.append(readAst3_3(curDir+'/'+files))
		allData_ = pd.concat(allData_).reset_index(drop=True)
		if sigma is not None:
			allData_['detect']  = allData_['detect'] * (1.0857 / allData_['dm'] > sigma)
		allData_ = allData_[allData_['detect']==True].reset_index(drop=True)
		self.dataBand = allData_['filter'].drop_duplicates().tolist()
		'''
		for i in range(len(allData_)):
			if allData_.loc[i, 'filter'] not in self.dataBand:
				self.dataBand.append(allData_.loc[i, 'filter'])
		'''
		allData_['mark'] = np.array([True for i in range(len(allData_))])

		# Organize data
		AB_to_Vega_tel = []
		AB_to_Vega_col = {'U':1.08,'B':-0.15,'V':-0.01,'R':0.13,'I':-0.37}
		self.allData = {}
		for item in self.dataBand:
			self.allData[item] = allData_[['MJD','m','dm','mark']][allData_['filter']==item].reset_index(drop=True)
			tel_ = item.split('_')[0]
			if tel_ in AB_to_Vega_tel:
				col_ = item.split('_')[-1]
				self.allData[item]['m'] = self.allData[item]['m'] - AB_to_Vega_col[col_]
			
		
		if AB:
			To_AB = {'UVOT_UVOT.B':0.13,'UVOT_UVOT.V':0.01,'UVOT_UVOT.U':-1.02,'UVOT_UVW1':-1.51,'UVOT_UVW2':-1.73,'UVOT_UVM2':-1.69}
			for k, v in To_AB.items():
				self.allData[k]['m'] = self.allData[k]['m'] - v



				
class PhotometricPlotter:

	def __init__(self, PhotometricData=None):
		if PhotometricData:
			self.name=PhotometricData.name
			if self.name:
				self.name = PhotometricData.name
				self.z = PhotometricData.z
				self.ra = PhotometricData.ra
				self.dec = PhotometricData.dec
				self.EBVmilky = PhotometricData.EBVmilky
			self.database = PhotometricData.database.copy()
			self.dataBand = PhotometricData.dataBand.copy()
			self.allData = PhotometricData.allData.copy()
			self.Tmax = PhotometricData.Tmax
			self.AB = PhotometricData.AB
		self.band_mag_max = {}
		self.other_MJD_max = {}
		self.interpolate = {}
		self.interpolate_method = {}
		self.interpolate_err = True
		self.database = ['ztf', 'atlas','lick','tnt','swift','asassn'] # Database to read
		self.UBVRI_bands = ['UVW2', 'UVM2', 'UVW1','UVOT.U','u',
			'U', 'UVOT.B', 'B', 'g', 'UVOT.V', 'V', 'c', 'r', 'o', 'G', 'R',  'i', 'I', 'z', 'L', 'w']
		self.central_wavelengths = {
			    'u': 3560,  'g': 4830, 'r': 6260, 'i': 7670,
			    'z': 8890, 'y': 9600, 'w':5985, 
			    'U': 3600, 'UVOT.U': 3469,  'B': 4380, 'V': 5450,'UVOT.B': 4380, 'UVOT.V': 5450, 'R': 6410,
			    'G': 6730, 'I': 7980, 'J': 12200, 'H': 16300,
			    'K': 21900, 'S': 2030, 'D': 2231, 'A': 2634,'UVW2': 2030, 'UVM2': 2231, 'UVW1': 2634,
			    'F': 1516, 'N': 2267, 'o': 6790, 'c': 5330,
			    'W': 33526, 'Q': 46028, 'C': 35075, 'E': 44366,
			}
		#All values in 1e-11 erg/s/cm2/Angs
		self.zp = {'u': 859.5, 'g': 466.9, 'r': 278.0, 'i': 185.2, 'z': 137.8, 'y': 118.2, 'w': 245.7, 
		      'U': 417.5, 'UVOT.U': 363.8, 'B': 632.0, 'V': 363.1,'UVOT.B': 632.0, 'UVOT.V': 363.1, 'R': 217.7, 'G': 240.0, 'I': 112.6, 'J': 31.47, 'H': 11.38,
		      'K': 3.961, 'S': 536.2, 'D': 463.7, 'A': 412.3, 'UVW2': 536.2, 'UVM2': 463.7, 'UVW1': 412.3, 'F': 4801., 'N': 2119., 'o': 236.2, 'c': 383.3,
		      'W': 0.818, 'Q': 0.242, 'C': 0.676, 'E': 0.273, 'UVOT.U':363.8, 'UVOT.V':363.1, 'UVOT.B':632.0, }
		#Filter widths (in Angs)
		self.filter_width = {'u': 458,  'g': 928, 'r': 812, 'i': 894,  'z': 1183, 'y': 628, 'w': 2560,
		                'U': 485,  'UVOT.U': 658,  'B': 831, 'UVOT.B': 831,'V': 827, 'UVOT.V': 827,'R': 1389, 'G': 4203, 'I': 899, 'J': 1759, 'H': 2041,
		                'K': 2800, 'S':671, 'UVW2': 671, 'D':446,'UVM2': 446, 'A':821,'UVW1': 821,  'F': 268,  'N': 732, 'o': 2580, 'c': 2280,
		                'W': 6626, 'Q': 10422, 'C': 7432, 'E': 10097}
		#Extinction coefficients in A_lam / E(B-V). Uses York Extinction Solver (http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/YorkExtinctionSolver/coefficients.cgi)
		self.Rf = Rf

	@classmethod
	def read_salt(cls, filename, name=None, Tmax=0, AB=False):
		cls_ = cls()
		cls_.name = name
		if cls_.name:
			cls_.name = name
			with open(workDirectory+name+'/'+name+'_results/para.txt','r') as f: 
				line     = f.readline().split()
				cls_.z   = float(line[1])
				line     = f.readline().split()
				cls_.ra  = float(line[1])
				line     = f.readline().split()
				cls_.dec = float(line[1])
				cls_.EBVmilky = coordsToEbv(cls_.ra, cls_.dec)
		cls_.Tmax = Tmax
		cls_.AB = AB

		# Get data
		cls_.dataBand = []
		cls_.allData = {}
		names = []
		with open(filename,'r') as f:
			line = f.readline()
			skiprows = 0
			while(line[0] == '@' or line[0] == '#'):
				skiprows += 1
				if line[0] == '#':
					names.append(line.split()[0][1:])
				line = f.readline()

		names.pop(-1)
		
		data = pd.read_csv(filename, skiprows=skiprows, sep=' ',names=names)
		data['mark'] = np.array([True for i in range(len(data))])
		data.rename(columns={'Date':'MJD', 'Mag':'m', 'Magerr':'dm', 'Filter':'filter'}, inplace=True)
		for i in range(len(data)):
			data.loc[i,'filter'] = data.loc[i,'filter'].split('::')[-1]
			if data.loc[i,'filter'] not in cls_.dataBand:
				cls_.dataBand.append(data.loc[i,'filter'])
		for item in cls_.dataBand:
			cls_.allData['UBVRI '+item] = data[['MJD','m','dm','mark']][data['filter']==item].reset_index(drop=True)
		return cls_

	@classmethod
	def read_Phot(cls, filename, name=None, Tmax=0, AB=False):
		cls_ = cls()
		cls_.name = name
		if cls_.name:
			cls_.name = name
			with open(workDirectory+name+'/'+name+'_results/para.txt','r') as f: 
				line     = f.readline().split()
				cls_.z   = float(line[1])
				line     = f.readline().split()
				cls_.ra  = float(line[1])
				line     = f.readline().split()
				cls_.dec = float(line[1])
				cls_.EBVmilky = coordsToEbv(cls_.ra, cls_.dec)
		cls_.Tmax = Tmax
		cls_.AB = AB

		# Get data
		cls_.dataBand = []
		cls_.allData = {}
		allData_ = pd.read_csv(filename)
		dataBand_ = allData_['filter'].drop_duplicates().tolist()
		for item in dataBand_:
			if 'stacked' not in item and 'UBVRI' not in item and 'SN' not in item:
				cls_.dataBand.append(item)
			elif 'ASASSN' in item:
				cls_.dataBand.append(item)

		# Organize data
		for item in dataBand_:
			if item[:2] != 'SN':
				cls_.allData[item] = allData_[['MJD','m','dm','mark']][allData_['filter']==item].reset_index(drop=True)
				cls_.allData[item] = cls_.allData[item].sort_values(by='MJD').reset_index(drop=True)

		return cls_

	def get_other(self, otherfile, othername, z=0, maxMJD=0, DM=0, DM_err=0, milky_ebv=0, host_ebv=0, host_rv=3.1, texp=0, stack=False, OAC=False, other_basicfile=None, 
		choose_source=None, remove_filters=None, remove_source_band=None, fit_tmax=False, fit_mmax=True):
		try:
			haha=self.other_mag_max
		except:
			self.other_mag_max = {}
		try:
			haha=self.other_texp
		except:
			self.other_texp = {}
		try:
			haha=self.other_z
		except:
			self.other_z = {}
		try:
			haha=self.other_MJD_max
		except:
			self.other_MJD_max = {}
		try:
			haha=self.other_DM
		except:
			self.other_DM = {}
		try:
			haha=self.other_DM_err
		except:
			self.other_DM_err = {}
		try:
			haha=self.other_milky_ebv
		except:
			self.other_milky_ebv = {}
		try:
			haha=self.other_host_ebv
		except:
			self.other_host_ebv = {}
		try:
			haha=self.other_host_rv
		except:
			self.other_host_rv = {}
		try:
			haha=self.other_basic = {}
		except:
			self.other_basic = {}
		sources = []
		if type(otherfile) == type('haha'):
			otherfile = [otherfile]
			if other_basicfile is not None:
				other_basicfile = [other_basicfile]
			othername = [othername]
			maxMJD = [maxMJD]
			DM = [DM]
			DM_err = [DM_err]
			milky_ebv = [milky_ebv]
			host_ebv = [host_ebv]
			host_rv = [host_rv]
			z = [z]
			texp = [texp]


		if other_basicfile is None:
			other_basicfile = [None for i in range(len(othername))]
		if len(otherfile) > len(maxMJD):
			maxMJD = np.zeros(len(otherfile))
			DM = np.zeros(len(otherfile))
			DM_err = np.zeros(len(otherfile))
			milky_ebv = np.zeros(len(otherfile))
			host_ebv = np.zeros(len(otherfile))
			host_rv = np.zeros(len(otherfile))
			z = np.zeros(len(otherfile))
			texp = np.zeros(len(texp))
		for (file, basicfile, name, maxmjd_, dm_, dm_err_, milky_ebv_, host_ebv_, host_rv_, z_, texp_) in zip(otherfile, other_basicfile, othername, maxMJD, DM, DM_err, milky_ebv, host_ebv, host_rv, z, texp):
			self.other_MJD_max[name] = maxmjd_
			self.other_DM[name] = dm_
			self.other_DM_err[name] = dm_err_
			self.other_milky_ebv[name] = milky_ebv_
			self.other_host_ebv[name] = host_ebv_
			self.other_host_rv[name] = host_rv_
			self.other_z[name] = z_
			self.other_texp[name] = texp_
			if basicfile is not None:
				basic_para = pd.read_csv(basicfile)
				if self.other_MJD_max[name] == 0:
					self.other_MJD_max[name] = timeToMjd(basic_para.loc[0, 'maxdate'].replace('/',''))
				self.other_basic[name] = {'ra':basic_para.loc[0, 'ra'], 'dec':basic_para.loc[0, 'dec'], 
										'redshift':basic_para.loc[0, 'redshift']}
				if ':' in self.other_basic[name]['ra']:
					self.other_basic[name]['ra'] = Angle(self.other_basic[name]['ra'] + ' hours').to('deg').value
					self.other_basic[name]['dec'] = Angle(self.other_basic[name]['dec'] + ' deg').value	
			else:
				self.other_basic[name] = {'redshift':z_, 'ra':None, 'dec':None}
			other_data = pd.read_csv(file)
			if OAC == True:
				other_data.rename(columns={'time':'MJD', 'magnitude':'m','e_magnitude':'dm','band':'filter'}, inplace=True)
				mark_no_dm = (other_data['dm'].isna())+(other_data['dm']==0)
				other_data['dm'][mark_no_dm] = 0.02
				mark_good = (other_data['MJD'].notna())*(other_data['m'].notna())*(other_data['filter'].notna())*(other_data['dm']<99)*(other_data['m']!=np.inf)
				other_data = other_data[mark_good].reset_index(drop=True)
				if choose_source is not None:
					if type(choose_source) == type('haha'):
						choose_source = [choose_source]
					len_data = len(other_data)
					mark_source = np.array([False for i in range(len_data)])
					len_source = len(choose_source)
					for i in range(len_source):
						mark_source = mark_source + (other_data['source'] == choose_source[i])
					other_data = other_data[mark_source].reset_index(drop=True)
				other_data['MJD'] = other_data['MJD'].astype('float')
				other_data['m'] = other_data['m'].astype('float')
				other_data['dm'] = other_data['dm'].astype('float')
				if remove_source_band is not None:
					if type(remove_source_band) == type('haha'):
						remove_source_band = [remove_source_band]
					for source_band_ in remove_source_band:
						sourve_, band_ = remove_source_band.split('_')
						mark_remove = (other_data['source'] == sourve_) * (other_data['filter'] == band_)
						other_data = other_data[~mark_remove]
				#print(len(mark_good), len(other_data))
				#exit()
				mark_swift = (other_data['source'] == '2014Ap&SS.354...89B') * ((other_data['filter'] == 'U') + (other_data['filter'] == 'V') + (other_data['filter'] == 'B')
					+ (other_data['filter'] == 'u') + (other_data['filter'] == 'v') + (other_data['filter'] == 'b'))
				other_data['filter'][mark_swift] = 'UVOT.' + other_data['filter'][mark_swift]
				mark_swift = (other_data['source'] == '2014Ap&SS.354...89B') * ((other_data['filter'] == 'uvw1') + (other_data['filter'] == 'uvm2') + (other_data['filter'] == 'uvw2')
					+ (other_data['filter'] == 'UVOT.u') + (other_data['filter'] == 'UVOT.b') + (other_data['filter'] == 'UVOT.v'))
				if mark_swift.any():
					swift_AB = 1
				else:
					swift_AB = 0
				if swift_AB:
					#mark_swift = (other_data['source'] == '2014Ap&SS.354...89B')
					To_VEGA = {'UVOT.B':-0.13,'UVOT.V':-0.01,'UVOT.U':1.02,'UVW1':1.51,'UVW2':1.73,'UVM2':1.69}
					for swift_i in other_data['filter'][mark_swift].index:
						#print(name, other_data.loc[swift_i,'filter'])
						other_data.loc[swift_i,'filter'] = other_data.loc[swift_i,'filter'].upper()
						other_data.loc[swift_i,'m'] = other_data.loc[swift_i,'m'] - To_VEGA[other_data.loc[swift_i,'filter']]
				other_data['mark'] = True
			other_bands = other_data['filter'].drop_duplicates().tolist()
			for item in other_bands:
				if remove_filters is not None and item in remove_filters:
					continue
				if 'mark' not in other_data.keys():
					other_data['mark'] = True
				one_band_data = other_data[['MJD','m','dm','mark']][other_data['filter']==item].reset_index(drop=True)
				mark = one_band_data['mark']
				v_ = one_band_data[mark].sort_values(by='MJD').reset_index(drop=True)
				if stack == True:
					if not v_.empty:
						length = len(v_)
						v_['stack'] = np.array([False for i in range(length)])
						v_.loc[length] = [0, 0, 0, True, False]
						head = 0
						v_.loc[head, 'stack'] = True
						for i in range(1, length+1):
							if int(v_.loc[i, 'MJD']) != int(v_.loc[head, 'MJD']):
								# if debug < 5:
								# 	 debug += 1
								v_.loc[head, 'MJD'] = v_.loc[head:(i-1), 'MJD'].sum()/(i-head)
								v_.loc[head, 'm'] = v_.loc[head:(i-1), 'm'].sum()/(i-head)
								v_.loc[head, 'dm'] = v_.loc[head:(i-1), 'dm'].sum()/(i-head)/np.sqrt(i-head)
								head = i
								if head != length:
									v_.loc[head, 'stack'] = True
						v_ = v_[(v_['stack']==True)].reset_index(drop=True).drop(labels='stack', axis=1)
				len_v = len(v_)
				#print(name, len_v, v_.loc[0, 'MJD'], self.other_MJD_max[name], v_.loc[len_v-1, 'MJD'])
				if len_v < 3 or not (v_.loc[0, 'MJD'] < self.other_MJD_max[name] < v_.loc[len_v-1, 'MJD']) or not fit_mmax:
					#m_max_index = np.argsort(v_['m'].to_numpy())[0]
					m_max = v_['m'].min()
				else:
					t_max, m_max = find_curve_min(v_['MJD'], v_['m'], v_['dm'], self.other_MJD_max[name], Name=name)
					if fit_tmax == True and item == 'B':
						self.other_MJD_max[name] = t_max
						print(name, t_max)
				rename_dict = {"u'":"u","g'":"g","r'":"r","i'":"i","z'":"z",'Rc':'R','Ic':'I'}
				if item in rename_dict.keys():	
					item_ = rename_dict[item]
				else:
					item_ = item
				self.other_mag_max[name + ' ' + item_] = m_max
				self.allData[name + ' ' + item_] = v_
			if OAC == True:
				if choose_source is not None:
					sources.append(','.join(choose_source))
				else:
					sources.append(','.join(other_data['source'].drop_duplicates().tolist())) 
			else:
				sources.append(None)
		return sources

	def remove_other_lc(self, othername):
		if type(othername) == type('haha'):
			othername = [othername]
		keys = list(self.allData.keys())
		for item in keys:
			name_ = item.split()[0]
			if name_ in othername:
				self.allData.pop(item)


	def get_other_snpy(self, name, z=None, ra=None, dec=None, err=None, model='EBV_model', shapeParam='dm15', save=None, get_Mmax=True, source=None, method='model'):
		try:
			haha=self.other_mag_max
		except:
			self.other_mag_max = {}
		try:
			haha=self.other_snpy_params
		except:
			self.other_snpy_params = {}
		filename = self.name + '_results/' + name + '_snpy.txt'
		if z is not None:
			self.other_basic[name]['redshift'] = z
		if ra is not None:
			self.other_basic[name]['ra'] = ra
		if dec is not None:
			self.other_basic[name]['dec'] = dec
		with open(filename, 'w') as f:
			f.writelines('%s %s %s %s\n' %(name, self.other_basic[name]['redshift'], self.other_basic[name]['ra'], self.other_basic[name]['dec']))
			for item in self.allData.keys():
				if name in item:
					name_, band_ = item.split()
					if name == name_ and band_ in ['U','B','V','g','r','R','i','I','UVW2','UVM2','UVW1','UVOT.U','UVOT.B','UVOT.V']:
						if band_ in ['B','V','R','I']:
							f.writelines('filter ' + band_ + 's\n')
						elif band_ in ['u','g','r','i','z']:
							f.writelines('filter ' + band_ + '_s\n')
						elif band_ in ['UVOT.U','UVOT.B','UVOT.V']:
							f.writelines('filter ' + band_.split('.')[1] + '_UVOT\n')
						else:
							f.writelines('filter ' + band_ + '\n')
						#f.writelines('filter ' + band_ + '\n')
						mark = self.allData[item]['mark']
						data_to_save_ = self.allData[item][mark].reset_index()
						if err is not None:
							data_to_save_['dm'][data_to_save_['dm']<err] = err
						for j in range(len(data_to_save_)):
							f.writelines('%s %s %s\n' %(data_to_save_.loc[j, 'MJD'], 
								data_to_save_.loc[j, 'm'], data_to_save_.loc[j, 'dm']))

		s = get_sn(filename)
		if method == 'model':
			s.choose_model(model, stype=shapeParam)
			s.set_restbands()
			bands = []
			fit_bands = ['Us','Bs','Vs','g','r','Rs','i','Is','B','V']
			'''
			s.Bs.mask_epoch(-30+self.other_MJD_max[name],20+self.other_MJD_max[name])
			s.Bs.template(method='polynomial', n=4)
			#print('Bs')
			print(s.Bs.Tmax, s.Bs.e_Tmax, s.Bs.Mmax, s.Bs.e_Mmax)
			s.Bs.plot()
			plt.show()
			exit()
			'''
			for item in s.data.keys():
				if item in fit_bands:
					bands.append(item)
			if self.other_MJD_max[name] != 0:
				print(self.other_MJD_max[name])
				print(bands)
				for band in bands:
					s.data[band].mask_epoch(-15+self.other_MJD_max[name], 60+self.other_MJD_max[name])
					#s.data[band].plot()
					#plt.show()
			if 'Bs' in bands:
				s.fit(['Bs'], EBVhost=0)
			elif 'B' in bands:
				s.fit(['Bs'], EBVhost=0)
				#print(s.parameters)
			s.fit(bands=bands)
			s.summary()
			plt.show()
			exit()
			if self.other_MJD_max[name] == 0:
				if 'Bs' in bands:
					self.other_MJD_max[name] = s.get_max('Bs')[0]
			self.other_snpy_params[name] = {}
			# ['DM', 'dm15', 'EBVhost', 'Tmax']
			for item in s.parameters.keys():
				self.other_snpy_params[name][item] = [s.parameters[item], s.errors[item] + s.systematics()[item]]
			if 'Bs' in bands:
				self.other_snpy_params[name]['true_Tmax'] = [s.get_max('Bs')[0], s.Bs.e_Tmax] #?
			self.other_snpy_params[name]['EBVmilky'] = s.EBVgal
			if get_Mmax == True:
				for item in bands:
					if item in ['Us', 'Bs', 'Vs', 'Rs', 'Is']:
						band_ = item[0]
					else:
						band_ = item
					s.choose_model('max_model', stype=shapeParam)
					s.fit(bands=[item])
					self.other_mag_max[name + ' ' + band_] = s.get_max(item)[1]
		if save is not None:
			with open(self.name+'_results/'+save+'.txt', 'a') as f:
				#f.writelines('Name dm15 dm15_err t0 t0_err\n')
				if source is None:
					source = '?'
				for k,v in self.other_snpy_params.items():
					f.writelines('%s %s %s %s %s %s\n'%(k, v['dm15'][0], v['dm15'][1], v['Tmax'][0], v['Tmax'][1], source))

	def get_other_salt_py(self, othername, fit_phase_range = (-15,45), source='salt2', err=0.02, save=None, lc_source=None, UBVRI_bands=None):
		if UBVRI_bands is None:
			UBVRI_bands=['U','B','g','V','r','R','i','I']
		self.SaveSALT(UBVRI=True, UBVRI_bands=UBVRI_bands, filename='Salt_%s.dat'%othername, new=True, err=0.02, othername=othername)
		self.get_salt_model_py(self.name+'_results/Salt_%s.dat'%othername, fit_phase_range=fit_phase_range, 
			source=source, othername=othername, save=save, lc_source=lc_source,)

	def get_salt_model_py(self, salt_file, fit_phase_range = (-15,45), model_phase_range=None, source='salt2', othername=None, save=None, lc_source=None, use_model=True):
		if model_phase_range is None:
			model_phase_range = fit_phase_range
		data = sncosmo.read_lc(salt_file,format='salt2')
		data.rename_column('Date','time')
		data.rename_column('Filter','band')
		#data.sort('time')
		ab = sncosmo.get_magsystem('ab')
		vega = sncosmo.get_magsystem('vega')
		phase_mask = (data['time'] > fit_phase_range[0]) * (data['time'] < fit_phase_range[1])
		data = data[phase_mask]
		data['flux'] = 0.
		data['fluxerr'] = 0.
		data['zp'] = 0.
		zp_dict = {'STANDARD::U': 14.133309018839299, 'STANDARD::B': 15.254431605251767, 'STANDARD::V': 14.824030186077428, 'STANDARD::R': 15.022942199703149, 
				   'STANDARD::I': 14.551056359274279, 'SDSS::u': 12.188604209568233, 'SDSS::g': 14.344175630135116, 'SDSS::r': 14.233186338790151, 'SDSS::i': 13.889746270307079}
		for i in range(len(data)):
			if data['MagSys'][i] == 'AB':
				data['flux'][i] = ab.band_mag_to_flux(data['Mag'][i], data['band'][i])
				#data['zp'][i] = zp_dict[data['band'][i]]
				data['zp'][i] = zp_dict[data['band'][i]]
				#data['zpsys'] = 'ab'
			elif data['MagSys'][i] == 'VEGA':
				data['flux'][i] = vega.band_mag_to_flux(data['Mag'][i], data['band'][i])
				data['zp'][i] = zp_dict[data['band'][i]]
				#data['zpsys'] = 'vega'
			data['fluxerr'][i] = data['flux'][i] * data['Magerr'][i] / 1.0857

		if source == 'salt2':
			source = sncosmo.get_source("salt2", "2.4")
		else:
			source = sncosmo.get_source(source)
		#model = sncosmo.Model(source='salt2')
		model = sncosmo.Model(source=source,
		                          effects=[sncosmo.F99Dust()],
		                          effect_names=['mw'],
		                          effect_frames=['obs'])
		if othername is not None:
			ra = self.other_basic[othername]['ra']
			dec = self.other_basic[othername]['dec']
			z = self.other_basic[othername]['redshift']
			EBVmilky = coordsToEbv(ra, dec)
		else:
			z = self.z
			EBVmilky = self.EBVmilky
		model.set(z=z, mwebv=EBVmilky)
		result, fitted_model = sncosmo.fit_lc(data, model, ['t0', 'x0', 'x1', 'c'])
		sncosmo.plot_lc(data, model=fitted_model, errors=result.errors)
		plt.show()
		# save parameters
		if othername is not None:
			t0 = result.parameters[1]
			t0_err = result.errors[result.param_names[1]]
			x1_ = result.parameters[3]
			dx1_ =  result.errors[result.param_names[3]]
			sampling = 0.1
			B_Tmax = [fitted_model.source.peakphase('STANDARD::B',sampling=sampling)+t0, t0_err]
			#B_Tmax.append(abs(B_Tmax-t0))
			flux_all_ = fitted_model.bandfluxcov(band='STANDARD::B',time=B_Tmax[0])
			B_max = [fitted_model.bandmag('STANDARD::B','vega',B_Tmax[0]),np.sqrt(flux_all_[1])/flux_all_[0]*1.0857]
			dm15_fit = [1.09 - 0.161*x1_ + 0.013*x1_*x1_ - 0.00130*x1_*x1_*x1_,
							abs(-0.161 + 0.013*2*x1_ - 0.0013*3*x1_*x1_)*dx1_]
			dm15_ = fitted_model.source.bandmag('STANDARD::B','vega',B_Tmax[0]+15) - fitted_model.source.bandmag('STANDARD::B','vega',B_Tmax[0])
			dm15_flux_all_ = fitted_model.bandfluxcov(band='STANDARD::B',time=[B_Tmax[0], B_Tmax[0]+15*(1+z)])
			dm15err_ = np.sqrt(np.sum(np.diag(dm15_flux_all_[1])/dm15_flux_all_[0]**2))*1.0857
			print(t0, t0_err, B_Tmax[0], B_Tmax[1])
			print(dm15_fit[0], dm15_fit[1], dm15_, dm15err_)
			exit()
			if othersave is not None:
				othersave = self.name + '_results/' + othersave + '.txt'
				with open(othersave, 'a') as f:
					f.writelines('%s %s %s %s %s %s'%(othername, dm15_, dm15_err, t0, t0_err, lc_source))

		else:
			self.salt_para = {}
			for i in range(1, 5):
				self.salt_para[result.param_names[i].upper()] = [result.parameters[i], result.errors[result.param_names[i]]]
			#self.salt_para['B_Tmax'] = [self.salt_para['T0'][0], self.salt_para['T0'][1]]
			#self.salt_para['B_Mmax'] = 
			sampling = 0.1
			self.salt_para['B_Tmax'] = [fitted_model.source.peakphase('STANDARD::B',sampling=sampling)+self.salt_para['T0'][0],self.salt_para['T0'][1]]
			flux_all_ = fitted_model.bandfluxcov(band='STANDARD::B',time=self.salt_para['B_Tmax'][0])
			self.salt_para['B_Mmax'] = [fitted_model.source.bandmag('STANDARD::B','vega',self.salt_para['B_Tmax'][0]),np.sqrt(flux_all_[1])/flux_all_[0]*1.0857]
			x1_, dx1_ = self.salt_para['X1'][0], self.salt_para['X1'][1]
			#self.salt_para['dm15'] = [1.09 - 0.161*x1_ + 0.013*x1_*x1_ - 0.00130*x1_*x1_*x1_,
			#				abs(-0.161 + 0.013*2*x1_ - 0.0013*3*x1_*x1_)*dx1_]
			dm15_ = fitted_model.source.bandmag('STANDARD::B','vega',self.salt_para['B_Tmax'][0]+15) - fitted_model.source.bandmag('STANDARD::B','vega',self.salt_para['B_Tmax'][0])
			dm15_flux_all_ = fitted_model.bandfluxcov(band='STANDARD::B',time=[self.salt_para['B_Tmax'][0], self.salt_para['B_Tmax'][0]+15*(1+z)])
			dm15err_ = np.sqrt(np.sum(np.diag(dm15_flux_all_[1])/dm15_flux_all_[0]**2))*1.0857
			self.salt_para['dm15'] = [dm15_, dm15err_]
			print(self.salt_para['B_Tmax'], self.salt_para['T0'][0])
			print(self.salt_para['dm15'])
			print([1.09 - 0.161*x1_ + 0.013*x1_*x1_ - 0.00130*x1_*x1_*x1_,abs(-0.161 + 0.013*2*x1_ - 0.0013*3*x1_*x1_)*dx1_])
			#from astropy import cosmology
			#cosmo=cosmology.WMAP9
			#print(fitted_model._parameters[0], cosmo.distmod(fitted_model._parameters[0]).value)

			# save synthetic lc
			self.salt = {}
			MJD_grid = np.linspace(model_phase_range[0], model_phase_range[1], int(model_phase_range[1] - model_phase_range[0])*2+1)
			all_bands = pd.Series(data['band']).drop_duplicates().tolist()

			for band in all_bands:
				band_ = band.split('::')[1]
				if band_.upper() == band_:
					magsys_ = 'vega'
				else:
					magsys_ = 'ab'
				m_ = fitted_model.bandmag(band=band, magsys=magsys_, time=MJD_grid)
				flux_all_ = fitted_model.bandfluxcov(band=band,time=MJD_grid)
				dm_ = np.sqrt(np.diag(flux_all_[1]))/flux_all_[0]*1.0857
				mark_ = dm_ < 99
				self.salt['UBVRI ' + band_] = pd.DataFrame({'MJD':MJD_grid,
															'm':m_,
															'dm':dm_,
															'mark':mark_})[mark_].reset_index(drop=True)


	def get_salt_model(self, salt_file, salt_para_file):
		# Get data
		self.salt = {}
		names = []
		with open(salt_file,'r') as f:
			line = f.readline()
			skiprows = 0
			while(line[0] == '@' or line[0] == '#'):
				skiprows += 1
				if line[0] == '#':
					names.append(line.split()[0][1:].replace(':',''))
				line = f.readline()

		names.pop(-1)
		
		data = pd.read_csv(salt_file, skiprows=skiprows, sep=' ',names=names)
		# Flux to mag
		data['m'] = -2.5*np.log10(data['Flux']) + data['ZP']
		data['dm'] = 1.0857 * data['Fluxerr'] / data['Flux']
		data['mark'] = data['dm'] < 1

		data.rename(columns={'Date':'MJD', 'Filter':'filter'}, inplace=True)
		for i in range(len(data)):
			data.loc[i,'filter'] = data.loc[i,'filter'].split('::')[-1]
		salt_bands = data['filter'].drop_duplicates().tolist()
		for item in salt_bands:
			self.salt['UBVRI '+item] = data[['MJD','m','dm','mark']][data['filter']==item].reset_index(drop=True)
		self.salt_para = {}
		with open(salt_para_file, 'r') as f:
			line = f.readline()
			while(line):
				line_split = line.split()
				paranames = ['X0','X1','Color','RestFrameMag_0_B','DayMax']
				if line_split == []:
					pass
				elif line_split[0] in paranames:
					self.salt_para[line_split[0]] = [float(line_split[1]), float(line_split[2])]
				line = f.readline()
		x1_, dx1_ = self.salt_para['X1'][0], self.salt_para['X1'][1]
		self.salt_para['dm15'] = [1.09 - 0.161*x1_ + 0.013*x1_*x1_ - 0.00130*x1_*x1_*x1_,
						abs(-0.161 + 0.013*2*x1_ - 0.0013*3*x1_*x1_)*dx1_]
		self.salt_para['B_Mmax'] = self.salt_para['RestFrameMag_0_B']
		self.salt_para['B_Tmax'] = self.salt_para['DayMax']

	def get_snpy_model(self, sn_object, bandName=None, phase_range=None):
		self.snpy = {}
		if phase_range is None:
			phase_range = [-15 + sn_object.Tmax, 60 + sn_object.Tmax]
		MJD = np.linspace(phase_range[0], phase_range[1], int(phase_range[1] - phase_range[0])*2 + 1)
		if bandName is None:
			bandName = sn_object.data.keys()
		else:
			bandName = bandName
		for item in bandName:
			model_data = sn_object.model(item, MJD)
			if item in ['Bs', 'Vs', 'Is', 'Rs', 'Us']:
				item_ = item[:-1]
			elif item in ['u_s', 'g_s', 'r_s', 'i_s', 'z_s']:
				item_ = item[0]
			else:
				item_ = item
			self.snpy['UBVRI ' + item_] = pd.DataFrame({'MJD':MJD, 'm':model_data[0], 'dm':model_data[1], 'mark':model_data[2]})
		self.snpy_para = {}
		for item in sn_object.parameters.keys():
			self.snpy_para[item] = [sn_object.parameters[item], sn_object.errors[item] + sn_object.systematics()[item]]

	def get_fit_power_law(self, band, fit_phase_range, t0_range, mcmc, nondetection=None, peak_percent=0.4, DM=30, DM_err=0.15, host_ebv=0., host_rv=3.1, texp=None):
		data_mark = self.allData[band]['mark']
		phase_mark = self.allData[band]['MJD'] < fit_phase_range[1]
		mark = data_mark * phase_mark
		t_mjd = self.allData[band]['MJD'][mark].to_numpy()
		m = self.allData[band]['m'][mark].to_numpy()
		dm = self.allData[band]['dm'][mark].to_numpy()
		band_max = self.get_band_max(band)
		#print(band_max)
		if type(band_max) == type([1,2]):
			band_max = band_max[0]
		mark = m > (band_max + 2.5*np.log10(1/peak_percent))
		t_mjd = t_mjd[mark]
		m = m[mark]
		dm = dm[mark]
		mark = t_mjd > fit_phase_range[0]
		milky_ebv = self.EBVmilky
		fit_power_law(t_mjd, m, dm, t0_range, nondetection=nondetection, band=band, mcmc=mcmc, mark=mark, DM=DM, DM_err=DM_err, milky_ebv=milky_ebv, host_ebv=host_ebv, host_rv=host_rv, texp=texp)

	def get_UBVRI(self, stack=False, bands=None, remove_bands=None):
		all_band = list(self.allData.keys())
		for band_ in all_band:
			if 'UBVRI' in band_:
				self.allData.pop(band_)
		data_UBVRI = {}
		if stack:
			stack_str = 'stacked '
		else:
			stack_str = ''
		if bands is None:
			bands = self.dataBand.copy()
		if remove_bands is not None:
			remove_pos = []
			for band_i, band_ in enumerate(bands):
				if band_ in remove_bands:
					remove_pos.append(band_i)
			if remove_bands != []:
				remove_len = len(remove_bands)
				for remove_i in range(remove_len-1, -1, -1):
					bands.pop(remove_pos[remove_i])
		for item_ in bands:
			band_ = item_.split('_')[-1]
			if band_ in self.UBVRI_bands:
				item = stack_str + item_
				if band_ not in data_UBVRI.keys():
					data_UBVRI[band_] = self.allData[item].copy()
				else:
					data_UBVRI[band_] = pd.concat([data_UBVRI[band_], self.allData[item].copy()])
		for k, v in data_UBVRI.items():
			v = v.sort_values(by='MJD').reset_index(drop=True)
			self.allData['UBVRI '+k] = v
	'''
	def fit_template(self, band, othername, otherband, guess_max_mjd, guess_mag_shift, magscale=50, plot = False):
		other = othername + ' ' + otherband
		if other not in self.interpolate.keys():
			self.interpolate_lc(other, method='linear')
		mark_phase = (self.allData[other][''])
		def compare_(max_mjd, mag_shift):
			return np.sum(((self.allData[band]['m'] - self.interpolate[other](self.allData[band]['MJD'] - max_mjd)[0] - mag_shift)*magscale)**2)
		guess = [guess_max_mjd, guess_mag_shift]
		res = optimize.minimize(compare_, guess).x
		max_mjd = res[0]
		mag_shift = res[1]
		if plot == True:
			fig, ax = plt.subplots()
			ax.plot(self.allData[band]['MJD'] - max_mjd, self.allData[band]['m'] - mag_shift, c = 'r', lable = self.name)
			ax.plot(self.allData[other]['MJD'], self.allData[other]['m'], c = 'b', lable = othername)
			ax.legend()
			plt.show()
		return max_mjd, mag_shift
	'''

	def mask_epoch(self, min_epoch, max_epoch, bandName=None):
		if bandName is None:
			bandName = self.allData.keys()
		elif type(bandName) == 'haha':
			bandName = [bandName]

		for band_i, band in enumerate(bandName):
			data_band_ = self.allData[band]
			mark_ = (data_band_['MJD'] > min_epoch) * (data_band_['MJD'] < max_epoch)
			data_band_['mark'][~mark_] = False


	def RemoveOutlierBySmooth(self, method='savgol', bandName=None, window=3, poly=1, remove_bad=True):
		if bandName:
			if type(bandName) == type('str'):
				bandName = [bandName]
		else:
			bandName = self.dataBand.copy()

		for item in bandName:
			if remove_bad == True:
				mark_ = self.allData[item]['mark']
			else:
				mark_ = [True for i in range(len(self.allData[item]))]
			data_band_ = self.allData[item][mark_]
			if not data_band_.empty:
				#print(item)
				#smoothedData = non_uniform_savgol(data_band_['MJD'].to_numpy(), \
				#	data_band_['m'].to_numpy(), window, poly)
				if method == 'savgol':
					smoothedData = signal.savgol_filter(data_band_['m'].to_numpy(), window, poly)
				elif method == 'gp':
					flux_ = np.power(10, 0.4*(23.9 - data_band_['m']))
					eflux_ = flux_*data_band_['dm']/1.0857
					col_ = item.split('_')[-1].split('.')[-1]
					gp = GP.fit_gp(data_band_['MJD'].to_numpy(),
						           flux_,
						           eflux_,
						           [col_ for i in range(len(data_band_))])
					gp.train(gp_mean='mean', opt_routine = 'minimize')
					gp_jd, gp_flux, gp_flux_errors, gp_ws = gp.predict(x_pred=data_band_['MJD'].to_numpy(), returnv=True)
					smoothedData = -2.5*np.log10(gp_flux[0])+23.9 
				else:
					Exception('not known method')
				deviation = abs(data_band_['m'] - smoothedData)
				maskOutlier = (((deviation > 3 * data_band_['dm']) * (deviation > 0.2))==True)
				self.allData[item].loc[maskOutlier.index, 'mark'] = False

	def RemoveOutlierByIndex(self, bandName, index):
		if type(index) == type(10):
			index = [index]
		mask = np.array([True for i in range(len(self.allData[bandName]))])
		for i in range(len(index)):
			mask *= (self.allData[bandName].index != index[i])
		print(self.allData[bandName]['MJD'][~mask])
		self.allData[bandName][~mask] = False 

	def RemoveByHand(self, bandName=None, UBVRI=None, stack=False, figsize=(12,8)):
		if bandName:
			if type(bandName) == type('str'):
				bandName = [bandName]
		elif UBVRI is not None:
			bandName = []
			for item in self.allData.keys():
				if item.split()[0] == 'UBVRI':
					bandName.append(item)
		else:
			bandName = self.dataBand.copy()

		if stack == True and UBVRI is None:
			for i in range(len(bandName)):
				bandName[i] = 'stacked ' + bandName[i]

		for item in bandName:
			if not self.allData[item].empty:
				fig, ax = plt.subplots(figsize=figsize)
				ax.set_title('Photometry')
				ax.set_xlabel('MJD')
				ax.set_ylabel('Apparent Magnitude')
				ax.set_ylim(float(self.allData[item]['m'].max())+1, float(self.allData[item]['m'].min())-1)
				if '_' in item:
					label = item.split('_')[-2].split()[-1].upper() + ' ' + item.split('_')[-1] + '-band'
				else:
					label = item.split()[-1]
				mark_ = self.allData[item]['mark']
				line1 = ax.errorbar(self.allData[item][mark_]['MJD'],
					self.allData[item][mark_]['m'],
					yerr=self.allData[item][mark_]['dm'],
					capsize = 7, 
					label=label,
					marker='o', linestyle='', picker=True, pickradius=5, ms=5).lines[0]
				plt.legend()

				line2 = ax.errorbar(self.allData[item][~mark_]['MJD'], 
								     self.allData[item][~mark_]['m'],
								     yerr=self.allData[item][~mark_]['dm'],
								     capsize = 7, 
									 marker='x', linestyle='', picker=True, pickradius=5, ms=5).lines[0]

				delete_mode = [1]

				def delete_parttern(event):
					if event.key == 'd':
						delete_mode[0] = 1
						ax.set_title('delete_pattern')
					elif event.key == 'r':
						delete_mode[0] = 0
						ax.set_title('recover_pattern')
					fig.canvas.draw_idle()


				def onpick(event):
					thisline = event.artist
					if type(thisline) == type(line1) and thisline.get_marker() in ['o', 'x']:
						ind = event.ind
						xdata = thisline.get_xdata()
						ydata = thisline.get_ydata()
						mark_ = np.array([True for i in range(len(xdata))])
						mark_[ind] = False
						pick_x = xdata[~mark_]
						pick_y = ydata[~mark_]
						marker = thisline.get_marker()
						if marker == 'o' and delete_mode[0] == 1:
							line1.set_xdata(xdata[mark_])
							line1.set_ydata(ydata[mark_])
							xdata_ = line2.get_xdata()
							ydata_ = line2.get_ydata()
							line2.set_xdata(np.concatenate((pick_x, xdata_)))
							line2.set_ydata(np.concatenate((pick_y, ydata_)))
						elif marker == 'x' and delete_mode[0] == 0:
							line2.set_xdata(xdata[mark_])
							line2.set_ydata(ydata[mark_])
							xdata_ = line1.get_xdata()
							ydata_ = line1.get_ydata()
							line1.set_xdata(np.concatenate((pick_x, xdata_)))
							line1.set_ydata(np.concatenate((pick_y, ydata_)))
						fig.canvas.draw_idle()

				def close_fig(event):
					xdata_mask = line2.get_xdata()
					xdata = self.allData[item]['MJD']				
					for i in range(len(xdata)):
						if xdata[i] in xdata_mask:
							self.allData[item].loc[i, 'mark'] = False
						else:
							self.allData[item].loc[i, 'mark'] = True

				fig.canvas.mpl_connect('pick_event', onpick)
				fig.canvas.mpl_connect('key_press_event', delete_parttern)
				fig.canvas.mpl_connect('close_event', close_fig)

				plt.show()


	def Stack(self, snr=3):
		# Generate stacked data
		# debug = 0
		all_band = list(self.allData.keys())
		for band_ in all_band:
			if 'stacked' in band_:
				self.allData.pop(band_)
		allData_ = self.allData.copy()
		for k, v in allData_.items():
			if 'UBVRI' not in k:
				mark = self.allData[k]['mark']
				v_ = v[mark].reset_index(drop=True)
				if not v_.empty:
					length = len(v_)
					v_['stack'] = np.array([False for i in range(length)])
					v_.loc[length] = [0, 0, 0, True, False]
					head = 0
					v_.loc[head, 'stack'] = True
					for i in range(1, length+1):
						#if int(v_.loc[i, 'MJD']) != int(v_.loc[head, 'MJD']):
						if abs(v_.loc[i, 'MJD'] - v_.loc[head, 'MJD']) > 0.5:
							
							# if debug < 5:
							# 	 debug += 1
							v_.loc[head, 'MJD'] = v_.loc[head:(i-1), 'MJD'].sum()/(i-head)
							v_.loc[head, 'm'] = v_.loc[head:(i-1), 'm'].sum()/(i-head)
							#v_.loc[head, 'dm'] = v_.loc[head:(i-1), 'dm'].sum()/(i-head)/np.sqrt(i-head)
							v_.loc[head, 'dm'] = np.sqrt((v_.loc[head:(i-1), 'dm']**2).sum())/(i-head)
							if 1.0857/v_.loc[head, 'dm'] < snr:
								v_.loc[head, 'mark'] = False
							head = i
							if head != length:
								v_.loc[head, 'stack'] = True
					v_ = v_[(v_['stack']==True)].reset_index(drop=True).drop(labels='stack', axis=1)
				self.allData['stacked ' + k] = v_	
	
	def plot_bolo(self, Dir, bol_file=None, std_err=0, DM_err=0.15, save=None, figsize=(10,8), t_exp=None, dt_exp=None, scale=1e43):
		if save:
			save = self.name + '_results/' + save
		if Dir[-1] != '/':
			Dir = Dir + '/'
		if bol_file is None:
			bol_file = 'Minim_data.dat'
		fig, ax = plt.subplots(figsize=figsize)
		ax.set_xlabel('Days from $B$-band Maximum', fontsize=15)
		ax.set_ylabel('Bolometric Flux [$\\times 10^{43}$erg s$^{-1}$]', fontsize=15)
		orgdata = np.loadtxt(Dir+bol_file, usecols=[0,1,2])
		fitdata = np.loadtxt(Dir+'Minim_bestfit.dat')

		if save is not None:
			save_latex = save.split('.')[0] + '.txt'
			with open(save_latex, 'w') as f:
				writelines = []
				if len(orgdata)%3 == 0:
					n_rows = int(len(orgdata)/3)
				else:
					n_rows = int(len(orgdata)/3) + 1
				i = 0
				while(i<n_rows-2):
					line_ = '%.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n'%(orgdata[i,0], orgdata[i,1], orgdata[i,2], orgdata[i+n_rows,0], orgdata[i+n_rows,1], orgdata[i+n_rows,2], orgdata[i+2*n_rows,0], orgdata[i+2*n_rows,1], orgdata[i+2*n_rows,2])
					writelines.append(line_)
					i += 1
				if len(orgdata)%3 == 2:
					line_ = '%.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n'%(orgdata[i,0], orgdata[i,1], orgdata[i,2], orgdata[i+n_rows,0], orgdata[i+n_rows,1], orgdata[i+n_rows,2], orgdata[i+2*n_rows,0], orgdata[i+2*n_rows,1], orgdata[i+2*n_rows,2])
					writelines.append(line_)
					i += 1
					line_ = '%.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n'%(orgdata[i,0], orgdata[i,1], orgdata[i,2], orgdata[i+n_rows,0], orgdata[i+n_rows,1], orgdata[i+n_rows,2])
					writelines.append(line_)
				else:
					line_ = '%.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n'%(orgdata[i,0], orgdata[i,1], orgdata[i,2], orgdata[i+n_rows,0], orgdata[i+n_rows,1], orgdata[i+n_rows,2])
					writelines.append(line_)
					i += 1
					line_ = '%.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\\n'%(orgdata[i,0], orgdata[i,1], orgdata[i,2], orgdata[i+n_rows,0], orgdata[i+n_rows,1], orgdata[i+n_rows,2])
					writelines.append(line_)
				f.writelines(writelines)
		

		tmax, Lmax = find_curve_min(orgdata[:,0], -orgdata[:,1]/scale, orgdata[:,2]/scale, 0, Name=None, method='polynomial',n=5, bounds=[-14, 14], get_err=True, Nboot=1000)
		tmax, dtmax = tmax
		Lmax, dLmax = Lmax
		tmax = self.Tmax + tmax*(1+self.z)
		dtmax = dtmax * (1+self.z)
		dLmax = np.sqrt((Lmax*DM_err/1.0857)**2 + dLmax**2*std_err)
		if t_exp is not None and dt_exp is not None:
			t_rise = (tmax - t_exp)/(1+self.z)
			dt_rise = np.sqrt(dtmax**2 + dt_exp**2)/(1+self.z)
			print(Lmax, dLmax, t_rise, dt_rise, tmax, dtmax)
			m_Ni56, dm_Ni56 = get_m_Ni56(Lmax, dLmax, t_rise, dt_rise)
			print(m_Ni56, dm_Ni56)
			exit()
		print(tmax, dtmax, Lmax, dLmax)
		ax.errorbar(orgdata[:,0], orgdata[:,1]/scale, yerr=orgdata[:,2]/scale, linestyle='', capsize=7, c='r', alpha=alpha)
		ax.plot(orgdata[:,0], orgdata[:,1]/scale, linestyle='', marker='o', c='r', label='bolometric light curve')
		ax.plot(fitdata[:,0], fitdata[:,1]/scale, c='b', label='Arnett models')
		text_x = (fitdata[0,0] + fitdata[-1,0])/2
		text_y = np.max(fitdata[:,1])/scale
		para = []
		with open(Dir+'Bestfit_par_radio.dat', 'r') as f:
			line = f.readline()
			line = f.readline()
			line = f.readline()
			n = 2
			while(n<12):
				params = line.replace('\n', '').split(':')[1].split(', ')
				para.append(params)
				line = f.readline()
				n += 1
		for i in range(1,len(para)):
			para[i][0] = '%.2f'%(float(para[i][0]))
			para[i][1] = '%.2f'%(float(para[i][1]))
		text_sepe = 0.15
		ax.text(text_x, text_y, '$t_{0}$=%s$\\pm$%sd'%(para[1][0], para[1][1]), fontsize=15)
		ax.text(text_x, text_y - 1*text_sepe, '$M_{\\rm Ni}$=%s$\\pm$%s $M_{\\rm \\odot}$'%(para[2][0], para[2][1]), fontsize=15)
		ax.text(text_x, text_y - 2*text_sepe, '$t_{\\rm lc}$=%s$\\pm$%s day'%(para[3][0], para[3][1]), fontsize=15)
		ax.text(text_x, text_y - 3*text_sepe, '$t_{\\rm \\gamma}$=%s$\\pm$%s day'%(para[5][0], para[5][1]), fontsize=15)
		#ax.text(text_x, text_y - 2*text_sepe, '$\\kappa$=%s$\\pm$%s cm$^2$ g$^{-1}$'%(para[6][0], para[6][1]), fontsize=15)
		#ax.text(text_x, text_y - 3*text_sepe, '$M_{\\rm ej}$=%s$\\pm$%s $M_{\\rm \\odot}$'%(para[7][0], para[7][1]), fontsize=15)
		#ax.text(text_x, text_y - 4*text_sepe, '$v_{\\rm exp}$=%s$\\pm$%s km s$^{-1}$'%(para[8][0], para[8][1]), fontsize=15)
		if save is not None:
			fig.savefig('%s.pdf'%save, bbox_inches='tight')
			with open('%s_snpy.txt'%save, 'w') as f:
				f.writelines('%s %s %s %s\n'%(self.name, self.z, self.ra, self.dec))
				f.writelines('filter B')
				data_len = len(orgdata)
				orgdata[:,0] = orgdata[:,0] * (1+self.z) + self.Tmax
				orgdata[:,1] = -np.log10(orgdata[:,1]/1e50)
				for i in range(data_len):
					f.writelines('%.3f %.3f %.3f\n'%(orgdata[i,0], orgdata[i,1], 0.1))
		plt.show()

	def interpolate_lc(self, bandName, method='gp', interpolate_region=None, n=5, plot=False):
		mark_ = self.allData[bandName]['mark']
		if interpolate_region is not None:
			mark_ = mark_ * (self.allData[bandName]['MJD'] - self.Tmax > interpolate_region[0]) * (self.allData[bandName]['MJD'] - self.Tmax < interpolate_region[1])
		banddata = self.allData[bandName][mark_].reset_index(drop=True)
		if method == 'gp':
			color = bandName.split()[-1].split('_')[-1]
			UVOT_tran_dict = {'UVW1':'A','UVM2':'D','UVW2':'S','UVOT.U':'U'}
			if color in UVOT_tran_dict.keys():
				color = UVOT_tran_dict[color]
			flux_ = self.zp[color]*1e-11*np.power(10, -0.4*banddata['m'])
			if self.interpolate_err == True:
				eflux_ = flux_*banddata['dm']/1.0857
			else:
				eflux_ = 0.05*flux_
			gp = GP.fit_gp(banddata['MJD'].to_numpy(),
				           flux_,
				           eflux_,
				           [color for i in range(len(flux_))])
			gp.train(gp_mean='mean', opt_routine = 'minimize')
			def gp_function(x_pred):
				gp_jd, gp_flux, gp_flux_errors, gp_ws = gp.predict(x_pred=x_pred, returnv=True)
				m_interp_ = -2.5*np.log10(gp_flux[0]/(self.zp[color]*1e-11))
				dm_interp_ = gp_flux_errors[0]/gp_flux[0]*1.0857
				return m_interp_, dm_interp_
			self.interpolate[bandName] = gp_function
		elif method in ['linear', 'cubic']:
			def interp1d_(x_pred):
				f_interp = interp1d(banddata['MJD'], banddata['m'], kind=method, bounds_error=False, fill_value=999)
				m_interp_ = f_interp(x_pred)
				try:
					dm_interp_ = np.ones(len(x_pred))*np.mean(banddata['dm'])
					return m_interp_, dm_interp_
				except:
					dm_interp_ = np.mean(banddata['dm'])
					return [m_interp_], [dm_interp_]
			self.interpolate[bandName] = interp1d_
		elif method == 'line':
			params, params_cov = optimize.curve_fit(linear_f, banddata['MJD'], banddata['m'], sigma=banddata['dm'])
			def interp_line(x_pred):
				#if type(x_pred) == type([1.]):
				x_pred = np.array(x_pred)
				m_interp_ = linear_f(x_pred, params[0], params[1])
				try:
					dm_interp_ = np.ones(len(x_pred))*np.mean(banddata['dm'])
					return m_interp_, dm_interp_
				except:
					dm_interp_ = np.mean(banddata['dm'])
					return [m_interp_], [dm_interp_]
			self.interpolate[bandName] = interp_line
		else:
			if method == 'polynomial':
				spline_f = fit1dcurve.Interpolator(type='polynomial',x=banddata['MJD'],y=banddata['m'],dy=banddata['dm'],n=n)
			else:
				spline_f = fit1dcurve.Interpolator(type=method,x=banddata['MJD'],y=banddata['m'],dy=banddata['dm'])
			def interp_line(x_pred):
				x_pred = np.array(x_pred)
				m_interp_ = spline_f(x_pred)[0]
				try:
					dm_interp_ = np.ones(len(x_pred))*np.mean(banddata['dm'])
					return m_interp_, dm_interp_
				except:
					dm_interp_ = np.mean(banddata['dm'])
					return [m_interp_], [dm_interp_]
			self.interpolate[bandName] = interp_line
		if plot == True:
			plot_MJD = np.linspace(banddata['MJD'].min(), banddata['MJD'].max(), len(banddata['MJD'])*2)
			plt.plot(banddata['MJD'], banddata['m'], linestyle='', marker='o')
			plt.plot(plot_MJD, self.interpolate[bandName](plot_MJD)[0])
			plt.show()
			go = input('go?(y/n): ')
			if go != 'y':
				exit()
		#elif type(method) == type(1):

		self.interpolate_method[bandName] = method

	def fit_template(self, othername, band, other_band=None, MJD_shift=None, Mag_shift=None, interpolate_method='linear', phase_range=None, plot=True):
		if phase_range is not None:
			phase_mark = (self.allData[band]['MJD'] > phase_range[0]) * (self.allData[band]['MJD'] < phase_range[1])
		else:
			len_ = len(self.allData[band]['MJD'])
			phase_mark = np.array([True for i in range(len_)])
		fit_data = self.allData[band][phase_mark].reset_index(drop=True)
		if other_band is None:
			other_band = band
		other_key = othername + ' ' + other_band
		if other_key not in self.interpolate.keys():
			self.interpolate_lc(other_key, method=interpolate_method)
		if MJD_shift is None:
			MJD_shift = np.mean(fit_data['MJD']) - np.mean(self.allData[other_key]['MJD'])
		if Mag_shift is None:
			Mag_shift = np.mean(fit_data['m']) - np.mean(self.allData[other_key]['m'])
		if plot == True:
			fig, ax = plt.subplots()
			ax.plot(self.allData[band]['MJD'] - MJD_shift, self.allData[band]['m'] - Mag_shift, c = 'r', label = self.name, linestyle='',marker='o')
			ax.plot(self.allData[other_key]['MJD'], self.allData[other_key]['m'], c = 'b', label = othername, linestyle='',marker='o')
			ax.invert_yaxis()
			ax.legend()
			plt.show()
		def fit_function(params):
			difference = fit_data['m'] - params[1] - self.interpolate[other_key]((fit_data['MJD'] - params[0]))[0]
			difference = np.sum(difference**2*1e4)
			print(difference)
			return difference

		guess = [MJD_shift, Mag_shift]
		print(guess)
		res = optimize.minimize(fit_function, guess).x
		MJD_shift = res[0]
		Mag_shift = res[1]
		if plot == True:
			fig, ax = plt.subplots()
			ax.plot(self.allData[band]['MJD'] - MJD_shift, self.allData[band]['m'] - Mag_shift, c = 'black', label = self.name+'_all', linestyle='',marker='o')
			ax.plot(fit_data['MJD'] - MJD_shift, fit_data['m'] - Mag_shift, c = 'r', label = self.name+'_fit', linestyle='',marker='o')
			ax.plot(self.allData[other_key]['MJD'], self.allData[other_key]['m'], c = 'b', label = othername, linestyle='',marker='o')
			ax.plot(self.allData[other_key]['MJD'], self.interpolate[other_key](self.allData[other_key]['MJD'])[0], c = 'b', linestyle='--')
			ax.invert_yaxis()
			ax.legend()
			plt.show()
		return MJD_shift, Mag_shift

	def Plot_color(self, Color, EBV, phase=False, save=False, phase_range=None, Othername=None, xlabel='Days from $B$-band Maximum', interpolate=1, remove_bad = True,day_sep_max=0.01,
		interpolate_method='gp', figsize=(16,6), legend_ncol=1, out_legend=False,otherstyle='point', mark_color_value=0.2, no_other_err=False, inset_phase=None, inset_box=None, texp=None,
		face_fill=None):
		#Color = [['B','V'], ['g','r']]
		if face_fill is None:
			if Othername is None:
				face_fill = ['none']
			else:
				face_fill = ['none' for i in range(len(Othername)+1)]
		if inset_phase is not None:
			inset_plot_flag = True
		else:
			inset_plot_flag = False
		if not Othername:
			othername = ['UBVRI']
		elif type(Othername) == str:
			othername = ['UBVRI', Othername]
		else:
			othername = Othername.copy()
			othername.insert(0, 'UBVRI')
		othercolor = ['black','r','b','darkgreen','purple','orange','pink','brown','gold','green','salmon','mediumorchid','lime','darkgoldenrod']
		othershape = ['o','^','*','p','<','X','D','H','v','>','s','P','h','d','1','2','3','4','8']
		otherline = [
	 ('dashed', 'dashed'),    # Same as '--'
	 ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dashdot', 'dashdot'),  # Same as '-.'
     ('long dash with offset', (5, (10, 3))),
     ('long long dash with offset', (5, (20, 3))),
     ('densely dashed',        (0, (5, 1))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('dotted', 'dotted'),]    # Same as (0, (1, 1)) or ':'
		if Othername is not None:
			if texp is not None:
				maxMJD =  self.other_texp.copy()
			else:
				maxMJD =  self.other_MJD_max.copy()
		else:
			maxMJD = {}
		if texp is not None:
			maxMJD['UBVRI'] = texp
		else:
			maxMJD['UBVRI'] = self.Tmax*1
		len_color = len(Color)
		nrows = int((len_color+1)/2)
		if len_color == 1:
			ncols = 1
		else:
			ncols = 2
		fig, axs = plt.subplots(nrows, ncols, sharex='col', figsize=figsize)
		if nrows == 1:
			if ncols == 1:
				axs = np.array([[axs]])
			else:
				axs = np.array([axs])
		plt.tick_params(labelsize=15)
		#label_list = np.zeros(len(othername))
		plot_lines = []
		plot_flag = {}
		for name_ in othername:
			plot_flag[name_] = False
		for color_i, color in enumerate(Color):
			ymin = 999
			ymax = -999
			xmin = 999
			xmax = -999
			row_ = int(color_i/2)
			col_ = color_i%2
			axs[row_, col_].set_xlabel(xlabel,fontsize=20)
			axs[row_, col_].set_ylabel('%s - %s [mag]'%(color[0],color[1]),fontsize=20)
			if inset_plot_flag:
				axins_ = axs[row_, col_].inset_axes(inset_box)
				axins_.tick_params(labeltop=True, labelbottom=False)
			zorder=999
			for name_i, name_ in enumerate(othername):
				print(name_)
				zorder=zorder-2
				band_name1_ = name_ + ' ' + color[0]
				band_name2_ = name_ + ' ' + color[1]
				if band_name1_ in self.allData.keys() and band_name2_ in self.allData.keys():
					if remove_bad == True:
						mark_1 = self.allData[band_name1_]['mark']
						mark_2 = self.allData[band_name2_]['mark']
					else:
						mark_1 = [True for i in range(len(self.allData[band_name1_]))]
						mark_2 = [True for i in range(len(self.allData[band_name1_]))]
					data_name1_ = self.allData[band_name1_][mark_1]
					data_name2_ = self.allData[band_name2_][mark_2]
					'''
					if not label_list[name_i]:
						if name_i == 0:
							label_ = self.name
						else:
							label_ = name_
						label_list[name_i] = 1
					else:
						label_ = None
					'''
					if name_i ==0:
						label_ = self.name
					else:
						label_ = name_
					if phase_range is None:
						phase_min = np.max([data_name1_['MJD'][0],self.allData[band_name2_]['MJD'][0]])
						phase_max = np.min([data_name1_['MJD'][len(data_name1_)-1],
							data_name2_['MJD'][len(data_name2_)-1]])
						phase_range_ = [phase_min-0.1, phase_max+0.1]
					else:
						phase_range_ = phase_range
					phase_mark1 = (data_name1_['MJD'] - maxMJD[name_] > phase_range_[0]  - self.Tmax)  \
						* (data_name1_['MJD'] - maxMJD[name_] < phase_range_[1]  - self.Tmax)
					phase_mark2 = (data_name2_['MJD'] - maxMJD[name_] > phase_range_[0]  - self.Tmax)  \
						* (data_name2_['MJD'] - maxMJD[name_] < phase_range_[1]  - self.Tmax)
					'''
					else:
						phase_mark1 = data_name1_['mark']
						phase_mark2 = data_name2_['mark']
					'''
					data_color1 = data_name1_[phase_mark1].reset_index(drop=True)
					data_color2 = data_name2_[phase_mark2].reset_index(drop=True)
					if interpolate == 0:
						x_interp = []
						pos2_ = []
						good = [True for i in range(len(data_color1))]
						for i1 in range(len(data_color1)):
							pos2_.append(np.argsort(np.abs(data_color2['MJD'] - data_color1.loc[i1, 'MJD']))[0])
							if abs(data_color2.loc[pos2_[-1], 'MJD'] - data_color1.loc[i1, 'MJD']) > day_sep_max:
								good[i1] = False
						x_interp = data_color1['MJD'].to_numpy()[good]
						data_color = (data_color1['m'].to_numpy() - data_color2['m'].to_numpy()[pos2_])[good]
						'''
						if name_ == 'SN1999aa':
							print(data_color1['m'].to_numpy()[good])
							print(data_color2['m'].to_numpy()[pos2_][good])
							print(data_color)
							exit()
						'''
						#data_color_err = np.sqrt((data_color1['dm']*data_color1['dm']).to_numpy() 
						#	+ (data_color2['dm']*data_color2['dm']).to_numpy()[pos2_])[good]
						data_color_err = ((data_color1['dm']*data_color1['dm']).to_numpy() 
							+ (data_color2['dm']*data_color2['dm']).to_numpy()[pos2_])[good]
						data_color_err = np.sqrt(data_color_err.astype('float'))
						mark_color = data_color_err < mark_color_value
						if name_ == 'iPTF16abc':
							mark_color[0] = True
					elif interpolate == 1:
						x_interp = data_color1['MJD'].to_numpy()
						if band_name2_ not in self.interpolate.keys():
							self.interpolate_lc(band_name2_, method=interpolate_method)
						inter_color2, inter_color2_err = self.interpolate[band_name2_](x_interp)
						inter_color2 = inter_color2
						inter_color2_err = inter_color2_err
						data_color = data_color1['m'] - inter_color2
						data_color_err = np.sqrt(data_color1['dm']*data_color1['dm'] + inter_color2_err*inter_color2_err)
						#mark_color = data_color_err < 0.25
						mark_color = data_color_err < mark_color_value
						
					elif interpolate == 2:
						x_interp = pd.concat([data_color1['MJD'],data_color2['MJD']]).sort_values().reset_index(drop=True).to_numpy()

						if band_name1_ not in self.interpolate.keys():
							self.interpolate_lc(band_name1_, method=interpolate_method)
						inter_color1, inter_color1_err = self.interpolate[band_name1_](x_interp)
						inter_color1 = inter_color1[0]
						inter_color1_err = inter_color1_err[0]
					

						if band_name2_ not in self.interpolate.keys():
							self.interpolate_lc(band_name2_, method=interpolate_method)
						inter_color2, inter_color2_err = self.interpolate[band_name2_](x_interp)
						inter_color2 = inter_color2[0]
						inter_color2_err = inter_color2_err[0]
						'''
						flux_1 = np.power(10, 0.4*(23.9 - data_color1['m']))
						eflux_1 = flux_1*data_color1['dm']/1.0857
						
						gp_1 = GP.fit_gp(data_color1['MJD'].to_numpy(),
							           flux_1,
							           eflux_1,
							           [color[0] for i in range(len(data_color1))])
						gp_1.train(gp_mean='mean', opt_routine = 'minimize')
						gp_jd_1, gp_flux_1, gp_flux_errors_1, gp_ws_1 = gp_1.predict(x_pred=x_interp, returnv=True)
						inter_color1 = -2.5*np.log10(gp_flux_1[0])+23.9
						inter_color1_err = gp_flux_errors_1[0]/gp_flux_1[0]*1.0857

						flux_2 = np.power(10, 0.4*(23.9 - data_color2['m']))
						eflux_2 = flux_2*data_color2['dm']/1.0857

						gp_2 = GP.fit_gp(data_color2['MJD'].to_numpy(),
							           flux_2,
							           eflux_2,
							           [color[1] for i in range(len(data_color2))])
						gp_2.train(gp_mean='mean', opt_routine = 'minimize')
						gp_jd_2, gp_flux_2, gp_flux_errors_2, gp_ws_2 = gp_2.predict(x_pred=x_interp, returnv=True)
						inter_color2 = -2.5*np.log10(gp_flux_2[0])+23.9
						inter_color2_err = gp_flux_errors_2[0]/gp_flux_2[0]*1.0857
						'''
						data_color = inter_color1 - inter_color2
						data_color_err = np.sqrt(inter_color1_err*inter_color1_err + inter_color2_err*inter_color2_err)
						#mark_color = data_color_err < 0.25
						mark_color = data_color_err < mark_color_value
					'''
					for mark_i, mark_color_ in enumerate(mark_color):
						if not mark_color_:
							for data2_i in range(len(data_color2)):
								if abs(data_color2.loc[data2_i, 'MJD'] - data_color1.loc[mark_i, 'MJD']) < 1:
									data_color[mark_i] = data_color1.loc[mark_i, 'm'] - data_color2.loc[data2_i, 'm']
									data_color_err[mark_i] = np.sqrt(data_color1.loc[mark_i, 'dm']**2 + data_color2.loc[data2_i, 'dm']**2)
									mark_color[mark_i] = True
									break
					'''
					if name_i == 0:
						phase = (x_interp - maxMJD[name_])/(1+self.z)
						if inset_plot_flag:
							inset_phase_mark = (phase > inset_phase[0])*(phase < inset_phase[1])*mark_color
						if otherstyle == 'point':
							zorder_ = 999
						else:
							zorder_ = 1
						axs[row_, col_].errorbar(phase[mark_color], data_color[mark_color] - EBV[name_i][color_i], yerr=data_color_err[mark_color],
						 capsize=7,
						 linestyle='',
						 c = othercolor[name_i],
						 alpha=alpha,
						 zorder=zorder_-1)
						line_, = axs[row_, col_].plot(phase[mark_color], data_color[mark_color] - EBV[name_i][color_i],
						 marker = othershape[name_i],
						 markerfacecolor=face_fill[name_i],
						 markeredgecolor=othercolor[name_i],
						 markersize=markersize,
						 linestyle='',
						 c = othercolor[name_i],
						 zorder=zorder_,
						 label = label_)
						if inset_plot_flag:
							axins_.errorbar(phase[inset_phase_mark], data_color[inset_phase_mark] - EBV[name_i][color_i], yerr=data_color_err[inset_phase_mark],
							 capsize=7,
							 linestyle='',
							 c = othercolor[name_i],
							 alpha=alpha,
							 zorder=zorder_-1)
							axins_.plot(phase[inset_phase_mark], data_color[inset_phase_mark] - EBV[name_i][color_i],
							 marker = othershape[name_i],
							 markerfacecolor=face_fill[name_i],
							 markeredgecolor=othercolor[name_i],
							 markersize=markersize,
							 linestyle='',
							 c = othercolor[name_i],
							 zorder=zorder_)
					else:
						phase = (x_interp - maxMJD[name_])/(1+self.other_z[name_])
						if otherstyle == 'point':
							if inset_plot_flag:
								inset_phase_mark = (phase > inset_phase[0])*(phase < inset_phase[1])*mark_color
							if no_other_err == False:
								axs[row_, col_].errorbar(phase[mark_color], data_color[mark_color] - EBV[name_i][color_i], yerr=data_color_err[mark_color],
								 capsize=7,
								 linestyle='',
								 c = othercolor[name_i],
								 zorder=zorder-1,
								 alpha=alpha,)
								if inset_plot_flag:
									axins_.errorbar(phase[inset_phase_mark], data_color[inset_phase_mark] - EBV[name_i][color_i], yerr=data_color_err[inset_phase_mark],
										 capsize=7,
										 linestyle='',
										 c = othercolor[name_i],
										 zorder=zorder-1,
										 alpha=alpha,)
							line_, = axs[row_, col_].plot(phase[mark_color], data_color[mark_color] - EBV[name_i][color_i],
								 marker = othershape[name_i],
								 markerfacecolor=face_fill[name_i],
								 markeredgecolor=othercolor[name_i],
								 markersize=markersize,
								 linestyle='',
								 zorder=zorder,
								 c = othercolor[name_i],
								 label = label_)
							if inset_plot_flag:
								axins_.plot(phase[inset_phase_mark], data_color[inset_phase_mark] - EBV[name_i][color_i],
									 marker = othershape[name_i],
									 markerfacecolor=face_fill[name_i],
									 markeredgecolor=othercolor[name_i],
									 markersize=markersize,
									 linestyle='',
									 zorder=zorder,
									 c = othercolor[name_i])
						else:
							if otherstyle == 'interpolate':
								other_interp = fit1dcurve.Interpolator(type='hyperspline',x=phase[mark_color],
									y=data_color[mark_color] - EBV[name_i][color_i],dy=data_color_err[mark_color])
								len_plot = len(phase[mark_color])
								other_xplot = np.linspace(phase[mark_color][0], phase[mark_color][len_plot-1], 100)
								other_yplot = other_interp(other_xplot)[0]
							else:
								other_xplot = phase[mark_color]
								other_yplot = data_color[mark_color] - EBV[name_i][color_i]
							line_, = axs[row_, col_].plot(other_xplot, other_yplot, linestyle=otherline[name_i][1], c=othercolor[name_i], label = label_, zorder=zorder)
							if inset_plot_flag:
								inset_phase_mark = (other_xplot > inset_phase[0])*(other_xplot < inset_phase[1])
								axins_.plot(other_xplot[inset_phase_mark], other_yplot[inset_phase_mark], linestyle=otherline[name_i][1], c=othercolor[name_i], zorder=zorder)
					if plot_flag[name_] == False:
						plot_lines.append(line_)
						plot_flag[name_] = True
					ymin = np.min([ymin, np.min(data_color[mark_color] - EBV[name_i][color_i])])
					ymax = np.max([ymax, np.max(data_color[mark_color] - EBV[name_i][color_i])])
					xmin = np.min([xmin, np.min(phase[mark_color])])
					xmax = np.max([xmax, np.max(phase[mark_color])])
			#axs[row_, col_].annotate(xy=(0.1,0.9), text='%s - %s'%(color[0],color[1]), ha='center', va='center', xycoords='axes fraction', fontsize=15)
			if Othername is not None:
				if out_legend == False:
					axs[row_, col_].legend(fontsize=15, ncol=legend_ncol)
			xmin = xmin - 0.1 * (xmax - xmin)
			xmax = xmax + 0.1 * (xmax - xmin)
			ymin = ymin - 0.1 * (ymax - ymin)
			ymax = ymax + 0.1 * (ymax - ymin)
			axs[row_, col_].set_xlim(xmin, xmax)
			axs[row_, col_].set_ylim(ymin, ymax)
			#Ni56_dat = np.loadtxt('/Users/liujialian/work/SN_catalog/P2016Nimixing.dat')
			#line_, = axs[row_, col_].plot(Ni56_dat[:,0]-18.07, Ni56_dat[:,1], linestyle='', marker='v', markerfacecolor='none', markeredgecolor='g', label='P2016_mixing_0.25')
			#plot_lines.append(line_)

		if out_legend == True:
			if len(Color) > 1:
				axs[0, 0].legend(handles=plot_lines, bbox_to_anchor=(0., 1.02, 2., .102), loc='lower left',
                      ncol=legend_ncol, mode="expand", borderaxespad=0., fontsize=15)
			else:
				axs[0, 0].legend(handles=plot_lines, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncol=legend_ncol, mode="expand", borderaxespad=0., fontsize=15)

		#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
        #             ncol=len(othername), mode="expand", borderaxespad=0., fontsize=15)
		if save:
			save = self.name + '_results/' + save
			fig.savefig('%s.pdf'%save, bbox_inches='tight')
		plt.show()

		#test curve
		'''
		fig, ax = plt.subplots()
		ax.errorbar(data_color1['MJD'], data_color1['m'], data_color1['dm'], label=color[0], marker='o', c='b', capsize = 7, linestyle='')
		ax.plot(x_interp, inter_color1, linestyle='-', c='b')
		ax.plot(x_interp, inter_color1+inter_color1_err, linestyle='--', c='b')
		ax.plot(x_interp, inter_color1-inter_color1_err, linestyle='--', c='b')
		ax.errorbar(data_color2['MJD'], data_color2['m'], data_color2['dm'], label=color[1], marker='o', c='g', capsize = 7, linestyle='')
		ax.plot(x_interp, inter_color2, linestyle='-', c='g')
		ax.plot(x_interp, inter_color2+inter_color2_err, linestyle='--', c='g')
		ax.plot(x_interp, inter_color2-inter_color2_err, linestyle='--', c='g')
		plt.show()
		'''
		

	def get_band_max(self, bandName, tmax=None, correct_ebv=False, host_ebv=0, host_rv=3.1, verbose=False, method='hyperspline', n=10, bounds=[-14, 30], get_err=False, Nboot=50, plot=False):
		col_ = bandName.split('_')[-1].split()[-1]
		if tmax is None:
			tmax = self.Tmax
		if bandName not in self.band_mag_max.keys():	
			mark = self.allData[bandName]['mark']
			band_max = find_curve_min(self.allData[bandName]['MJD'][mark],  self.allData[bandName]['m'][mark], self.allData[bandName]['dm'][mark], tmax, Name=bandName,
				method=method,n=n,bounds=bounds,get_err=get_err,Nboot=Nboot,plot=plot)
			if verbose == True:
				print(band_max)
			self.band_mag_max[bandName] = band_max[1]
		else:
			band_max = self.band_mag_max[bandName]
		if correct_ebv == True:
			band_max = band_max - self.Rf[col_]*3.1/3.011*self.EBVmilky - self.Rf[col_]*host_rv/3.011*host_ebv
		return band_max

	def Plot(self, seperate=False, bandToPlot=None, stack=False, phase=False, UBVRI=None, salt=False, snpy=False, save=False, phase_range=None, interpolate=None, DM=0,
		othername=None, figsize=(8,12), mag_normalize_flag=True, xlabel='Days from $B$-band Maximum', plot_band_pos = None, plot_sepe = None, remove_bad=True, Rr=False,
		Ii=False, label_color=False, legend_ncol=1, plot_Co=None, correct_ebv=False, host_ebv=0, host_rv=3.1, DM_err=0, out_legend=False, sepe_plot=False,
		Plot_nondetect=None, color_text_distance=5., otherstyle='point', inset_plot=None, inset_type=None, residual_ref=None, inset_box=None, texp=None,linear_not_fit=[],
		DM_axi=None, label_left=[], label_left_distance=1.5, linear_fit=None, linear_mcmc=False):
		'''
		Parameters
		dataName: The name of database, string or list, such as 'atlas', ['ztf'].
		stack: Whether stack the data
		'''
		

		plotDict = {'UVW2':0, 'UVM2':5, 'UVW1':5,'UVOT.U':5.5,'u':5.5,
			'U':6,'UVOT.B':7,'B':8,'g':12, 'UVOT.V':9,'V':10,  'c':14, 'r': 16, 'o':18, 'R':20, 'i': 22, 'I':24, 'z':26, 'L':28,'G':30,'w':32}
		colorDict = {'c':'darkgreen', 'o':'r',
			'g':'g', 'r':'r', 'i':'brown','u':'purple',
			'U':'mediumorchid','B':'b', 'V':'lime', 'R':'salmon', 'I':'darkgoldenrod',
			'UVOT.U':'saddlebrown','UVOT.B':'darkorange','UVOT.V':'gold',
			'UVW1':'pink','UVW2':'slateblue','UVM2':'black','z':'gray','L':'r','bolo':'black','G':'darkgreen','w':'r'}
		shape_list = ['o','^','*','p','<','X','D','H','v','>','s','P','h','d','1','2','3','4','8']
		shape_dict = {'B':'o','V':'v','g':'^','r':'<','R':'>','i':'s','I':'P','UVW2':'o','UVM2':'v','UVW1':'s','UVOT.U':'<','UVOT.B':'>','UVOT.V':'*'} #'o','v','^','<','>','s','P','D','X'
		if othername is not None:
			othercolor = ['r','b','darkgreen','purple', 'orange','pink','brown','gold','green','salmon','mediumorchid','lime','darkgoldenrod',]
			othershape = ['D','^','*','p','<','X','H','v','>','s','P','h','d','1','2','3','4','8']
			otherline = [
				 ('solid', 'solid'),      # Same as (0, ()) or '-'
				 ('dashdot', 'dashdot'),  # Same as '-.'
			     ('dashed', 'dashed'),    # Same as '--'
			     ('long dash with offset', (5, (10, 3))),
			     ('long long dash with offset', (5, (20, 3))),
			     ('densely dashed',        (0, (5, 1))),
			     ('densely dashdotted',    (0, (3, 1, 1, 1))),
			     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
			     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
			     ('dotted', 'dotted'),]    # Same as (0, (1, 1)) or ':'
		if not UBVRI:
			tel_list = []
			for item in self.allData.keys():
				if '_' not in item:
					continue
				if item.split()[0] != 'UBVRI':
					tel_ = item.split('_')[-2].split()[-1].lower()
					if tel_ not in tel_list:
						tel_list.append(tel_)

			shapeDict = {}
			for i in range(len(tel_list)):
				shapeDict[tel_list[i]]=shape_list[i]
		if stack == True:
			for k, v in colorDict.copy().items():
				colorDict['stacked ' + k] = v

		if bandToPlot is None:
			bandToPlot = self.dataBand.copy()
			#print(bandToPlot)
			#exit()
		elif type(bandToPlot) == type('str'):
			bandToPlot = [bandToPlot]
		# Stacked or not?
		if stack == True:
			for i in range(len(bandToPlot)):
				bandToPlot[i] = 'stacked ' + bandToPlot[i]
			if linear_not_fit is not None:
				for i in range(len(linear_not_fit)):
					linear_not_fit[i] = 'stacked ' + linear_not_fit[i]
		if sepe_plot == True:
			sepe_ = 0.
		elif plot_sepe is not None:
			sepe_ = plot_sepe
		else:
			sepe_ = 1.4
		if UBVRI is not None:
			bandToPlot = []
			quene = self.UBVRI_bands
			if type(UBVRI) == list:
				for i in range(len(UBVRI)):
					UBVRI[i] = 'UBVRI ' + UBVRI[i]
				keys = UBVRI
			elif salt:
				keys = self.salt.keys()
			elif snpy:
				keys = self.snpy.keys()
			elif type(UBVRI) == list:
				for i in range(len(UBVRI)):
					UBVRI[i] = 'UBVRI ' + UBVRI[i]
				keys = UBVRI
			else:
				keys = self.allData.keys()
			for item in keys:
				if item.split()[0] == 'UBVRI':
					bandToPlot.append(item)
			n_ = 0
			for item in quene:
				item_ = 'UBVRI ' + item
				if item_ in bandToPlot:
					if item == 'I' and 'UBVRI i' in bandToPlot and Ii == True:
						n_ -= 1
						plotDict[item] = sepe_*n_
					elif item == 'R' and 'UBVRI r' in bandToPlot and Rr == True:
						n_ -= 1
						plotDict[item] = sepe_*n_
					elif item == 'i' and Ii == False:
						plotDict[item] = sepe_*n_ + sepe_*0
					elif item == 'z':
						plotDict[item] = sepe_*n_ + sepe_*1
					elif item == 'UVW2':
						plotDict[item] = -1*sepe_*3
						n_ -= 1
					elif item == 'UVM2':
						plotDict[item] = -1*sepe_*1
						n_ -= 1
					elif item == 'UVW1':
						plotDict[item] = -1*sepe_*1
						n_ -= 1
					elif item == 'UVOT.U':
						plotDict[item] = -1*sepe_*1
					else:
						plotDict[item] = sepe_*n_
					n_ += 1
		else:
			#quene = ['UVW2', 'UVM2', 'UVW1','UVOT.U', 'U','UVOT.B','B', 'g', 'UVOT.V','V',  'c', 'r', 'R', 'o', 'i', 'I']
			quene = self.UBVRI_bands.copy()
			bandToPlot_ = []
			for item in bandToPlot:
				bandToPlot_.append(item.split('_')[-1])
			n_ = 0
			for item in quene:
				if item in bandToPlot_:
					if (item == 'I' and 'i' in bandToPlot_ and Ii==True) or (item == 'R' and 'r' in bandToPlot_ and Rr==True):
						print('haha')
						n_ -= 1
						plotDict[item] = sepe_*n_
					elif item == 'i':
						plotDict[item] = sepe_*n_ + sepe_*1
					elif item == 'r':
						plotDict[item] = sepe_*n_ + sepe_*1
					elif item == 'z':
						plotDict[item] = sepe_*n_ + sepe_*1
					elif item == 'UVW2':
						#plotDict[item] = -1*sepe_*3
						plotDict[item] = -1*sepe_*3.5
						n_ -= 1
					elif item == 'UVM2':
						#plotDict[item] = -1*sepe_*1
						plotDict[item] = -1*sepe_*3.
						n_ -= 1
					elif item == 'UVW1':
						#plotDict[item] = -1*sepe_*1
						plotDict[item] = -1*sepe_*2
						n_ -= 1
					elif item == 'UVOT.U':
						plotDict[item] = -1*sepe_*1
					elif item == 'U':
						plotDict[item] = sepe_*n_ - sepe_*1
					else:
						plotDict[item] = sepe_*n_
					n_ += 1
			if plot_band_pos is not None:
				for k, v in plot_band_pos.items():
					plotDict[k] = v
		if 'L' in plotDict.keys():
			plotDict['L'] = plotDict['r']
		if 'w' in plotDict.keys():
			plotDict['w'] = plotDict['r']
		# Generate the figure
		if seperate == True and sepe_plot == False:
			plotNumber = 1 
		else:
			plotNumber = 0
		minMag = 99
		maxMag = -99
		band_first = list(self.allData.keys())[0]
		if remove_bad == True:
			mark_ = self.allData[band_first]['mark']
		else:
			mark_ = [True for i in range(len(self.allData[band_first]))] 
		minDate = self.allData[band_first][mark_]['MJD'].min()
		maxDate = self.allData[band_first][mark_]['MJD'].max()
		for item in bandToPlot:
			#if not self.allData[item].empty:
			if item in self.allData.keys():
				if not self.allData[item].empty:
					if UBVRI is not None:
						col_ = item.split()[-1]
					else:
						col_ = item.split('_')[-1]
					if remove_bad == True:
						mark_ = self.allData[item]['mark']
					else:
						mark_ = [True for i in range(len(self.allData[item]))]
					data_band_ =  self.allData[item][mark_]
					minMag_ = data_band_['m'].min() - plotDict[col_]*plotNumber - DM
					maxMag_ = data_band_['m'].max() - plotDict[col_]*plotNumber - DM
					minDate_ = data_band_['MJD'].min()
					maxDate_ = data_band_['MJD'].max()
					if minMag_ < minMag:
						minMag = minMag_
					if maxMag_ > maxMag:
						maxMag = maxMag_
					if minDate_ < minDate:
						minDate = minDate_
					if maxDate_ > maxDate:
						maxDate = maxDate_
		

		plotDistance = 2

		if phase:
			phase_flag = 1
			if texp is not None:
				Tmax_ = texp
			elif self.Tmax:
				Tmax_ = self.Tmax
			elif salt:
				Tmax_ = self.salt_para['B_Tmax'][0]
			elif snpy:
				Tmax_ = self.snpy_para['Tmax'][0]
			else:
				Exception('Tmax needs to be set!')
		else:
			phase_flag = 0
			Tmax_ = 0
		maxDate = maxDate - Tmax_
		minDate = minDate - Tmax_
		if sepe_plot == True:
			sub_label_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
			space = 0.1
			sub_n = len(bandToPlot)
			if Rr == True:
				sub_n -= 1
			if Ii == True:
				sub_n -= 1
			nrows = int((sub_n+1)/2)
			if sub_n == 1:
				ncols = 1
			else:
				ncols = 2
			all_bands = []
			if UBVRI is not None:
				for band_ in bandToPlot:
					all_bands.append(band_.split()[-1])
			else:
				for band_ in bandToPlot:
					all_bands.append(band_.split('_')[-1])
			all_bands = list(set(all_bands))
			pos_dict = {}
			band_n = 0
			last_band = None
			for UBVRI_band in self.UBVRI_bands:
				if UBVRI_band in all_bands:
					if last_band == 'r' or last_band == 'i':
						if Rr == True and last_band == 'r' and UBVRI_band == 'R':
							band_n -= 1
						elif Ii == True and last_band == 'i' and UBVRI_band == 'I':
							band_n -= 1						
					row_pos_ = int(band_n/2)
					col_pos_ = band_n%2
					pos_dict[UBVRI_band] = [row_pos_, col_pos_]
					band_n += 1
					last_band = UBVRI_band 
			fig, axs = plt.subplots(nrows, ncols, sharex='col', figsize=figsize)
			fig.subplots_adjust(wspace=0, hspace=0)
			if nrows == 1:
				if ncols == 1:
					axs = np.array([[axs]])
				else:
					axs = np.array([axs])
		else:
			fig, axs = plt.subplots(figsize=figsize)
			axs = [[axs]]
		inset_residual_flag = False
		inset_plot_flag = False
		if inset_plot is not None:
			if inset_type == 'plot':
				inset_plot_flag = True
			else:
				inset_residual_flag = True
			axs_inset = [None for i in range(np.size(axs))]
		#print(inset_plot_flag, inset_residual_flag)
		#exit()
		#fig, ax = plt.subplots(figsize=figsize)
		#ax.set_title('Photometry')
		plt.tick_params(labelsize=15)
		if phase:
			if sepe_plot == True:
				fig.supxlabel(xlabel,fontsize=20,y=0.01)
			else:
				axs[0][0].set_xlabel(xlabel,fontsize=20)
		else:
			if sepe_plot == True:
				fig.supxlabel('MJD',fontsize=20,y=0.03)
			else:
				axs[0][0].set_xlabel('MJD',fontsize=20)
		if seperate == True:
			constants_str = ' + Constants'
		else:
			constants_str = ''
		if DM == 0:
			if sepe_plot == True:
				fig.supylabel('Apparent Magnitude' + constants_str,fontsize=20,x=0.05)
			else:
				axs[0][0].set_ylabel('Apparent Magnitude' + constants_str,fontsize=20)
		else:
			if sepe_plot == True:
				fig.supylabel('Absolute Magnitude',fontsize=20,x=0.02)
			else:
				axs[0][0].set_ylabel('Absolute Magnitude' + constants_str,fontsize=20)
		
		# print(self.allData)
		#label=item.split('_')[-2].upper() + ' ' + item.split('_')[-1] + '-band')
		axs[0][0].axvline(220, linestyle='--', c='grey')
		if UBVRI is None:
			tel_done = {}
			for item in tel_list:
				tel_done[item.lower()] = 0
		plot_color_text = {}
		for item in plotDict.keys():
			plot_color_text[item] = [-9999,-9999]
		label_ = None
		if othername:
			label_list = np.zeros(1+len(othername))
		for item in bandToPlot:
			# print(item)
			#if not self.allData[item].empty:
			if UBVRI is not None:
				col_ = item.split()[-1]
				fmt = 'o'
				if othername:
					c = 'black'
					fmt = shape_dict[col_]
					if not label_list[0]:
						label_list[0] = 1
						label_ = self.name
					else:
						label_ = None
					axs[0][0].plot(0,
						-999,
						marker = 'o',
						markerfacecolor='none',
						markeredgecolor=c,
						markersize=markersize,
						c = c,
						linestyle='',
						label =label_)
				else:
					c = colorDict[item.split()[-1]]
				fmt = 'o'
			else:
				tel_ = item.split('_')[-2].split()[-1].lower()
				col_ = item.split('_')[-1]
				fmt = shapeDict[tel_]
				c = colorDict[item.split('_')[-1]]
			if sepe_plot == True:
				row_pos = pos_dict[col_][0]
				col_pos = pos_dict[col_][1]
			else:
				row_pos = 0
				col_pos = 0
			if col_pos == 1:
				axs[row_pos][col_pos].tick_params(axis='y',labelright=True, labelleft=False)
			if inset_residual_flag:
				inset_index_ = row_pos*ncols + col_pos
				axs_inset[inset_index_] = axs[row_pos][col_pos].inset_axes(inset_box[inset_index_])
				axs_inset[inset_index_].tick_params(labeltop=True, labelbottom=False, labelright=True, labelleft=False)
				axs_inset[inset_index_].invert_yaxis()
				band_name = residual_ref + ' ' + col_
				other_phase_ref = (self.allData[band_name]['MJD'] - self.other_texp[residual_ref]).to_numpy()/(1+self.other_z[residual_ref]*phase_flag)
				other_mark = self.allData[band_name]['mark']
				if correct_ebv==True:
					extinction_correct = (self.Rf[col_]*3.1/3.011*self.other_milky_ebv[residual_ref] + 
						self.Rf[col_]*self.other_host_rv[residual_ref]/3.011*self.other_host_ebv[residual_ref])
				else:
					extinction_correct = 0
				data_band_ref_ = self.allData[band_name][other_mark].reset_index(drop=True)
				if otherstyle == 'line':
					ref_interp_linear_ = interp1d(other_phase_ref[other_mark], data_band_ref_['m'] - self.other_DM[residual_ref] - extinction_correct, 
						kind='linear', bounds_error=False, fill_value=np.nan)
					interp_min = other_phase_ref[other_mark].min()
					interp_max = other_phase_ref[other_mark].max()
					def ref_interp_(t):
						value = ref_interp_linear_(t)
						mark = (t > interp_min)*(t < interp_max)
						return [value, mark]
				else:
					ref_interp_ = fit1dcurve.Interpolator(type='hyperspline',x=other_phase_ref[other_mark],
										y=data_band_ref_['m'] - self.other_DM[residual_ref] - extinction_correct, dy=data_band_ref_['dm'])
			elif inset_plot_flag:
				inset_index_ = row_pos*ncols + col_pos
				axs_inset[inset_index_] = axs[row_pos][col_pos].inset_axes(inset_box[inset_index_])
				axs_inset[inset_index_].tick_params(labeltop=True, labelbottom=False, labelright=True, labelleft=False)
				axs_inset[inset_index_].invert_yaxis()
				def ref_interp_(t):
					return [t*0, t>-999]
				

			if '_' in item:
				label = item.split('_')[-2].split()[-1].upper()
				if tel_ == 'uvot':
					label  = 'swift ' + label
			else:
				label = ''
			if item in self.allData.keys():
				if not self.allData[item].empty:
					if remove_bad == True:
						mark = self.allData[item]['mark']
					else:
						mark = np.array([True for i in range(len(self.allData[item]))])
					'''
					if UBVRI is not None:
						col_ = item.split()[-1]
					else:
						tel_ = item.split('_')[-2].split()[-1].lower()
						col_ = item.split('_')[-1]
					if '_' in item:
						label = item.split('_')[-2].split()[-1].upper()
						if tel_ == 'uvot':
							label  = 'swift ' + label
					else:
						label = ''
					'''
					if UBVRI is not None:
						pass
					elif not tel_done[tel_]:
						axs[row_pos][col_pos].plot((self.allData[item]['MJD'][mark][0] - Tmax_)/(1+self.z*phase_flag),
							   self.allData[item]['m'][mark][0] - plotDict[col_]*plotNumber - DM,
							   marker=shapeDict[tel_],
							   markerfacecolor='none',
							   markeredgecolor='grey',
							   markersize=markersize,
							   linestyle='',
							   label=label)
						if Plot_nondetect is not None:
							if tel_ + '_' + col_ in Plot_nondetect:
								for non_detect_key in Plot_nondetect.keys():
									if non_detect_key.split('_')[0] == tel_:
										non_detect_col_ = non_detect_key.split('_')[-1]
										label_nondetect = '%s $%s$-band\n non-detection'%(label, non_detect_col_)
										axs[row_pos][col_pos].plot((Plot_nondetect[non_detect_key]['MJD'] - Tmax_)/(1+self.z*phase_flag),
											Plot_nondetect[non_detect_key]['m'] - plotDict[non_detect_col_]*plotNumber - DM,
											linestyle='',marker='v',alpha=alpha,c=colorDict[non_detect_col_],label=label_nondetect)
										maxMag = np.max([maxMag, Plot_nondetect[non_detect_key]['m'] - plotDict[non_detect_col_]*plotNumber - DM])
								#axs[row_pos][col_pos].plot(60250.633196 - Tmax_,
								#			18.88 - plotDict['r']*plotNumber - DM,
								#			linestyle='',marker='v',alpha=alpha,c=colorDict['o'],label='ATLAS o-band\n non-detection')
						tel_done[tel_] = 1						
					#label_ = None
					'''
					if UBVRI is not None:
						fmt = 'o'
						if othername:
							c = 'black'
							fmt = shape_dict[col_]
							if not label_list[0]:
								label_list[0] = 1
								label_ = self.name
							else:
								label_ = None
						else:
							c = colorDict[item.split()[-1]]
					else:
						fmt = shapeDict[tel_]
						c = colorDict[item.split('_')[-1]]
					'''
					data_length = len(self.allData[item])
					phase_mark = np.array([True for j in range(data_length)])
					if phase_range:
						phase_mark = (self.allData[item]['MJD'] > phase_range[0]) * (self.allData[item]['MJD'] < phase_range[1])
					data_band_ = self.allData[item][mark*phase_mark].reset_index(drop=True)
					if inset_plot_flag:
						phase_inset_mark = (data_band_['MJD']  > inset_plot[0]) * (data_band_['MJD'] < inset_plot[1])
					else:
						phase_inset_mark = data_band_['mark']
					#if inset_residual_flag:
					#	ref_phase_mark = (data_band_['MJD'] - Tmax_)/(1+self.z*phase_flag) > other_phase_ref.min()
					#else:
					#	ref_phase_mark = data_band_['m'] < 99
					if correct_ebv == True:
						data_band_['m'] = data_band_['m'] - self.Rf[col_]*3.1/3.011*self.EBVmilky - self.Rf[col_]*host_rv/3.011*host_ebv
					if sepe_plot == True:
						minMag = data_band_['m'].min() - plotDict[col_]*plotNumber - DM
						maxMag = data_band_['m'].max() - plotDict[col_]*plotNumber - DM
						minDate = data_band_['MJD'].min()
						maxDate = data_band_['MJD'].max()
					#if DM_err != 0:
					#	data_band_['dm'] = np.sqrt(data_band_['dm']**2 + DM_err**2)
					lenData = len(data_band_)
					if otherstyle == 'point':
						axs[row_pos][col_pos].errorbar((data_band_['MJD'] - Tmax_)/(1+self.z*phase_flag),
							data_band_['m'] - plotDict[col_]*plotNumber - DM,
							yerr=np.sqrt(data_band_['dm'].astype('float')**2 + DM_err**2),
							capsize = 7,
							linestyle='',
							alpha=alpha,
							c=c,
							zorder=998.)
						axs[row_pos][col_pos].plot((data_band_['MJD'] - Tmax_)/(1+self.z*phase_flag),
							data_band_['m'] - plotDict[col_]*plotNumber - DM,
							marker = fmt,
							linestyle='',
							markerfacecolor='none',
							markeredgecolor=c,
							markersize=markersize,
							c = c,
							zorder=999.)
					else:
						axs[row_pos][col_pos].errorbar((data_band_['MJD'] - Tmax_)/(1+self.z*phase_flag),
							data_band_['m'] - plotDict[col_]*plotNumber - DM,
							yerr=np.sqrt(data_band_['dm'].astype('float')**2 + DM_err**2),
							capsize = 7,
							linestyle='',
							alpha=alpha,
							c=c,)
						axs[row_pos][col_pos].plot((data_band_['MJD'] - Tmax_)/(1+self.z*phase_flag),
							data_band_['m'] - plotDict[col_]*plotNumber - DM,
							marker = fmt,
							linestyle='',
							markerfacecolor='none',
							markeredgecolor=c,
							markersize=markersize,
							c = c,)
					if inset_residual_flag or inset_plot_flag:
						ref_phase_mark = ref_interp_((data_band_['MJD'] - Tmax_)/(1+self.z*phase_flag))[1]*phase_inset_mark
						yplot = data_band_['m'][ref_phase_mark] - DM - ref_interp_((data_band_['MJD'][ref_phase_mark] - Tmax_)/(1+self.z*phase_flag))[0]

						'''
						if item == 'UBVRI g':
							print((data_band_['MJD'][ref_phase_mark] - Tmax_)/(1+self.z*phase_flag))
							print(data_band_['m'][ref_phase_mark] - DM)
							print(data_band_ref_)
							print(ref_interp_((data_band_['MJD'][ref_phase_mark] - Tmax_)/(1+self.z*phase_flag))[0])
							exit()
						'''
						axs_inset[inset_index_].errorbar((data_band_['MJD'][ref_phase_mark] - Tmax_)/(1+self.z*phase_flag),
							yplot,
							yerr=np.sqrt(data_band_['dm'][ref_phase_mark].astype('float')**2 + DM_err**2),
							capsize = 7,
							linestyle='',
							alpha=alpha,
							c=c,
							zorder=998.)
						axs_inset[inset_index_].plot((data_band_['MJD'][ref_phase_mark] - Tmax_)/(1+self.z*phase_flag),
							yplot,
							marker = fmt,
							linestyle='',
							markerfacecolor='none',
							markeredgecolor=c,
							markersize=markersize,
							c = c,
							zorder=999.)
						if item == 'UBVRI V':
							#axs_inset[inset_index_].text(3, -14.75, 'GOTO L', fontsize=15)
							#axs_inset[inset_index_].plot([1.13,2.95], [-15.37,-15], c='black')
							axs_inset[inset_index_].text(5.5, 1, 'GOTO L', fontsize=15)
							axs_inset[inset_index_].plot([2,5], [1.3,1], c='black')
						if inset_residual_flag:
							axs_inset[inset_index_].set_ylim(yplot.max() + 0.2, yplot.min() - 0.2)
					if linear_fit is not None and item not in linear_not_fit:
						mark_linear = (data_band_['MJD'] - Tmax_ > linear_fit[0])*(data_band_['MJD'] - Tmax_ < linear_fit[1])
						if mark_linear.sum() > 1:
							params, params_cov = optimize.curve_fit(linear_f, (data_band_['MJD'][mark_linear] - Tmax_)/(1+self.z*phase_flag), 
								data_band_['m'][mark_linear] - plotDict[col_]*plotNumber - DM,
								sigma=data_band_['dm'][mark_linear])
							print(item, params[0], params_cov[0][0]**0.5)
							axs[row_pos][col_pos].plot(np.array(linear_fit)/(1+self.z*phase_flag), linear_f(np.array(linear_fit)/(1+self.z*phase_flag), params[0], params[1]), linestyle='--', color='gray')
							if linear_mcmc:
								quick_mcmc(linear_f, (data_band_['MJD'][mark_linear] - Tmax_)/(1+self.z*phase_flag), 
									data_band_['m'][mark_linear] - plotDict[col_]*plotNumber - DM,
									data_band_['dm'][mark_linear],
									params, 
									[(-np.inf, -np.inf),(np.inf, np.inf)],
									plot=False)
							#print(linear_f(460, params[0], params[1])+31.02)
							#axs[row_pos][col_pos].text(linear_fit[1]/(1+self.z*phase_flag), linear_f(linear_fit[1], params[0], params[1]), '%.4f\n [mag/d]'%params[0], fontsize=15)
					if interpolate is not None:
						if item not in self.interpolate.keys():
							self.interpolate_lc(item, method=interpolate)
						x_pred = np.linspace(data_band_['MJD'][0], data_band_['MJD'][lenData-1], 1000)
						x_pred = np.concatenate([data_band_['MJD'], x_pred])
						x_pred = np.sort(x_pred)
						'''
						if self.interpolate_method[item] == 'gp':
							gp_jd, gp_flux, gp_flux_errors, gp_ws = self.interpolate[item].predict(x_pred=x_pred, returnv=True)
							m_interp_ = -2.5*np.log10(gp_flux[0]/(self.zp[item.split()[-1].split('_')[-1]]*1e-11))
							dm_interp_ = gp_flux_errors[0]/gp_flux[0]*1.0857
							ax.plot(x_pred - Tmax_, m_interp_ - plotDict[col_]*plotNumber, linestyle='-', c = c)
							ax.plot(x_pred - Tmax_, m_interp_ + dm_interp_ - plotDict[col_]*plotNumber, linestyle='--', c = c)
							ax.plot(x_pred - Tmax_, m_interp_ - dm_interp_ - plotDict[col_]*plotNumber, linestyle='--', c = c)
						'''
						m_interp_, dm_interp_ = self.interpolate[item](x_pred)
						axs[row_pos][col_pos].plot((x_pred - Tmax_)/(1+self.z*phase_flag), m_interp_ - plotDict[col_]*plotNumber, linestyle='-', c = c)
						axs[row_pos][col_pos].plot((x_pred - Tmax_)/(1+self.z*phase_flag), m_interp_ + dm_interp_ - plotDict[col_]*plotNumber, linestyle='--', c = c)
						axs[row_pos][col_pos].plot((x_pred - Tmax_)/(1+self.z*phase_flag), m_interp_ - dm_interp_ - plotDict[col_]*plotNumber, linestyle='--', c = c)
					if othername:
						#othercolor = ['r', 'b', 'purple', 'orange', 'brown', 'darkgreen', 'pink','gray']
						#othershape = ['^','*','p','<','X','D','H','h']
						other_zorder=900
						other_xshift = {}
						for index, othername_ in enumerate(othername):
							other_zorder = other_zorder - 2
							band_name = othername_ + ' ' + col_
							if band_name in self.allData.keys():
								if not label_list[index+1]:
									label_ = othername_
									label_list[index+1] = 1
								else:
									label_ = None
								if texp is not None:
									other_phase = (self.allData[band_name]['MJD'] - self.other_texp[othername_]).to_numpy()/(1+self.other_z[othername_]*phase_flag)
								else:
									other_phase = (self.allData[band_name]['MJD'] - self.other_MJD_max[othername_]).to_numpy()/(1+self.other_z[othername_]*phase_flag)
								if phase_range:
									other_mark = (other_phase  > (phase_range[0] - Tmax_)/(1+self.z*phase_flag)) * (other_phase < (phase_range[1] - Tmax_)/(1+self.z*phase_flag))
								else:
									other_mark = self.allData[band_name]['mark']
								if correct_ebv==True:
									extinction_correct = (self.Rf[col_]*3.1/3.011*self.other_milky_ebv[othername_] + 
										self.Rf[col_]*self.other_host_rv[othername_]/3.011*self.other_host_ebv[othername_])
								else:
									extinction_correct = 0
								if mag_normalize_flag == True:
									if DM != 0:
										mag_normalize = self.other_DM[othername_]
									elif DM_axi is not None:
										mag_normalize = self.other_DM[othername_] - DM_axi
									else:
										mag_normalize = (self.other_mag_max[band_name] - extinction_correct - 
											self.get_band_max(item, correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_rv))#self.allData[item]['m'][mark*phase_mark].min()
								else:
									mag_normalize = 0
								if sepe_plot == True:
									mag_normalize += plotDict[col_]
								'''
								ax.errorbar(other_phase[other_mark],
											self.allData[band_name]['m'][other_mark] - plotDict[col_] - mag_normalize,
											yerr=self.allData[band_name]['dm'][other_mark],
											fmt = othershape[index],
											c = othercolor[index],
											label = label_,
											capsize = 7)
								'''
								other_band_ = self.allData[band_name][other_mark].reset_index(drop=True)
								if inset_plot_flag:
									other_inset_mark = (other_phase[other_mark]  > (inset_plot[0] - Tmax_)/(1+self.z*phase_flag)) * (other_phase[other_mark] < (inset_plot[1] - Tmax_)/(1+self.z*phase_flag))
								else:
									other_inset_mark = other_band_['mark']
								if correct_ebv == True:
									other_band_['m'] = other_band_['m'] - extinction_correct
								if self.other_DM_err[othername_] != 0 and DM != 0:
									other_band_['dm'] = np.sqrt(other_band_['dm']**2 + self.other_DM_err[othername_]**2)
								#if other_band_.loc[0,'m'] - plotDict[col_] - mag_normalize < 0:
								#	print(othername_, band_name)
								if sepe_plot == True:
									othercolor_tmp = othercolor[index]
								else:
									othercolor_tmp = c
								if otherstyle == 'point':
									axs[row_pos][col_pos].errorbar(other_phase[other_mark],
												other_band_['m'] - plotDict[col_] - mag_normalize,
												yerr=other_band_['dm'],
								   				c=othercolor_tmp,
												capsize = 7,
												linestyle='',
												zorder=other_zorder-1,
												alpha=alpha)
									axs[row_pos][col_pos].plot(other_phase[other_mark],
												other_band_['m'] - plotDict[col_] - mag_normalize,
												marker = othershape[index],
												markerfacecolor='none',
								   				markeredgecolor=othercolor_tmp,
								   				markersize=markersize,
								   				linestyle='',
								   				zorder=other_zorder,
								   				c=othercolor_tmp)
									#c = othercolor[index],
									if sepe_plot == True:
										other_label_color = othercolor[index]
									else:
										other_label_color = 'gray'
									axs[0][0].plot(0,
											-999,
											marker = othershape[index],
											markerfacecolor='none',
								   			markeredgecolor=other_label_color,
											markersize=markersize,
											linestyle='',
											c = other_label_color,
											label = label_)
									if inset_residual_flag or inset_plot_flag:
										inset_mark = ref_interp_(other_phase[other_mark])[1]*other_inset_mark
										axs_inset[inset_index_].errorbar(other_phase[other_mark][inset_mark],
												other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize - ref_interp_(other_phase[other_mark][inset_mark])[0],
												yerr=other_band_['dm'][inset_mark],
								   				c=othercolor_tmp,
												capsize = 7,
												linestyle='',
												zorder=other_zorder-1,
												alpha=alpha)
										axs_inset[inset_index_].plot(other_phase[other_mark][inset_mark], 
											other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize - ref_interp_(other_phase[other_mark][inset_mark])[0],
											marker = othershape[index],
											markerfacecolor='none',
								   			markeredgecolor=othercolor[index],
											markersize=markersize,
											linestyle='',
											c=othercolor_tmp, zorder=other_zorder)
								elif otherstyle == 'line':
									other_interp = fit1dcurve.Interpolator(type='hyperspline',x=other_phase[other_mark],
										y=other_band_['m'] - plotDict[col_] - mag_normalize,dy=other_band_['dm'])
									axs[row_pos][col_pos].plot(other_phase[other_mark],
												other_band_['m'] - plotDict[col_] - mag_normalize,
												linestyle=otherline[index][1],
												markerfacecolor='none',
								   				markeredgecolor=othercolor[index],
								   				markersize=markersize,
								   				zorder=other_zorder,
								   				c=othercolor_tmp)
									if band_name == 'SN2018oh g':
										phase_18oh = (np.array([58144.60000000019, 58152.10000000019]) - self.other_texp[othername_])/(1+self.other_z[othername_]*phase_flag)
										m_18oh = np.array([20.852, 15.500]) - extinction_correct
										dm_18oh = np.sqrt(np.array([0.223, 0.010])**2 + self.other_DM_err[othername_]**2)
										axs[row_pos][col_pos].errorbar(phase_18oh,
												m_18oh - plotDict[col_] - mag_normalize,
												yerr=dm_18oh,
								   				c=othercolor_tmp,
												capsize = 7,
												linestyle='',
												zorder=other_zorder-1,
												alpha=alpha)
										axs[row_pos][col_pos].plot(phase_18oh,
													m_18oh - plotDict[col_] - mag_normalize,
													marker = othershape[index],
													markerfacecolor='none',
									   				markeredgecolor=othercolor[index],
									   				markersize=markersize,
									   				linestyle='',
									   				zorder=other_zorder,
									   				c=othercolor_tmp)
									#c = othercolor[index],
									if sepe_plot == False:
										axs[0][0].plot([0,1],[-999,-999],linestyle=otherline[index][1],c = 'gray',label = label_)
									else:
										axs[0][0].plot([0,1],[-999,-999],linestyle=otherline[index][1],c = othercolor[index],label = label_)
									if inset_residual_flag or inset_plot_flag:
										inset_mark = ref_interp_(other_phase[other_mark])[1]*other_inset_mark
										axs_inset[inset_index_].plot(other_phase[other_mark][inset_mark], 
											other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize - ref_interp_(other_phase[other_mark][inset_mark])[0],
											linestyle=otherline[index][1],
											c=othercolor_tmp, zorder=other_zorder)
										if band_name == 'SN2018oh g':
											axs_inset[inset_index_].errorbar(phase_18oh,
												m_18oh - plotDict[col_] - mag_normalize - ref_interp_(phase_18oh)[0],
												yerr=dm_18oh,
								   				c=othercolor_tmp,
												capsize = 7,
												linestyle='',
												zorder=other_zorder-1,
												alpha=alpha)
											axs_inset[inset_index_].plot(phase_18oh, 
												m_18oh - plotDict[col_] - mag_normalize - ref_interp_(phase_18oh)[0],
												marker = othershape[index],
												markerfacecolor='none',
									   			markeredgecolor=othercolor[index],
												markersize=markersize,
												linestyle='',
												c=othercolor_tmp, zorder=other_zorder)
											axs_inset[inset_index_].plot([1,3],[-13,-13], c=othercolor_tmp)
											axs_inset[inset_index_].plot([7.5,6],[-17.5,-14], c=othercolor_tmp)
											axs_inset[inset_index_].text(3.1, -12.5, 'SN2018oh', c=othercolor_tmp, fontsize=15)
										'''
										if band_name == 'Piro_mix_0.25 g':
											print(other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize - ref_interp_(other_phase[other_mark][inset_mark])[0])
											exit()
											print(other_phase[other_mark][inset_mark])
											print(other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize)
											print(ref_interp_(other_phase[other_mark][inset_mark])[0])
											exit()
										'''										
								else:
									other_interp = fit1dcurve.Interpolator(type='hyperspline',x=other_phase[other_mark],
										y=other_band_['m'] - plotDict[col_] - mag_normalize,dy=other_band_['dm'])
									len_plot = len(other_phase[other_mark])
									'''
									guess = 0
									shift_x = optimize.minimize(lambda x: other_interp(x)[0], guess, bounds=[(guess-20, guess+20)]).x[0]
									if col_ == 'B' and not inset_residual_flag:
										other_xshift[othername_] = shift_x
									if other_xshift.get(othername_, 0):
										phase_shift = other_xshift[othername_]
									else:
										phase_shift = 0
									'''
									#if DM == 0.:
									#	shift_y = self.other_mag_max[item] - self.get_band_max(item, correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_rv)
									#else:
									phase_shift = 0
									shift_y = 0.
									other_xplot = np.linspace(other_phase[other_mark][0], other_phase[other_mark][len_plot-1], 100)
									other_yplot = other_interp(other_xplot)[0] - shift_y
									axs[row_pos][col_pos].plot(other_xplot - phase_shift, other_yplot, linestyle=otherline[index][1], c=othercolor_tmp, zorder=other_zorder)
									axs[0][0].plot([0,1],[-999,-999],linestyle=otherline[index][1],c = othercolor[index],label = label_)
									#axs[row_pos][col_pos].plot(other_phase[other_mark],
									#			other_band_['m'] - plotDict[col_] - mag_normalize,
									#			marker = fmt,
									#			markerfacecolor='none',
								   	#			markeredgecolor=othercolor[index],
								   	#           linestyle='',
								   	#			zorder=other_zorder,
								   	#			c=othercolor[index])
									if inset_residual_flag or inset_plot_flag:
										inset_mark = ref_interp_(other_xplot)[1]
										if inset_plot_flag:
											inset_mark = inset_mark*(other_xplot>(inset_plot[0]-Tmax_)/(1+self.other_z[othername_*phase_flag]))*(other_xplot<(inset_plot[1]-Tmax_)/(1+self.other_z[othername_]*phase_flag))
										axs_inset[inset_index_].plot(other_xplot[inset_mark], other_yplot[inset_mark] - ref_interp_(other_xplot)[0][inset_mark], 
											linestyle=otherline[index][1],c=othercolor[index], zorder=other_zorder)
								text_x_ = other_phase[other_mark].tolist()[-1] + color_text_distance
								text_y_ = other_band_['m'].tolist()[-1] - plotDict[col_]*plotNumber - mag_normalize
								if text_x_ > plot_color_text[col_][0] and col_ not in label_left:
									plot_color_text[col_] = [text_x_, text_y_]
								maxMag = np.max([maxMag, (other_band_['m'] - plotDict[col_] - mag_normalize).max()])
								minMag = np.min([minMag, (other_band_['m'] - plotDict[col_] - mag_normalize).min()])

					if salt:
						mark_salt_ = self.salt[item]['mark']
						salt_data_ = self.salt[item][mark_salt_].copy()
						salt_filters = ''
						for salt_filter_ in self.salt.keys():
							salt_filters += salt_filter_.split()[-1]
						axs[row_pos][col_pos].set_title('SN %s %s fit with SALT2'%(self.name[2:], salt_filters), fontsize=20)
						axs[row_pos][col_pos].plot((salt_data_['MJD'] - Tmax_)/(1+self.z*phase_flag), salt_data_['m'] - plotDict[col_]*plotNumber, c = 'black')
						axs[row_pos][col_pos].plot((salt_data_['MJD'] - Tmax_)/(1+self.z*phase_flag), salt_data_['m'] + salt_data_['dm'] - plotDict[col_]*plotNumber, c = 'black', linestyle='-.')
						axs[row_pos][col_pos].plot((salt_data_['MJD'] - Tmax_)/(1+self.z*phase_flag), salt_data_['m'] - salt_data_['dm'] - plotDict[col_]*plotNumber, c = 'black', linestyle='-.')
						salt_text_dict = {'X0':'$\\rm X_0$','X1':'$\\rm X_1$','C':'C','B_Mmax':'$\\rm m_B$','B_Tmax':'$\\rm t_{max}(B)$'}
						y_ = maxMag - 2
						for k, v in salt_text_dict.items():
							axs[row_pos][col_pos].text(self.salt_para['B_Tmax'][0] - Tmax_ - 9, y_ + 0.8, v + ' = %.3f $\\pm$ %.3f'%(self.salt_para[k][0], self.salt_para[k][1]), fontsize=12)
							y_ += 0.4
						lenData = len(salt_data_)
						text_x_ = salt_data_.loc[lenData-1, 'MJD'] + color_text_distance - Tmax_
						text_y_ = salt_data_.loc[lenData-1, 'm'] - plotDict[col_]*plotNumber+0.5
						if text_x_ > plot_color_text[col_][0]:
							plot_color_text[col_] = [text_x_, text_y_]
					elif snpy:
						mark_snpy_ = self.snpy[item]['mark']
						snpy_data_ = self.snpy[item][mark_snpy_].reset_index(drop=True).copy()
						axs[row_pos][col_pos].set_title('SN %s BVgri fit with SNooPy2'%self.name[2:], fontsize=20)
						axs[row_pos][col_pos].plot((snpy_data_['MJD'] - Tmax_)/(1+self.z*phase_flag), snpy_data_['m'] - plotDict[col_]*plotNumber, c = 'black')
						axs[row_pos][col_pos].plot((snpy_data_['MJD'] - Tmax_)/(1+self.z*phase_flag), snpy_data_['m'] + snpy_data_['dm'] - plotDict[col_]*plotNumber, c = 'black', linestyle='-.')
						axs[row_pos][col_pos].plot((snpy_data_['MJD'] - Tmax_)/(1+self.z*phase_flag), snpy_data_['m'] - snpy_data_['dm'] - plotDict[col_]*plotNumber, c = 'black', linestyle='-.')
						snpy_text_dict = {'DM':'DM','dm15':'$\\rm \\Delta m_{15}(B)$','EBVhost':'$\\rm EBV_{host}$','Tmax':'$\\rm t_{max}(B)$'}
						y_ = maxMag - 2
						for k, v in snpy_text_dict.items():
							axs[row_pos][col_pos].text(self.snpy_para['Tmax'][0] - Tmax_ - 10, y_ + 1, v + ' = %.3f $\\pm$ %.3f'%(self.snpy_para[k][0], self.snpy_para[k][1]), fontsize=12)
							y_ += 0.5
						lenData = len(snpy_data_)
						text_x_ = snpy_data_.loc[lenData-1, 'MJD']+color_text_distance - Tmax_
						text_y_ = snpy_data_.loc[lenData-1, 'm'] - plotDict[col_]*plotNumber+0.5
						if text_x_ > plot_color_text[col_][0]:
							plot_color_text[col_] = [text_x_, text_y_]
					else:
						'''
						if col_ == 'L':
							text_x_ = data_band_.loc[0, 'MJD'] - 7. - Tmax_
							text_y_ = data_band_.loc[0, 'm'] - plotDict[col_]*plotNumber - DM
						elif col_ == 'o':
							text_x_ = data_band_.loc[0, 'MJD'] - 1.5 - Tmax_
							text_y_ = data_band_.loc[0, 'm'] - plotDict[col_]*plotNumber - DM
						'''
						if col_ in label_left:
							text_x_ = data_band_.loc[0, 'MJD'] - label_left_distance - Tmax_
							text_y_ = data_band_.loc[0, 'm'] - plotDict[col_]*plotNumber - DM
						else:
							text_x_ = data_band_.loc[lenData-1, 'MJD']+color_text_distance - Tmax_
							text_y_ = data_band_.loc[lenData-1, 'm'] - plotDict[col_]*plotNumber - DM
					if text_x_ > plot_color_text[col_][0]:
						plot_color_text[col_] = [text_x_, text_y_]

			if othername and label_color == True and sepe_plot == True:
				other_zorder=900
				for index, othername_ in enumerate(othername):
					other_zorder=other_zorder-2
					band_name = othername_ + ' ' + col_
					if band_name in self.allData.keys():
						if not label_list[index+1]:
							label_ = othername_
							label_list[index+1] = 1
						else:
							label_ = None
						if texp is not None:
							other_phase = (self.allData[band_name]['MJD'] - self.other_texp[othername_]).to_numpy()/(1+self.other_z[othername_]*phase_flag)
						else:
							other_phase = (self.allData[band_name]['MJD'] - self.other_MJD_max[othername_]).to_numpy()/(1+self.other_z[othername_]*phase_flag)
						if phase_range:
							other_mark = (other_phase  > (phase_range[0] - Tmax_)/(1+self.z*phase_flag)) * (other_phase < (phase_range[1] - Tmax_)/(1+self.z*phase_flag))
						else:
							other_mark = self.allData[band_name]['mark']
						if correct_ebv==True:
							extinction_correct = (self.Rf[col_]*3.1/3.011*self.other_milky_ebv[othername_] + 
								self.Rf[col_]*self.other_host_rv[othername_]/3.011*self.other_host_ebv[othername_])
						else:
							extinction_correct = 0
						if mag_normalize_flag == True:
							if DM != 0:
								mag_normalize = self.other_DM[othername_]
							elif DM_axi is not None:
								mag_normalize = self.other_DM[othername_] - DM_axi
							else:
								if item not in self.allData.keys():
									if item == 'UBVRI R':
										#mag_normalize = self.other_mag_max[band_name] - self.allData['UBVRI r']['m'][mark*phase_mark].min()
										mag_normalize = (self.other_mag_max[band_name] - extinction_correct -
											self.get_band_max('UBVRI r', correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_ebv))
									elif item == 'UBVRI r':
										#mag_normalize = self.other_mag_max[band_name] - self.allData['UBVRI R']['m'][mark*phase_mark].min()
										mag_normalize = (self.other_mag_max[band_name] - extinction_correct -
											self.get_band_max('UBVRI R', correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_ebv))
									elif item == 'UBVRI I':
										#mag_normalize = self.other_mag_max[band_name] - self.allData['UBVRI i']['m'][mark*phase_mark].min()
										mag_normalize = (self.other_mag_max[band_name] - extinction_correct -
											self.get_band_max('UBVRI i', correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_ebv))
									elif item == 'UBVRI i':
										#mag_normalize = self.other_mag_max[band_name] - self.allData['UBVRI I']['m'][mark*phase_mark].min()
										mag_normalize = (self.other_mag_max[band_name] - extinction_correct -
											self.get_band_max('UBVRI I', correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_ebv))
								else:
									mag_normalize = (self.other_mag_max[band_name] - extinction_correct - 
									self.get_band_max(item, correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_ebv))
						else:
							mag_normalize = 0
						if sepe_plot == True:
							mag_normalize += plotDict[col_]
						'''
						ax.errorbar(other_phase[other_mark],
									self.allData[band_name]['m'][other_mark] - plotDict[col_] - mag_normalize,
									yerr=self.allData[band_name]['dm'][other_mark],
									fmt = othershape[index],
									c = othercolor[index],
									label = label_,
									capsize = 7)
						'''
						other_band_ = self.allData[band_name][other_mark].reset_index(drop=True)
						if inset_plot_flag:
							other_inset_mark = (other_phase[other_mark]  > (inset_plot[0] - Tmax_)/(1+self.z*phase_flag)) * (other_phase[other_mark] < (inset_plot[1] - Tmax_)/(1+self.z*phase_flag))
						else:
							other_inset_mark = other_band_['mark']
						if correct_ebv == True:
							other_band_['m'] = other_band_['m'] - extinction_correct
						if self.other_DM_err[othername_] != 0 and DM != 0:
							other_band_['dm'] = np.sqrt(other_band_['dm']**2 + self.other_DM_err[othername_]**2)
						#print(other_band_.loc[0,'m'] - plotDict[col_] - mag_normalize)
						#if other_band_.loc[0,'m'] - plotDict[col_] - mag_normalize < 0:
						#	print(othername_, band_name, other_band_.loc[0,'m'], plotDict[col_], mag_normalize)
						#print(othername_, band_name, mag_normalize)
						if otherstyle == 'point':
							axs[row_pos][col_pos].errorbar(other_phase[other_mark],
										other_band_['m'] - plotDict[col_] - mag_normalize,
										yerr=other_band_['dm'],
										c = othercolor[index],
										capsize = 7,
										linestyle='',
										zorder=other_zorder-1,
										alpha=alpha)
							axs[row_pos][col_pos].plot(other_phase[other_mark],
										other_band_['m'] - plotDict[col_] - mag_normalize,
										marker = othershape[index],
										markerfacecolor='none',
								   		markeredgecolor=othercolor[index],
								   		markersize=markersize,
								   		zorder=other_zorder,
								   		linestyle='',
										c = othercolor[index])
							axs[0][0].plot(0,
									-999,
									marker = othershape[index],
									markerfacecolor='none',
								   	markeredgecolor=othercolor[index],
									markersize=markersize,
									linestyle='',
									c = othercolor[index],
									label = label_)
							if inset_residual_flag or inset_plot_flag:
								inset_mark = ref_interp_(other_phase[other_mark])[1]*other_inset_mark
								axs_inset[inset_index_].errorbar(other_phase[other_mark][inset_mark],
												other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize - ref_interp_(other_phase[other_mark][inset_mark])[0],
												yerr=other_band_['dm'][inset_mark],
								   				c=othercolor[index],
												capsize = 7,
												linestyle='',
												zorder=other_zorder-1,
												alpha=alpha)
								axs_inset[inset_index_].plot(other_phase[other_mark][inset_mark], 
									other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize - ref_interp_(other_phase[other_mark][inset_mark])[0],
									marker = othershape[index],
									markerfacecolor='none',
						   			markeredgecolor=othercolor[index],
									markersize=markersize,
									linestyle='',
									c=othercolor[index], zorder=other_zorder)
						elif otherstyle == 'line':
							other_interp = fit1dcurve.Interpolator(type='hyperspline',x=other_phase[other_mark],
								y=other_band_['m'] - plotDict[col_] - mag_normalize,dy=other_band_['dm'])
							axs[row_pos][col_pos].plot(other_phase[other_mark],
										other_band_['m'] - plotDict[col_] - mag_normalize,
										linestyle=otherline[index][1],
										markerfacecolor='none',
						   				markeredgecolor=othercolor[index],
						   				markersize=markersize,
						   				zorder=other_zorder,
						   				c=othercolor[index])
							#c = othercolor[index],
							#axs[0][0].plot(0,
							#		-999,
							#		marker = 'o',
							#		linestyle='',
							#		c = othercolor[index],
							#		label = label_)
							axs[0][0].plot([0,1],[-999,-999],linestyle=otherline[index][1],c = othercolor[index],label = label_)
							if inset_residual_flag or inset_plot_flag:
								inset_mark = ref_interp_(other_phase[other_mark])[1]*other_inset_mark
								axs_inset[inset_index_].plot(other_phase[other_mark][inset_mark], 
									other_band_['m'][inset_mark] - plotDict[col_] - mag_normalize - ref_interp_(other_phase[other_mark][inset_mark])[0],
									linestyle=otherline[index][1],
									c=othercolor[index], zorder=other_zorder)


								#inset_mark = ref_interp_(other_phase[other_mark])[1]*other_inset_mark
								#axs_inset[inset_index_].plot(other_phase[other_mark][inset_mark], other_band_['m'][inset_mark] - ref_interp_(other_phase[other_mark][inset_mark])[0],
								#			linestyle=otherline[index][1],c=othercolor[index], zorder=other_zorder)
						else:
							other_interp = fit1dcurve.Interpolator(type='hyperspline',x=other_phase[other_mark],
								y=other_band_['m'] - plotDict[col_] - mag_normalize,dy=other_band_['dm'])
							'''
							guess = 0
							shift_x = optimize.minimize(lambda x: other_interp(x)[0], guess, bounds=[(guess-20, guess+20)]).x[0]
							if col_ == 'B' and not inset_residual_flag:
								other_xshift[othername_] = shift_x
							if other_xshift.get(othername_, 0):
								phase_shift = other_xshift[othername_]
							else:
								phase_shift = 0
							'''
							#if DM == 0.:
							#	shift_y = self.other_mag_max[item] - self.get_band_max(item, correct_ebv=correct_ebv, host_ebv=host_ebv, host_rv=host_rv)
							#else:
							phase_shift = 0
							shift_y = 0.
							len_plot = len(other_phase[other_mark])
							other_xplot = np.linspace(other_phase[other_mark][0], other_phase[other_mark][len_plot-1], 100)
							#print(shift_y)
							other_yplot = other_interp(other_xplot)[0] - shift_y
							axs[row_pos][col_pos].plot(other_xplot - phase_shift, other_yplot, linestyle=otherline[index][1],c=othercolor[index], zorder=other_zorder)
							axs[0][0].plot([0,1],[-999,-999],linestyle=otherline[index][1],c = othercolor[index],label = label_)
							if inset_residual_flag or inset_plot_flag:
								inset_mark = ref_interp_(other_xplot)[1]
								if inset_plot_flag:
									inset_mark = inset_mark*(other_xplot>(inset_plot[0]-Tmax_)/(1+self.other_z[othername_*phase_flag]))*(other_xplot<(inset_plot[1]-Tmax_)/(1+self.other_z[othername_]*phase_flag))
								axs_inset[inset_index_].plot(other_xplot[inset_mark], other_yplot[inset_mark] - ref_interp_(other_xplot)[0][inset_mark], 
									linestyle=otherline[index][1],c=othercolor[index], zorder=other_zorder)
						maxMag = np.max([maxMag, (other_band_['m'] - plotDict[col_] - mag_normalize).max()])
						minMag = np.min([minMag, (other_band_['m'] - plotDict[col_] - mag_normalize).min()])
						maxDate = np.max([maxDate, (other_phase[other_mark]).max()])
						minDate = np.min([minDate, (other_phase[other_mark]).min()])
			if DM != 0. and maxMag > -10:
				maxMag = -10
			if sepe_plot == True:
				if phase_range:
					#axs[row_pos][col_pos].set_xlim(phase_range[0] - Tmax_ - 10, phase_range[1] - Tmax_ + 20)
					#len_phase = (phase_range[1] - phase_range[0])*0.1
					len_phase = 0.5
					axs[row_pos][col_pos].set_xlim(phase_range[0] - Tmax_ - len_phase, phase_range[1] - Tmax_ + len_phase)
				else:
					axs[row_pos][col_pos].set_xlim(minDate - 10, maxDate + 30)
				axs[row_pos][col_pos].set_ylim(maxMag+0.1*(maxMag - minMag), minMag-0.1*(maxMag - minMag))
		if sepe_plot == True and out_legend == True:
			SN_legend = axs[0][0].legend(bbox_to_anchor=(0., 1.02, 2., .102), loc='lower left',
		                      ncol=legend_ncol, mode="expand", borderaxespad=0., fontsize=15)
			axs[0][0].add_artist(SN_legend)
		if label_color == True and sepe_plot == True:
			color_lines = []
			for i_row in range(nrows):
				color_lines_ = []
				for i_col in range(ncols):
					color_lines_.append([])
				color_lines.append(color_lines_)
			for item_band in bandToPlot:			
				col_ = item_band.split()[-1].split('_')[-1]
				if sepe_plot == True:
					label_color_ = '%s'%col_
					row_pos = pos_dict[col_][0]
					col_pos = pos_dict[col_][1]
				else:
					label_color_ = '%s-%.1f'%(col_, plotDict[col_])
					row_pos = 0
					col_pos = 0
				if otherstyle == 'point':
					color_line_, = axs[row_pos][col_pos].plot(0,
					   -999,
					   marker=shape_dict[col_],
					   markerfacecolor='none',
					   markeredgecolor='grey',
					   markersize=markersize,
					   linestyle='',
					   alpha=0.,
					   label=label_color_)
				else:
					if Rr or Ii:
						if (Rr and col_ == 'r') or (Ii and col_ == 'i'):
							linestyle = '--'
						else:
							linestyle = '-'
						color_line_, = axs[row_pos][col_pos].plot([0,1],[-999,-999],linestyle=linestyle,label=label_color_,c='gray')
					else:
						color_line_, = axs[row_pos][col_pos].plot([0],[-999],linestyle='',label=label_color_)
				color_lines[row_pos][col_pos].append(color_line_)
			for i_row in range(nrows):
				for i_col in range(ncols):
					if color_lines[i_row][i_col] != []:
						axs[i_row][i_col].legend(handles=color_lines[i_row][i_col], fontsize=15, frameon=False)
			maxDate += 0.33*(maxDate - minDate)
		if seperate == True and label_color == False:
			for k,v in plot_color_text.items():
				if plot_color_text[k] != [-9999,-9999]:
					if plotDict[k] < 0:
						plotDict_ = k+'+{:.1f}'.format(-plotDict[k])
					else:
						plotDict_ = k+'-{:.1f}'.format(plotDict[k])
					if othername and len(othername) != 1:
						axs[row_pos][col_pos].text(v[0], v[1], plotDict_, c='black',fontsize=15, verticalalignment='center')
					else:
						axs[row_pos][col_pos].text(v[0], v[1], plotDict_, c=colorDict[k],fontsize=15, verticalalignment='center')
			if plot_Co is not None:
				for item_Co in plot_Co:
					axs[0][0].plot(item_Co[0], item_Co[1], linestyle='--', c='black')
					axs[0][0].text(np.mean(item_Co[0]), np.mean(item_Co[1]), '$^{56}$Co decay', c='black', fontsize=15)
					#axs[0][0].text(item_Co[0][0] + 0.3*(item_Co[0][1]-item_Co[0][0]), item_Co[1][0] + 0.3*(item_Co[1][1]-item_Co[1][0]), '$^{56}$Co decay', c='black', fontsize=15)
		if sepe_plot == False:
			#axs[row_pos][col_pos].axvline(60253.887219 - Tmax_, linestyle='--', label='Epoch of the\nfirst spectrum')
			if phase_range:
				#axs[row_pos][col_pos].set_xlim(phase_range[0] - Tmax_ - 10, phase_range[1] - Tmax_ + 20)
				axs[row_pos][col_pos].set_xlim(phase_range[0] - Tmax_ - 0.05*(phase_range[1] - phase_range[0]), phase_range[1] - Tmax_ + 0.1*(phase_range[1] - phase_range[0]))
			else:
				axs[row_pos][col_pos].set_xlim(minDate - 10, maxDate + 30)
			axs[row_pos][col_pos].set_ylim(maxMag+0.05*(maxMag - minMag), minMag-0.05*(maxMag - minMag))
			if UBVRI is None:
				ncol_ = len(tel_list)
				if ncol_ > 4:
					ncol_ = 4
				if out_legend == True:
					l1 = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
		                      ncol=ncol_, mode="expand", borderaxespad=0., fontsize=15)
				else:
					l1 = plt.legend(ncol=legend_ncol, fontsize=15)
			elif othername:
				l1 = plt.legend(fontsize=15, ncol=legend_ncol)
			if label_color == True and sepe_plot == False:
				color_lines = []
				for k,v in plot_color_text.items():
					if seperate == True:
						if plotDict[k] < 0:
							plotDict_ = k+'+{:.1f}'.format(-plotDict[k])
						else:
							plotDict_ = k+'-{:.1f}'.format(plotDict[k])
					else:
						plotDict_ = k
					if v != [-9999,-9999]:
						color_line_, = axs[row_pos][col_pos].plot([0,1],[-999,-999],linestyle='',marker='o',label=plotDict_,c=colorDict[k])
						color_lines.append(color_line_)
				plt.legend(handles=color_lines, fontsize=15, frameon=False)
				plt.gca().add_artist(l1)

			if plot_Co is not None:
				n_Co = 0
				for item_Co in plot_Co:
					axs[row_pos][col_pos].plot(item_Co[0], item_Co[1], linestyle='--', c='black')
					if n_Co == 0:
						axs[row_pos][col_pos].text(np.mean(item_Co[0]), np.mean(item_Co[1]), '$^{56}$Co decay', c='black', fontsize=15)
						n_Co = 1
			if DM_axi is not None:
				axs_M = axs[row_pos][col_pos].twinx()
				axs_M.set_ylabel('Absolute Magnitude' + constants_str,fontsize=20)
				axs_M.set_ylim(maxMag+0.1*(maxMag - minMag) - DM_axi, minMag-0.1*(maxMag - minMag) - DM_axi)
		else:
			if sub_n % 2 == 1:
				axs[-1][1].axis('off')
				axs[-2][1].tick_params(axis='x',labelbottom=True)
			for sub_fig_n in range(sub_n):
				row_pos = int(sub_fig_n/2)
				col_pos = sub_fig_n%2
				if plot_Co is not None:
					axs[row_pos][col_pos].plot(plot_Co[-1][0], plot_Co[-1][1], linestyle='--', c='black')
					axs[row_pos][col_pos].text(np.mean(plot_Co[-1][0]), np.mean(plot_Co[-1][1]), '$^{56}$Co decay\n 0.0224 [mag/d]', c='black', fontsize=15)
		if save:
			save = self.name + '_results/' + save
			fig.savefig('%s.pdf'%save, bbox_inches='tight')
		plt.show()

	def SED_curve(self, bandName, MJD, interpolate=None, save=None, fit_blackbody=False, guess_T=None, guess_A=None, bound_T=[None], bound_A=[None], plot_fit=False, fit_err=False):
		sed_wave = []
		sed_flux = []
		sed_flux_error = []
		bad = False
		if interpolate is None:
			'''
			for band in bandName:
				band_ = band.split()[-1].split('_')[-1]
				pos_ = pos_[pos_]
				if not pos_.index.empty:
					pos_ = pos_.index[0]
					break
			'''
			for band in bandName:
				if band == bandName[0]:
					limit = 1e-7
				else:
					limit = 0.2
				mark_ = self.allData[band]['mark']
				data_band_ = self.allData[band][mark_].reset_index(drop=True)
				pos_ = np.argsort(np.abs(data_band_['MJD'] - MJD))[0]
				if abs(data_band_.loc[pos_, 'MJD'] - MJD) > limit:
					print(MJD)
					print(abs(data_band_.loc[pos_, 'MJD'] - MJD), band)
					bad = True
				band_ = band.split()[-1].split('_')[-1]
				if band_ in self.UBVRI_bands:
					sed_wave.append(self.central_wavelengths[band_])
					sed_flux.append(self.zp[band_]*1e-11*10**(-0.4*(data_band_['m'][pos_])))
					sed_flux_error.append(data_band_['dm'][pos_]/1.0857*sed_flux[-1])
		elif interpolate is not None:
			for band in bandName:
				if band == bandName[0]:
					limit = 1e-7
					mark_ = self.allData[band]['mark']
					data_band_ = self.allData[band][mark_].reset_index(drop=True)
					pos_ = np.argsort(np.abs(data_band_['MJD'] - MJD))[0]
					if abs(data_band_.loc[pos_, 'MJD'] - MJD) > limit:
						print(abs(data_band_.loc[pos_, 'MJD'] - MJD), band)
						bad = True
					band_ = band.split()[-1].split('_')[-1]
					if band_ in self.UBVRI_bands:
						sed_wave.append(self.central_wavelengths[band_])
						sed_flux.append(self.zp[band_]*1e-11*10**(-0.4*(data_band_['m'][pos_])))
						sed_flux_error.append(data_band_['dm'][pos_]/1.0857*sed_flux[-1])
				else:
					if band not in self.interpolate.keys():
						self.interpolate_lc(band, method=interpolate)
					'''
					if self.interpolate_method[band] == 'gp':
						gp_jd, gp_flux, gp_flux_errors, gp_ws = self.interpolate[band].predict(x_pred=[MJD], returnv=True)
						m_ = -2.5*np.log10(gp_flux[0][0])+23.9
						dm_ = gp_flux_errors[0][0]/gp_flux[0][0]*1.0857
					'''
					m_, dm_ = self.interpolate[band]([MJD])
					m_ = m_[0]
					dm_ = dm_[0]
					band_ = band.split()[-1].split('_')[-1]
					if band_ in self.UBVRI_bands:
						sed_wave.append(self.central_wavelengths[band_])
						sed_flux.append(self.zp[band_]*1e-11*10**(-0.4*m_))
						sed_flux_error.append(dm_/1.0857*sed_flux[-1])
					else:
						Exception('Not known band!')

		quene_wave = np.argsort(sed_wave)
		sed_wave = np.array(sed_wave)[quene_wave]
		sed_flux = np.array(sed_flux)[quene_wave]
		sed_flux_error = np.array(sed_flux_error)[quene_wave]
		sed_curve = np.array([sed_wave, sed_flux, sed_flux_error])

		A=T=T_err=None
		if fit_blackbody == True and bad == False:
			scale_fit = sed_flux.min()
			#scale_fit=1
			sed_flux_fit = sed_flux/scale_fit
			sed_flux_error_fit = sed_flux_error/scale_fit

			if guess_T is not None:
				guess = [1e-10, guess_T]
			else:
				guess = [1e-10, 5000]
			if guess_A is not None:
				guess[0] = guess_A
			if bound_T[0] is not None:
				bounds = ([0, bound_T[0]], [np.inf, bound_T[1]])
			else:
				bounds = ([0, 10], [np.inf, 100000])
			if bound_A[0] is not None:
				bounds[0][0] = bound_A[0]
				bounds[1][0] = bound_A[1]
			if fit_err == True:
				sigma=sed_flux_error_fit
			else:
				sigma=None
			params,params_covariance=optimize.curve_fit(blackbody_func, sed_wave, sed_flux_fit, 
				p0=guess, maxfev=500000, bounds=bounds, sigma=sigma)
			A = params[0]*scale_fit
			T = params[1]
			T_err = np.sqrt(params_covariance[1,1])
			#print(blackbody_func(5000, 1, 5000))
			#exit()
			print(A, T, T_err)
			if plot_fit==True:
				fig, ax = plt.subplots()
				ax.set_title('T=%.1f$\\pm$%.1f'%(T, T_err))
				wave_plot = np.sort(np.concatenate([np.linspace(3000,10000,100), sed_wave]))
				ax.errorbar(sed_wave, sed_flux, yerr=sed_flux_error, linestyle='', marker='o', capsize=5)
				ax.plot(wave_plot, blackbody_func(wave_plot, A, T))
				plt.show()

		if save is not None and bad == False:
			save = self.name + '_results/' + save
			with open(save, 'w') as f:
				if fit_blackbody == True:
					f.writelines('# %.4e %.1f %.1f\n'%(A, T, T_err))
				for i in range(len(bandName)):
					f.writelines('%.3f %e %e\n'%(sed_wave[i], sed_flux[i], sed_flux_error[i]))

		return sed_curve, bad, [A, T, T_err]

	def get_bb_bolo(self, bandName=['UBVRI B', 'UBVRI g', 'UBVRI V',  'UBVRI r', 'UBVRI i'], phase_range=None, filter0='i', filter1='B', DM=30., host_ebv=0., 
		overwrite=False,interpolate_method=None, interpolate_region=None, ref_band=None, n=5, plot=False, band_require=None, res=0.25, LC_file=None):
		#LC_file = self.name+'_lc_LCF.csv'
		if overwrite == True and LC_file is None:
			LC_file = self.name+'_lc_LCF.csv'
			self.SavePhot(LC_file, remove_bad=True, bandName=bandName, MOSFIT=False, phase_range=phase_range, LCF=True,
				interpolate_method=interpolate_method, interpolate_region=interpolate_region, ref_band=ref_band, n=n, plot=plot)
		else:
			LC_file = self.name+'_lc_LCF.csv'
		filters = []
		for band_ in bandName:
			filter_ = band_.split()[-1].split('_')[-1]
			if 'UVOT' in filter_:
				filter_ = filter_[-1] + '_S'
			filters.append(filter_)
		all_filters = ''.join(filters)
		lc = get_LC_fitting(self.name + '_results/' + LC_file, DM, self.EBVmilky, self.z, filters, host_ebv)
		t = get_bb_bolo(lc,filter0=filter0,filter1=filter1,save=None,overwrite=overwrite, z=self.z, band_require=band_require, res=res)
		t = t[t['filts']==all_filters]
		MJD = t['MJD']
		m = -2.5*np.log10(t['L_mcmc']*1e-35)
		#d_flux = flux*dm/1.0857
		dm = t['dL_mcmc0']/t['L_mcmc'] * 1.0857
		bolo_df = pd.DataFrame({'MJD':MJD, 'm':m, 'dm':dm})
		bolo_df['mark'] = True
		bolo_df['filter'] = 'bolo'
		self.allData['bolo'] = bolo_df

	def get_color(self, band1, band2, t, phase_range=None, method='hyperspline', n=5, Nboot=50):
		t1 = self.allData[band1]['MJD'].to_numpy()
		m1 = self.allData[band1]['m'].to_numpy()
		dm1 = self.allData[band1]['dm'].to_numpy()
		mark1 = (t1 > phase_range[0])*(t1 < phase_range[1])
		t1 = t1[mark1]
		m1 = m1[mark1]
		dm1 = dm1[mark1]
		t2 = self.allData[band2]['MJD'].to_numpy()
		m2 = self.allData[band2]['m'].to_numpy()
		dm2 = self.allData[band2]['dm'].to_numpy()
		mark2 = (t2 > phase_range[0])*(t2 < phase_range[1])
		t2 = t2[mark2]
		m2 = m2[mark2]
		dm2 = dm2[mark2]
		return get_color(t1, m1, dm1, t2, m2, dm2, t, method=method, n=n, Nboot=Nboot)

	def plot_SED(self, bandName, save=None, figsize=(8,6), separate=1e-13, interpolate=None,fit_blackbody=False, 
		guess_T=None, guess_A=None, bound_T=[None], bound_A=[None], show_T=False, fit_err=False):
		blackbody_paras_list = []
		fig, ax = plt.subplots(figsize=figsize)
		title_bandName = bandName.copy()
		title_bandName_ = []
		for i in range(len(bandName)):
			title_bandName[i] = bandName[i].split()[-1].split('_')[-1]
		for i in range(len(self.UBVRI_bands)):
			if self.UBVRI_bands[i] in title_bandName:
				title_bandName_.append(self.UBVRI_bands[i])
		ax.set_title(', '.join(title_bandName_))
		ax.set_xlabel('Wavelength $\\rm [\\AA]$', fontsize=15)
		ax.set_ylabel('$F_{\\lambda}+offsets$', fontsize=15)
		plot_number = 0
		len_firstband = len(self.allData[bandName[0]])
		if type(guess_T) != type(['haha']) :
			guess_T = [guess_T for j in range(len_firstband)]
		if type(guess_A) != type(['haha']) :
			guess_A = [guess_A for j in range(len_firstband)]
		if len(bound_T) == 1:
			bound_T = [bound_T for j in range(len_firstband)]
		if len(bound_A) == 1:
			bound_A = [bound_A for j in range(len_firstband)]
		for i in range(len_firstband):
			MJD = self.allData[bandName[0]].loc[i,'MJD']
			#print(i, MJD)
			save_ = save.split('.')[0]+'_%s'%MJD+'.txt'
			SED_curve, bad, blackbody_paras = self.SED_curve(bandName=bandName, MJD=MJD, save=save_, interpolate=interpolate, fit_blackbody=fit_blackbody, 
				guess_T=guess_T[i], bound_T=bound_T[i], bound_A=bound_A[0], plot_fit=False, fit_err=fit_err)
			if bad == False:
				blackbody_paras.append(MJD)
				blackbody_paras_list.append(blackbody_paras)
				if fit_blackbody == True:
					ax.errorbar(SED_curve[0], SED_curve[1] - plot_number, SED_curve[2], capsize=7, marker='o', linestyle='', c='black')
				else:
					ax.errorbar(SED_curve[0], SED_curve[1] - plot_number, SED_curve[2], marker='o', c='black')
				sign_ = '+'
				if MJD - self.Tmax < 0:
					sign_ = '-'
				if fit_blackbody == True and show_T == True:
					ax.text(SED_curve[0][-1] + 0.05*(SED_curve[0][-1]-SED_curve[0][0]), SED_curve[1][-1] - plot_number, 
						sign_+'%.1f d\n T=%.1f'%(MJD - self.Tmax, blackbody_paras[1]), verticalalignment='center')
					wave_plot = np.sort(np.concatenate([np.linspace(SED_curve[0].min(),SED_curve[0].max(),100), SED_curve[0]]))
					ax.plot(wave_plot, blackbody_func(wave_plot, blackbody_paras[0], blackbody_paras[1]) - plot_number, c='black')
				else:
					ax.text(SED_curve[0][-1] + 0.05*(SED_curve[0][-1]-SED_curve[0][0]), SED_curve[1][-1] - plot_number, 
						sign_+'%.1f d'%(MJD - self.Tmax), verticalalignment='center')
				plot_number += separate
		if save is not None:
			plt.savefig(self.name + '_results/' + save, bbox_inches='tight')
		plt.show()
		blackbody_paras_list = np.array(blackbody_paras_list)

		if fit_blackbody==True:
			fig, ax = plt.subplots()
			if self.Tmax is not None:
				ax.set_xlabel('Days since Discovery') 
				Tmax_ = self.Tmax
			else:
				ax.set_xlabel('MJD')
				Tmax_ = 0
			ax.set_ylabel('Temperature [K]')
			ax.errorbar(blackbody_paras_list[:,3] - Tmax_, blackbody_paras_list[:,1], yerr=blackbody_paras_list[:,2], linestyle='', marker='o', capsize=5)
			if save is not None:
				plt.savefig(self.name + '_results/' + 'T_' + save, bbox_inches='tight')
			plt.show()

	def SavePhot(self, save, bandName=None, remove_bad = False, MOSFIT=False, phase_range=None, LCF=False, include=['UBVRI','stacked','normal'], 
			interpolate_method=None, interpolate_region=None, ref_band=None, n=5, plot=False):
		save = self.name + '_results/' + save
		if bandName is None:
			bandName = self.allData.keys()
		elif type(bandName) == type('haha'):
			bandName = [bandName]
		save_data = []
		if interpolate_method is not None:
			ref_mark = [True for i in range(len(self.allData[ref_band]))]
			if remove_bad == True:
				ref_mark = self.allData[ref_band]['mark']
			if phase_range is not None:
				ref_mark = ref_mark * (self.allData[ref_band]['MJD'] > phase_range[0]) * (self.allData[ref_band]['MJD'] < phase_range[1])
		for band_i, band in enumerate(bandName):
			band_split = band.split()
			if len(band_split) > 1:
				if band_split[0] not in include:
					continue
			elif len(band_split) == 1:
				if 'normal' not in include:
					continue
			else:
				continue
			mark_ = [True for i in range(len(self.allData[band]))]
			if remove_bad == True:
				mark_ = self.allData[band]['mark']
			if phase_range is not None:
				mark_ = mark_ * (self.allData[band]['MJD'] > phase_range[0]) * (self.allData[band]['MJD'] < phase_range[1])
			#print(interpolate_method)
			#print(band)
			#print(ref_band)
			if interpolate_method is not None and band != ref_band:
				data_band_ = self.allData[ref_band][ref_mark].copy()
				mjd_mark = (data_band_['MJD'] >= self.allData[band]['MJD'][mark_].min() - 0.1) * (data_band_['MJD'] <= self.allData[band]['MJD'][mark_].max() + 0.1)
				data_band_ = data_band_[mjd_mark].reset_index(drop=True)
				self.interpolate_lc(band, method=interpolate_method, interpolate_region=interpolate_region, n=n, plot=plot)
				data_band_['m'] = self.interpolate[band](data_band_['MJD'])[0]
				if interpolate_method == 'gp':
					data_band_['dm'] = self.interpolate[band](data_band_['MJD'])[1]
			else:
				data_band_ = self.allData[band][mark_].copy()
			data_band_['filter'] = band
			data_band_['band'] = band_split[-1].split('_')[-1].split('.')[-1]
			data_band_['source'] = '%10s'%band_split[-1].split('_')[0].split('.')[0]
			#print(data_band_)
			save_data.append(data_band_)
		save_data = pd.concat(save_data)
		if MOSFIT == True:
			save_data['event'] = self.name
			save_data.rename(columns={'MJD':'time', 'm':'magnitude', 'dm':'e_magnitude'}, inplace=True)
			save_data['source'] = 'This work'
			save_data = save_data[['event','time','magnitude','e_magnitude','band','source']]
		elif LCF == True:
			save_data['mark'] = ~save_data['mark']
			save_data.rename(columns={ 'm':'mag', 'dm':'dmag','band':'filt','mark':'nondet'}, inplace=True)
			save_data = save_data[['MJD','mag','dmag','filt','source','nondet']]
			save_data['filt'][(save_data['filt']=='U')*(save_data['source']=='%10s'%('UVOT'))] = 'U_S'
			save_data['filt'][(save_data['filt']=='B')*(save_data['source']=='%10s'%('UVOT'))] = 'B_S'
			save_data['filt'][(save_data['filt']=='V')*(save_data['source']=='%10s'%('UVOT'))] = 'V_S'
		else:
			save_data = save_data[['MJD','m','dm','mark','filter']]
		save_data.to_csv(save, index=0)

	def SaveSnpy(self, filename='lc.txt', bandToPlot=None, stack=True, UBVRI=None, UBVRI_bands=None, err=None, standard_filter=['U','B','V','R','I']):
		filename = self.name + '_results/' + filename
		if not self.name:
			raise Exception('need a name!')
		To_AB = {'UVOT.B':0.13,'UVOT.V':0.01,'UVOT.U':-1.02,'UVW1':-1.51,'UVW2':-1.73,'UVM2':-1.69}
		if UBVRI is not None:
			if not UBVRI_bands:
				UBVRI_bands = self.UBVRI_bands
			with open(filename,'w') as f:
				f.writelines('%s %s %s %s\n' %(self.name, self.z, self.ra, self.dec))
				for item in self.allData.keys():
					if item.split()[0] == 'UBVRI':
						band_ = item.split()[1]
						if band_ in UBVRI_bands:
							offset = 0.
							if 'UVOT' in band_:
								offset = To_AB[band_]
								band_ = band_.split('.')[-1] + '_UVOT'
							elif band_ in ['c','o']:
								band_ = 'ATLAS_' + band_
							elif band_ in standard_filter:
								#band_ = band_+'kait'
								band_ = band_+'s'
								#band_ = band_
							elif band_ in ['u','g','r','i','z']:
								band_ = band_+'_s'
							elif band_ in ['UVW2','UVM2','UVW1']:
								offset = To_AB[band_]
							if self.AB == True:
								offset = 0.
							f.writelines('filter ' + band_ + '\n')
							mark = self.allData[item]['mark']
							data_to_save_ = self.allData[item][mark].reset_index()
							if err is not None:
								data_to_save_['dm'][data_to_save_['dm']<err] = err
							for j in range(len(data_to_save_)):
								f.writelines('%s %s %s\n' %(data_to_save_.loc[j, 'MJD'], 
									data_to_save_.loc[j, 'm']-offset, data_to_save_.loc[j, 'dm']))
		else:
			if not bandToPlot:
				bandToPlot = self.dataBand.copy()
			if type(bandToPlot) == type('str'):
				bandToPlot = [bandToPlot]
			print(bandToPlot)
			if stack == True:
				for i in range(len(bandToPlot)):
					bandToPlot[i] = 'stacked ' + bandToPlot[i]
			with open(filename,'w') as f:
				f.writelines('%s %s %s %s\n' %(self.name, self.z, self.ra, self.dec))
				for i in range(len(bandToPlot)):
					filter_line = 'filter ' + bandToPlot[i].split(' ')[-1]
					if 'ZTF' in filter_line:
						filter_line = filter_line.lower()
					elif 'lick' in filter_line:
						filter_line = 'filter ' + filter_line.split('_')[-1] + 'kait'
					elif 'TNT' in filter_line:
						band_ = bandToPlot[i].split('_')[-1]
						if band_ in ['U','B','V','R','I']:
							band_ = band_ + 's'
						elif band_ in ['u','g','r','i','z']:
							band_ = band_ + '_s'
						filter_line = 'filter ' + band_
					elif 'swift' in filter_line:
						color_ = filter_line.split('_')[-1]
						if color_ in ['UVOT.U','UVOT.B','UVOT.V']:
							filter_line = 'filter ' + color_.split('.')[-1] + '_UVOT'
						else:
							filter_line = 'filter ' + color_
					f.writelines('%s\n' %filter_line)
					mark = self.allData[bandToPlot[i]]['mark']
					data_to_save_ = self.allData[bandToPlot[i]][mark].reset_index()

					for j in range(len(data_to_save_)):
						f.writelines('%s %s %s\n' %(data_to_save_.loc[j, 'MJD'], 
							data_to_save_.loc[j, 'm'], data_to_save_.loc[j, 'dm']))



	def SaveSALT(self, filename, UBVRI=True, new=True, UBVRI_bands=None, err=None, othername=None):
		if not self.name:
			raise Exception('need a name!')
		filename = self.name + '_results/' + filename
		if new:
			'''
			if os.path.exists(filename):
				if os.path.exists(filename+'_bk'):
					os.system('rm  %s_bk'%filename)
				os.system('cp %s %s_bk'%(filename, filename))
				os.system('rm %s'%filename)
			'''
			with open(filename, 'w') as f:
				if othername is not None:
					write_name = othername[2:]
					ra = self.other_basic[othername]['ra']
					dec = self.other_basic[othername]['dec']
					z = self.other_basic[othername]['redshift']
					EBVmilky = coordsToEbv(ra, dec)
				else:
					write_name = self.name
					ra = self.ra
					dec = self.dec
					z = self.z
					EBVmilky = self.EBVmilky
				f.writelines('@SN %s\n'%write_name)
				f.writelines('@RA %s\n'%ra)
				f.writelines('@DEC %s\n'%dec)
				f.writelines('@Z_HELIO %s\n'%z)
				f.writelines('@MWEBV %s\n'%EBVmilky)
				f.writelines('@SELECTED_FOR_SNLS3_TRAINING 1\n')
				f.writelines('@SURVEY SNLS3_LC\n')
				f.writelines('#Date :\n')
				f.writelines('#Mag :\n')
				f.writelines('#Magerr :\n')
				f.writelines('#Filter : instrument and band\n')
				f.writelines('#MagSys : magnitude system\n')
				f.writelines('#end :\n')
				if UBVRI is not None:
					if not UBVRI_bands:
						UBVRI_bands = self.UBVRI_bands
					#MAGSYS_dict = {'U':'VEGA','B':'VEGA','V':'VEGA','R':'VEGA','I':'VEGA',
					#			'g':'AB','r':'AB','i':'AB'}
					#INSTRUMENT_dict = {'U':'STANDARD','B':'STANDARD','V':'STANDARD','R':'STANDARD','I':'STANDARD',
					#			'g':'SDSS','r':'SDSS','i':'SDSS'}
					MAGSYS_dict = {}
					INSTRUMENT_dict = {}
					for item in UBVRI_bands:
						if item == item.upper():
							MAGSYS_dict[item] = 'VEGA'
							INSTRUMENT_dict[item] = 'STANDARD'
						else:
							MAGSYS_dict[item] = 'AB'
							INSTRUMENT_dict[item] = 'SDSS'
					if othername is None:
						data_label = 'UBVRI'
					else:
						data_label = othername
					for k, v in self.allData.items():
						if k.split()[0] == data_label:
							band_ = k.split()[1]
							if band_ in UBVRI_bands:
								for i in range(len(v)):
									if v.loc[i, 'mark']:
										if err is not None:
											f.writelines('%s %s %s %s::%s %s\n'
												%(v.loc[i, 'MJD'], v.loc[i, 'm'], np.max([err, v.loc[i, 'dm']]), INSTRUMENT_dict[band_], band_, MAGSYS_dict[band_]))
										else:
											f.writelines('%s %s %s %s::%s %s\n'
												%(v.loc[i, 'MJD'], v.loc[i, 'm'], v.loc[i, 'dm'], INSTRUMENT_dict[band_], band_, MAGSYS_dict[band_]))

		else:
			if os.path.exists(self.name):
				if os.path.exists(self.name+'_bk'):
					os.system('rm -rf %s_bk'%self.name)
				os.system('cp -rf %s %s_bk'%(self.name, self.name))
				os.system('rm -rf %s'%self.name)
			os.mkdir(self.name)
			with open(self.name + '/lightfile', 'w') as f:
				f.writelines('NAME %s\n'%self.name)
				f.writelines('RA %s\n'%self.ra)
				f.writelines('DEC %s\n'%self.dec)
				f.writelines('Redshift %s\n'%self.z)
				f.writelines('MWEBV %s\n'%self.EBVmilky)
			if UBVRI is not None:
				MAGSYS_dict = {'U':'VEGA','B':'VEGA','V':'VEGA','R':'VEGA','I':'VEGA',
							'g':'AB','r':'AB','i':'AB'}
				INSTRUMENT_dict = {'U':'STANDARD','B':'STANDARD','V':'STANDARD','R':'STANDARD','I':'STANDARD',
							'g':'SDSS','r':'SDSS','i':'SDSS'}
				for k, v in self.allData.items():
					if k.split()[0] == 'UBVRI':
						band_ = k.split()[1]
						with open(self.name + '/lc2fit_%s_%s.dat'%(band_,MAGSYS_dict[band_]), 'w') as f:
							f.writelines('#Date :\n')
							f.writelines('#Mag :\n')
							f.writelines('#Magerr :\n')
							f.writelines('#end :\n')
							f.writelines('@INSTRUMENT %s :\n'%INSTRUMENT_dict[band_])
							f.writelines('@BAND %s\n'%band_)
							f.writelines('@MAGSYS %s\n'%MAGSYS_dict[band_])
							for i in range(len(v)):
								if v.loc[i, 'mark']:
									f.writelines('%s %s %s\n'%(v.loc[i, 'MJD'], v.loc[i, 'm'], v.loc[i, 'dm']))

	def SaveHaffet(self, filename, bandToPlot=None,stack=False,UBVRI=True, UBVRI_bands=None):
		filename = self.name + '_results/' + filename
		if not self.name:
			raise Exception('need a name!')
		if UBVRI is not None:
			if not UBVRI_bands:
				UBVRI_bands = self.UBVRI_bands
			dataToSave = []
			for k, v in self.allData.items():
				if k.split()[0] == 'UBVRI':
					band_ = k.split()[1]
					if band_ in UBVRI_bands:
						pd_ = v.copy()
						mark = pd_['mark']
						pd_ = pd_[mark].reset_index()
						pd_['filter'] = [band_ for j in range(len(pd_))]
						pd_ = pd_[['filter','MJD','m','dm']]
						pd_['MJD'] = pd_['MJD'] + 24e5+0.5
						pd_.columns = ['filter','jdobs','mag','emag']
						dataToSave.append(pd_.copy())
			dataToSave = pd.concat(dataToSave).reset_index(drop=True)
			dataToSave.to_csv(self.name+'HaffetLcs.csv', index=False)
		else:
			if not bandToPlot:
				bandToPlot = self.dataBand.copy()
			if type(bandToPlot) == type('str'):
				bandToPlot = [bandToPlot]
			if stack == True:
				for i in range(len(bandToPlot)):
					bandToPlot[i] = 'stacked ' + bandToPlot[i]
			dataToSave = []
			for i in range(len(bandToPlot)):
				pd_ = self.allData[bandToPlot[i]].copy()
				pd_['filter'] = [bandToPlot[i].split('_')[-1] for j in range(len(pd_))]
				pd_ = pd_[['filter','MJD','m','dm']]
				pd_['MJD'] = pd_['MJD'] + 24e5+0.5
				pd_.columns = ['filter','jdobs','mag','emag']
				dataToSave.append(pd_.copy())
			dataToSave = pd.concat(dataToSave).reset_index(drop=True)
			dataToSave.to_csv(filename + '.csv', index=False)

	
	def Latex_table(self, bands=None, divide_swift=True, swift=False, stack=False, outfile='lc_tex_table.txt', MJD_format='%.1f', single=False, line_end='\\\\'):
		outfile = self.name + '_results/' + outfile
		if bands is None:
			bands = np.array(self.dataBand)
		else:
			bands = np.array(bands)
		mask = np.array([True for i in range(len(bands))])

		for i, item in enumerate(bands):
			if item.split('_')[0].upper() == 'UVOT':
				mask[i] = False
		if divide_swift:
			if swift:
				bands = bands[~mask]
			else:
				bands = bands[mask]
		table_data = []
		for item in self.dataBand:
			if item in bands:
				tel_ = item.split('_')[0].upper()
				filter_ = item.split('_')[1]
				if stack == True:
					band_ = 'stacked ' + item
				else:
					band_ = item
				data_ = self.allData[band_].copy()
				data_ = data_[data_['mark']]
				data_['tel'] = tel_
				data_['filter'] = filter_
				table_data.append(data_)
		table_data = pd.concat(table_data)

		table_data = table_data.sort_values(by='MJD').reset_index(drop=True)
		table_len = len(table_data)
		for i in range(table_len):
			table_data.loc[i, 'MJD'] = MJD_format%table_data.loc[i, 'MJD']
		table_data['MJD'] = table_data['MJD'].astype('str')
		if single == True:
			writelines = []
			with open(outfile, 'w') as f:
				for i in range(table_len):
					line_ = '%s & %.3f & %.3f & %s & %s %s\n'%(table_data.loc[i, 'MJD'], table_data.loc[i, 'm'], table_data.loc[i, 'dm'],
						table_data.loc[i, 'filter'], table_data.loc[i, 'tel'], line_end)
					writelines.append(line_)
				f.writelines(writelines)
		else:
			filter_list_ = table_data['filter'].drop_duplicates().tolist()
			filter_quene = ['UVW2', 'UVM2', 'UVW1','UVOT.U','UVOT.B','UVOT.V',
				'U',  'B', 'V','R','I','u','b','v','g','c','r','o','i']
			filter_list = []
			for item_quene in filter_quene:
				if item_quene in filter_list_:
					filter_list.append(item_quene)
			filter_index = {}
			for i, item in enumerate(filter_list):
				filter_index[item] = i + 1

			with open(outfile, 'w') as f:
				table_i = 0
				start = 0
				while(1):
					f.writelines(r'\begin{landscape}' + '\n')
					f.writelines(r'\begin{table}' + '\n')
					f.writelines(r'\centering' + '\n')
					if table_i == 0:
						f.writelines(r'\caption{Ground-based Optical Photometry of ' + 'SN ' + self.name[2:] + '.}\n')
					else:
						f.writelines(r'\contcaption{A table continued from the previous one.}' + '\n')
						f.writelines(r'\label{tab:continued}' + '\n')
					f.writelines(r'\begin{tabular}{ccccccccccccc}' + '\n')
					f.writelines(r'\hline' + '\n')
					line_ = ['MJD']
					for item in filter_list:
						line_.append('\\textit{%s} (mag)'%item.split('.')[-1])
					if not swift:
						line_.append('data source')
					col_n = len(line_)
					f.writelines(' & '.join(line_) + line_end + '\n')
					f.writelines(r'\hline' + '\n')
					MJD_ = table_data.loc[0, 'MJD']
					tel_ = table_data.loc[0, 'tel']
					line_ = [MJD_]
					line_.extend(['...' for i in range(len(filter_list))])
					if not swift:
						line_.append(tel_)

					m_ = table_data.loc[start, 'm']
					dm_ = ('%.3f'%table_data.loc[start, 'dm']).split('.')[1]
					line_[filter_index[table_data.loc[start, 'filter']]] = '%.3f(%s)'%(m_, dm_)
					line_i = 0
					for i in range(start+1,table_len):
						m_ = table_data.loc[i, 'm']
						dm_ = ('%.3f'%table_data.loc[i, 'dm']).split('.')[1]
						if table_data.loc[i, 'MJD'] == MJD_ and table_data.loc[i, 'tel'] == tel_:
							line_[filter_index[table_data.loc[i, 'filter']]] = '%.3f(%s)'%(m_, dm_)
						else:
							line_i += 1
							if line_i == 46:
								start = i
								break
							f.writelines(' & '.join(line_) + line_end + '\n')
							MJD_ = table_data.loc[i, 'MJD']
							tel_ = table_data.loc[i, 'tel']
							line_ = [MJD_]
							line_.extend(['...' for i in range(len(filter_list))])
							if not swift:
								line_.append(tel_)
							line_[filter_index[table_data.loc[i, 'filter']]] = '%.3f(%s)'%(m_, dm_)
					f.writelines(' & '.join(line_) + line_end + '\n')
					f.writelines(r'\hline' + '\n')
					f.writelines(r'\multicolumn{5}{l}{Note: Uncertainties, in units of 0.001 mag, are $1\sigma$.}\\' + '\n')
					f.writelines(r'\end{tabular}%' + '\n')
					if table_i == 0:
						if swift:
							f.writelines(r'\label{tab:swift}%' + '\n')
						else:
							f.writelines(r'\label{tab:optical}%' + '\n')
					else:
						f.writelines(r'\label{tab:addlabel}%' + '\n')
					f.writelines(r'\end{table}%' + '\n')
					f.writelines(r'\end{landscape}' + '\n\n')
					if i == table_len - 1:
						break
					table_i += 1


def angle_complete(x):
	if abs(float(x)) < 10:
		x = '0'+str(x)
	else:
		x = str(x)
	return x

def ds9_region(reffile, target_name, outfile='ds9.reg'):
	outfile = target_name + '_results/' + outfile
	refdata = np.loadtxt(reffile, usecols=(1,2))
	with open(outfile, 'w') as f:
		f.writelines('# Region file format: DS9 version 4.1\n')
		f.writelines('global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
		f.writelines('fk5\n')
		len_ = len(refdata)
		for i in range(len_):
			f.writelines('circle(%.6f,%.6f,'%(refdata[i][0], refdata[i][1])+'7.810") # text={%d}\n'%(i+1))


def ref_star_table(BVfile, grifile, target_name, outfile='ref_star.txt'):
	outfile = target_name + '_results/' + outfile
	BVdata = np.loadtxt(BVfile, dtype='str', usecols=(1,2,3,4,6,7))
	gridata = np.loadtxt(grifile, dtype='str', usecols=(3,4,6,7,9,10))
	row_n = len(BVdata)
	col_n = len(BVdata[0])+len(gridata[0])
	data = np.ones((row_n, col_n)).astype('str')
	for i in range(row_n):
		data[i] = np.concatenate((BVdata[i], gridata[i]))
		ra_ = Angle(data[i, 0], unit='degree').hms
		ra_h = angle_complete(int(ra_[0]))
		ra_m = angle_complete(int(ra_[1]))
		ra_s = angle_complete('%.3f'%ra_[2])
		data[i,0] = '{:s}:{:s}:{:s}'.format(ra_h, ra_m, ra_s)
		dec_ = Angle(data[i, 1], unit='degree').dms
		dec_d = angle_complete(int(dec_[0]))
		dec_m = angle_complete(int(dec_[1]))
		dec_s = angle_complete('%.2f'%dec_[2])
		if dec_[0]>0:
			sign_ = '+'
		else:
			sign_ = ''
		data[i,1] = '{:s}{:s}:{:s}:{:s}'.format(sign_, dec_d, dec_m, dec_s)

	mask = [True for i in range(row_n)]
	for i in range(row_n):
		for j in range(2,col_n):
			if data[i, j] == 'INDEF':
				mask[i] = False
				break
	
	data = data[mask]
	data[:, 4] = (data[:, 2].astype('float') + data[:, 4].astype('float')).astype('str')
	data[:, 5] = np.sqrt(-data[:, 3].astype('float')**2 + data[:, 5].astype('float')**2)
	for i in range(row_n):
		data[i, 5] = '%.3f'%float(data[i,5])
	for i in range(len(data)):
		for j in range(5):
			data[i, 3+2*j] = data[i, 3+2*j].split('.')[1]

	data_change = data[:, 2:4].copy()
	data[:, 2:4] = data[:, 4:6].copy()
	data[:, 4:6] = data_change
	
	with open(outfile, 'w') as f:
		# head
		f.writelines("\\tablehead{\n")
		f.writelines("\\colhead{ID} & \\colhead{$\\alpha$} & \\colhead{$\\delta$} & \\colhead{$B$(mag) & \\colhead{$V$(mag) & \\colhead{$g$(mag) & \\colhead{$r$(mag) & \\colhead{$i$(mag) } \\\\\n")
		f.writelines("}\n")

		# data
		f.writelines("\\startdata\n")
		for i in range(len(data)):
			f.writelines("{:d} & ".format(i+1) + "{:s} & {:s} &".format(*data[i, :2]) + 
				" & ".join(['{:s}({:s})' for i in range(5)]).format(*data[i, 2:]) + "\\\\\n")
		f.writelines("\\enddata\n")

def blackbody_func(lam, A, T):
	return 1/(lam**5 * (np.exp(1/(lam*T)*1.43877688e+08)-1)) * 3.74177185e+27 * A #lam : AA; T: K



def linear_f(x, k, b):
	return k*x+b

def f_power_law(t, A, t0, p):
	#return np.power(A*(t - t0), p)
	return A*np.power(t - t0, p)

def fit_power_law(t_mjd, m, dm, t0_range, nondetection=None, band=None, mcmc=False, mark=None, DM=30, DM_err=0.15, milky_ebv=0., host_ebv=0., host_rv=3.1, texp=None):
	z = 0.0104
	fontsize_=18
	colorDict = {'c':'darkgreen', 'o':'orange',
			'g':'g', 'r':'r', 'i':'brown','u':'purple',
			'U':'mediumorchid','B':'b', 'V':'lime', 'R':'salmon', 'I':'darkgoldenrod',
			'UVOT.U':'saddlebrown','UVOT.B':'darkorange','UVOT.V':'gold',
			'UVW1':'pink','UVW2':'slateblue','UVM2':'black','z':'gray','L':'black','bolo':'black'}
	otherline = [
	 ('dashed', 'dashed'),    # Same as '--'
	 ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dashdot', 'dashdot'),  # Same as '-.'
     ('long dash with offset', (5, (10, 3))),
     ('long long dash with offset', (5, (20, 3))),
     ('densely dashed',        (0, (5, 1))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('dotted', 'dotted'),]    # Same as (0, (1, 1)) or ':'
	if band is None:
		band = 'r'
	else:
		band = band.split()[-1].split('_')[-1]
	bad_mark = np.array([True for i in range(len(m))])
	#bad_mark[3] = False
	#m[0]=20.3
	t_mjd = t_mjd[bad_mark]
	m = m[bad_mark]
	dm = dm[bad_mark]
	mark = mark[bad_mark]
	marker_size = 7.5
	#t_mjd = np.insert(t_mjd, 0, 60251.523750)
	#m = np.insert(m, 0, 20.620615347360612)
	#dm = np.insert(dm, 0, 0.3230619512195122)
	#mark = np.insert(mark, 0, True)

	#data_16abc = pd.read_csv('/Users/liujialian/work/SN_catalog/iPTF16abc_gPTF.csv')
	#data_16abc = data_16abc.rename(columns={'time':'MJD','magnitude':'m','e_magnitude':'dm'})
	#data_16abc_g = data_16abc[(data_16abc['band']=='g')*(data_16abc['dm']!=0.)]
	#wrk_g_m_max = 13.959406181403232
	#abc_g_m_max = 15.937598262751035
	#min_x, min_y = find_curve_min(data_16abc_g['MJD'], data_16abc_g['m'], data_16abc_g['dm'], 57500, Name='iPTF16abc', method='polynomial', n=5, bounds=[-14, 30], get_err=True)
	#print(min_x, min_y)
	#exit()
	
	np.random.seed(123456)
	dm[dm < 0.02] = 0.02
	color = colorDict[band]
	flux = np.power(10, -0.4*m)
	d_flux = flux*dm/1.0857
	scale_min = flux.min()
	scale_max = flux.max()
	scale_flux = flux / scale_min
	scale_d_flux = d_flux / scale_min
	t_mjd_unfit = None

	
	if mark is not None:
		t_mjd_unfit = t_mjd[~mark]
		t_mjd = t_mjd[mark]
		scale_flux_unfit = scale_flux[~mark]
		scale_flux = scale_flux[mark]
		scale_d_flux_unfit = scale_d_flux[~mark]
		scale_d_flux = scale_d_flux[mark]
	#scale_d_flux = np.ones(len(flux))
	if nondetection is not None:
		t_nondetect = nondetection[0]
		m_nondetect = nondetection[1]
		flux_nondetect = np.power(10, -0.4*m_nondetect)
		scale_flux_nondetect = flux_nondetect / scale_min
		def log_likelihood(theta, t_mjd, scale_flux, scale_d_flux):
			A, t0, p = theta
			if texp is not None:
				t0 = texp
			if f_power_law(t_nondetect, A, t0, p) > scale_flux_nondetect:
				return -np.inf
			return -0.5*np.sum(((scale_flux - f_power_law(t_mjd, A, t0, p))/scale_d_flux)**2)
	else:
		def log_likelihood(theta, t_mjd, scale_flux, scale_d_flux):
			A, t0, p = theta
			if texp is not None:
				t0 = texp
			return -0.5*np.sum(((scale_flux - f_power_law(t_mjd, A, t0, p))/scale_d_flux)**2)
	def log_prior(theta):
		A, t0, p = theta
		if 0 < A and t0_range[0] <= t0 <= t0_range[1] and 0 < p:
			return 0.0
		return -np.inf
	def log_probability(theta, t_mjd, scale_flux, scale_d_flux):
		lp = log_prior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + log_likelihood(theta, t_mjd, scale_flux, scale_d_flux)
	#A  =  scale_max / scale_min

	if texp is not None:
		t0 = texp
		n_para = 2
	else:
		n_para = 3
		if nondetection is not None:
			t0 = t0_range[-1]
		else:
			#t0 = np.mean(t0_range)
			t0 = t0_range[0]
	if nondetection is not None:
		p  = 1
	else:
		p = 2
	A = scale_flux[-1]/(t_mjd[-1] - t0)**p
	#print(t0, t_nondetect)
	#print(f_power_law(t_nondetect, A, t0, p), scale_flux_nondetect)
	#print(scale_flux)
	#exit()
	guess = [A, t0, p]
	print(guess)
	#print(guess)
	#exit()
	bounds = [(0, np.inf),(t0_range[0],t0_range[1]),(0, np.inf)]
	nll = lambda *args: -log_likelihood(*args)
	res = optimize.minimize(nll, guess, args=(t_mjd, scale_flux, scale_d_flux), bounds=bounds)
	if texp is not None:
		res.x[1] = texp
	A, t0, p = res.x
	print('least square: ', A, t0, p)
	#params, params_cov = optimize.curve_fit(f_power_law, t_mjd, scale_flux, sigma=scale_d_flux, p0=[A, t0, p])
	#params, params_cov = optimize.curve_fit(f_power_law, [-15.4,-14.4,-13.4], [-15535,-14077,-11969], sigma=[842,317,227])
	#print('curve_fit: ', params, params_cov)
	#print(np.diag(params_cov)**0.5)
	#exit()
	#print(f_power_law(t_nondetect, A, t0, p), scale_flux_nondetect)
	#print(scale_flux)
	#exit()

	phase = t_mjd - res.x[1]
	phase_unfit = t_mjd_unfit - res.x[1]
	plot_t = np.linspace(0, phase[-1] + 0.5, 100)
	if mcmc == True:
		rand_start = np.ones([32,3])
		rand_start[0] = [0, 0, 0]
		for i in range(1,32):
			rand_start[i][0] = np.random.randn()*res.x[0]*0.1
			#rand_start[i][1] = (np.random.rand()*(t0_range[1] - t0_range[0]) + t0_range[0]) - res.x[1]
			rand_start[i][1] = np.random.rand()*0.05 - 0.025 
			rand_start[i][2] = np.random.randn()*res.x[2]*0.1
		start = res.x + rand_start
		fig, ax = plt.subplots()
		ax.plot(phase, scale_flux, linestyle='', marker='o', label='data')
		ax.errorbar(phase, scale_flux, yerr=scale_d_flux, linestyle='', capsize=7, alpha=alpha)
		ax.plot(plot_t, f_power_law(plot_t + res.x[1], res.x[0], res.x[1], res.x[2]), label='fit')
		ax.legend()
		plt.show()
		print(log_probability(res.x, t_mjd, scale_flux, scale_d_flux))
		for i in range(32):
			start_log_prob_tempt = log_probability(start[i], t_mjd, scale_flux, scale_d_flux)
			if not np.isfinite(start_log_prob_tempt):
				print(i, start_log_prob_tempt, start[i])
		time.sleep(3)
		nwalkers, ndim = start.shape
		steps = 20000
		AutocorrError = emcee.autocorr.AutocorrError
		while(1):
			sampler = emcee.EnsembleSampler(
				nwalkers, ndim, log_probability, args=(t_mjd, scale_flux, scale_d_flux)
			)
			sampler.run_mcmc(start, steps, progress=True)

			
			try:
				tau = sampler.get_autocorr_time()
			except AutocorrError:
				steps *= 2
			else:
				break	
		samples = sampler.get_chain()
		fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
		labels = ['A', 't0', 'p']
		for i in range(ndim):
		    ax = axes[i]
		    ax.plot(samples[:, :, i], "k", alpha=0.3)
		    ax.set_xlim(0, len(samples))
		    ax.set_ylabel(labels[i])
		    ax.yaxis.set_label_coords(-0.1, 0.5)

		axes[-1].set_xlabel("step number")
		plt.show()

		flat_samples = sampler.get_chain(discard=int(2.5*np.max(tau)), thin=int(np.max(tau)/2), flat=True)
		print(flat_samples.shape)
		log_likelihood_value = []
		for i in range(flat_samples.shape[0]):
			log_likelihood_value.append(log_likelihood(flat_samples[i], t_mjd, scale_flux, scale_d_flux))
		max_log_likelihood = np.max(log_likelihood_value)
		for i in range(flat_samples.shape[0]):
			if log_likelihood_value[i] == max_log_likelihood:
				pos_max = i
		theta = flat_samples[pos_max]
		print(theta)
		print('best: ', log_probability(theta, t_mjd, scale_flux, scale_d_flux), log_probability(theta, t_mjd, scale_flux, scale_d_flux)*(-2)/(len(scale_flux) - n_para))
		A, t0, p = theta
		fig = corner.corner(flat_samples, labels=labels, truths=[A, t0, p])
		plt.show()
		t0_value = np.percentile(flat_samples[:,1], [16, 50, 84])
		pow_value  = np.percentile(flat_samples[:,2], [16, 50, 84])
		A_value = np.percentile(flat_samples[:,0], [16, 50, 84])
		
		pos_mid = np.argsort(np.abs(flat_samples[:,1] - t0_value[1]))[1]
		theta_mid = flat_samples[pos_mid]
		pos_small = np.argsort(np.abs(flat_samples[:,1] - t0_value[0]))[0]
		theta_small = flat_samples[pos_small]
		pos_large = np.argsort(np.abs(flat_samples[:,1] - t0_value[2]))[0]
		theta_large = flat_samples[pos_large]
		print('mid: ', log_probability(theta_mid, t_mjd, scale_flux, scale_d_flux), log_probability(theta_mid, t_mjd, scale_flux, scale_d_flux)*(-2)/(len(scale_flux) - n_para))
		print(theta_mid)
		print('small: ', log_probability(theta_small, t_mjd, scale_flux, scale_d_flux), log_probability(theta_small, t_mjd, scale_flux, scale_d_flux)*(-2)/(len(scale_flux) - n_para))
		print(theta_small)
		print('large: ', log_probability(theta_large, t_mjd, scale_flux, scale_d_flux), log_probability(theta_large, t_mjd, scale_flux, scale_d_flux)*(-2)/(len(scale_flux) - n_para))
		print(theta_large)
		print(t0_value, pow_value)
		A, t0, p = theta_mid
		theta = theta_mid
		phase = t_mjd - t0
		phase_unfit = t_mjd_unfit - t0
		plot_t = np.linspace(0, phase[-1] + 1.5, 100)
	left, width = 0.1, 0.8
	bottom, height = 0.09, 0.65
	spacing = 0.015
	rect_scatter = [left, bottom + 0.2 + spacing, width, height]
	rect_residual = [left, bottom, width, 0.2]
	fig = plt.figure(figsize=(7,7))

	ax = fig.add_axes(rect_scatter)
	#ax.set_title('%s band, $t_0$=%.2f, index=%.2f'%(band, t0, p))
	#ax.plot(60251.523750, np.power(10, -0.4*19.667) / scale_min, marker='v', linestyle='', label='last non-detection g-ZTF')
	#ax.axvline(60251.523750, linestyle='--', c='g', label='last non-detection g-ZTF')
	#ax.axvline(60252.229, linestyle='--', c='r', label='first detection L-GOTO')
	ax.tick_params(labelsize=fontsize_)
	ax.tick_params(axis="x", labelbottom=False)
	ax.set_ylabel('Scaled Flux',fontsize=fontsize_)
	#ax.text(t_mjd_unfit[0]-0.15, scale_flux_unfit[0]+0.5, 'L-GOTO')
	ax.plot(phase/(1+z), scale_flux, linestyle='', marker='o', label='SN2023wrk', c=color, zorder=999, markersize=marker_size)
	ax.errorbar(phase/(1+z), scale_flux, yerr=scale_d_flux, linestyle='', capsize=7, alpha=alpha, c=color, zorder=998)
	if nondetection is not None:
		ax.plot((t_nondetect-t0)/(1+z), scale_flux_nondetect, linestyle='', marker='v', label='limit', c=color, markersize=marker_size)
	if t_mjd_unfit is not None:
		if len(t_mjd_unfit) != 0:
			ax.plot(phase_unfit/(1+z), scale_flux_unfit, linestyle='', marker='x', label='SN2023wrk unused', c=color, markersize=marker_size)
			ax.errorbar(phase_unfit/(1+z), scale_flux_unfit, yerr=scale_d_flux_unfit, linestyle='', capsize=7, alpha=alpha, c=color)
	ax.plot(plot_t/(1+z), f_power_law(plot_t + t0, A, t0, p), label='SN2023wrk fit', c=color, zorder=995, markersize=marker_size)
	#ax.text(0.5, 0.2, '$F_{%s}\\propto(t-%.2f)^{%.2f}$'%(band, t0, p), c=color, transform = ax.transAxes, fontsize=15)
	ax.text(0.5, 0.2, '$F_{%s}\\propto(t-t_0)^{%.2f}$'%(band, p), c=color, transform = ax.transAxes, fontsize=20)
	#if mcmc == True:
	#	ax.plot(plot_t + theta_small[1], f_power_law(plot_t + theta_small[1], *theta_small), linestyle='--', c='black')
	#	ax.plot(plot_t + theta_large[1], f_power_law(plot_t + theta_large[1], *theta_large), linestyle='--', c='black')
	
	residual = (scale_flux - f_power_law(phase + t0, A, t0, p)) / f_power_law(phase + t0, A, t0, p)
	ax_residual = fig.add_axes(rect_residual)
	ax_residual.tick_params(labelsize=fontsize_)
	ax_residual.sharex(ax)
	#ax_residual.set_xlabel('Days from $B$-band Maximum',fontsize=15)
	ax_residual.set_xlabel('Days since First Light',fontsize=fontsize_)
	ax_residual.set_ylabel('Fractional Residual',fontsize=fontsize_)
	ax_residual.plot(phase/(1+z), residual, linestyle='', marker='o',c=color, markersize=marker_size)
	ax_residual.errorbar(phase/(1+z), residual, yerr=scale_d_flux / f_power_law(phase + t0, A, t0, p), linestyle='', capsize=7, alpha=alpha,c=color)
	if t_mjd_unfit is not None:
		if len(t_mjd_unfit) != 0:
			residual_unfit = (scale_flux_unfit - f_power_law(phase_unfit + t0, A, t0, p)) / f_power_law(phase_unfit + t0, A, t0, p)
			ax_residual.plot(phase_unfit/(1+z), residual_unfit, linestyle='', marker='x',c=color, markersize=marker_size)
			ax_residual.errorbar(phase_unfit/(1+z), residual_unfit, yerr=scale_d_flux_unfit / f_power_law(phase_unfit + t0, A, t0, p), linestyle='', capsize=7, alpha=alpha,c=color)
	ax_residual.plot([plot_t[0], plot_t[-1]], [0,0], markersize=marker_size)

	data_16abc = pd.read_csv('/Users/liujialian/work/SN_catalog/iPTF16abc_gPTF.csv')
	data_16abc = data_16abc.rename(columns={'time':'MJD','magnitude':'m','e_magnitude':'dm'})
	filt_ = 'g'
	data_16abc_g = data_16abc[(data_16abc['band']==filt_)*(data_16abc['dm']!=0.)]
	#wrk_g_m_max = 13.959406181403232
	#abc_g_m_max = 15.937598262751035
	#print(find_curve_min(data_16abc_g['MJD'], data_16abc_g['m'], data_16abc_g['dm'], method='polynomial', n=10))
	#data_16abc_g['m'] = data_16abc_g['m'] - 15.937598262751035 + 13.959406181403232#
	#print(- 35.08 + 33.42 - 0.0779*Rf[filt_] + 0.042*Rf[filt_])
	#exit()
	data_16abc_g['m'] = data_16abc_g['m'] - 35.08 + 33.42 - 0.0779*Rf[filt_] + 0.042*Rf[filt_]
	data_16abc_g['MJD'] = (data_16abc_g['MJD'] - 57481.209 - (t0 - 60251.28608070811))/(1.0234) #60251.28608070811 60251.47791733341
	data_16abc_g = data_16abc_g[data_16abc_g['MJD']<8] #60254.5 60258.5545

	flux_16abc = np.power(10, -0.4*data_16abc_g['m'])
	d_flux_16abc = flux_16abc*data_16abc_g['dm']/1.0857
	scale_flux_16abc = flux_16abc / scale_min
	scale_d_flux_16abc = d_flux_16abc / scale_min
	#params, params_cov = optimize.curve_fit(f_power_law, data_16abc_g['MJD'], scale_flux_16abc, sigma=scale_d_flux_16abc, p0=[A, t0, p])
	#print(params)

	ax.plot(data_16abc_g['MJD'], scale_flux_16abc, linestyle='', marker='v', label='iPTF16abc', c='r', zorder=990, markersize=marker_size)
	ax.errorbar(data_16abc_g['MJD'], scale_flux_16abc, yerr=scale_d_flux_16abc, linestyle='', capsize=7, alpha=alpha, c='r', zorder=989)
	#ax.plot(plot_t + t0, f_power_law(plot_t + t0, *params), label='iPTF16abc fit', c='r')
	#residual_16abc = (scale_flux_16abc - f_power_law(data_16abc_g['MJD']*(1+z) + t0, A, t0, p)) / f_power_law(data_16abc_g['MJD']*(1+z) + t0, A, t0, p)
	#ax_residual.plot(data_16abc_g['MJD'], residual_16abc, linestyle='', marker='v',c='r')
	#ax_residual.errorbar(data_16abc_g['MJD'], residual_16abc, yerr=scale_d_flux_16abc / f_power_law(data_16abc_g['MJD']*(1+z) + t0, A, t0, p), linestyle='', capsize=7, alpha=alpha,c='r')

	'''
	zorder=900
	P2016_Ni_CSM = pd.read_csv('/Users/liujialian/work/SN_catalog/P2016_0.1CSM_1e11_0.15Ni_lc.csv')
	P2016_Ni_CSM = P2016_Ni_CSM.rename(columns={'time':'MJD','magnitude':'m','e_magnitude':'dm'})
	P2016_Ni_CSM_g = P2016_Ni_CSM[(P2016_Ni_CSM['band']=='V')]
	#P2016_Ni_CSM_g['MJD'] = P2016_Ni_CSM_g['MJD'] * 1.0104 + 60251.47791733341#60251.28608070811 60251.47791733341
	P2016_Ni_CSM_g = P2016_Ni_CSM_g[P2016_Ni_CSM_g['MJD']<3.] #60254.5 60258.5545
	flux_P2016_Ni_CSM = np.power(10, -0.4*P2016_Ni_CSM_g['m'])
	scale_flux_P2016_Ni_CSM = flux_P2016_Ni_CSM / flux_P2016_Ni_CSM.max() * 45.44
	ax.plot(P2016_Ni_CSM_g['MJD'], scale_flux_P2016_Ni_CSM, linestyle=otherline[1][1], label='P2016_0.1CSM_1e11_0.15Ni', c='black', zorder=zorder)
	zorder+=2

	P2016_Ni = pd.read_csv('/Users/liujialian/work/SN_catalog/P2016_mix_0.25_lc.csv')
	P2016_Ni = P2016_Ni.rename(columns={'time':'MJD','magnitude':'m','e_magnitude':'dm'})
	P2016_Ni_g = P2016_Ni[(P2016_Ni['band']=='g')]
	#P2016_Ni_g['MJD'] = P2016_Ni_g['MJD'] * 1.0104 + 60251.47791733341#60251.28608070811 60251.47791733341
	P2016_Ni_g = P2016_Ni_g[P2016_Ni_g['MJD']<3.] #60254.5 60258.5545
	flux_P2016_Ni = np.power(10, -0.4*P2016_Ni_g['m'])
	scale_flux_P2016_Ni = flux_P2016_Ni / flux_P2016_Ni.max() * 46.7
	ax.plot(P2016_Ni_g['MJD'], scale_flux_P2016_Ni, linestyle=otherline[2][1], label='P2016_mix_0.25', c='darkgreen', zorder=zorder)
	zorder+=2

	M2020 = pd.read_csv('/Users/liujialian/work/SN_catalog/M2020_17cbv_0.03_0.06_lc.csv')
	M2020 = M2020.rename(columns={'time':'MJD','magnitude':'m','e_magnitude':'dm'})
	M2020_g = M2020[(M2020['band']=='g')]
	#M2020_g['MJD'] = M2020_g['MJD'] * 1.0104 + 60251.47791733341#60251.28608070811 60251.47791733341
	M2020_g = M2020_g[M2020_g['MJD']<3.] #60254.5 60258.5545
	flux_M2020 = np.power(10, -0.4*M2020_g['m'])
	scale_flux_M2020 = flux_M2020 / flux_M2020.max() * 43
	ax.plot(M2020_g['MJD'], scale_flux_M2020, linestyle='dotted', label='M2020_17cbv_0.03_0.06', c='blue', zorder=zorder)
	zorder+=2
	'''

	ax.legend(fontsize=14.5)
	plt.savefig('Power_fit_%s.pdf'%band,bbox_inches='tight')
	plt.show()
	if mcmc == False:
		t0_value = t0
		pow_value = p
	#save data
	if os.path.exists('power_law.csv'):
		power_law_data = pd.read_csv('power_law.csv')
	else:
		power_law_data = pd.DataFrame(columns=['MJD','phase','abs_m','dm','powerlaw_abs_m','DM_err','band'])
	if texp is not None: 
		MJD = np.concatenate([t_mjd_unfit, t_mjd])
		phase = MJD - texp
		abs_m = m - DM - milky_ebv * Rf[band] * 3.1/3.011 + Rf[band]*host_rv/3.011*host_ebv
		powerlaw_abs_m = -2.5*np.log10(scale_min * f_power_law(phase + texp, A, t0, p)) - DM - milky_ebv * Rf[band] * 3.1/3.011 + Rf[band]*host_rv/3.011*host_ebv
		new_data = pd.DataFrame({'MJD':MJD,'phase':phase,'abs_m':abs_m,'dm':dm,'powerlaw_abs_m':powerlaw_abs_m})
		new_data['DM_err'] = DM_err
		new_data['band'] = band
		power_law_data = pd.concat([power_law_data, new_data])
		power_law_data.to_csv('power_law.csv', index=False)
	return t0_value, pow_value

def mu_to_d(mu):
	return 10**(0.2*mu)*1e-5

def d_to_mu(d):
	#d in Mpc
	return 5*np.log10(d*1e5)

def SDSS_to_Johnson(m_SDSS):
	filter_SDSS = m_SDSS.keys()
	filter_flag = {'umag':False,'gmag':False,'rmag':False,'imag':False,'zmag':False,'Umag':False,'Bmag':False,'Vmag':False,'Rmag':False,'Imag':False}
	for filter_ in filter_SDSS:
		filter_flag[filter_] = True

	#if filter_flag['umag']*filter_flag['gmag']:
	#	m_SDSS['Bmag'], m_SDSS['e_Bmag'] = cal_trans(m_SDSS['gmag'], m_SDSS['e_gmag'], 0.175, 0.002, m_SDSS['umag'], m_SDSS['gmag'], m_SDSS['e_umag'], m_SDSS['e_gmag'], 0.15, 0.003)
	#	filter_flag['Bmag'] = True
	if filter_flag['gmag']*filter_flag['rmag']:
		m_SDSS['Bmag'], m_SDSS['e_Bmag'] = cal_trans(m_SDSS['gmag'], m_SDSS['e_gmag'], 0.313, 0.003, m_SDSS['gmag'], m_SDSS['rmag'], m_SDSS['e_gmag'], m_SDSS['e_rmag'], 0.219, 0.002)
		filter_flag['Bmag'] = True

	#if filter_flag['Bmag']*filter_flag['umag']:
	#	m_SDSS['Umag'], m_SDSS['e_Umag'] = cal_trans(m_SDSS['gmag'], m_SDSS['e_gmag'], 0.965, 0.0201, m_SDSS['umag'], m_SDSS['gmag'], m_SDSS['e_umag'], m_SDSS['e_gmag'], -0.78, 0.0202)
	#	filter_flag['Umag'] = True

	if filter_flag['Bmag']*filter_flag['umag']:
		m_SDSS['Umag'], m_SDSS['e_Umag'] = cal_trans(m_SDSS['Bmag'], m_SDSS['e_Bmag'], 0.79, 0.02, m_SDSS['umag'], m_SDSS['gmag'], m_SDSS['e_umag'], m_SDSS['e_gmag'], -0.93, 0.02)
		filter_flag['U'] = True

	if filter_flag['gmag']*filter_flag['rmag']:
		m_SDSS['Vmag'], m_SDSS['e_Vmag'] = cal_trans(m_SDSS['gmag'], m_SDSS['e_gmag'], -0.565, 0.001, m_SDSS['gmag'], m_SDSS['rmag'], m_SDSS['e_gmag'], m_SDSS['e_rmag'], -0.016, 0.001)
		filter_flag['Vmag'] = True

	if filter_flag['rmag']*filter_flag['imag']:
		m_SDSS['Rmag'], m_SDSS['e_Rmag'] = cal_trans(m_SDSS['rmag'], m_SDSS['e_rmag'], -0.153, 0.003, m_SDSS['rmag'], m_SDSS['imag'], m_SDSS['e_rmag'], m_SDSS['e_imag'], -0.117, 0.003)
		filter_flag['Rmag'] = True

	if filter_flag['imag']*filter_flag['zmag']:
		m_SDSS['Imag'], m_SDSS['e_Imag'] = cal_trans(m_SDSS['imag'], m_SDSS['e_imag'], -0.386, 0.004, m_SDSS['imag'], m_SDSS['zmag'], m_SDSS['e_imag'], m_SDSS['e_zmag'], -0.397, 0.001)
		filter_flag['Imag'] = True
	return filter_flag

def SDSS_to_Johnson_(m_SDSS):
	filter_SDSS = m_SDSS.keys()
	filter_flag = {'u':False,'g':False,'r':False,'i':False,'z':False,'U':False,'B':False,'V':False,'R':False,'I':False}
	for filter_ in filter_SDSS:
		filter_flag[filter_] = True

	#if filter_flag['u']*filter_flag['g']:
	#	m_SDSS['B'], m_SDSS['e_B'] = cal_trans(m_SDSS['g'], m_SDSS['g_err'], 0.175, 0.002, m_SDSS['u'], m_SDSS['g'], m_SDSS['u_err'], m_SDSS['g_err'], 0.15, 0.003)
	#	filter_flag['B'] = True
	if filter_flag['g']*filter_flag['r']:
		m_SDSS['B'], m_SDSS['B_err'] = cal_trans(m_SDSS['g'], m_SDSS['g_err'], 0.313, 0.003, m_SDSS['g'], m_SDSS['r'], m_SDSS['g_err'], m_SDSS['r_err'], 0.219, 0.002)
		filter_flag['B'] = True

	#if filter_flag['B']*filter_flag['u']:
	#	m_SDSS['U'], m_SDSS['U_err'] = cal_trans(m_SDSS['g'], m_SDSS['g_err'], 0.965, 0.0201, m_SDSS['u'], m_SDSS['g'], m_SDSS['u_err'], m_SDSS['g_err'], -0.78, 0.0202)
	#	filter_flag['U'] = True

	if filter_flag['B']*filter_flag['u']:
		m_SDSS['U'], m_SDSS['U_err'] = cal_trans(m_SDSS['B'], m_SDSS['B_err'], 0.79, 0.02, m_SDSS['u'], m_SDSS['g'], m_SDSS['u_err'], m_SDSS['g_err'], -0.93, 0.02)
		filter_flag['U'] = True

	if filter_flag['g']*filter_flag['r']:
		m_SDSS['V'], m_SDSS['V_err'] = cal_trans(m_SDSS['g'], m_SDSS['g_err'], -0.565, 0.001, m_SDSS['g'], m_SDSS['r'], m_SDSS['g_err'], m_SDSS['r_err'], -0.016, 0.001)
		filter_flag['V'] = True

	if filter_flag['r']*filter_flag['i']:
		m_SDSS['R'], m_SDSS['R_err'] = cal_trans(m_SDSS['r'], m_SDSS['r_err'], -0.153, 0.003, m_SDSS['r'], m_SDSS['i'], m_SDSS['r_err'], m_SDSS['i_err'], -0.117, 0.003)
		filter_flag['R'] = True

	if filter_flag['i']*filter_flag['z']:
		m_SDSS['I'], m_SDSS['I_err'] = cal_trans(m_SDSS['i'], m_SDSS['i_err'], -0.386, 0.004, m_SDSS['i'], m_SDSS['z'], m_SDSS['i_err'], m_SDSS['z_err'], -0.397, 0.001)
		filter_flag['I'] = True
	return filter_flag

		
def cal_trans(f0, f0_err, c, c_err, f1, f2, f1_err, f2_err, z, z_err):
	mag_trans = f0 + c*(f1 - f2) + z
	#mag_trans_err = np.sqrt((c1*f1_err)**2 + (c_err*f1)**2 + (c2*f2_err)**2 + (c_err*f2)**2 + z_err**2)
	mag_trans_err = np.sqrt(f0_err**2 + (c_err*(f1-f2))**2 + c**2*(f1_err**2 + f2_err**2) + z_err**2)
	return mag_trans, mag_trans_err

def get_color(t1, m1, dm1, t2, m2, dm2, t, method='hyperspline', n=10, Nboot=50):
	if method == 'polynomial':
		spline_f1 = fit1dcurve.Interpolator(type='polynomial',x=t1,y=m1,dy=dm1,n=n)
		spline_f2 = fit1dcurve.Interpolator(type='polynomial',x=t2,y=m2,dy=dm2,n=n)
	else:
		spline_f1 = fit1dcurve.Interpolator(type=method,x=t1,y=m1,dy=dm1,)
		spline_f2 = fit1dcurve.Interpolator(type=method,x=t2,y=m2,dy=dm2,)
	color = spline_f1(t)[0] - spline_f2(t)[0]
	if type(color) == np.float64:
		color = np.array([color])
	np.random.seed(123456)
	color_list = [color]
	for i in range(Nboot):
		spline_f1.draw()
		spline_f2.draw()
		'''
		if method == 'polynomial':
			spline_f1 = fit1dcurve.Interpolator(type='polynomial',x=t1,y=m1,dy=dm1,n=n)
			spline_f2 = fit1dcurve.Interpolator(type='polynomial',x=t2,y=m2,dy=dm2,n=n)
		else:
			spline_f1 = fit1dcurve.Interpolator(type=method,x=t1,y=m1,dy=dm1,)
			spline_f2 = fit1dcurve.Interpolator(type=method,x=t2,y=m2,dy=dm2,)
		'''
		color_ = spline_f1(t)[0] - spline_f2(t)[0]
		if type(color_) == np.float64:
			color_ = np.array([color_])
		color_list.append(color_)
	color_list = np.array(color_list)
	color_err = []
	for i in range(len(color)):
		color_err.append(color_list[:, i].std(ddof=0))
	return np.array([color, color_err])

def find_curve_min(x, y, dy, guess, Name=None, method='hyperspline', n=10, bounds=[-14, 30], get_err=False, Nboot=50, plot=False):
	mark = (x>guess+bounds[0])*(x<guess+bounds[1])
	len_data = len(x[mark])
	#if Name=='iPTF16abc':
	#	spline_f = fit1dcurve.Interpolator(type='polynomial',x=x[mark],y=y[mark],dy=dy[mark],n=5)
	#	plot_x = np.linspace(57485,57530, 100)
	#	plt.plot(plot_x, spline_f(plot_x)[0])
	#	plt.plot(x[mark], y[mark], marker='o', linestyle='')
	#	plt.show()
	#	min_x = optimize.minimize(lambda x: spline_f(x)[0], guess, bounds=[(guess+bounds[0], guess+bounds[1])]).x[0]
	#	min_y = spline_f(min_x)[0]
	#	print(min_x,min_y)
	#	exit()
	if method == 'polynomial':
		spline_f = fit1dcurve.Interpolator(type='polynomial',x=x[mark],y=y[mark],dy=dy[mark],n=n)
	else:
		spline_f = fit1dcurve.Interpolator(type=method,x=x[mark],y=y[mark],dy=dy[mark])
	
	min_x = optimize.minimize(lambda x: spline_f(x)[0], guess, bounds=[(guess+bounds[0], guess+bounds[1])]).x[0]
	min_y = spline_f(min_x)[0]
	
	if Name is None:
		fig, ax = plt.subplots()
		ax.plot(x[mark], y[mark], linestyle='', marker='o', c='r')
		ax.plot(np.linspace(bounds[0]+guess,bounds[1]+guess,100), spline_f(np.linspace(bounds[0]+guess,bounds[1]+guess,100))[0], c='b')
		plt.show()
		exit()
	if plot == True:
		fig, ax = plt.subplots()
		ax.plot(x[mark], y[mark], linestyle='', marker='o', c='r')
		ax.plot(np.linspace(bounds[0]+guess,bounds[1]+guess,100), spline_f(np.linspace(bounds[0]+guess,bounds[1]+guess,100))[0], c='b')
		plt.show()


	if get_err == True:
		min_x_list = []
		min_y_list = []
		np.random.seed(123456)
		for i in range(Nboot):
			if method == 'polynomial':
				spline_f = fit1dcurve.Interpolator(type='polynomial',x=x[mark],y=y[mark]+np.random.normal(y[mark], dy[mark]),dy=dy[mark],n=n)
			else:
				spline_f = fit1dcurve.Interpolator(type=method,x=x[mark],y=y[mark]+np.random.normal(y[mark], dy[mark]),dy=dy[mark])
			min_x_ = optimize.minimize(lambda x: spline_f(x)[0], guess, bounds=[(guess+bounds[0], guess+bounds[1])]).x[0]
			min_y_ = spline_f(min_x)[0]
			min_x_list.append(min_x_)
			min_y_list.append(min_y_)
		min_x_err = np.std(min_x_list)
		min_y_err = np.std(min_y_list)
		min_x = [min_x, min_x_err]
		min_y = [min_y, min_y_err]

	#print(min_x, min_y)
	#exit()
	return [min_x, min_y]

def get_m_Ni56(L_max, dL_max, t_rise, dt_rise):
	#L_max 10**43 erg/s
	#t_rise day
	k = 6.45*np.exp(-(t_rise/8.8)) + 1.45*np.exp(-(t_rise/111.3))
	dk = (6.45/8.8*np.exp(-(t_rise/8.8)) + 1.45/111.3*np.exp(-(t_rise/111.3)))*dt_rise
	m_Ni56 = L_max/k #M_sun
	dm_Ni56 = m_Ni56 * np.sqrt((dk/k)**2 + (dL_max/L_max)**2)
	return m_Ni56, dm_Ni56

def get_LC_fitting(LC_file, DM, milky_ebv, redshift, filters, host_ebv=0., host_rv=3.1):
	lc = LC.read(LC_file)
	lc.meta['dm'] = DM
	lc.meta['ebv'] = milky_ebv
	lc.meta['host_ebv'] = host_ebv
	lc.meta['redshift'] = redshift
	extinction = {}
	for filter_ in filters:
		extinction[filter_] = milky_ebv * Rf[filter_] * 3.1/3.011 + Rf[filter_]*host_rv/3.011*host_ebv
	lc.meta['extinction'] = extinction
	lc.calcAbsMag()
	return lc

def get_bb_bolo(lc, filter0='i',filter1='B',save=None, overwrite=False, bk_file = 't_temp.dat', z=0, band_require=None, res=0.25, plot_para='T'):
	if save is None:
		save = './bb_bolo'
	if os.path.exists(bk_file) and not overwrite:
		t = LC.read(bk_file)
	else:
		t = calculate_bolometric(lc, outpath=save, z=lc.meta['redshift'], save_table_as=bk_file, filter0=filtdict[filter0], filter1=filtdict[filter1], res=res)
	#plot_bolometric_results(t)
	#plt.show()
	if band_require is not None:
		mark = (t['dtemp']/t['temp'] < 1)*(t['filts']==band_require)
	else:
		mark = (t['dtemp']/t['temp'] < 1)
	###other
	cow_data = np.loadtxt('2018cow_sedfitresult.dat',skiprows=1)
	z_cow = 0.014
	t0_cow = 58284.34
	cow_data = cow_data[(cow_data[:,0] - t0_cow)/(1+z_cow) < 40]
	###
	if plot_para == 'T':
		plt.errorbar((t['MJD'][mark]-60578.75)/(1+z), t['temp_mcmc'][mark]*1e3, yerr=[t['dtemp0'][mark]*1e3,t['dtemp1'][mark]*1e3], capsize=7,  alpha=0.3, c='r')
		plt.plot((t['MJD'][mark]-60578.75)/(1+z), t['temp_mcmc'][mark]*1e3, marker='o', linestyle='', markersize=7.5, label='AT2024wpp', c='r')
		#plt.errorbar((t['MJD'][mark]-60578.75)/(1+z), t['temp'][mark]*1e3, yerr=t['dtemp'][mark]*1e3, capsize=7,  alpha=0.3, c='r')
		#plt.plot((t['MJD'][mark]-60578.75)/(1+z), t['temp'][mark]*1e3, marker='o', linestyle='', markersize=7.5, label='AT2024wpp', c='r')
		plt.errorbar((cow_data[:,0]-t0_cow)/(1+z_cow), cow_data[:,7], yerr=cow_data[:,8], capsize=7, alpha=0.3, c='b')
		plt.plot((cow_data[:,0]-t0_cow)/(1+z_cow), cow_data[:,7], marker='o', linestyle='', markersize=7.5, label='AT2018cow', c='b')
		plt.ylabel('Temperature [K]')
		save_fig = 'SN2024wpp_results/T_compare.pdf'
	elif plot_para == 'L':
		plt.errorbar((t['MJD'][mark]-60578.75)/(1+z), t['L_bol_mcmc'][mark]*1e7, yerr=[t['dL_bol_mcmc0'][mark]*1e7,t['dL_bol_mcmc1'][mark]*1e7], capsize=7, alpha=0.3, c='r')
		plt.plot((t['MJD'][mark]-60578.75)/(1+z), t['L_bol_mcmc'][mark]*1e7, marker='o', linestyle='', markersize=7.5, label='AT2024wpp', c='r')
		plt.errorbar((cow_data[:,0]-t0_cow)/(1+z_cow), cow_data[:,1], yerr=cow_data[:,2], capsize=7, alpha=0.3, c='b')
		plt.plot((cow_data[:,0]-t0_cow)/(1+z_cow), cow_data[:,1], marker='o', linestyle='', markersize=7.5, label='AT2018cow', c='b')
		plt.ylabel('Blackbody luminosity [erg s$^{-1}$]',fontsize=15)
		plt.yscale('log')
		save_fig = 'SN2024wpp_results/bol_compare.pdf'
	plt.xlabel('Days from First Light',fontsize=15)
	plt.legend()
	plt.savefig(save_fig, bbox_inches='tight')
	plt.show()
	exit()
	return t

def SiFTO_to_dm15(sb):
	sb_ = sb - 1
	return 1. - 1.63*sb_ + 2.03*sb_*sb_ - 1.82*sb_*sb_*sb_



		
		

