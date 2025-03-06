import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import extinction
import scipy.optimize as optimize
import scipy.integrate as itg
import scipy.signal as signal
import matplotlib.collections as collections
import emcee
import corner
#from snpy import Vega


#from speclite import filters
from myTools.filters import get_zp, get_response, get_Vega
from astropy.io import fits
from myTools.timeTransform import *
from scipy.interpolate import interp1d
from myTools.GalDust import coordsToEbv
from snpy.utils import fit1dcurve

plt.rcParams["font.family"] = 'Times New Roman'
markersize = 7.5

class SpectraData:
	'''
	Instruction:
	Contains spectra data and some methods to deal with the data.

	Parameter:
	SpectraDir: The absolute directory containing the spectra data
	'''

	def __init__(self, name, SpectraDir, z=None, Rv=3.1, EBVmilky=0., Tmax=0., rebin=2.0, basicfile=None, EBVhost=0., RVhost=3.1, rebin_dict={}):
		self.name = name
		if basicfile is not None:
			basic_para = pd.read_csv(basicfile)
			ra = basic_para.loc[0, 'ra']
			dec = basic_para.loc[0, 'dec']
			self.z = basic_para.loc[0, 'redshift']
			self.Tmax = timeToMjd(basic_para.loc[0, 'maxdate'].replace('/',''))
			self.EBVmilky = coordsToEbv(ra, dec)
		else:
			self.z   = z
			self.EBVmilky = EBVmilky
			self.Tmax = Tmax
		self.EBVhost = EBVhost
		self.RVhost = RVhost
		self.Rv  = Rv
		print(self.EBVmilky, self.EBVmilky*self.Rv)
		print(self.z)
		self.feature_endpoints = None
		self.velocities = {}
		self.other_spectraDf = pd.DataFrame({})
		self.max_index = None
		self.tel_dict = {'Xinglong  2.16m':'XLT','Other':'Other','2m4-01':'LJT',
			'1.22m Reflector':'Pennar1.22','1.82m Reflector':'Copernico','shane':'Shane','NOT':'NOT'}
		self.inst_dict_none = {'Shane':'Kast'}
		self.inst_dict = {'BFOSC':'BFOSC','Other':'Other','yf01':'YFOSC','ALFOSC_FASU':'ALFOSC_FASU',
			'Andor iDus DU440A-BU2':'B&C','AFOSC + Andor iKon-L DZ936N-BEX2-DD-9HF':'AFOSC','FOSC-ES32':'FOSC-ES32'}

		self.Na_EBV = {}


		dataFormat = ['txt', 'dat', 'flm', 'spec', 'ascii', 'xlsx', 'ecsv', 'mrt', 'd'] # File of these formats will be recognized as spectra data files.

		# List to contain spectra data and corresponding parameters
		spectra = []
		obsDate = []
		mjd = []
		tel = []
		expt = []
		inst = []
		#units = []

		# Find spectra data files in the input directory and store the data
		for curDir, dirs, files in os.walk(SpectraDir):
			# Check whether exists a fits file
			if 'fits' in ''.join(map(lambda x: x.split('.')[-1], files)):
				for item in files:
					if item.split('.')[-1] in dataFormat:
						spectra_data_ = np.loadtxt(curDir + '/' + item).T
						if len(spectra_data_) == 2:
							flux_err_ = np.ones(len(spectra_data_[1]))*np.median(spectra_data_[1])*0.1
							spectra_data_ = np.append(spectra_data_, [flux_err_], axis=0)
						spectra.append(spectra_data_)
					elif item.split('.')[-1] == 'fits':
						hdul = fits.open(curDir + '/' + item)
						hdr = hdul[0].header
						if 'JD' in list(hdr):
							date_ = jdToDate(float(hdr['JD']))
						elif 'DATE_OBS' in list(hdr):
							date_ = hdr['DATE_OBS']
						else:
							date_ = hdr['DATE-OBS']

						obsDate.append(date_)
						mjd.append(timeToMjd(date_))
						try:
							tel.append(hdr['TELESCOP'])
						except:
							tel.append('what?')
						try:
							expt.append(hdr['EXPTIME'])
						except:
							expt.append('?')
						try:
							inst.append(hdr['INSTRUME'])
						except:
							inst.append('what?')
						'''
						if 'BUNIT' in list(hdr.keys()):
							units.append(hdr['BUNIT'])
						else:
							print(hdr['DATE-OBS'])
							units.append('None')
						'''
						hdul.close()


			# Check whether exists a csv file
			elif 'csv' in ''.join(map(lambda x: x.split('.')[-1], files)):
				if os.path.exists(curDir + '/.DS_Store'):
					os.remove(curDir + '/.DS_Store')
				for item in files:
					if item.split('.')[-1] == 'csv':
						df = pd.read_csv(curDir + '/' + item)
						for i in range(len(df)):
							date_ = 'T'.join(df['Obs-date'][i].split())
							obsDate.append(date_)
							mjd.append(df['JD'][i]-2400000.5)
							tel.append(df['Telescope'][i])
							expt.append(df['Exp-time'][i])
							inst.append(df['Instrument'][i])
							#units.append(df['Spec. units'])
				#print(len(obsDate))
				#print(len(files))
				#exit()
				good = 0
				for i in range(len(files)-1, 0, -1):
					#print(i)
					for item in files:
						data_format = item.split('.')[-1]
						if data_format in dataFormat:
							if len(item.split('_')) == 5:
								JD_ = float(item.split('_')[1])
								date_, hms_ = jdToDate(JD_).split('T')
							elif item.split('_')[0] == 'SN':
								date_ = item.split('_')[2]
								hms_ = item.split('_')[3].replace('-',':')
							else:
								date_ = item.split('_')[1]
								hms_ = item.split('_')[2].replace('-',':')
							#print(obsDate)
							#print(date_,hms_,obsDate[-i].split('T')[0],obsDate[-i].split('T')[1])
							if date_ == obsDate[-i].split('T')[0]:
								#print(date_,hms_,obsDate[-i].split('T')[0],obsDate[-i].split('T')[1]) 
								if hms_.split('.')[0][:-1] in obsDate[-i].split('T')[1]:
									#print(1)
									skiprows=0
									if 'DESI' in item:
										skiprows=72
									spectra_data_ = np.loadtxt(curDir + '/' + item, skiprows=skiprows).T
									mark_nan = np.isnan(spectra_data_[1])
									spectra_data_ = spectra_data_[:, ~mark_nan] 
									if spectra_data_[1][0] > 1e-3:
										if '_LT_' in item:
											spectra_data_[1] = 10**spectra_data_[1]
										else:
											spectra_data_[1] = spectra_data_[1]*1e-15
									#spectra.append(np.loadtxt(curDir + '/' + item).T)
									if len(spectra_data_) == 2:
										flux_err_ = np.ones(len(spectra_data_[1]))*np.median(spectra_data_[1])*0.1
										spectra_data_ = np.append(spectra_data_, [flux_err_], axis=0)
									spectra.append(spectra_data_)
									good += 1
									#print(good, i)
									break

		#rebin
		if rebin is not None:
			for spectra_i in range(len(spectra)):
				if '%.2f'%mjd[spectra_i] in rebin_dict.keys():
					rebin_ = rebin_dict['%.2f'%mjd[spectra_i]]
				else:
					rebin_ = rebin
				#print('rebin_')
				xlist = spectra[spectra_i][0].copy()
				ylist = spectra[spectra_i][1].copy()
				if len(spectra[spectra_i]) > 2:
					y_errlist = spectra[spectra_i][2].copy()
				else:
					y_errlist = np.ones(len(xlist))*np.mean(ylist)*0.1
				bin_size_ = rebin_*(len(xlist)-1)/(xlist[-1]-xlist[0])
				#print(bin_size_, len(xlist), xlist[-1]-xlist[0])
				if bin_size_ > 1.5:
					if bin_size_ != int(bin_size_):
						bin_size = int(bin_size_) +1
					else:
						bin_size = int(bin_size_)
					xlist_len = len(xlist)
					left_number = xlist_len%bin_size
					if left_number != 0:
						xlist = np.delete(xlist,np.arange(left_number))
						ylist = np.delete(ylist,np.arange(left_number))
						y_errlist = np.delete(y_errlist,np.arange(left_number))
					xlist_ = np.zeros(int(xlist_len/bin_size))
					ylist_ = np.zeros(int(xlist_len/bin_size))
					y_errlist_ = np.zeros(int(xlist_len/bin_size))
					for bin_i in range(bin_size):
						xlist_ = xlist_ + xlist[bin_i::bin_size]
						ylist_ = ylist_ + ylist[bin_i::bin_size]
						y_errlist_ = y_errlist_ + y_errlist[bin_i::bin_size]**2
					spectra[spectra_i] = np.array([xlist_/bin_size, ylist_/bin_size, np.sqrt(y_errlist_)/bin_size])

		# Store the data in a pandas DataFrame
		phase = (np.array(mjd) - self.Tmax)/(1+self.z)
		print(len(obsDate),len(spectra))
		self.spectraDf = pd.DataFrame({
			'obsDate': obsDate,
			'spectra': spectra,
			'MJD':mjd,
			'phase':phase,
			'tel':tel,
			'inst':inst,
			'expt':expt
			})


		self.spectraDf = self.spectraDf.sort_values(by='MJD').reset_index(drop=True)
		

		# Clear unused arrays
		spectra = []
		obsDate = []

	def get_other_spectra(self, file_Dir, rebin=2.0):
		read_extension = ['flm','txt','ascii','dat','spec','mrt','d','csv']
		data_file = os.listdir(file_Dir)
		other_name = []
		other_ebv = []
		other_rv = []
		other_ebvhost = []
		other_rvhost = []
		other_z = []
		other_index = []
		other_phase = []
		other_scale = []
		spectra = []
		if file_Dir[-1] != '/':
			file_Dir = file_Dir + '/'
		for file in data_file:
			if file.split('.')[-1] in read_extension:
				abs_dir = file_Dir+file
				lines = None
				with open(abs_dir, 'r') as f:
					other_name.append(f.readline().split()[1])
					other_ebv.append(float(f.readline().split()[1]))
					other_rv.append(float(f.readline().split()[1]))
					#other_ebvhost.append(float(f.readline().split()[1]))
					#other_rvhost.append(float(f.readline().split()[1]))
					other_ebvhost.append(0.)
					other_rvhost.append(3.1)
					other_z.append(float(f.readline().split()[1]))
					other_index.append(int(f.readline().split()[1]))
					phase_ = float(f.readline().split()[1])/(1+other_z[-1])
					if phase_ < 0:
						phase_ = '%.1f'%phase_
					else:
						phase_ = '+%.1f'%phase_
					other_phase.append(phase_)
					#other_phase.append(float(f.readline().split()[1]))
					line_data = f.readline()
				with open(abs_dir, 'r') as f:
					if ',' in line_data:
						lines = f.readlines()
						for line_i, line_ in enumerate(lines):
							lines[line_i] = lines[line_i].replace(',', ' ')
				if lines is not None:
					with open(abs_dir, 'w') as f:
						f.writelines(lines)
				spectra_data_ = np.loadtxt(abs_dir)[:,0:2].T
				if spectra_data_[1][0] > 1e-3:
					if '_LLT_' in file:
						spectra_data_[1] = 10**spectra_data_[1]
					else:
						spectra_data_[1] = spectra_data_[1]*1e-15
				#spectra.append(np.loadtxt(curDir + '/' + item).T)
				spectra_data_mark = spectra_data_[1] > 0
				spectra_data_ = np.array([spectra_data_[0][spectra_data_mark], spectra_data_[1][spectra_data_mark]])

				#rebin

				if rebin is not None:
					xlist = spectra_data_[0].copy()
					ylist = spectra_data_[1].copy()
					bin_size_ = rebin*100/(xlist[101]-xlist[1])
					if bin_size_ > 1.5:
						if bin_size_ != int(bin_size_):
							bin_size = int(bin_size_) +1
						else:
							bin_size = int(bin_size_)
						xlist_len = len(xlist)
						left_number = xlist_len%bin_size
						if left_number != 0:
							xlist = np.delete(xlist,np.arange(left_number))
							ylist = np.delete(ylist,np.arange(left_number))
						xlist_ = np.zeros(int(xlist_len/bin_size))
						ylist_ = np.zeros(int(xlist_len/bin_size))
						for bin_i in range(bin_size):
							xlist_ = xlist_ + xlist[bin_i::bin_size]
							ylist_ = ylist_ + ylist[bin_i::bin_size]
						spectra_data_ = np.array([xlist_/bin_size, ylist_/bin_size])  
				spectra.append(spectra_data_)
		
		other_spectra = pd.DataFrame({'name':other_name,
									'ebv':other_ebv,
									'rv':other_rv,
									'ebvhost':other_ebvhost,
									'rvhost':other_rvhost,
									'z':other_z,
									'index':other_index,
									'phase':other_phase,
									'Spectra':spectra})
		if self.other_spectraDf.empty:
			self.other_spectraDf = other_spectra
		else:
			self.other_spectraDf = pd.concat([self.other_spectraDf, other_spectra])
		
	
	def spec_calibartion(self, index, PPlotter, band, interpolate_method = 'gp', filters_dict=None, interpolate_region=None):
		if filters_dict is None:
			filters_dict = {'U':'Generic_Bessell.U+Vega', 'B':'Generic_Bessell.B+Vega', 'V':'Generic_Bessell.V+Vega', 'R':'Generic_Bessell.R+Vega', 'I':'Generic_Bessell.I+Vega',
							'u':'SLOAN_SDSS.u+AB', 'g':'SLOAN_SDSS.g+AB', 'r':'SLOAN_SDSS.r+AB', 'i':'SLOAN_SDSS.i+AB', 'z':'SLOAN_SDSS.z+AB',}
			band_ = band.split('_')[-1].split()[-1]
		else:
			band_ = band
		Vega_wave, Vega_flux = get_Vega()
		Vega_interp_ = interp1d(Vega_wave, Vega_flux, kind='linear',bounds_error=False,fill_value=0.0)
		#sdss = filters.load_filters('sdss2010noatm-*')
		#sdss = filters.load_filters('sdss2010-*')
		#filters.plot_filters(sdss, wavelength_limits=(3000, 11000))
		#plt.show()
		#exit()
		#bess = filters.load_filters('bessell-*')
		#sdss_dict = {'u':0,'g':1,'r':2,'i':3,'z':4}
		#bess_dict = {'U':0,'B':1,'V':2,'R':3,'I':4}
		#bess_zero = {'U':0.79,'B':-0.102,'V':0.008,'R':0.193,'I':0.443}
		MJD = self.spectraDf.loc[index, 'MJD']
		if band not in PPlotter.interpolate.keys():
			PPlotter.interpolate_lc(band, method = interpolate_method, interpolate_region=interpolate_region)
		#if PPlotter.interpolate_method[band] == 'gp':
		#	gp_jd, gp_flux, gp_flux_errors, gp_ws = PPlotter.interpolate[band].predict(x_pred=[MJD], returnv=True)
		#	inter_color2, inter_color2_err = PPlotter.interpolate[band]([MJD])
		#	m_phot = -2.5*np.log10(gp_flux[0][0])+23.9
		#	dm_phot = gp_flux_errors[0][0]/gp_flux[0][0]*1.0857
		#else:
		m_phot, dm_phot = PPlotter.interpolate[band]([MJD])
		#test
		Vega_AB = {'U': 0.814991770746154, 'B': -0.10398310727753568, 'V': 0.005847314234088685, 'R': 0.188308577547879, 'I': 0.4367023749120875, 
		'u': 0.9476655871364414, 'g': -0.10235568752400326, 'r': 0.13877328046277881, 'i': 0.3533331159596216, 'z': 0.5251741178070901}
		'''
		for band_test in ['U','B','V','R','I','u','g','r','i','z']:
			filter_name, filter_sys = filters_dict[band_test].split('+')
			filter_sys = 'AB'
			wave_, resp_ = get_response(filter_name+'.dat')
			nu_resp_ = 3e10/(wave_/1e8)[::-1]
			resp_interp_ = interp1d(nu_resp_, resp_[::-1], kind='linear',bounds_error=False,fill_value=0.0)
			nu_spec_ = 3e10/(Vega_wave*1e-8)[::-1]
			flux_nu_ = Vega_flux[::-1]*1e8*3e10/nu_spec_**2
			flux_nu_interp_ = interp1d(nu_spec_, flux_nu_, kind='linear',bounds_error=False,fill_value=0.0)
			flux_fliter = itg.quad(lambda x: flux_nu_interp_(x)*resp_interp_(x), nu_resp_[0], nu_resp_[-1])[0]
			flux_response = itg.quad(lambda x: resp_interp_(x), nu_resp_[0], nu_resp_[-1])[0]
			flux_ratio = flux_fliter/flux_response
			m_spec = -2.5*np.log10(flux_ratio) - 48.6
			Vega_AB[band_test] = m_spec
		print(Vega_AB)
		exit()
		'''

		#if band_ in sdss_dict.keys():
		if band_ in filters_dict.keys():
			filter_name, filter_sys = filters_dict[band_].split('+')
			wave_, resp_ = get_response(filter_name+'.dat')
			zp_ = get_zp(filter_name, filter_sys)
			if filter_sys == 'AB':
				#resp_ = sdss[sdss_dict[band_]].response
				#nu_resp_ = 3e10/(sdss[sdss_dict[band_]]._wavelength/1e8)[::-1]
				nu_resp_ = 3e10/(wave_/1e8)[::-1]
				resp_interp_ = interp1d(nu_resp_, resp_[::-1], kind='linear',bounds_error=False,fill_value=0.0)
				nu_spec_ = 3e10/(self.spectraDf.loc[index, 'spectra'][0]*1e-8)[::-1]
				flux_nu_ = self.spectraDf.loc[index, 'spectra'][1][::-1]*1e8*3e10/nu_spec_**2
				flux_nu_interp_ = interp1d(nu_spec_, flux_nu_, kind='linear',bounds_error=False,fill_value=0.0)
				flux_fliter = itg.quad(lambda x: flux_nu_interp_(x)*resp_interp_(x), nu_resp_[0], nu_resp_[-1])[0]
				flux_response = itg.quad(lambda x: resp_interp_(x), nu_resp_[0], nu_resp_[-1])[0]
				flux_ratio = flux_fliter/flux_response
				m_spec = -2.5*np.log10(flux_ratio) - 48.6
				#plt.plot(sdss[sdss_dict[band_]]._wavelength, sdss[sdss_dict[band_]].response)
				#plt.show()
				#exit()
			elif filter_sys == 'Vega':
				resp_interp_ = interp1d(wave_, resp_, kind='linear',bounds_error=False,fill_value=0.0)
				flux_interp_ = interp1d(self.spectraDf.loc[index, 'spectra'][0], 
					self.spectraDf.loc[index, 'spectra'][1], kind='linear',bounds_error=False,fill_value=0.0)
				flux_fliter = itg.quad(lambda x: flux_interp_(x)*resp_interp_(x), wave_[0], wave_[-1])[0]
				flux_Vega = itg.quad(lambda x: Vega_interp_(x)*resp_interp_(x), wave_[0], wave_[-1])[0]
				flux_ratio = flux_fliter/flux_Vega
				m_spec = -2.5*np.log10(flux_ratio)
				if 'Bessell' in filter_name:
					bess_zero = {'U':0.79,'B':-0.102,'V':0.008,'R':0.193,'I':0.443}
					m_spec += bess_zero[band.split('_')[-1].split()[-1]]
			else:
				raise Exception('Unknown magnitude system!')
		else:
			raise Exception('Unknown magnitude band!')

		#elif band_ in bess_dict.keys():	
		#flux_cal = self.spectraDf.loc[index, 'spec'][1]*10**(-0.4*(m_phot - m_spec))
		#flux_cal_err = dm_phot*flux_cal/1.0857

		cal_ratio = 10**(-0.4*(m_phot[0] - m_spec))
		cal_flux_relative_err = dm_phot[0]/1.0857
		pos_ = np.argmax(self.spectraDf.loc[index, 'spectra'][0] > PPlotter.central_wavelengths[band.split('_')[-1].split()[-1]])
		print(self.spectraDf.loc[index, 'spectra'][0][pos_])
		print(self.spectraDf.loc[index, 'spectra'][1][pos_])
		#print(flux_ratio*3e10/(6260**2*1e-8))
		print(zp_*10**(-0.4*m_phot[0]))
		print(MJD, m_phot[0], m_spec, cal_ratio)
		#exit()
		return cal_ratio, cal_flux_relative_err

	def get_rest_(self, index, err=False):
		# Determine extinction        
		mag_ext = extinction.fitzpatrick99(self.spectraDf.loc[index, 'spectra'][0], 
										   self.Rv * self.EBVmilky,
										   self.Rv)

		# Correct flux to rest-frame
		rest_flux = self.spectraDf.loc[index, 'spectra'][1] * 10 ** (0.4 * mag_ext)
		rest_fluxerr = self.spectraDf.loc[index, 'spectra'][2] * 10 ** (0.4 * mag_ext)
		rest_wave = self.spectraDf.loc[index, 'spectra'][0] / (1 + self.z)
		
		if self.EBVhost != 0.:
			mag_ext = extinction.fitzpatrick99(rest_wave, 
										   self.RVhost * self.EBVhost,
										   self.RVhost)
			rest_flux = rest_flux * 10 ** (0.4 * mag_ext)
			rest_fluxerr = rest_flux * 10 ** (0.4 * mag_ext)
		if err == True:
			return rest_wave, rest_flux, rest_fluxerr
		else:
			return rest_wave, rest_flux

	def get_rest_other(self, name, index):
		# Determine extinction   
		pos_index = np.argmax((self.other_spectraDf['name'] == name) * (self.other_spectraDf['index'] == index))
		spectrum_ = self.other_spectraDf.iloc[pos_index]
		mag_ext = extinction.fitzpatrick99(spectrum_['Spectra'][0], 
										   spectrum_['rv'] * spectrum_['ebv'],
										   spectrum_['rv'])

		# Correct flux to rest-frame
		rest_wave = spectrum_['Spectra'][0] / (1 + spectrum_['z'])
		rest_flux = spectrum_['Spectra'][1] * 10 ** (0.4 * mag_ext)

		if  spectrum_['ebvhost'] != 0.:
			mag_ext = extinction.fitzpatrick99(rest_wave, 
										   spectrum_['rvhost'] * spectrum_['ebvhost'],
										   spectrum_['rvhost'])
			rest_flux = rest_flux * 10 ** (0.4 * mag_ext)
		return rest_wave, rest_flux, self.other_spectraDf.loc[pos_index, 'phase']

	def Plot(self, rest_frame=True, save=False, index=None, figsize=(8,12), scale_range=[4000, 8000], scale_method='max', seperate=0.5, normalize_flag=False,
		label_spec=False, lines_name=None, lines_position=None, lines_ymax=None, smooth_w=50, mask_high=None, wave_range=None, label_tel=False, log_flux=True, 
		smooth=False, text_tail=True, legend_loc = 'upper right', lines_ymin=None, label_telluric=None, legend_ncol=1, label_x=None, smooth_dict={}, plot_v=False,
		other_name=None, other_dir=None, other_params=None, other_index_dict=None, w_to_v=False, v_range=None, line_rest_w=None, set_ylim=None, set_xlim=None,
		phase_fmt = '%.1f',line_v_region=None, label_uniform=False, add_box=None):
		tel_dict = {'Xinglong  2.16m':'XLT','Other':'Other','2m4-01':'LJT', 'OHP-2m':'OHP','Lick-3m':'Shane',
			'1.22m Reflector':'Pennar1.22','1.82m Reflector':'Copernico','shane':'Shane','TNG':'TNG','EBE':'EBE','ESOU':'ESOU','Keck2':'Keck II'}
		inst_dict_none = {'Shane':'Kast'}
		inst_dict = {'YFOSC':'LJT'}

		if other_name is not None:
			other_EBV, other_Tmax, other_z = other_params
			if other_dir is None:
				other_dir = '/Users/liujialian/work/'+self.name+'/Si_spectra/'+other_name
			other_b = SpectraData(other_name, other_dir, z=other_z, EBVmilky=other_EBV, Tmax=other_Tmax, rebin=2)
			#print(other_b.spectraDf[['MJD','phase','inst','obsDate']])
			#exit()
		tel_list = self.spectraDf['tel'].copy()
		inst_list = self.spectraDf['inst'].copy()
		color_dict = ['black', 'b', 'purple','orange', 'brown', 'r', 'pink','cyan', 'darkgreen',]
		for i in range(10):
			color_dict.append('darkgreen')
		tel_color_dict = {}
		tel_done = {}
		color_i = 0
		for tel_i in range(len(tel_list)):
			if tel_list[tel_i] in tel_dict.keys():
				tel_list[tel_i] = tel_dict[tel_list[tel_i]]
			elif inst_list[tel_i] in inst_list:
				tel_list[tel_i] = inst_dict[inst_list[tel_i]]
			if tel_list[tel_i] not in tel_color_dict.keys():
				tel_color_dict[tel_list[tel_i]] = color_dict[color_i]
				tel_done[tel_list[tel_i]] = False
				color_i += 1

		if save:
			save = self.name + '_results/' + save
		plotNumber = 0
		fig, ax = plt.subplots(figsize=figsize)
		plt.tick_params(labelsize=15)
		if w_to_v:
			ax.set_xlabel('Velocity [km s$^{-1}$]', fontsize=20)
		elif rest_frame:
			ax.set_xlabel('Rest Wavelength [$\\rm \\AA$]', fontsize=20)
		else:
			ax.set_xlabel('Wavelength [$\\rm \\AA$]', fontsize=20)
		if normalize_flag == True:
			scale_str = 'Normalized'
		else:
			scale_str = 'Scaled'
		if log_flux:
			log_flux_ = ' Log '
			scale_str = ''
		else:
			log_flux_ = ' '
		#if w_to_v == True and len(index) == 1:
		#	scale_str = ''
		if seperate:	
			ax.set_ylabel(scale_str + log_flux_ + 'Flux Density $f_{\lambda}$ + Constants', fontsize=20)
		else:
			ax.set_ylabel(scale_str + log_flux_ + 'Flux Density $f_{\lambda}$', fontsize=20)
			seperate=0
		#ax.set_ylabel('FLux [$\\rm erg\\ cm^{-2}\\ s^{-1}\\ \\AA^{-1}$]')
		if lines_ymax is not None:
			fig_ymax = np.max(lines_ymax)
		else:
			fig_ymax = -np.inf
		fig_ymin = np.inf
		if index is None:
			index = self.spectraDf.index
		label_y = None
		plot_target_done = False
		plot_other_done = False
		for i in index:
			if rest_frame:
				wave_, flux_ = self.get_rest_(i)
				if other_name is not None and str(i) in other_index_dict.keys():
					other_i = other_index_dict[str(i)]
					other_wave_, other_flux_ = other_b.get_rest_(other_index_dict[str(i)])
					other_flag = True
				else:
					other_flag = False
			else:
				wave_ = self.spectraDf.loc[i, 'spectra'][0].copy()
				flux_ = self.spectraDf.loc[i, 'spectra'][1].copy()
			if str(i) in smooth_dict.keys():
				print(i)
				smooth_w_ = int(smooth_dict[str(i)]*(len(wave_)-1)/(wave_[-1] - wave_[0]))  # AA to true smooth_w_
				smooth_w_ = smooth_w_ + (smooth_w_%2 + 1)%2
			else:
				smooth_w_ = int(smooth_w*(len(wave_)-1)/(wave_[-1] - wave_[0]))  # AA to true smooth_w_
				smooth_w_ = smooth_w_ + (smooth_w_%2 + 1)%2
			#smooth_w_ = int(smooth_w*10/(wave_[11] - wave_[1]))  # AA to true window
			#smooth_w_ = smooth_w_ + (smooth_w_%2 + 1)%2
			if smooth:
				flux_ = my_filter(flux_, smooth_w_, 1)
				if scale_method == 'max':
					scale_ = flux_[(wave_ > scale_range[0])*(wave_<scale_range[1])].max()
				elif scale_method == 'min':
					scale_ = flux_[(wave_ > scale_range[0])*(wave_<scale_range[1])].min()
				else:
					scale_ = 1.0
				if other_flag:
					if scale_method == 'max':
						try:
							other_scale_ = my_filter(other_flux_, smooth_w_, 1)[(other_wave_ > scale_range[0])*(other_wave_<scale_range[1])].max()
						except:
							print(other_i)
							exit()
					elif scale_method == 'min':
						other_scale_ = my_filter(other_flux_, smooth_w_, 1)[(other_wave_ > scale_range[0])*(other_wave_<scale_range[1])].min()
					else:
						other_scale_ = 1.0
			else:
				if scale_method == 'max':
					scale_ = my_filter(flux_[(wave_ > scale_range[0])*(wave_<scale_range[1])], smooth_w_, 1).max()
				elif scale_method == 'min':
					scale_ = my_filter(flux_[(wave_ > scale_range[0])*(wave_<scale_range[1])], smooth_w_, 1).min()
				else:
					scale_ = 1.0
				if other_flag:
					if scale_method == 'max':
						try:
							other_scale_ = my_filter(other_flux_, smooth_w_, 1)[(other_wave_ > scale_range[0])*(other_wave_<scale_range[1])].max()
						except:
							print(other_i)
							exit()
					elif scale_method == 'min':
						other_scale_ = my_filter(other_flux_, smooth_w_, 1)[(other_wave_ > scale_range[0])*(other_wave_<scale_range[1])].min()
					else:
						other_scale_ = 1.0
			scaled_flux_ = flux_/scale_
			if other_flag:
				other_scaled_flux_ = other_flux_/other_scale_
			if wave_range is not None:
				mask_wave = (wave_ > wave_range[0])*(wave_ < wave_range[1])
				wave_ = wave_[mask_wave]
				flux_ = flux_[mask_wave]
				scaled_flux_ = scaled_flux_[mask_wave]
				if other_flag:
					other_mask_wave = (other_wave_ > wave_range[0])*(other_wave_ < wave_range[1])
					other_wave_ = other_wave_[other_mask_wave]
					other_flux_ = other_flux_[other_mask_wave]
					other_scaled_flux_ = other_scaled_flux_[other_mask_wave]
			if log_flux:
				scaled_flux_ = np.log10(scaled_flux_)
				mark_not_nan = ~np.isnan(scaled_flux_)
				scaled_flux_ = scaled_flux_[mark_not_nan]
				wave_ = wave_[mark_not_nan]
				if other_flag:
					other_scaled_flux_ = np.log10(other_scaled_flux_)
					other_mark_not_nan = ~np.isnan(other_scaled_flux_)
					other_scaled_flux_ = other_scaled_flux_[other_mark_not_nan]
					other_wave_ = other_wave_[other_mark_not_nan]
			#elif normalize_flag:
			#	def continuum(wave):
			#		return scaled_flux_[0] + (scaled_flux_[-1] - scal)
			if mask_high:
				mask_high_ = scaled_flux_ < mask_high
				wave_ = wave_[mask_high_]
				scaled_flux_ = scaled_flux_[mask_high_]
			

			'''
			#print(fig_ymin, self.spectraDf.loc[i, 'phase'])
			if i == index[0]:
				fig_ymax = scaled_flux_.max()
			elif i == index[-1]:
				fig_ymin = scaled_flux_.min()
			'''
			if w_to_v == True:
				for line_i, line_ in enumerate(lines_name):
					v_ = f_rela(wave_/line_rest_w[line_i])
					mark_v_ = (v_ > v_range[0])*(v_ < v_range[1])
					v_ = v_[mark_v_]
					fv_ = scaled_flux_[mark_v_] * line_rest_w[line_i] / 7500
					def continuum_(x):
						return fv_[0] + (fv_[-1]-fv_[0])/(v_[-1]-v_[0]) * (x - v_[0])
					fv_ = fv_ - fv_.min()
					#fv_ = fv_ - continuum_(v_)
					ax.plot(v_, fv_-plotNumber, c='gray')
					label_line = False
					for mark_line_region in line_v_region[line_i]:
						mark_line_v_ = (v_ > mark_line_region[0])*(v_ < mark_line_region[1])
						if not label_line:
							ax.plot(v_[mark_line_v_], fv_[mark_line_v_]-plotNumber, c=color_dict[line_i], label=line_)
							label_line = True
						else:
							ax.plot(v_[mark_line_v_], fv_[mark_line_v_]-plotNumber, c=color_dict[line_i])
					fig_ymin = np.min([fv_.min()-plotNumber, fig_ymin])
					fig_ymax = np.max([fv_.max()-plotNumber, fig_ymax])
					plotNumber += seperate
			elif label_spec == True:
				if self.spectraDf.loc[i, 'phase'] > 0:
					sign_ = '+'
				else:
					sign_ = ''
				ax.plot(wave_, scaled_flux_-plotNumber, label=sign_+'%.1f d'%self.spectraDf.loc[i, 'phase'])
				if scaled_flux_.max() != np.inf:
					fig_ymax = np.max([scaled_flux_.max()-plotNumber, fig_ymax])
				if scaled_flux_.min() != -np.inf:
					fig_ymin = np.min([scaled_flux_.min()-plotNumber, fig_ymin])
			else:
				if label_tel:
					if tel_done[tel_list[i]] == False:
						ax.plot(wave_, scaled_flux_-plotNumber, label=tel_list[i], c=tel_color_dict[tel_list[i]])
						tel_done[tel_list[i]] = True
					else:
						ax.plot(wave_, scaled_flux_-plotNumber, c=tel_color_dict[tel_list[i]])
					color = tel_color_dict[tel_list[i]]
				else:
					if plot_target_done:
						target_label=None
					else:
						target_label=self.name
						plot_target_done = True
					ax.plot(wave_, scaled_flux_-plotNumber, c='black', label=target_label)
					color = 'black'

				if label_x is not None:
					text_x = label_x
					if label_y is None:
						label_y = scaled_flux_[wave_ > (wave_[-1]-500)].mean()
					if label_uniform==False:
						label_y = scaled_flux_[-1]
					text_y = label_y

				elif text_tail == True:
					'''
					if np.abs(scaled_flux_[-1]) == np.inf:
						text_x = wave_[-10]
						text_y = smooth_scaled_flux_[-10]
					'''
					text_x = wave_[-1]
					text_y = scaled_flux_[-1]
					#text_y = scaled_flux_[wave_ > (wave_[-1]-1000)].mean()
				else:
					if wave_[-1] < 8500:
						if np.abs(scaled_flux_[-1]) == np.inf:
							text_x = wave_[-10]
							text_y = scaled_flux_[-10]
						else:
							text_x = wave_[-1]
							text_y = scaled_flux_[-1]
					elif wave_[-1] > 10200:
						text_x = 9000
						text_y = scaled_flux_[(wave_>9000)].max()
					else:
						text_x = 8400
						text_y = scaled_flux_[(wave_>8400)].max()
				if wave_range is not None and label_x is None:
					text_x = wave_range[1] + 0.05*(wave_range[1] - wave_range[0])
				#if wave_[-1] < 8000 and wave[0] > 4400:
				#	text_y
				if self.spectraDf.loc[i, 'phase'] > 0:
					sign_ = '+'
				else:
					sign_ = ''
				phase_text  = phase_fmt%self.spectraDf.loc[i, 'phase']
				ax.text(text_x, text_y-plotNumber, sign_+ phase_text + ' d', fontsize=15, c=color, verticalalignment='center')
				if other_flag:
					if plot_other_done:
						other_label=None
					else:
						other_label=other_name
						plot_other_done = True
					ax.plot(other_wave_, other_scaled_flux_-plotNumber, c='red', linestyle='--', label=other_label)
					if other_b.spectraDf.loc[other_i, 'phase'] > 0:
						other_sign_ = '+'
					else:
						other_sign_ = ''
					phase_text  = phase_fmt%other_b.spectraDf.loc[other_i, 'phase']
					ax.text(text_x+150, text_y-plotNumber, other_sign_+ phase_text + ' d', 
						fontsize=15, c='red', verticalalignment='center') #170
				if scaled_flux_.max() != np.inf:
					fig_ymax = np.max([scaled_flux_.max()-plotNumber, fig_ymax])
				if scaled_flux_.min() != -np.inf:
					fig_ymin = np.min([scaled_flux_.min()-plotNumber, fig_ymin])

			plotNumber += seperate
		if label_telluric is not None:
			if lines_name is None:
				lines_position = []
				lines_name = []
				lines_ymax = []
			if rest_frame==True:
				if wave_range is not None:
					if wave_range[0] < 6854/(1+self.z) and wave_range[1] > 6947/(1+self.z):
						lines_ymax.append(label_telluric)
						lines_name.append(r'$\bigoplus$')
						lines_position.append([6854/(1+self.z),6947/(1+self.z)])
					if wave_range[0] < 7171/(1+self.z) and wave_range[1] > 7403/(1+self.z):
						lines_ymax.append(label_telluric)
						lines_name.append(r'$\bigoplus$')
						lines_position.append([7171/(1+self.z),7403/(1+self.z)])
					if wave_range[0] < 7585/(1+self.z) and wave_range[1] > 7706/(1+self.z):
						lines_ymax.append(label_telluric)
						lines_name.append(r'$\bigoplus$')
						lines_position.append([7585/(1+self.z),7706/(1+self.z)])
				else:
					lines_position.append([6854/(1+self.z),6947/(1+self.z)])
					lines_position.append([7171/(1+self.z),7403/(1+self.z)])
					lines_position.append([7585/(1+self.z),7706/(1+self.z)])
					lines_ymax.append(label_telluric)
					lines_ymax.append(label_telluric)
					lines_ymax.append(label_telluric)
					lines_name.append(r'$\bigoplus$')
					lines_name.append(r'$\bigoplus$')
					lines_name.append(r'$\bigoplus$')
			else:
				lines_position.append([6854,6947])
				lines_position.append([7171,7403])
				lines_position.append([7585,7706])
				lines_ymax.append(label_telluric)
				lines_ymax.append(label_telluric)
				lines_ymax.append(label_telluric)
				lines_name.append(r'$\bigoplus$')
				lines_name.append(r'$\bigoplus$')
				lines_name.append(r'$\bigoplus$')
		if lines_name is not None and w_to_v == False:
			fig_ymax += 0.17
			#fig_ymax = np.max(lines_ymax)

			for i in range(len(lines_name)):
				if len(lines_position[i]) == 2:
					if plot_v == False:
						collection = collections.BrokenBarHCollection.span_where(np.linspace(lines_position[i][0],lines_position[i][1],100),
							ymin=fig_ymin-plotNumber-0.1,ymax=lines_ymax[i],where=np.ones(100)>0,facecolor='gray',alpha=0.5, zorder=0)
						ax.add_collection(collection)
					else:
						ax.vlines(lines_position[i][0], ymin=fig_ymin-plotNumber-0.1, ymax=lines_ymax[i], color='grey', linestyle='-', zorder=0)
						for line_wave_ in np.linspace(lines_position[i][0],lines_position[i][1],5)[1:-1]:
							ax.vlines(line_wave_, ymin=fig_ymin-plotNumber-0.1, ymax=lines_ymax[i], color='grey', linestyle='--', zorder=0)
						ax.vlines(lines_position[i][1], ymin=fig_ymin-plotNumber-0.1, ymax=lines_ymax[i], color='grey', linestyle='-', zorder=0)
				else:
					if lines_ymin is None:
						ax.vlines(lines_position[i][0], ymin=fig_ymin-plotNumber-0.1, ymax=lines_ymax[i], color='grey', linestyle='--', zorder=0)
					else:
						if lines_ymin[i] is None:
							ax.vlines(lines_position[i][0], ymin=fig_ymin-plotNumber-0.1, ymax=lines_ymax[i], color='grey', linestyle='--', zorder=0)
						else:
							ax.vlines(lines_position[i][0], ymin=lines_ymin[i], ymax=lines_ymax[i], color='grey', linestyle='--', zorder=0)
				ax.text(np.mean(lines_position[i]), lines_ymax[i], lines_name[i], fontsize=15, horizontalalignment='center',verticalalignment='bottom')
		#ax.set_ylim(fig_ymin-plotNumber-0.1, fig_ymax+0.1)
		#ax.plot([6400,6400],[-1.83,-1.95],c='black')
		#ax.text(6400,-2.0,'CII $\\rm \\lambda$6580',c='black',fontsize=15,horizontalalignment='center')
		if not log_flux:
			ax.set_ylim(fig_ymin-0.1, fig_ymax+0.1)
		else:
			ax.set_ylim(fig_ymin-0.1, fig_ymax+0.15)
		#ax.set_ylim(-20, fig_ymax+0.15)
		#ax.set_ylim(-4, 1)
		#ax.set_ylim(0, 0.35)
		if set_ylim is not None:
			ax.set_ylim(set_ylim[0], set_ylim[1])
		if w_to_v == True and len(index)==1:
			ax.text(7500, 0.45, '+%.0f d'%self.spectraDf.loc[index[0],'phase'], c='black', fontsize=15)
		if set_xlim is not None:
			ax.set_xlim(set_xlim[0], set_xlim[1])
		elif wave_range is not None and w_to_v == False:
			ax.set_xlim(wave_range[0] - 50, wave_range[1] + 50)
		if label_spec or w_to_v:
			ax.legend(fontsize=15, loc=legend_loc, ncol=legend_ncol)
		elif label_tel:
			ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
	                      ncol=np.max([legend_ncol,len(tel_done)%legend_ncol]), mode="expand", borderaxespad=0., fontsize=15)
		elif other_name is not None:
			ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
	                      ncol=np.max([legend_ncol,len(tel_done)%legend_ncol]), mode="expand", borderaxespad=0., fontsize=15)
		if add_box is not None:
			add_box = np.array(add_box)
			#add_box = [[1,2],[3,4]]
			ax.plot(add_box[0], np.ones(2)*add_box[1,0], c='purple')
			ax.plot(add_box[0], np.ones(2)*add_box[1,1], c='purple')
			ax.plot(np.ones(2)*add_box[0,0], add_box[1], c='purple')
			ax.plot(np.ones(2)*add_box[0,1], add_box[1], c='purple')
		if save:
			fig.savefig('%s.pdf'%save, bbox_inches='tight', dpi=500)
		plt.show()

	def Plot_analysis(self, index, othername=None, save=None, lines_position=None, lines_name=None, figsize=(8,6), wave_range=None, ymax=None, seperate=0.6,show_lam=False, 
		log_flux=False, rest_frame=True, scale_range=[6800,7000], scale_method='max', window=1, smooth=False, label_text_pos=[7500., 9000.], line_fontsize=12, label_list=None,
		line_len_origin=0.2, ymin=None, attach=None, smooth_w_toAA=True, SYN=False, space=None, phase_fmt='%.1f', xaxi_y=None, only_other=False, label_spec=False, log_y=False, xaxi_x=0.02):
		xmin, xmax=[None,None]
		if save:
			save = self.name + '_results/' + save
		if type(seperate) != type([]):
			seperate = [seperate for i in range(len(index))]
		if type(index) != list:
			index = [index]
		if not othername:
			name_list = [self.name]
		elif type(othername) != list:
			name_list = [self.name, othername]
		else:
			name_list = othername
			name_list.insert(0, self.name)
		len_index = len(index)
		nrows = int((len_index+1)/2)
		if len_index == 1:
			ncols = 1
		else:
			ncols = 2
		color_dict = ['black', 'r', 'b', 'purple', 'orange', 'brown', 'cyan', 'pink','gray']
		#fig, axs = plt.subplots(nrows, ncols, sharex='col', sharey='row', figsize=figsize)
		fig, axs = plt.subplots(nrows, ncols, sharex='col', figsize=figsize)
		fig.subplots_adjust(wspace=0, hspace=0)
		if nrows == 1:
			if ncols == 1:
				axs = np.array([[axs]])
			else:
				axs = np.array([axs])
		plt.tick_params(labelsize=15)
		if nrows == 1:
			y_correct = 0.02
		else:
			y_correct = 0.0
		if xaxi_y is None:
			xaxi_y = nrows*0.025 - y_correct
		if rest_frame:
			fig.supxlabel('Rest Wavelength [$\\rm \\AA$]', fontsize=20, y=xaxi_y)
		else:
			fig.supxlabel('Wavelength [$\\rm \\AA$]', fontsize=20, y=xaxi_y)
		if log_flux:
			log_flux_ = ' Log '
		else:
			log_flux_ = ' '
		if seperate != 0 and log_flux==False:
			scale_str = 'Scaled'
		else:
			scale_str = ''
		if scale_method is None:
			scale_str = ''
			y_unit = ' [erg s$^{-1}$ cm$^{-2}$ $\\rm \\AA^{-1}$]'
		else:
			y_unit = ''
		if xaxi_x is None:
			if seperate[0] != 0:
				xaxi_x = 0.01
			else:
				xaxi_x = 0.02
		if seperate[0] != 0:	
			fig.supylabel(scale_str + log_flux_ + 'Flux Density $f_{\lambda}$ + Constants', fontsize=20, x=xaxi_x)
		else:
			fig.supylabel(scale_str + log_flux_ + 'Flux Density $f_{\lambda}$' + y_unit, fontsize=20, x=xaxi_x)
		'''
		fig.supxlabel('Rest Wavelength [$\\rm \\AA$]', fontsize=20, y=0.05)
		if log_flux == False:
			fig.supylabel('Scaled Flux Density + Constants', fontsize=20, x=0.05)
		else:
			fig.supylabel('Scaled Log Flux Density + Constants', fontsize=20, x=0.05)
		'''
		#label_list = np.zeros(len(othername))
		if label_list is None:
			label_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n']
		if space is None:
			if log_flux:
				space = 0.02
			else:
				#space = 0.2
				space = 0.2
		for index_i, indexs_ in enumerate(index):
			if attach is not None:
				attach_data = []
				#attach_text = []
				#line_flag = 0
			pos_correct = 0
			row_ = int(index_i/2)
			col_ = index_i%2

			for name_i, name_ in enumerate(name_list):
				if name_i == 0:
					if only_other==True:
						continue
					for index_i_, index_ in enumerate(indexs_):
						wave_SYN = None
						if index_i_ != 0:
							pos_correct += 1
						if rest_frame:
							wave_, flux_ = self.get_rest_(index_)
						else:
							wave_ = self.spectraDf.loc[index_, 'spectra'][0].copy()
							flux_ = self.spectraDf.loc[index_, 'spectra'][1].copy()
						if smooth_w_toAA == True:
							smooth_w_ = int(window*(len(wave_)-1)/(wave_[-1] - wave_[0]))  # AA to true smooth_w_
							smooth_w_ = smooth_w_ + (smooth_w_%2 + 1)%2
							#window_ = int(window*10/(wave_[11] - wave_[1]))  # AA to true window_
							#window_ = window_ + (window_%2 + 1)%2
						else:
							smooth_w_ = window
						flux_smooth = my_filter(flux_, smooth_w_, 1)
						if smooth:
							flux_ = flux_smooth
						if scale_method == 'max':
							scale_ = flux_smooth[(wave_ > scale_range[0])*(wave_<scale_range[1])].max()
						elif scale_method == 'min':
							scale_ = flux_smooth[(wave_ > scale_range[0])*(wave_<scale_range[1])].min()
						elif scale_method is None:
							scale_ = 1
						if wave_range is not None:
							mark_ = (wave_ > wave_range[0])*(wave_ < wave_range[1])
							wave_ = wave_[mark_]
							flux_ = flux_[mark_]
							flux_smooth = flux_smooth[mark_]
							if lines_position is not None:
								for i in range(len(index)):
									lines_position_mean_ = []
									len_lines_position_i = len(lines_position[i])
									for position_i in range(len_lines_position_i):
										lines_position_mean_.append(np.mean(lines_position[i][position_i]))
									lines_position_mean_ = np.array(lines_position_mean_)
									mark_ = (lines_position_mean_ > wave_range[0])*(lines_position_mean_ < wave_range[1])
									if type(mark_) != type(np.array([True, False])):
										mark_ = np.array([mark_])
									for position_i in range(len_lines_position_i-1, -1, -1):
										if mark_[position_i] == False:
											lines_position[i].pop(position_i)
									#lines_position[i] = lines_position[i][mark_]
									lines_name[i] = lines_name[i][mark_]
						scaled_flux_ = flux_/scale_
						flux_smooth = flux_smooth/scale_
						if log_flux == True:
							scaled_flux_ = np.log10(scaled_flux_)
							flux_smooth = np.log10(flux_smooth)
						xmin = wave_[0]
						xmax = wave_[-1]
						
						phase_ = self.spectraDf.loc[index_, 'phase']
						if phase_ >= 0.:
							phase_ = '+'+phase_fmt%abs(phase_)
						else:
							phase_ = '-'+phase_fmt%abs(phase_)

						if lines_position is not None and index_i_ == 0:
							axs[row_, col_].plot(5000, scaled_flux_.max()+0.2, alpha=0)#0.68
							if SYN == True and (self.other_spectraDf['name']=='SYN++').any():
								mark_pos = (self.other_spectraDf['name'] == 'SYN++') * (self.other_spectraDf['index'] == index_)
								if mark_pos.any() == True:
									if rest_frame:
										wave_SYN, flux_SYN, phase_SYN = self.get_rest_other('SYN++', index_)
									else:
										pos_index = np.argmax(mark_pos)
										spectrum_ = self.other_spectraDf.iloc[pos_index]
										wave_SYN = spectrum_['Spectra'][0].copy()
										flux_SYN = spectrum_['Spectra'][1].copy()
										phase_SYN = self.other_spectraDf.loc[pos_index, 'phase']
									if scale_method == 'max':
										scale_ = flux_SYN[(wave_SYN > scale_range[0])*(wave_SYN<scale_range[1])].max()
									else:
										scale_ = flux_SYN[(wave_SYN > scale_range[0])*(wave_SYN<scale_range[1])].min()
									if wave_range is not None:
										mark_ = (wave_SYN > wave_range[0])*(wave_SYN < wave_range[1])
										wave_SYN = wave_SYN[mark_]
										flux_SYN = flux_SYN[mark_]
									xmin = np.min([xmin, wave_SYN[0]])
									xmax = np.max([xmax, wave_SYN[-1]])
									scaled_flux_SYN = flux_SYN/scale_
									if log_flux == True:
										scaled_flux_SYN = np.log10(scaled_flux_SYN)
									wave_line = wave_SYN
									scaled_flux_line = scaled_flux_SYN	
								else:
									wave_line = wave_
									scaled_flux_line = scaled_flux_
							else:
								wave_line = wave_
								scaled_flux_line = scaled_flux_
							for line_pos_, line_name_ in zip(lines_position[index_i], lines_name[index_i]):
								if type(line_pos_) != type(['haha']):
									line_pos_ = [line_pos_]
								line_x_list = []
								line_y_list = []
								for item in line_pos_:
									line_index_ = np.argmax(wave_line > item)
									line_x_list.append(wave_line[line_index_])
									line_y_list.append(scaled_flux_line[line_index_]+space)
								line_y_max = np.max(line_y_list)
								line_len = line_len_origin
								no_lines = len(line_pos_)
								for i in range(no_lines):
									axs[row_, col_].plot([line_x_list[i],line_x_list[i]],[line_y_list[i],line_y_max+line_len],c='black')
								if no_lines > 1:
									axs[row_, col_].plot([line_x_list[0],line_x_list[-1]],[line_y_max+line_len,line_y_max+line_len],c='black')
									line_len *= 2
									axs[row_, col_].plot([np.mean(line_x_list),np.mean(line_x_list)],[line_y_max+line_len*0.5,line_y_max+line_len],c='black')
								if show_lam == False:
									all_plot_line_name_ = line_name_.split(',')
									plot_line_name_ = all_plot_line_name_[0].split('$')[0]
									len_all_plot_name_ = len(all_plot_line_name_)
									if len_all_plot_name_ != 1:
										for i_plot_name_ in range(1, len_all_plot_name_):
											if 'I' in all_plot_line_name_[i_plot_name_]:
												plot_line_name_ = plot_line_name_ + ', ' + all_plot_line_name_[i_plot_name_].split('$')[0]
								else:
									plot_line_name_ = line_name_
								axs[row_, col_].text(np.mean(line_x_list), line_y_max+line_len+space, '%s'%plot_line_name_, 
									verticalalignment='center', rotation=90, rotation_mode='anchor', fontsize=line_fontsize, c='black')
							'''
							if index_i == 0:
								axs[row_, col_].plot([4830,4830],[-11.5, -12],c='black')
								axs[row_, col_].text(4830, -12, 'FeII', verticalalignment='top', horizontalalignment='center', fontsize=line_fontsize, c='black')
							if index_i == 2:
								axs[row_, col_].plot([3550,3550],[1.8, 2.1],c='black')
								axs[row_, col_].text(3550, 2.15, 'CoIII', 
									verticalalignment='center', rotation=90, rotation_mode='anchor', fontsize=line_fontsize, c='black')
								axs[row_, col_].plot([3250,3250],[0.5, 0.8],c='black')
								axs[row_, col_].text(3250, 0.85, 'CoIII, FeIII, CoII', 
									verticalalignment='center', rotation=90, rotation_mode='anchor', fontsize=line_fontsize, c='black')
							'''
						if wave_[-1] < label_text_pos[0]:
							text_x = wave_[-1]
							#text_y = flux_smooth[(wave_>(wave_[-1]-1000))*(wave_<(wave_[-1]+1))].min()
							text_y = scaled_flux_[(wave_>(wave_[-1]-1000))*(wave_<(wave_[-1]+1))].min()
						#elif wave_[-1] > 10200: 
						#	text_x = 9000
						#	text_y = flux_smooth[(wave_>9000)].max()/scale_
						else:
							text_x = label_text_pos[0]
							#text_y = flux_smooth[(wave_>label_text_pos[0])*(wave_<label_text_pos[1])].min()
							text_y = scaled_flux_[(wave_>label_text_pos[0])*(wave_<label_text_pos[1])].min()
						if label_spec == True:
							label_  = name_ + ' %sd'%phase_
						else:
							label_ = None
							if othername is not None:
								axs[row_, col_].text(text_x, text_y-seperate[index_i]*(name_i + pos_correct)-space, 
										'%sd %s'%(phase_, name_[4:]), color=color_dict[name_i], fontsize=12, verticalalignment='top')
							else:
								axs[row_, col_].text(text_x, text_y-seperate[index_i]*(name_i + pos_correct)+space, 
										'%sd %s'%(phase_, name_[4:]), color=color_dict[name_i], fontsize=12, verticalalignment='top')
						axs[row_, col_].plot(wave_, scaled_flux_-seperate[index_i]*(name_i + pos_correct), color=color_dict[name_i], label=label_)
						if wave_SYN is not None:
							axs[row_, col_].plot(wave_SYN, scaled_flux_SYN-seperate[index_i]*(name_i + pos_correct), color='r', linestyle='dashdot')
						if attach is not None:
							attach_data.append([wave_, scaled_flux_])
							#attach_text.append([text_x, text_y, '%sd %s'%(phase_, name_[4:]), color_dict[name_i]])

				else:
					index_ = indexs_[0]
					mark_pos = (self.other_spectraDf['name'] == name_) * (self.other_spectraDf['index'] == index_)
					if mark_pos.any() == False:
						pos_correct -= 1
						continue
					#z_ = self.other_spectraDf.loc[mark_pos, 'z']
					if rest_frame:
						wave_, flux_, phase_ = self.get_rest_other(name_, index_)
						#phase_ = float(phase_)/(1+z_)
						#if phase_ < 0:
						#	phase_ = '%.1f'%phase_
						#else:
						#	phase_ = '+%.1f'%phase_
					else:
						pos_index = np.argmax(mark_pos)
						spectrum_ = self.other_spectraDf.iloc[pos_index]
						wave_ = spectrum_['Spectra'][0].copy()
						flux_ = spectrum_['Spectra'][1].copy()
						phase_ = self.other_spectraDf.loc[pos_index, 'phase']
					phase_ = float(phase_)
					if phase_ >= 0.:
						phase_ = '+'+phase_fmt%abs(phase_)
					else:
						phase_ = '-'+phase_fmt%abs(phase_)
					if smooth_w_toAA == True:
						smooth_w_ = int(window*(len(wave_)-1)/(wave_[-1] - wave_[0]))  # AA to true smooth_w_
						smooth_w_ = smooth_w_ + (smooth_w_%2 + 1)%2
						#window_ = int(window*10/(wave_[11] - wave_[1]))  # AA to true window_
						#window_ = window_ + (window_%2 + 1)%2
					else:
						smooth_w_ = window
					#print(smooth_w_)
					flux_smooth = my_filter(flux_, smooth_w_, 1)
					if smooth:
						flux_ = flux_smooth
					if scale_method == 'max':
						scale_ = flux_smooth[(wave_ > scale_range[0])*(wave_<scale_range[1])].max()
					elif scale_method == 'min':
						scale_ = flux_smooth[(wave_ > scale_range[0])*(wave_<scale_range[1])].min()
					elif scale_method is None:
						scale_ = 1
					if wave_range is not None:
						mark_ = (wave_ > wave_range[0])*(wave_ < wave_range[1])
						wave_ = wave_[mark_]
						flux_ = flux_[mark_]
						flux_smooth = flux_smooth[mark_]
					if xmin is None:
						xmin = wave_[0]
					else:
						xmin = np.min([xmin, wave_[0]])
					if xmax is None:
						xmax = wave_[-1]
					else:
						xmax = np.min([xmax, wave_[-1]])
					scaled_flux_ = flux_/scale_
					flux_smooth = flux_smooth/scale_
					if log_flux == True:
						scaled_flux_ = np.log10(scaled_flux_)
						flux_smooth = np.log10(flux_smooth)
					if wave_[-1] < label_text_pos[0]:
						text_x = wave_[-1]
						#text_y = flux_smooth[(wave_>(wave_[-1]-1000))*(wave_<(wave_[-1]+1))].min()
						text_y = scaled_flux_[(wave_>(wave_[-1]-1000))*(wave_<(wave_[-1]+1))].min()
					#elif wave_[-1] > 10200: 
					#	text_x = 9000
					#	text_y = flux_smooth[(wave_>9000)].max()
					else:
						text_x = label_text_pos[0]
						#text_y = flux_smooth[(wave_>label_text_pos[0])*(wave_<label_text_pos[1])].min()
						text_y = scaled_flux_[(wave_>label_text_pos[0])*(wave_<label_text_pos[1])].min()
						#print(text_y, text_y-seperate[index_i]*(name_i + pos_correct)-space)
					if label_spec == True:
						label_  = name_ + ' %sd'%phase_
					else:
						label_ = None
						if name_[:2] != 'SN':
							plot_name = name_
						else:
							plot_name = name_[4:]
						axs[row_, col_].text(text_x, text_y-seperate[index_i]*(name_i + pos_correct)-space, 
							'%sd %s'%(phase_, plot_name), color=color_dict[name_i], fontsize=12, verticalalignment='top')
					
					if name_ in ['SN2009dc','SN2012hn']:
						pos_dict = {'SN2009dc':2.3, 'SN2012hn':1.8}
						color_dict_temp = {'SN2009dc':'purple', 'SN2012hn':'orange'}
						axs[row_, col_].plot(wave_, scaled_flux_-pos_dict[name_], color=color_dict_temp[name_])
						axs[row_, col_].text(text_x, text_y-pos_dict[name_]-space, '%sd %s'%(phase_, name_[4:]), color=color_dict_temp[name_])
						continue
					
					axs[row_, col_].plot(wave_, scaled_flux_-seperate[index_i]*(name_i + pos_correct), color=color_dict[name_i], label=label_)
					if attach is not None:
						attach_index = attach[index_i][name_i-1]
						axs[row_, col_].plot(attach_data[attach_index][0], attach_data[attach_index][1]-seperate[index_i]*(name_i + pos_correct), color=color_dict[0], zorder=0)
						#axs[row_, col_].text(attach_text[attach_index][0], attach_text[attach_index][1]-seperate[index_i]*(name_i + pos_correct)-space, 
						#	attach_text[attach_index][2], color=attach_text[attach_index][3], fontsize=12, verticalalignment='top')
			if col_ == 1:
				axs[row_, col_].tick_params(axis='y',labelright=True, labelleft=False)
			if ncols * nrows != 1:
				axs[row_, col_].text(0.9,0.9,'(%s)'%label_list[row_*2+col_],horizontalalignment='left',verticalalignment='bottom',transform=axs[row_, col_].transAxes)
			axs[row_, col_].set_xlim(xmin, xmax)
			if ymax is not None:
				if ymin is not None:
					axs[row_, col_].set_ylim(bottom=ymin[index_i], top=ymax[index_i])
				else:
					axs[row_, col_].set_ylim(bottom=None, top=ymax[index_i])
			if log_y == True:
				axs[row_, col_].set_yscale('log')
		'''
		plt.axvline(4659, linestyle='--', c='gray')
		plt.text(3700, 1., '[Fe III]',fontsize=15)
		plt.axvline(7155, linestyle='--', c='gray')
		plt.text(6250, 1., '[Fe II]',fontsize=15)
		plt.axvline(7378, linestyle='--', c='gray')
		plt.text(7500, 1., '[Ni II]',fontsize=15)
		'''
		if len_index%2 == 1 and len_index > 2:
			axs[-1][1].axis('off')
			axs[-2][1].tick_params(axis='x',labelbottom=True)
		name_list.pop(0)
		if label_spec == True:
			plt.legend(fontsize=12)
		if save:
			fig.savefig('%s.pdf'%save, bbox_inches='tight')
		plt.show()


	def Calculate_EBV(self, index, MW=False):
		# Poznanski_EBV
		if MW == True:
			MW_str = 'MW_'
		else:
			MW_str = ''
		if '5893_' + MW_str + str(index) not in self.velocities.keys():
			print('Calclulate Na I first!')
		else:
			Na_name = '5893_' + MW_str + str(index)
			EBV = np.power(10, 1.17*self.velocities[Na_name][4]-1.85)
			EBV_err =  ((EBV*np.log(10)*1.17*self.velocities[Na_name][5])**2 + (EBV*0.68)**2)**0.5
			self.Na_EBV[str(index) + MW_str] = np.array([EBV, EBV_err])

	def Calculate_dSi(self, phase_range=None):
		if phase_range is None:
			phase_range = [-1.5, 11.5]
		phase_mark = (self.spectraDf['phase'] > phase_range[0]) * (self.spectraDf['phase'] < phase_range[1])
		index_list = self.spectraDf.index[phase_mark].tolist()
		phase_list = self.spectraDf['phase'][phase_mark].tolist()
		vSi_list = []
		UvSi_list = []
		for i in range(len(phase_list)):
			vSi_list.append(self.velocities['6355_'+str(index_list[i])][0])
			UvSi_list.append(self.velocities['6355_'+str(index_list[i])][1])
		params, params_cov = optimize.curve_fit(linear_f, phase_list, 
							vSi_list,
							sigma=UvSi_list)
		print('dvSi: ', params[0], params_cov[0][0]**0.5)

	def read_velocities(self, filename):
		with open(filename, 'r') as f:
			#line = f.readline()
			line = f.readline()
			while(line):
				line_split = line.split()
				self.velocities[line_split[0]] = np.array(line_split[2:]).astype(np.float64)
				line = f.readline()

	def save_velocities(self, filename, correct=None):
		filename = self.name + '_results/' + filename
		with open(filename, 'w') as f:
			for k, v in self.velocities.items():
				index_ = int(k.split('_')[-1])
				if correct is None:
					k_c = k
				else:
					correct = np.array(correct)
					index_ += (index_ >= correct).sum()
					k_c = '_'.join(k.split('_')[:-1]) + '_' + str(index_)
				phase_ = self.spectraDf.loc[index_, 'phase']
				f.writelines('%s %.1f %s %s %s %s %s %s\n' %(k_c, phase_, *v))

	def table_velocities(self, rest_wavelength_list, plot_para_list, save, format_list, line_aligns=None, para_aligns=None, units=None, index=None, phase_range=None):
		save = self.name + '_results/' + save
		if type(rest_wavelength_list) == str or type(rest_wavelength_list) == int:
			rest_wavelength_list = [rest_wavelength_list]
		if type(plot_para_list) == str:
			plot_para_list = [plot_para_list]
		if type(format_list) == str:
			format_list = [format_list]
		if len(format_list) != len(plot_para_list):
			raise Exception('The size of format_list and plot_para_list must equal!')
		if line_aligns is None:
			line_aligns = {}
			for rest_wavelength in rest_wavelength_list:
				line_aligns[str(rest_wavelength)] = str(rest_wavelength)
		if para_aligns is None:
			para_aligns = {}
			for plot_para in plot_para_list:
				para_aligns[plot_para] = plot_para
		all_data = {}
		all_phase = np.array([])
		for i in range(len(rest_wavelength_list)):
			rest_wavelength = str(rest_wavelength_list[i])
			all_data[rest_wavelength] = {}
			plot_n = 0
			for plot_para in plot_para_list:
				index_list = []
				phase_list = []
				para_list = []
				para_err_list = []
				para_dict = {'velocity':0, 'FWHM':2, 'pEW':4}
				for k, v in self.velocities.items():
					k_split = k.split('_')
					index_ = int(k_split[1])
					wavelength_ = k_split[0]
					go = 1
					if index is not None:
						if index_ not in index:
							go = 0
					if wavelength_ == rest_wavelength and go == 1:
						index_list.append(index_)
						phase_list.append(float('%.3f'%self.spectraDf.loc[index_, 'phase']))
						para_list.append(self.velocities[k][para_dict[plot_para]])
						para_err_list.append(self.velocities[k][para_dict[plot_para]+1])
				index_list = np.array(index_list)
				quene = np.argsort(index_list)
				phase_list = np.array(phase_list)[quene]
				para_list = np.array(para_list)[quene]
				para_err_list = np.array(para_err_list)[quene]
				if phase_range is not None:
					mark_phase = (phase_list > phase_range[0])*(phase_list < phase_range[1])
					phase_list = phase_list[mark_phase]
					para_list = para_list[mark_phase]
					para_err_list = para_err_list[mark_phase]
				all_data[rest_wavelength][plot_para] = pd.DataFrame({'phase':phase_list,
														  'para':para_list,
														  'para_err':para_err_list}).sort_values(by='phase').reset_index(drop=True)
				if plot_n == 0:
					all_phase = np.concatenate([all_phase, phase_list])
				plot_n += 1
		all_phase = np.sort(np.unique(all_phase))
		with open(save, 'w') as f:
			f.writelines('\\hline\n')
			write_line = [' ']
			write_n = 0
			para_n = len(plot_para_list)
			for rest_wavelength in rest_wavelength_list:
				if write_n == 0:
					write_line[0] = write_line[0] + ' & ' + '\\multicolumn{%.0f}{c}{%s}'%(para_n,line_aligns[str(rest_wavelength)])
					write_n = 1
				else:
					write_line.append('\\multicolumn{%.0f}{c}{%s}'%(para_n,line_aligns[str(rest_wavelength)]))
			write_line = ' && '.join(write_line)
			f.writelines(write_line + '\\\\\n')
			for wave_i, rest_wavelength in enumerate(rest_wavelength_list):
				f.writelines('\\cline{%.0f-%.0f}\n'%(wave_i*(para_n+1)+2,(wave_i+1)*(para_n+1)))
			write_line = ['phase']
			write_n = 0
			write_line_wave = []
			for rest_wavelength in rest_wavelength_list:
				write_line_para = []
				for plot_para in plot_para_list:
					write_line_para.append(para_aligns[plot_para])
				write_line_wave.append(' & '.join(write_line_para))
			for write_line_wave_ in write_line_wave:
				if write_n == 0:
					write_line[0] = write_line[0] + ' & ' + write_line_wave_
					write_n = 1
				else:
					write_line.append(write_line_wave_)
			write_line = ' && '.join(write_line)
			f.writelines(write_line + '\\\\\n')
			if units is not None:
				write_line = ['(days)']
				write_n = 0
				unit_n = 0
				write_line_wave = []
				for rest_wavelength in rest_wavelength_list:
					write_line_para = []
					for plot_para in plot_para_list:
						write_line_para.append(units[unit_n])
						unit_n += 1
					write_line_wave.append(' & '.join(write_line_para))
				for write_line_wave_ in write_line_wave:
					if write_n == 0:
						write_line[0] = write_line[0] + ' & ' + write_line_wave_
						write_n = 1
					else:
						write_line.append(write_line_wave_)
				write_line = ' && '.join(write_line)
				f.writelines(write_line + '\\\\\n')
			f.writelines('\\hline\n')
			for phase_ in all_phase:
				write_line = ['%.1f'%phase_]
				write_n = 0
				write_line_wave = []
				for rest_wavelength_i, rest_wavelength in enumerate(rest_wavelength_list):
					rest_wavelength = str(rest_wavelength)
					write_line_para = []
					if phase_ not in all_data[rest_wavelength][plot_para_list[0]]['phase'].tolist():
						for plot_para in plot_para_list:
							write_line_para.append('-')
						write_line_wave.append(' & '.join(write_line_para))
					else:
						for plot_para_i, plot_para in enumerate(plot_para_list):
							df_ = all_data[rest_wavelength][plot_para]
							pos_ = df_[df_['phase']==phase_].index[0]
							#if 'e' in format_list[plot_para_i]:
							#	para_ = e_to_times(format_list[plot_para_i]%df_.loc[pos_, 'para'])
							#	para_err_ = e_to_times(format_list[plot_para_i]%df_.loc[pos_, 'para'])
							#	write_line.append('$'+para_+'\\pm'+para_err_+'$')
							#else:
							#	write_line.append((format_list[plot_para_i]+'$\\pm$'+format_list[plot_para_i])%(df_.loc[pos_, 'para'], df_.loc[pos_, 'para_err']))
							if 'e' in format_list[plot_para_i]:
								if units is None:
									para_ = e_to_times(format_list[plot_para_i]%(df_.loc[pos_, 'para']))
									para_err_ = e_to_times(format_list[plot_para_i]%(df_.loc[pos_, 'para_err']))
									write_line_element = '$'+para_+'\\pm'+para_err_+'$'
								else:
									unit_n = rest_wavelength_i * 3 + plot_para_i
									e_number = int(units[unit_n].split('{')[1].split('}')[0])
									write_line_element = ('%.1f$\\pm$%.1f'%(df_.loc[pos_, 'para']/10**e_number, df_.loc[pos_, 'para_err']/10**e_number))
							else:
								write_line_element = (format_list[plot_para_i]+'$\\pm$'+format_list[plot_para_i])%(df_.loc[pos_, 'para'], df_.loc[pos_, 'para_err'])
							write_line_para.append(write_line_element)
						write_line_wave.append(' & '.join(write_line_para))
				for write_line_wave_ in write_line_wave:
					if write_n == 0:
						write_line[0] = write_line[0] + ' & ' + write_line_wave_
						write_n = 1
					else:
						write_line.append(write_line_wave_)
				write_line = ' && '.join(write_line)
				f.writelines(write_line + '\\\\\n')
			f.writelines('\\hline\n')

	def plot_velocities(self, rest_wavelength_list, plot_para, index=None, ylabel=None, save=False, feature_label=None, otherfiles=None,
	 xlabel='Days from $B$-band Maximum', figsize=(8,6), inverse_y=True, log_y=False, other_dir=None, phase_range=None, ratio=None, table_file=None):
		if table_file is not None:
			write_lines = ['Name & Phase & pEW(SiII 6355 \\AA) & pEW(CII 6580 \\AA) & pEW(CII 6580 \\AA)/pEW(SiII 6355 \\AA) \\\\\n',
					 ' & Day & \\AA & \\AA & \\\\\n']
			table_file = self.name + '_results/' + table_file
		if save:
			save = self.name + '_results/' + save
		if type(rest_wavelength_list) == str or type(rest_wavelength_list) == int:
			rest_wavelength_list = [rest_wavelength_list]
		fig, ax = plt.subplots(figsize=(8,6))
		plt.tick_params(labelsize=15)
		ax.set_xlabel(xlabel, fontsize=20)
		if ylabel:
			ax.set_ylabel(ylabel, fontsize=20)
		else:
			ax.set_ylabel('Velocity [km s$^{-1}$]')
		ymin = 9999999.
		ymax = -9999999.
		all_data = {}
		all_data[self.name] = {}
		color_dict = ['black','b','b','b','b','darkgreen','darkgreen','darkgreen','salmon','mediumorchid','lime','brown','darkgoldenrod']
		#color_dict = ['r','b','darkgreen','purple','orange','pink','gold','green','salmon','mediumorchid','lime','brown','darkgoldenrod']
		shape_dict = ['^','*','p','<','X','D','H','v','>','s','P','h','d','1','2','3','4','8']
		if otherfiles is not None:
			othername = []
			if other_dir is None:
				other_dir = self.name+'_results/'
			for file in otherfiles:
				othername.append(file.split('_')[0])
				all_data[othername[-1]] = {}
		for i in range(len(rest_wavelength_list)):
			rest_wavelength = str(rest_wavelength_list[i])
			index_list = []
			phase_list = []
			para_list = []
			para_err_list = []
			para_dict = {'velocity':0, 'FWHM':2, 'pEW':4}
			for k, v in self.velocities.items():
				k_split = k.split('_')
				index_ = int(k_split[-1])
				wavelength_ = '_'.join(k_split[:-1])
				go = 1
				if index is not None:
					if index_ not in index:
						go = 0
				if wavelength_ == rest_wavelength and go == 1:
					index_list.append(index_)
					phase_list.append(self.spectraDf.loc[index_, 'phase'])
					para_list.append(self.velocities[k][para_dict[plot_para]])
					para_err_list.append(self.velocities[k][para_dict[plot_para]+1])
			index_list = np.array(index_list)
			quene = np.argsort(index_list)
			phase_list = np.array(phase_list)[quene]
			para_list = np.array(para_list)[quene]
			para_err_list = np.array(para_err_list)[quene]
			if phase_range is not None:
				mark_phase = (phase_list > phase_range[0])*(phase_list < phase_range[1])
				phase_list = phase_list[mark_phase]
				para_list = para_list[mark_phase]
				para_err_list = para_err_list[mark_phase]
			if ratio is not None:
				all_data[self.name][rest_wavelength] = pd.DataFrame({'phase':phase_list,
															   'para':para_list,
															   'para_err':para_err_list}).sort_values(by='phase').reset_index(drop=True)
			if feature_label is not None:
				label = feature_label[i]
			elif otherfiles is not None:
				label = self.name
			else:
				label = ''
			if ratio is None:
				if otherfiles is not None:
					c = 'black'
				else:
					c = color_dict[i]
				if otherfiles is not None:
					marker = 'o'
				else:
					marker = shape_dict[i]
				if log_y:
					ax.plot(phase_list, np.log10(para_list), marker=marker, markerfacecolor='none', markeredgecolor=c, linestyle='', markersize=markersize, label=label, c=c)
					ax.errorbar(phase_list, np.log10(para_list), np.log10(para_err_list+np.abs(para_list))-np.log10(np.abs(para_list)), capsize=7, linestyle='', markersize=markersize, c=c, alpha=0.5)
				else:
					ax.plot(phase_list, para_list, marker=marker, markerfacecolor='none', markeredgecolor=c, linestyle='', markersize=markersize, label=label, c=c)
					ax.errorbar(phase_list, para_list, para_err_list, capsize=7, linestyle='', markersize=markersize, c=c, alpha=0.5)
				ymin = np.min([ymin, para_list.min() - para_err_list.max()])
				ymax = np.max([ymax, para_list.max() + para_err_list.max()])
			if otherfiles is not None:
				if other_dir is None:
					other_dir = self.name+'_results/'
				for file_i, file in enumerate(otherfiles):
					othername_ = file.split('_')[0]
					otherdata_ = np.loadtxt(other_dir+file, dtype='str')
					otherdata_len_ = len(otherdata_)
					correct_index = []
					for i in range(otherdata_len_):
						wavelength_ = otherdata_[i, 0].split('_')[0]
						if wavelength_ == rest_wavelength:
							correct_index.append(i)
					otherdata_ = otherdata_[correct_index]
					otherphase_ = otherdata_[:,1].astype('float')
					if phase_range is not None:
						mark_phase = (otherphase_ > phase_range[0]) * (otherphase_ < phase_range[1])
						otherdata_ = otherdata_[mark_phase]
						otherphase_ = otherphase_[mark_phase]
					otherpara_ = otherdata_[:,para_dict[plot_para]+2].astype('float')
					otherpara_err_ = otherdata_[:,para_dict[plot_para]+3].astype('float') 
					if ratio is not None:
						all_data[othername_][rest_wavelength] = pd.DataFrame({'phase':otherphase_,
																	   'para':otherpara_,
																	   'para_err':otherpara_err_}).sort_values(by='phase').reset_index(drop=True)
					else:
						if log_y:
							ax.plot(otherphase_, np.log10(np.abs(otherpara_)), capsize=7, marker=shape_dict[file_i], markerfacecolor='none', markeredgecolor=color_dict[file_i], linestyle='', markersize=markersize, c=color_dict[file_i], label=othername_)
							ax.errorbar(otherphase_, np.log10(np.abs(otherpara_)), np.log10(otherpara_err_+np.abs(otherpara_))-np.log10(np.abs(otherpara_)), capsize=7, linestyle='', markersize=markersize, c=color_dict[file_i], alpha=0.5)
						else:
							ax.plot(otherphase_, otherpara_, marker=shape_dict[file_i], markerfacecolor='none', markeredgecolor=color_dict[file_i], linestyle='', markersize=markersize, c=color_dict[file_i], label=othername_)
							ax.errorbar(otherphase_, otherpara_, otherpara_err_, capsize=7, linestyle='', markersize=markersize, c=color_dict[file_i], alpha=0.5)
						ymin = np.min([ymin, otherpara_.min() - otherpara_err_.max()])
						ymax = np.max([ymax, otherpara_.max() + otherpara_err_.max()])
		if ratio is not None:
			wavelength1, wavelength2 = ratio.split('_')
			name_n = 0
			for name, data in all_data.items():
				phase_list = []
				para_list = []
				para_err_list = []
				len1 = len(all_data[name][wavelength1])
				len2 = len(all_data[name][wavelength2])
				init2 = 0
				phase1 = all_data[name][wavelength1]['phase']
				para1 = all_data[name][wavelength1]['para']
				para_err1 = all_data[name][wavelength1]['para_err']
				phase2 = all_data[name][wavelength2]['phase']
				para2 = all_data[name][wavelength2]['para']
				para_err2 = all_data[name][wavelength2]['para_err']
				for i1 in range(len1):
					for i2 in range(init2, len2):
						if phase2[i2] == phase1[i1]:
							phase_list.append(phase2[i2])
							para_list.append(1/para2[i2]*para1[i1])
							para_err_list.append(1/para2[i2]*para1[i1]*np.sqrt((1/para1[i1]*para_err1[i1])**2 + (1/para2[i2]*para_err2[i2])**2))
							if table_file is not None:
								write_line_ = []
								write_line_.append(name)
								write_line_.append('%.1f'%phase_list[-1])
								write_line_.append('%.2f$\\pm$%.2f'%(para1[i1], para_err1[i1]))
								write_line_.append('%.2f$\\pm$%.2f'%(para2[i2], para_err2[i1]))
								write_line_.append('%.2f$\\pm$%.2f'%(para_list[-1], para_err_list[-1]))
								write_line_ = ' & '.join(write_line_) + ' \\\\\n'
								write_lines.append(write_line_)
							init2 = i2+1
				phase_list = np.array(phase_list)
				para_list = np.array(para_list)
				para_err_list = np.array(para_err_list)
				if name == self.name:
					c = 'black'
					marker = 'o'
					label = name
				else:
					c = color_dict[name_n]
					marker = shape_dict[name_n]
					label = name
					name_n += 1 
				if log_y:
					ax.errorbar(phase_list, np.log10(para_list), np.log10(para_err_list+np.abs(para_list))-np.log10(np.abs(para_list)), capsize=7, linestyle='', markersize=markersize, c=c, alpha=0.5)
					ax.plot(phase_list, np.log10(para_list), marker=marker, markerfacecolor='none', markeredgecolor=c, linestyle='', markersize=markersize, label=label, c=c)
				else:
					ax.errorbar(phase_list, para_list, para_err_list, capsize=7, linestyle='', markersize=markersize, c=c, alpha=0.5)
					ax.plot(phase_list, para_list, marker=marker, markerfacecolor='none', markeredgecolor=c, linestyle='', markersize=markersize, label=label, c=c)
				#ymin = np.min([ymin, para_list.min() - para_err_list.max()])
				ymin = np.min([ymin, para_list.min() - 0.5])
				ymax = np.max([ymax, para_list.max() + para_err_list.max()])

		edge_distance = 0.05
		if inverse_y:
			if log_y:
				ax.set_ylim((edge_distance+1)*np.log10(ymax)-edge_distance*np.log10(ymin), (edge_distance+1)*np.log10(ymin)-edge_distance*np.log10(ymax))
			else:
				ax.set_ylim((edge_distance+1)*ymax-edge_distance*ymin, (edge_distance+1)*ymin-edge_distance*ymax)
		else:
			if log_y:
				ax.set_ylim((edge_distance+1)*np.log10(ymin)-edge_distance*np.log10(ymax), (edge_distance+1)*np.log10(ymax)-edge_distance*np.log10(ymin))
			else:
				ax.set_ylim((edge_distance+1)*ymin-edge_distance*ymax, (edge_distance+1)*ymax-edge_distance*ymin)
		#ax.set_xlim(-19, -9)
		if feature_label is not None or otherfiles is not None:
			ax.legend(fontsize=14)
		if save:
			fig.savefig('%s.pdf'%save, bbox_inches='tight')
		if table_file is not None:
			with open('%s.txt'%table_file,'w') as f:
				f.writelines(write_lines)
		plt.show()



	def max_velocity(self, rest_wavelength, guess_wave=None, zone=600, method='guassian', window=5, save=True ,MC=30, subtract=False, gaussian_n = 1, calibrate=1, calibrate_err=0):
		if not self.max_index:
			phase_max = np.abs(self.spectraDf['phase']).min()
			for i in range(len(self.spectraDf)):
				if np.abs(self.spectraDf.loc[i, 'phase'])==phase_max:
					self.max_index = i
					break
		return self.Calculate_velocity(index=self.max_index, rest_wavelength=rest_wavelength, guess_wave=guess_wave, 
			subtract=subtract, zone=zone, method=method, window=window,save=save, MC=MC,gaussian_n=gaussian_n, calibrate=calibrate)

	def Calculate_velocity(self, index, rest_wavelength, guess_wave=None, zone=600, method='guassian', window=5, save=None, MC=30, subtract=False, guess=None, bounds=None, window_toAA=True,methods=None,
		kind = 'cubic', label=None, mask_region=None, velocity_formula = 'relativistic', guess_b=None, guess_r=None, calibrate=1, calibrate_err=0, fit_err=0, window_max=None, fit_smooth=True):
		np.random.seed(123456789)
		if velocity_formula == 'relativistic':
			lambda_to_velocity = f_rela
		else:
			lambda_to_velocity = f_common
		if save:
			save = self.name + '_results/' + save
		if not self.feature_endpoints:
			self.feature_endpoints = [0, -1]
		if not guess_wave:
			if type(rest_wavelength) == type(['haha']):
				guess_wave = np.mean(rest_wavelength)
			else:
				guess_wave = rest_wavelength
		if window_max is None:
			window_max = window + 30
		wave_p, flux_p, fluxerr_p = self.get_rest_(index, err=True)
		if window_toAA == True:
			window = int(window*(len(wave_p)-1)/(wave_p[-1] - wave_p[0]))  # AA to true window
			window = window + (window%2 + 1)%2
			window_max = int(window_max*(len(wave_p)-1)/(wave_p[-1] - wave_p[0]))  # AA to true window_max
			window_max = window_max + (window_max%2 + 1)%2
		plot_range = [guess_wave - zone, guess_wave + zone]
		mark_ = (wave_p > plot_range[0]) * (wave_p < plot_range[1])
		scale0 = flux_p[mark_].max()
		flux_p = flux_p / scale0
		fluxerr_p = fluxerr_p / scale0
		flux_psmooth = my_filter(flux_p, window, 1)
		mask_ = np.array([True for i in range(len(wave_p))])
		if mask_region is not None:
			for i in range(len(mask_region)):
				mark_ = mark_ * (~((wave_p > mask_region[i][0])*(wave_p < mask_region[i][1])))
		'''
		if window == 1:
			flux_smooth = flux_
		else:
			flux_smooth = signal.savgol_filter(flux_, window, 1)
		'''
		smooth_error_all = np.abs(flux_p - flux_psmooth)
		wave_ = wave_p[mark_]
		flux_ = flux_p[mark_]
		fluxerr_ = fluxerr_p[mark_]
		flux_smooth = flux_psmooth[mark_]

		fig, ax = plt.subplots()
		ax.set_xlabel('Wavelength [$\\rm \\AA$]')
		ax.set_ylabel('Scaled Flux')
		line1, = ax.plot(wave_, flux_, marker='o', linestyle = '', picker=True, pickradius=2, ms=2)
		ax.plot(wave_, flux_smooth, c='gray')
		line_blue = ax.axvline(wave_[self.feature_endpoints[0]],c='b')
		line_red = ax.axvline(wave_[self.feature_endpoints[1]],c='r')

		endpoints_pattern = ['b']
		
		def choose_endpoints(event):
			if event.key == 'b':
				endpoints_pattern[0] = 'b'
				ax.set_title('blue')
			elif event.key == 'r':
				endpoints_pattern[0] = 'r'
				ax.set_title('red')
			fig.canvas.draw_idle()

		def onpick(event):
			thisline = event.artist
			ind = event.ind
			xdata = thisline.get_xdata()
			if endpoints_pattern[0] == 'b':
				line_blue.set_xdata([xdata[ind[0]], xdata[ind[0]]])
				self.feature_endpoints[0] = ind[0]
			elif endpoints_pattern[0] == 'r':
				line_red.set_xdata([xdata[ind[0]], xdata[ind[0]]])
				self.feature_endpoints[1] = ind[0]
			fig.canvas.draw_idle()

		fig.canvas.mpl_connect('pick_event', onpick)
		fig.canvas.mpl_connect('key_press_event', choose_endpoints)

		plt.show()

		wave_fit = wave_[self.feature_endpoints[0] : (self.feature_endpoints[1]+1)]
		#flux_fit = flux_[self.feature_endpoints[0] : self.feature_endpoints[1]]
		if fit_smooth:
			flux_fit = flux_smooth[self.feature_endpoints[0] : (self.feature_endpoints[1]+1)]
		else:
			flux_fit = flux_[self.feature_endpoints[0] : (self.feature_endpoints[1]+1)]
		fluxerr_fit = fluxerr_[self.feature_endpoints[0] : (self.feature_endpoints[1]+1)]


		
		continuum = fcontinuum(wave_fit,  wave_fit, flux_smooth[self.feature_endpoints[0] : (self.feature_endpoints[1]+1)])
		if np.sum(flux_fit) > np.sum(continuum):
			line_type = 'emission'
		else:
			line_type = 'absorption'
		if subtract:
			flux_nor = flux_fit - continuum
			fluxerr_nor = fluxerr_fit - 0
		else:
			flux_nor = (flux_fit - continuum)/continuum
			fluxerr_nor = fluxerr_fit/continuum
		scale = np.abs(flux_nor).max()
		flux_nor = 1/scale*1000*flux_nor
		fluxerr_nor = 1/scale*1000*fluxerr_nor
		fig, ax = plt.subplots()
		if line_type == 'absorption':
			ax.set_ylabel('Normalized Flux Density $f_{\\lambda}$', fontsize=15)
		else:
			ax.set_ylabel('Scaled Flux Density $f_{\\lambda}$', fontsize=15)
		ax.set_xlabel('Rest Wavelength [$\\rm \\AA$]', fontsize=15)
		#if len(wave_fit) < 5:
		resolution = (wave_fit[-1] - wave_fit[0])/(len(wave_fit)-1)
		#else:
		#	resolution = (wave_fit[5] - wave_fit[0])/5
		if MC is None:
			random_edge = int(30 / resolution)
		else:
			random_edge = int(MC / resolution)
		print(random_edge)
		feature_endpoints_ = self.feature_endpoints.copy()
		print(feature_endpoints_)
		
		if method == 'gaussian':
			#def gaussians(x, params):

			#continuum = (flux_nor[0] - flux_nor[-1]) / (wave_fit[0] - wave_fit[-1]) * (wave_fit - wave_fit[0]) + flux_nor[0]
			if guess is None:
				if line_type == 'emission':
					guess = [1000, guess_wave, 10]
				else:
					guess = [-1000, guess_wave, 10]

			if bounds is None:
				bounds = ([-float('inf'),guess_wave-500,0],[float('inf'),guess_wave+500,float('inf')])
			params,params_covariance=optimize.curve_fit(gaussian, wave_fit, flux_nor, p0=guess, maxfev=500000, bounds=bounds)
			result_x = np.linspace(wave_fit[0], wave_fit[-1], 100)
			fit_result = gaussian(result_x, params[0], params[1], params[2])
			
			#print(params_covariance)
			

			print(params)


			
			ax.plot(wave_fit, flux_nor, c='b', label='data', linestyle='', markersize=markersize,marker='o')
			ax.plot(result_x, fit_result, c='r', label='fit')
			ax.axvline(x=params[1],linestyle='--')
			ax.legend()
			plt.show()
			go = int(input('go?: '))
			params[0] = params[0]*scale*1e-3
			params_covariance[0,0] = params_covariance[0,0]*scale**2*1e-6

			#velocity = (params[1] - rest_wavelength)/rest_wavelength * 300000 #km/s
			velocity = lambda_to_velocity(params[1]/rest_wavelength)

			if not subtract:
				pEW = np.abs(np.sqrt(2*np.pi)*params[0]*params[2])
			else:
				#pEW, pEW_integ_err = itg.quad(lambda x: gaussian(x, params[0], params[1], params[2]) / fcontinuum(x, wave_fit, flux_fit), wave_fit[0], wave_fit[-1])
				#flux
				pEW = np.abs(np.sqrt(2*np.pi)*params[0]*params[2])

			#FWHM = np.sqrt(2*np.log(2))*params[2]/rest_wavelength * 300000 * 2
			FWHM = lambda_to_velocity(2.355*params[2]/rest_wavelength+1)
			print(velocity)

			

			velocity_list = []
			FWHM_list = []
			pEW_list = []
			
			if MC is not None:
				for i in range(1000):
					print(i, end='\r')
					feature_endpoints_[0] = self.feature_endpoints[0] + np.random.randint(2*random_edge+1) - random_edge
					feature_endpoints_[1] = self.feature_endpoints[1] + np.random.randint(2*random_edge+1) - random_edge
					wave_fit = wave_[feature_endpoints_[0] : feature_endpoints_[1]]
					flux_fit = flux_smooth[feature_endpoints_[0] : feature_endpoints_[1]]
					continuum = fcontinuum(wave_fit,  wave_fit, flux_fit)
					if subtract:
						flux_nor = flux_fit - continuum
					else:
						flux_nor = (flux_fit - continuum)/continuum
					scale = np.abs(flux_nor).max()
					flux_nor = 1/scale*1000*flux_nor
					params_,params_covariance_=optimize.curve_fit(gaussian, wave_fit, flux_nor, p0=params, maxfev=500000, bounds=bounds)
					params_[0] = params[0]*scale*1e-3

					#velocity_ = (params_[1] - rest_wavelength)/rest_wavelength * 300000 #km/s
					velocity_ = lambda_to_velocity(params_[1]/rest_wavelength)
					if not subtract:
						pEW_ = np.abs(2.507*params_[0]*params_[2]) # np.sqrt(2*np.pi)=2.507
					else:
						'''
						def fpEW_(x):
							return (gaussian(x, params_[0], params[1], params[2])) / fcontinuum(x, wave_fit, flux_fit)
						pEW_, pEW_integ_err_ = itg.quad(fpEW_, wave_fit[0], wave_fit[-1])
						'''
						'''
						pEW_, pEW_integ_err_ = itg.quad(lambda x: gaussian(x, params[0], params[1], params[2]) / fcontinuum(x, wave_fit, flux_fit), wave_fit[0], wave_fit[-1])
						'''
						# flux
						pEW_ = np.abs(2.507*params_[0]*params_[2]) # np.sqrt(2*np.pi)=2.507
					#FWHM_ = 2.355*params_[2]/rest_wavelength * 300000 # np.sqrt(2*np.log(2))*2=2.355
					FWHM_ = lambda_to_velocity(2.355*params_[2]/rest_wavelength+1)
					velocity_list.append(velocity_) #km/s
					FWHM_list.append(FWHM_)
					pEW_list.append(pEW_)
				velocity_MCerr = np.std(velocity_list, ddof=1)
				FWHM_MCerr = np.std(FWHM_list, ddof=1)
				pEW_MCerr = np.std(pEW_list, ddof=1)
			else:
				velocity_MCerr = 0.
				FWHM_MCerr = 0.
				pEW_MCerr = 0.
			#print(params_covariance[1,1]**0.5/rest_wavelength * 300000)
			velocity_uncertainty = np.sqrt(velocity_MCerr**2 + (params_covariance[1,1]**0.5/rest_wavelength * 300000)**2*fit_err)
			FWHM_uncertainty = np.sqrt(FWHM_MCerr**2 + (params_covariance[2,2]*2.355/rest_wavelength * 300000)**2*fit_err)
			pEW_uncertainty = np.sqrt(pEW_MCerr**2 + 2.507**2*(params_covariance[0,0]*params[2]**2 + params_covariance[2,2]*params[0]**2)*fit_err)
		elif method == 'direct':
			if line_type == 'absorption':
				flux_fit_direct = flux_nor
				#flux_fit_direct = signal.savgol_filter(flux_nor, window, 1)
			else:
				flux_fit_direct = -1*flux_nor
				#flux_fit_direct = -1*signal.savgol_filter(flux_nor, window, 1)
			if fit_smooth == True:
				f_flux_fit = interp1d(wave_fit, flux_fit_direct, kind=kind)
			else:
				def f_flux_fit(x):
					f_interp = fit1dcurve.Interpolator(type='hyperspline',x=wave_fit,y=flux_fit_direct,dy=flux_fit_direct/flux_fit_direct)
					return f_interp(x)[0]
			#plt.plot(np.linspace(wave_fit[0], wave_fit[-1],1000), f_flux_fit(np.linspace(wave_fit[0], wave_fit[-1],1000)))
			#plt.show()
			#exit()
			if guess is None:
				guess = guess_wave
			res_v = optimize.minimize(f_flux_fit, guess, bounds=[(wave_fit[0], wave_fit[-1])]).x[0]
			min_ = f_flux_fit(res_v)
			half_ = min_*0.5
			wave_b = np.linspace(wave_fit[0],res_v,50)
			wave_r = np.linspace(res_v,wave_fit[-1],50)
			flux_b = f_flux_fit(wave_b)
			flux_r = f_flux_fit(wave_r)
			bounds_b_min = wave_b[flux_b<0.25*min_][0]
			bounds_b_max = wave_b[flux_b>0.75*min_][-1]
			bounds_b = [(bounds_b_min,bounds_b_max)]
			if guess_b is None:
				guess_b = (bounds_b_min+bounds_b_max)/2
			else:
				bounds_b = None
			bounds_r_max = wave_r[flux_r<0.25*min_][-1]
			bounds_r_min = wave_r[flux_r>0.75*min_][0]
			bounds_r = [(bounds_r_min,bounds_r_max)]
			if guess_r is None:
				guess_r = (bounds_r_min+bounds_r_max)/2
			else:
				bounds_r = None
			print(bounds_b, bounds_r)

			res_w_b = optimize.minimize(lambda x: (1/half_*(f_flux_fit(x)-half_)*10)**2, guess_b, bounds=bounds_b).x[0]
			res_w_r = optimize.minimize(lambda x: (1/half_*(f_flux_fit(x)-half_)*10)**2, guess_r, bounds=bounds_r).x[0]
			res_w = abs(res_w_r - res_w_b)
			if not subtract:
				pEW, pEW_integ_err = itg.quad(lambda x: -1*f_flux_fit(x), wave_fit[0], wave_fit[-1])
			else:
				#pEW, pEW_integ_err = itg.quad(lambda x: f_flux_fit(x)*scale*1e-3 / fcontinuum(x, wave_fit, flux_fit), wave_fit[0], wave_fit[-1])
				# flux
				pEW, pEW_integ_err = itg.quad(lambda x: f_flux_fit(x), wave_fit[0], wave_fit[-1])
			pEW = pEW * scale*1e-3
			

				
			#print(f_flux_fit(np.array([res_w_b, res_v, res_w_r])))
			#print((f_flux_fit(res_w_b)-half_)**2, (f_flux_fit(res_w_r)-half_)**2)
			flux_br = np.concatenate([np.linspace(wave_fit[0],wave_fit[-1],100), wave_fit]).sort()
			ax.plot(np.linspace(wave_fit[0],wave_fit[-1],100), f_flux_fit(np.linspace(wave_fit[0],wave_fit[-1],100)), c='b')
			ax.plot(wave_fit, flux_fit_direct, c='purple')
			ax.axvline(res_v, c = 'black')
			ax.axvline(res_w_b, c = 'b')
			ax.axvline(res_w_r, c = 'r')
			plt.show()

			#velocity = (res_v - rest_wavelength)/rest_wavelength * 300000 #km/s
			#FWHM = res_w/rest_wavelength * 300000
			velocity = lambda_to_velocity(res_v/rest_wavelength)
			FWHM = lambda_to_velocity(res_w_r/rest_wavelength) - lambda_to_velocity(res_w_b/rest_wavelength)
			print(velocity, FWHM, pEW, self.spectraDf.loc[index, 'phase'])

			go = int(input('go?: '))
			

			velocity_list = []
			FWHM_list = []
			pEW_list = []

			#print(self.feature_endpoints[0], self.feature_endpoints[1])
			for i in range(1000):
				print(i, end='\r')
				window_ = np.random.randint(window, window_max+1)
				#window_ = int(window_/resolution)
				window_ = window_ + (window_%2 + 1)%2
				flux_psmooth = my_filter(flux_p, window_, 1)
				flux_fit_ = flux_psmooth[mark_]
				if MC:
					feature_endpoints_[0] = self.feature_endpoints[0] + np.random.randint(2*random_edge+1) - random_edge
					feature_endpoints_[1] = self.feature_endpoints[1] + np.random.randint(2*random_edge+1) - random_edge	
				wave_fit = wave_[feature_endpoints_[0] : feature_endpoints_[1]]
				flux_fit_ = flux_fit_[feature_endpoints_[0] : feature_endpoints_[1]]
					#smooth_error = smooth_error_all[feature_endpoints_[0] : feature_endpoints_[1]]
				#else:
				#	smooth_error = smooth_error_all[self.feature_endpoints[0] : self.feature_endpoints[1]]
				#flux_fit_ = flux_fit + np.random.normal(scale=smooth_error)*smooth_error
				continuum = fcontinuum(wave_fit,  wave_fit, flux_fit_)
				if subtract:
					flux_nor = flux_fit_ - continuum
				else:
					flux_nor = (flux_fit_ - continuum)/continuum
				scale = np.abs(flux_nor).max()
				flux_nor = 1/scale*1000*flux_nor
				if line_type == 'emission':
					flux_fit_direct = -flux_nor
				else:
					flux_fit_direct = flux_nor
					
				f_flux_fit_ = interp1d(wave_fit, flux_fit_direct, kind=kind)
				try:
					res_v_ = optimize.minimize(f_flux_fit_, res_v, bounds=[(wave_fit[0]+1,wave_fit[-1]-1)]).x[0] #bounds=([wave_fit[0]+1,wave_fit[-1]-1])
				except Exception as e:
					print(e)
					plt.plot(wave_fit, f_flux_fit_(wave_fit), c='b')
					plt.plot(wave_fit, flux_nor, c='r', linestyle='', markersize=markersize, marker='o')
					plt.plot(wave_fit, flux_fit_direct, c='purple')
					plt.show()
				'''
				print(res_v_)
				
				plt.plot(wave_fit, f_flux_fit_(wave_fit), c='b', label='inter')
				plt.plot(wave_fit, flux_nor, c='r', linestyle='', markersize=markersize, marker='o', label='data')
				plt.plot(wave_fit, flux_fit_direct, c='purple', linestyle='', markersize=markersize, marker='o', label='smooth')
				plt.axvline(res_v_, c = 'black')
				plt.legend()
				plt.show()
				go = int(input('go?:'))
				'''
				half_ = f_flux_fit_(res_v_)*0.5
				bounds_b_ = [(wave_fit[0]+1, res_v_)]
				bounds_r_ = [(res_v_, wave_fit[-1]-1)]
				res_w_b_ = optimize.minimize(lambda x: (1/half_*(f_flux_fit_(x)-half_)*10)**2, res_w_b, bounds=bounds_b_).x[0]
				res_w_r_ = optimize.minimize(lambda x: (1/half_*(f_flux_fit_(x)-half_)*10)**2, res_w_r, bounds=bounds_r_).x[0]
				res_w_ = abs(res_w_r_ - res_w_b_)

				if not subtract:
					pEW_, pEW_integ_err_ = itg.quad(lambda x: -1*f_flux_fit_(x), wave_fit[0], wave_fit[-1])
				else:
					#pEW_, pEW_integ_err_ = itg.quad(lambda x: f_flux_fit_(x) / fcontinuum(x, wave_fit, flux_fit_), wave_fit[0], wave_fit[-1])
					#flux
					pEW_, pEW_integ_err_ = itg.quad(lambda x: f_flux_fit_(x), wave_fit[0], wave_fit[-1])
				pEW_ = pEW_ * scale*1e-3

				#rint(window_, feature_endpoints_[0], feature_endpoints_[1], pEW_)
				#plt.plot(wave_fit, f_flux_fit_(wave_fit), c='b', label='inter')
				#plt.plot(wave_fit, flux_nor, c='r', linestyle='', markersize=markersize, marker='o', label='data')
				#plt.plot(wave_fit, flux_fit_direct, c='purple', linestyle='', markersize=markersize, marker='o', label='smooth')
				#plt.legend()
				#plt.show()
				#go = int(input('go?: '))
				
				#velocity_ = (res_v_ - rest_wavelength)/rest_wavelength * 300000 #km/s
				#FWHM_ = res_w_/rest_wavelength * 300000
				velocity_ = lambda_to_velocity(res_v_/rest_wavelength)
				FWHM_ = lambda_to_velocity(res_w_r_/rest_wavelength) - lambda_to_velocity(res_w_b_/rest_wavelength)

				velocity_list.append(velocity_) #km/s
				FWHM_list.append(FWHM_)
				pEW_list.append(pEW_)

			velocity = np.mean(velocity_list)
			velocity_uncertainty = np.std(velocity_list, ddof=1)
			FWHM = np.mean(FWHM_list)
			FWHM_uncertainty = np.std(FWHM_list, ddof=1)
			pEW = np.mean(pEW_list)
			pEW_uncertainty = np.std(pEW_list, ddof=1)

		elif type(method) == type(gaussian):
			print(bounds)
			print(guess)
			params,params_covariance=optimize.curve_fit(method, wave_fit, flux_nor, sigma=fluxerr_nor, p0=guess, maxfev=500000, bounds=bounds)
			print(params)
			print(np.sum(((method(wave_fit, *params) - flux_nor)/fluxerr_nor)**2) / (len(wave_fit) - len(params)))
			#print(params_covariance)		
			params = [638.05303814,  674.6194568,  1878.97827889]
			#print(2.507*params[3]*params[5]*scale*1e-3)

			#result_x = np.linspace(wave_fit[0], wave_fit[-1], 100)
			#fig, ax = plt.subplots() 
			#ax.plot(wave_fit, flux_nor, c='b', label='smoothed data', linestyle='', markersize=markersize, marker='o')
			wave_plot = wave_[(feature_endpoints_[0] - 10) : (feature_endpoints_[1] + 10)]
			flux_plot = flux_smooth[(feature_endpoints_[0] - 10) : (feature_endpoints_[1] + 10)]
			fluxerr_plot = fluxerr_[(feature_endpoints_[0] - 10) : (feature_endpoints_[1] + 10)]
			ax.errorbar(wave_plot, flux_plot, yerr=fluxerr_plot, c='black', linestyle='', alpha=0.25)
			ax.plot(wave_plot, flux_plot, c='black', label='data')
			ax.plot(wave_fit, methods[0](wave_fit, *params)*scale*1e-3 + continuum, c='purple', label='[C I] 9825 $\\rm \\AA$', linestyle='--')
			ax.plot(wave_fit, methods[1](wave_fit, *params)*scale*1e-3 + continuum, c='green', label='[C I] 9850 $\\rm \\AA$', linestyle='--')
			ax.plot(wave_fit, method(wave_fit, *params)*scale*1e-3 + continuum, c='r', label='best fit')
			ax.plot(wave_fit, continuum, c='b', label='pseudocontinuum')
			ax.legend(fontsize=15)
			plt.savefig('SN2022pul_results/scatter_fit.pdf', bbox_inches='tight')
			plt.show()
			exit()

			### MCMC
			len_para = len(params)
			np.random.seed(123456789)
			def log_likelihood(theta, wave_fit, flux_nor, fluxerr_nor):
				return -0.5*np.sum(((flux_nor - method(wave_fit, *theta))/fluxerr_nor)**2)
			def log_prior(theta):
				for item_i, item in enumerate(theta):
					if item < bounds[0][item_i] or item > bounds[1][item_i]:
						return -np.inf
				return 0.0
			def log_probability(theta, wave_fit, flux_nor, fluxerr_nor):
				lp = log_prior(theta)
				if not np.isfinite(lp):
					return -np.inf
				return lp + log_likelihood(theta, wave_fit, flux_nor, fluxerr_nor)
			rand_start = np.zeros([32,len_para])
			for start_i in range(1,32):
				while(1):
					for para_i in range(len_para):
						rand_start[start_i][para_i] = np.random.randn()*params[para_i]*0.05
					if not np.isinf(log_prior(params + rand_start[start_i])):
						break
			start_MC = params + rand_start
			nwalkers, ndim = start_MC.shape
			steps = 20000
			AutocorrError = emcee.autocorr.AutocorrError
			while(1):
				sampler = emcee.EnsembleSampler(
					nwalkers, ndim, log_probability, args=(wave_fit, flux_nor, fluxerr_nor)
				)
				sampler.run_mcmc(start_MC, steps, progress=True)
				try:
					tau = sampler.get_autocorr_time()
				except AutocorrError:
					steps *= 2
				else:
					break
			samples = sampler.get_chain()
			fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
			labels = ['A', 'v', 'width']
			for i in range(ndim):
			    ax = axes[i]
			    ax.plot(samples[:, :, i], "k", alpha=0.3)
			    ax.set_xlim(0, len(samples))
			    ax.set_ylabel(labels[i])
			    ax.yaxis.set_label_coords(-0.1, 0.5)

			axes[-1].set_xlabel("step number")
			plt.show()
			flat_samples = sampler.get_chain(discard=int(2.5*np.max(tau)), thin=int(np.max(tau)/2), flat=True)
			log_likelihood_value = []
			for i in range(flat_samples.shape[0]):
				log_likelihood_value.append(log_likelihood(flat_samples[i], wave_fit, flux_nor, fluxerr_nor))
			max_log_likelihood = np.max(log_likelihood_value)
			for i in range(flat_samples.shape[0]):
				if log_likelihood_value[i] == max_log_likelihood:
					pos_max = i
			theta = flat_samples[pos_max]
			print(theta)
			print(np.sum(((method(wave_fit, *theta) - flux_nor)/fluxerr_nor)**2) / (len(wave_fit) - len(theta)))
			fig = corner.corner(flat_samples, labels=labels, truths=theta)
			plt.show()
			for i in range(len_para):
				print(labels[i])
				print(np.percentile(flat_samples[:,i], [16, 50, 84]))
			###

			go = int(input('go?: '))

			params_list = []
			for i in range(len(params)):
				params_list.append([])
			
			if MC is not None:
				for i in range(1000):
					print(i, end='\r')
					feature_endpoints_[0] = self.feature_endpoints[0] + np.random.randint(2*random_edge+1) - random_edge
					feature_endpoints_[1] = self.feature_endpoints[1] + np.random.randint(2*random_edge+1) - random_edge
					
					wave_fit = wave_[feature_endpoints_[0] : feature_endpoints_[1]]
					flux_fit = flux_smooth[feature_endpoints_[0] : feature_endpoints_[1]]
					continuum = fcontinuum(wave_fit,  wave_fit, flux_fit)
					if subtract:
						flux_nor = flux_fit - continuum
					else:
						flux_nor = (flux_fit - continuum)/continuum
					flux_nor = 1/scale*1000*flux_nor
					params_,params_covariance_=optimize.curve_fit(method, wave_fit, flux_nor, p0=params, maxfev=500000, bounds=bounds)
					#print(feature_endpoints_, scale)
					#print(2.507*params_[3]*params_[5]*scale*1e-3)
					#go = int(input('go?: '))
					#print(params_)
					#fig, ax = plt.subplots() 
					#ax.plot(wave_fit, flux_nor, c='b', label='smoothed data', linestyle='', markersize=markersize,marker='o')
					#ax.plot(result_x, gaussian(result_x, params[0], params[1], params[2]), c='purple', label='gaussian1')
					#ax.plot(result_x, gaussian(result_x, params[3], params[4], params[5]), c='green', label='gaussian2')
					#ax.plot(result_x, method(result_x, *params), c='r', label='all')
					#ax.legend()
					#plt.show()
					#go = input('go?: ')
					for i in range(len(params_list)):
						params_list[i].append(params_[i])
					params_MCerr_list = [np.std(params_list[i], ddof=1) for i in range(len(params_list))]
			else:
				#params_MCerr_list = [0 for i in range(len(params_list))]
				para_list = None
			#params_uncertainty = [np.sqrt(params_MCerr_list[i]**2 + params_covariance[i,i]*fit_err) for i in range(len(params_list))]
			if subtract:
				scale = scale*scale0
			return params, params_list, scale*1e-3*calibrate, wave_fit, flux_nor, params_covariance
		else:
			raise Excpetion('method %s is not included'%method)
		if subtract:
			pEW = pEW*scale0*calibrate
			pEW_uncertainty = np.sqrt((pEW_uncertainty*scale0*calibrate)**2 + (pEW*calibrate_err)**2)
		# Save
		self.velocities[str(rest_wavelength)+'_'+str(index)] = [velocity, velocity_uncertainty, FWHM, FWHM_uncertainty, pEW, pEW_uncertainty]
		if save is not None:
			with open(save, 'a') as f:
				phase_ = self.spectraDf.loc[index, 'phase']
				if label is not None:
					f.writelines('%s %.1f %s %s %s %s %s %s\n' %(str(rest_wavelength)+label+'_'+str(index), phase_, velocity, velocity_uncertainty, FWHM, FWHM_uncertainty, pEW, pEW_uncertainty))
				else:
					f.writelines('%s %.1f %s %s %s %s %s %s\n' %(str(rest_wavelength)+'_'+str(index), phase_, velocity, velocity_uncertainty, FWHM, FWHM_uncertainty, pEW, pEW_uncertainty))
		self.feature_endpoints = [0, -1]
		return [velocity, velocity_uncertainty, FWHM, FWHM_uncertainty, pEW, pEW_uncertainty] 

	def Calculate_double_velocity(self, index, zone=1000, window=5, save=None, MC=30, subtract=False, guess=None, bounds=None, rest_wavelength=[8579, 8579], guess_wave=8000,
		velocity_formula = 'relativistic', calibrate=1, calibrate_err=0, fit_err=0, window_max=None, mask_region=None):
		if save:
			save = self.name + '_results/' + save
		if velocity_formula == 'relativistic':
			lambda_to_velocity = f_rela
		else:
			lambda_to_velocity = f_common
		def gaussian_2(x, A1, mu1, sigma1, A2, mu2, sigma2):
			#mu2 = 6364*mu1/6300
			#sigma2 = 6364*sigma1/6300
			return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2)
		if guess is None:
			guess = [-1, 7900, 50, -1, 8100, 50]
		if bounds is None:
			bounds = ([-np.inf,7700,0,-np.inf,7900,0,],[np.inf,8200,np.inf,np.inf,8500,np.inf])
		#params, params_uncertainty, scale, wave_fit, flux_nor = self.Calculate_velocity(rest_wavelength=rest_wavelength,guess_wave=guess_wave,method=gaussian_2, 
		#	save=save,window=window,zone=zone,index=index,MC=MC,subtract=subtract,guess=guess,bounds=bounds,velocity_formula=velocity_formula,
		#	calibrate=calibrate, calibrate_err=calibrate_err, fit_err=fit_err, mask_region=mask_region)
		params, params_list, scale, wave_fit, flux_nor, params_covariance = self.Calculate_velocity(rest_wavelength=rest_wavelength,guess_wave=guess_wave,method=gaussian_2, 
			save=save,window=window,zone=zone,index=index,MC=MC,subtract=subtract,guess=guess,bounds=bounds,velocity_formula=velocity_formula,
			calibrate=calibrate, calibrate_err=calibrate_err, fit_err=fit_err, mask_region=mask_region)
		#velocity_HV = (params[1]-rest_wavelength[0])/rest_wavelength[0]*300000
		#FWHM_HV = 2.355*params[2]/rest_wavelength[0] * 300000
		velocity_HV = lambda_to_velocity(params[1]/rest_wavelength[0])
		FWHM_HV = 2.355*lambda_to_velocity(params[2]/rest_wavelength[0]+1)
		pEW_HV = np.abs(2.507*params[0]*params[2]*scale)
		#velocity_NV = (params[4]-rest_wavelength[1])/rest_wavelength[1]*300000
		#FWHM_NV = 2.355*params[5]/rest_wavelength[1] * 300000
		velocity_NV = lambda_to_velocity(params[4]/rest_wavelength[1])
		FWHM_NV = 2.355*lambda_to_velocity(params[5]/rest_wavelength[1]+1)
		pEW_NV = np.abs(2.507*params[3]*params[5]*scale)

		params_list = np.array(params_list)
		if params_list is not None:
			velocity_HV_MC_uncertainty = np.std(lambda_to_velocity(params_list[1]/rest_wavelength[0]), ddof=0)
			FWHM_HV_MC_uncertainty = np.std(2.355*lambda_to_velocity(params_list[2]/rest_wavelength[0]+1), ddof=0)
			pEW_HV_MC_uncertainty = np.std(np.abs(2.507*params_list[0]*params_list[2]*scale), ddof=0)
			velocity_NV_MC_uncertainty = np.std(lambda_to_velocity(params_list[4]/rest_wavelength[1]), ddof=0)
			FWHM_NV_MC_uncertainty = np.std(2.355*lambda_to_velocity(params_list[5]/rest_wavelength[1]+1), ddof=0)
			pEW_NV_MC_uncertainty = np.std(np.abs(2.507*params_list[3]*params_list[5]*scale), ddof=0)
		else:
			velocity_HV_MC_uncertainty = 0
			FWHM_HV_MC_uncertainty = 0
			pEW_HV_MC_uncertainty = 0
			velocity_NV_MC_uncertainty = 0
			FWHM_NV_MC_uncertainty = 0
			pEW_NV_MC_uncertainty = 0
		params_uncertainty = [params_covariance[i][i]**0.5 for i in range(len(params))]
		velocity_HV_FIT_uncertainty = np.abs(lambda_to_velocity((params_uncertainty[1]+params[1])/rest_wavelength[0]) - 
			lambda_to_velocity((-params_uncertainty[1]+params[1])/rest_wavelength[0]))/2
		FWHM_HV_FIT_uncertainty = np.abs(lambda_to_velocity((params_uncertainty[2]+params[2])/rest_wavelength[0]+1) - 
			lambda_to_velocity((-params_uncertainty[2]+params[2])/rest_wavelength[0]+1))/2
		pEW_HV_FIT_uncertainty = pEW_HV*np.sqrt(params_uncertainty[0]**2/params[0]**2 + params_uncertainty[2]**2/params[2]**2 + calibrate_err**2)

		velocity_NV_FIT_uncertainty = np.abs(lambda_to_velocity((params_uncertainty[4]+params[4])/rest_wavelength[1]) - 
			lambda_to_velocity((-params_uncertainty[4]+params[4])/rest_wavelength[1]))/2
		FWHM_NV_FIT_uncertainty = np.abs(lambda_to_velocity((params_uncertainty[5]+params[5])/rest_wavelength[1]+1) - 
			lambda_to_velocity((-params_uncertainty[5]+params[5])/rest_wavelength[1]+1))/2
		pEW_NV_FIT_uncertainty = pEW_NV*np.sqrt(params_uncertainty[3]**2/params[3]**2 + params_uncertainty[5]**2/params[5]**2 + calibrate_err**2)

		velocity_HV_uncertainty = np.sqrt(velocity_HV_MC_uncertainty**2 + velocity_HV_FIT_uncertainty**2)
		FWHM_HV_uncertainty = np.sqrt(FWHM_HV_MC_uncertainty**2 + FWHM_HV_FIT_uncertainty**2)
		pEW_HV_uncertainty = np.sqrt(pEW_HV_MC_uncertainty**2 + pEW_HV_FIT_uncertainty**2)

		velocity_NV_uncertainty = np.sqrt(velocity_NV_MC_uncertainty**2 + velocity_NV_FIT_uncertainty**2)
		FWHM_NV_uncertainty = np.sqrt(FWHM_NV_MC_uncertainty**2 + FWHM_NV_FIT_uncertainty**2)
		pEW_NV_uncertainty = np.sqrt(pEW_NV_MC_uncertainty**2 + pEW_NV_FIT_uncertainty**2)
		#print(velocity_NV_MC_uncertainty, velocity_NV_FIT_uncertainty)
		#print(FWHM_NV_MC_uncertainty, FWHM_NV_FIT_uncertainty)
		#print(pEW_NV_MC_uncertainty, pEW_NV_FIT_uncertainty)
		#velocity_HV_uncertainty = params_uncertainty[1]/rest_wavelength[0]*300000
		#FWHM_HV_uncertainty = params_uncertainty[2]/rest_wavelength[0]*300000
		#velocity_HV_uncertainty = lambda_to_velocity((params_uncertainty[1])/rest_wavelength[0]+1)
		#FWHM_HV_uncertainty = lambda_to_velocity(params_uncertainty[2]/rest_wavelength[0]+1)
		#pEW_HV_uncertainty = pEW_HV*np.sqrt(params_uncertainty[0]**2/params[0]**2 + params_uncertainty[2]**2/params[2]**2 + calibrate_err**2)

		#velocity_HV_uncertainty = np.std(lambda_to_velocity(params_list[1]/rest_wavelength[0]), ddof=0)
		#velocity_NV_uncertainty = params_uncertainty[4]/rest_wavelength[1]*300000
		#FWHM_NV_uncertainty = params_uncertainty[5]/rest_wavelength[1]*300000
		#velocity_NV_uncertainty = lambda_to_velocity((params_uncertainty[4])/rest_wavelength[0]+1)
		#FWHM_NV_uncertainty = lambda_to_velocity(params_uncertainty[5]/rest_wavelength[0]+1)
		#pEW_NV_uncertainty = pEW_HV*np.sqrt(params_uncertainty[3]**2/params[3]**2 + params_uncertainty[5]**2/params[5]**2 + calibrate_err**2)

		# Save
		save_name = ['_', '_']
		if rest_wavelength[0] == rest_wavelength[1]:
			save_name[0] = str(rest_wavelength[0])+'_HV_'+str(index)
			save_name[1] = str(rest_wavelength[1])+'_NV_'+str(index)
		else:
			save_name[0] = str(rest_wavelength[0])+'_'+str(index)
			save_name[1] = str(rest_wavelength[1])+'_'+str(index)
		self.velocities[save_name[0]] = [velocity_HV, velocity_HV_uncertainty, FWHM_HV, FWHM_HV_uncertainty, pEW_HV, pEW_HV_uncertainty]
		self.velocities[save_name[1]] = [velocity_NV, velocity_NV_uncertainty, FWHM_NV, FWHM_NV_uncertainty, pEW_NV, pEW_NV_uncertainty]
		if save is not None:
			with open(save, 'a') as f:
				f.writelines('%s %.1f %s %s %s %s %s %s\n' %(save_name[0], self.spectraDf.loc[index, 'phase'],
					velocity_HV, velocity_HV_uncertainty, FWHM_HV, FWHM_HV_uncertainty, pEW_HV, pEW_HV_uncertainty))
				f.writelines('%s %.1f %s %s %s %s %s %s\n' %(save_name[1], self.spectraDf.loc[index, 'phase'],
					velocity_NV, velocity_NV_uncertainty, FWHM_NV, FWHM_NV_uncertainty, pEW_NV, pEW_NV_uncertainty))

	def Latex_table(self, outfile='spec_tex_table.txt'):
		outfile = self.name + '_results/' + outfile
		tel_dict = {'Xinglong  2.16m':'XLT','Other':'Other','2m4-01':'LJT','OHP-2m':'OHP','TNG':'TNG','EBE':'EBE','ESOU':'ESOU','SALT':'SALT','GTC':'GTC',
			'1.22m Reflector':'Pennar1.22','1.82m Reflector':'Copernico','shane':'Shane', 'Keck II':'Keck II', 'Keck I':'Keck I','UH88':'UH88','ANU-2.3m':'ANU-2.3m'}
		inst_dict_none = {'Shane':'Kast'}
		inst_dict = {'BFOSC':'BFOSC','Other':'Other','yf01':'YFOSC','MISTRAL':'MISTRAL','TNG':'TNG','EBE':'EBE','ESOU':'ESOU','RSS':'RSS','OSIRIS':'OSIRIS',
			'Andor iDus DU440A-BU2':'B&C','AFOSC + Andor iKon-L DZ936N-BEX2-DD-9HF':'AFOSC','FOSC-ES32':'FOSC-ES32','AFOSC + Andor iKon-L DZ936N-BEX2-DD-9V6':'AFOSC',
			'DEIMOS: real science mosaic CCD subsystem with PowerPC in VME crate':'DEIMOS','SNIFS':'SNIFS','LRIS+LRISBLUE':'LRIS','LRS':'LRS','WiFeS':'WiFeS','LRISpBLUE':'LRISpBLUE'}
		with open(outfile, 'w') as f:
			f.writelines('MJD & Date & Phase & Range($\\rm \\AA$) & Exposure(s) & Instrument/Telescope\n')
			for i in range(len(self.spectraDf)):
				MJD_ = self.spectraDf.loc[i, 'MJD']
				date_ = mjdToDate(MJD_).split('T')[0].replace('-','')
				phase_ = self.spectraDf.loc[i, 'phase']
				Range_b_ = self.spectraDf.loc[i, 'spectra'][0, 0]
				Range_r_ = self.spectraDf.loc[i, 'spectra'][0, -1]
				expt_ = self.spectraDf.loc[i, 'expt']
				telescope_ = tel_dict[self.spectraDf.loc[i, 'tel']]
				instrument_ =  self.spectraDf.loc[i, 'inst']
				if instrument_ == 'what?':
					instrument_ = inst_dict_none[telescope_]
				else:
					instrument_ = inst_dict[instrument_]
				if telescope_ == 'Shane' and Range_r_ < 9000:
					instrument_ = 'HIRES'
				
				f.writelines('%.1f & %s & %.1f & %d-%d & %s & %s/%s \\\\\n'%(MJD_, date_, phase_, Range_b_, Range_r_, expt_, instrument_, telescope_))
			f.writelines('Relative to the $B$-band maximum light, MJD$_{B\\rm max}=%.3f$ \n'%self.Tmax)





def checkFits(files):
		# Check whether exists a fits file
		if 'fits' in ''.join(map(lambda x: x.split('.')[-1], files)):
			return 1
		else:
			return 0

def checkCsv(files):
		# Check whether exists a fits file
		if 'csv' in ''.join(map(lambda x: x.split('.')[-1], files)):
			return 1
		else:
			return 0

def gaussian(x, A, mu, sigma):
	return A*np.exp(-(x-mu)**2/2/sigma**2)

def fcontinuum(x, wave_fit, flux_fit):
	return (flux_fit[0] - flux_fit[-1]) / (wave_fit[0] - wave_fit[-1]) * (x - wave_fit[0]) + flux_fit[0]

#spectraData1 = SpectraData('/Users/liujialian/work/SN2022hrs/data/spectroscopy')
#print(spectraData1.spectraDf)
'''
class SpectraPlotter:

	def __init__(self, spectraData):
		self.spectraDf = spectraData.spectraDf

	def Plot(self):
		plotNumber = 0
		fig, ax = plt.subplots()
		ax.set_xlabel('Wavelength [$\\rm \\AA$]')
		ax.set_ylabel('Scaled Flux + Constants')
		#ax.set_ylabel('FLux [$\\rm erg\\ cm^{-2}\\ s^{-1}\\ \\AA^{-1}$]')
		for i in range(len(self.spectraDf)):
			ax.plot(self.spectraDf.loc[i, 'spectra'][0], 
				self.spectraDf.loc[i, 'spectra'][1]/self.spectraDf.loc[i, 'spectra'][1].mean()-plotNumber)
			plotNumber += 1
		plt.show()
'''

def quick_plot(filenames, style='line', z=0, scale=None, smooth=None, scale_method='max', scale_smooth=1):
	if smooth is not None:
		scale_smooth = smooth
	if type(filenames) == type('haha'):
		filenames = [filenames]
		z = [z]
	fig, ax = plt.subplots()
	for file_i, filename in enumerate(filenames):
		data = np.loadtxt(filename)
		if scale is not None:
			if scale_method == 'max':
				scale_ = np.max(my_filter(data[:,1][(data[:,0]/(1+z[file_i]) > scale[0])*(data[:,0]/(1+z[file_i]) < scale[1])], scale_smooth, 1))
			else:
				scale_ = np.min(my_filter(data[:,1][(data[:,0]/(1+z[file_i]) > scale[0])*(data[:,0]/(1+z[file_i]) < scale[1])], scale_smooth, 1))
		else:
			scale_ = 1
		if smooth is not None:
			data[:,1] = my_filter(data[:,1], smooth, 1)
		if style == 'line':
			ax.plot(data[:,0]/(1+z[file_i]), data[:,1]/scale_, label=filename.split('/')[-1])
		else:
			ax.plot(data[:,0]/(1+z[file_i]), data[:,1]/scale_, linestyle='', markersize=markersize, marker='o', ms=2, label=filename.split('/')[-1])
	plt.legend()
	plt.show()

def my_filter(data, window, n):
	if window == 1:
		return data
	else:
		return signal.savgol_filter(data, window, n)

def f_rela(x):
	return 1/(x*x+1)*3e5*(x*x-1)
def f_common(x):
	return (x-1)*3e5
def f_c_to_r(x):
	return f_rela(1+x/3e5)
def e_to_times(x):
	x = str(x)
	x_split = x.split('e')
	if x_split[1][0] == '+':
		x_split[1] = x_split[1:]
		while(x_split[1][0] == '0'):
			x_split[1] = x_split[1][1:]
	else:
		while(x_split[1][1] == '0'):
			x_split[1] = '-' + x_split[1][2:]
	return '%s\\times10^{%s}'%(x_split[0],x_split[1])

def spec_convolution(wave, flux, band, filters_dict=None):
	if filters_dict is None:
		filters_dict = {'U':'Generic_Bessell.U+Vega', 'B':'Generic_Bessell.B+Vega', 'V':'Generic_Bessell.V+Vega', 'R':'Generic_Bessell.R+Vega', 'I':'Generic_Bessell.I+Vega',
						'u':'SLOAN_SDSS.u+AB', 'g':'SLOAN_SDSS.g+AB', 'r':'SLOAN_SDSS.r+AB', 'i':'SLOAN_SDSS.i+AB', 'z':'SLOAN_SDSS.z+AB',}
		band_ = band.split('_')[-1].split()[-1]
	else:
		band_ = band
	Vega_wave, Vega_flux = get_Vega()
	Vega_interp_ = interp1d(Vega_wave, Vega_flux, kind='linear',bounds_error=False,fill_value=0.0)
	Vega_AB = {'U': 0.814991770746154, 'B': -0.10398310727753568, 'V': 0.005847314234088685, 'R': 0.188308577547879, 'I': 0.4367023749120875, 
	'u': 0.9476655871364414, 'g': -0.10235568752400326, 'r': 0.13877328046277881, 'i': 0.3533331159596216, 'z': 0.5251741178070901}
	if band_ in filters_dict.keys():
		filter_name, filter_sys = filters_dict[band_].split('+')
		wave_, resp_ = get_response(filter_name+'.dat')
		zp_ = get_zp(filter_name, filter_sys)
		if filter_sys == 'AB':
			nu_resp_ = 3e10/(wave_/1e8)[::-1]
			resp_interp_ = interp1d(nu_resp_, resp_[::-1], kind='linear',bounds_error=False,fill_value=0.0)
			nu_spec_ = 3e10/(wave*1e-8)[::-1]
			flux_nu_ = flux[::-1]*1e8*3e10/nu_spec_**2
			flux_nu_interp_ = interp1d(nu_spec_, flux_nu_, kind='linear',bounds_error=False,fill_value=0.0)
			flux_fliter = itg.quad(lambda x: flux_nu_interp_(x)*resp_interp_(x), nu_resp_[0], nu_resp_[-1])[0]
			flux_response = itg.quad(lambda x: resp_interp_(x), nu_resp_[0], nu_resp_[-1])[0]
			flux_ratio = flux_fliter/flux_response
			m_spec = -2.5*np.log10(flux_ratio) - 48.6
		elif filter_sys == 'Vega':
			resp_interp_ = interp1d(wave_, resp_, kind='linear',bounds_error=False,fill_value=0.0)
			flux_interp_ = interp1d(wave, 
				flux, kind='linear',bounds_error=False,fill_value=0.0)
			flux_fliter = itg.quad(lambda x: flux_interp_(x)*resp_interp_(x), wave_[0], wave_[-1])[0]
			flux_Vega = itg.quad(lambda x: Vega_interp_(x)*resp_interp_(x), wave_[0], wave_[-1])[0]
			flux_ratio = flux_fliter/flux_Vega
			m_spec = -2.5*np.log10(flux_ratio)
			#if 'Bessell' in filter_name:
			#	bess_zero = {'U':0.79,'B':-0.102,'V':0.008,'R':0.193,'I':0.443}
			#	m_spec += bess_zero[band.split('_')[-1].split()[-1]]
		else:
			raise Exception('Unknown magnitude system!')
	else:
		raise Exception('Unknown magnitude band!')
	return m_spec

def linear_f(x, k, b):
	return k*x+b






	