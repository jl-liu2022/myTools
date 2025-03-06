import numpy as np
import pandas as pd
import pytz
import astropy.units as u

from astropy.io import fits
from datetime import datetime, tzinfo

from myTools.timeTransform import timeToMjd

def angle_dis(ra1, dec1, ra2, dec2):
	ra1_ = ra1*d_to_r
	dec1_ = dec1*d_to_r
	ra2_ = ra2*d_to_r
	dec2_ = dec2*d_to_r
	return np.arccos(np.sin(dec1_)*np.sin(dec2_) + np.cos(dec1_)*np.cos(dec2_)*np.cos(ra2_-ra1_))/d_to_r*3600

def location_info(LOCATION):
	from astropy.coordinates import EarthLocation
	#天文台信息
	# bear_mountain      = EarthLocation(lat=41.3*u.deg, lon=-74*u.deg, height=390*u.m)
	# BM_utcoffset       = -4*u.hour    # Eastern Daylight Time
	Xinglong           = EarthLocation(lat=40.39*u.deg,  lon=-242.41*u.deg, height=950*u.m)
	Xinglong_utcoffset = -16*u.hour    
	lijiang            = EarthLocation(lat=26.7*u.deg,   lon=-259.98*u.deg, height=3193*u.m)
	lijiang_utcoffset  = -16*u.hour   
	Lick               = EarthLocation(lat=37.34*u.deg,  lon=-121.63*u.deg, height=1290*u.m)
	Lick_utcoffset     = -8*u.hour   
	Mauna_kea          = EarthLocation(lat=19.83*u.deg,  lon=-155.47*u.deg, height=4215*u.m)
	Mauna_kea_utcoffset= -10*u.hour    
	La_Silla           = EarthLocation(lat=-29.26*u.deg, lon=-70.73*u.deg, height=2347*u.m)
	La_Silla_utcoffset = -4*u.hour    
	Asiago             = EarthLocation(lat=45.84*u.deg,  lon=-348.45*u.deg, height=1366*u.m)
	Asiago_utcoffset   = -23*u.hour    
	Xinjiang           = EarthLocation(lat=43.5 *u.deg,  lon=-272.877*u.deg, height=2080*u.m)
	Xinjiang_utcoffset = -16*u.hour    
	RDLM               = EarthLocation(lat=28.76*u.deg,  lon=-17.89*u.deg, height=2326*u.m)
	RDLM_utcoffset     = -24*u.hour   
	if LOCATION=="XL":
		return Xinglong,Xinglong_utcoffset
	if LOCATION=="LJ":
		return lijiang,lijiang_utcoffset
	if LOCATION=="Lick":
		return Lick,Lick_utcoffset
	if LOCATION=="MK":
		return Mauna_kea,Mauna_kea_utcoffset
	if LOCATION=="LS":
		return La_Silla,La_Silla_utcoffset
	if LOCATION=="AS":
		return Asiago,Asiago_utcoffset
	if LOCATION=="XJ":
		return Xinjiang,Xinjiang_utcoffset
	if LOCATION=="RDLM":#Rouqe de los Muchachos Observatory
		return RDLM,RDLM_utcoffset

def distance_to_moon(LOCATION, ra, dec, date=None):

	from astropy.time import Time
	from astropy.coordinates import get_moon
	from astropy.coordinates import SkyCoord

	if date is None:
		date = str(Time.now().to_value(format='datetime')).split(' ')[0]
	[Telescope_loacation,utcoffset]=location_info(LOCATION)
	midnight = Time(date+' 00:00:00') + (1 - (Telescope_loacation.lon / (360*u.degree)))*24*u.hour
	try:
		ra = float(ra)
		dec = float(dec)
		target_coord = SkyCoord(float(ra), float(dec), unit='deg')
	except:
		target_coord = SkyCoord('%s %s'%(ra, dec), unit=(u.hourangle, u.deg))
	dis_to_moon = (get_moon(midnight).separation(target_coord)).to_value("degree")
	return dis_to_moon

def get_pstars(ra, dec, size, output_dir, filters="grizy", save=False):

    '''
    Attempt to download a templte image from the PS1 image cutout server.

    :param ra: Right Ascension of the target in degrees
    :type ra: float
    :param dec: Declination of the target in degrees
    :type dec: float
    :param size: Pixel size of image
    :type size: int
    :param filters: Name of filters we need, defaults to "grizy"
    :type filters: str, optional
    :return: Filepath of template file from PS1 webiste
    :rtype: str

    '''

    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import requests
    import sys,os

    try:
        ra = float(ra)
        dec = float(dec)
    except:
        target_coord = SkyCoord('%s %s'%(ra, dec), unit=(u.hourangle, u.deg))
        ra = target_coord.ra.to_value('degree')
        dec = target_coord.dec.to_value('degree')

    try:

        format='fits'
        delimiter = ','

        service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
        url = ("{service}?ra={ra}&dec={dec}&size={size}&format={format}&sep={delimiter}"
               "&filters={filters}").format(**locals())

        with requests.Session() as s:
            myfile = s.get(url)
            s.close()

        text = np.array([line.decode('utf-8') for line in myfile.iter_lines()])

        text = [text[i].split(',') for i in range(len(text))]

        df = pd.DataFrame(text)
        df.columns = df.loc[0].values
        table =Table.from_pandas( df.reindex(df.index.drop(0)).reset_index())


        url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
               "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())



        # sort filters from red to blue
        flist = ["yzirg".find(x) for x in table['filter']]

        table = table[np.argsort(flist)]

        # if color:
        #     if len(table) > 5:
        #         # pick 3 filters
        #         table = table[[0,len(table)//2,len(table)-1]]
        #     for i, param in enumerate(["red","green","blue"]):
        #         url = url + "&{}={}".format(param,table['filename'][i])
        # else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname1 = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname1, exc_tb.tb_lineno,e)
        url = None

    if url is not None and save==True:
        with fits.open(url[0],ignore_missing_end = True,lazy_load_hdus = True) as hdu:
            try:
                hdu.verify('silentfix+ignore')
                headinfo_template = hdu[0].header
                template_found  = True
                # save templates into original folder under the name template
                fits.writeto(os.path.join(output_dir, url[0].split('/')[-1]),
                             hdu[0].data,
                             headinfo_template,
                             overwrite=True,
                             output_verify = 'silentfix+ignore')
            except Exception as e:
                print(e)
    elif url is None and save == True:
        print('template not found')
    return url

#get_pstars('02:42:05.499', '-16:57:22.90', 240, output_dir='/Users/liujialian/work/scripts/myTools/PS_temp', filters="r")

def plot_TNS(TNS_file, save=None, n_cols=5):

	import matplotlib.image as mpimg
	import matplotlib.pyplot as plt
	from astropy.visualization import ImageNormalize,SquaredStretch,ZScaleInterval

	TNS_table = pd.read_csv(TNS_file)
	len_table = len(TNS_table)
	if len_table == 0:
		print('no object')
		exit()
	if len_table < n_cols:
		n_cols = len_table
	n_rows = int((len_table - 1) / n_cols) + 1
	fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
	if n_rows == 1:
		if n_cols == 1:
			axs = np.array([[axs]])
		else:
			axs = np.array([axs])
	for i in range(len_table):
		row = int(i / n_cols)
		col = i % n_cols
		target_name = TNS_table.loc[i, 'Name']
		target_ra = TNS_table.loc[i, 'RA']
		target_dec = TNS_table.loc[i, 'DEC']
		target_group = TNS_table.loc[i, 'Discovery Data Source/s']
		target_mag = TNS_table.loc[i, 'Discovery Mag/Flux']
		target_filter = TNS_table.loc[i, 'Discovery Filter']
		axs[row][col].set_xlabel(target_name + ' ' + target_filter + ' ' + str(target_mag))
		axs[row][col].set_ylabel(target_group)
		axs[row][col].tick_params(labelbottom=False, labelleft=False)
		url = get_pstars(target_ra, target_dec, 500, output_dir='/Users/liujialian/work/scripts/myTools/PS_temp', 
			filters="r")

		try:
			with fits.open(url[0], ignore_missing_end = True,lazy_load_hdus = True, cache=True) as hdu:
				vmin,vmax = (ZScaleInterval(nsamples = 1000)).get_limits(hdu[0].data)
				axs[row][col].imshow(hdu[0].data,
					vmin = vmin,
					vmax = vmax,
					origin = 'lower',
					cmap = 'viridis')
				axs[row][col].plot([250.5-10, 250.5-30], [250.5, 250.5], c='purple')
				axs[row][col].plot([250.5+10, 250.5+30], [250.5, 250.5], c='purple')
				axs[row][col].plot([250.5, 250.5], [250.5-10, 250.5-30], c='purple')
				axs[row][col].plot([250.5, 250.5], [250.5+10, 250.5+30], c='purple')
		except:
			print('image of %s not found'%target_name)
			print(url)
	if save is not None:
		plt.savefig(save, bbox_inches='tight')
	plt.show()

class obs_target:

	def __init__(self, name, discover_date, ra, dec, refer_date, refer_mag, refer_band, z, done=False, ZTF='no_ZTF', Type='ToR'):
		self.name = name
		self.discover_date = discover_date
		self.ra = ra
		self.dec = dec
		self.z = z
		self.done = done
		self.refer_date = refer_date
		self.refer_mag = refer_mag
		self.refer_band = refer_band
		self.refer_from_today = int(timeToMjd(refer_date) - timeToMjd(datetime.now(tz=pytz.UTC)))
		self.refer_phase = int(timeToMjd(refer_date) - timeToMjd(self.discover_date))
		self.ZTF = ZTF
		self.Type = Type
		self.data = pd.DataFrame({})

	def change_ref_obs(self, refer_date, refer_mag, refer_band):
		self.refer_date = refer_date
		self.refer_mag = refer_mag
		self.refer_band = refer_band
		self.refer_from_today = int(timeToMjd(refer_date) - timeToMjd(datetime.now(tz=pytz.UTC)))

	def initial_data(self, obs_date, obs_type):
		if type(obs_date) == type('haha'):
			obs_date = [obs_date]
			obs_type = [obs_type]
		from_today = np.ones(len(obs_date))
		phase = np.ones(len(obs_date))
		for i in range(len(from_today)):
			from_today[i] = int(timeToMjd(obs_date[i]) - timeToMjd(datetime.now(tz=pytz.UTC)))
			phase[i] = int(timeToMjd(obs_date[i]) - timeToMjd(self.discover_date))
		self.data = pd.DataFrame({'obs_date':obs_date,
								  'from_today':from_today,
								  'phase':phase,
								  'obs_type':obs_type})

	def add_data(self, obs_date, obs_type):
		if not self.data.empty:
			if ((self.data['obs_date']==obs_date)*(self.data['obs_type']==obs_type)).any():
				pass
			else:
				from_today = int(timeToMjd(obs_date) - timeToMjd(datetime.now(tz=pytz.UTC)))
				phase = int(timeToMjd(obs_date) - timeToMjd(self.discover_date))
				self.data.loc[len(self.data.index)] = [obs_date, from_today, phase, obs_type]
		else:
			self.initial_data(obs_date, obs_type)


	def info(self):
		print(self.name, self.ra, self.dec, self.discover_date, self.Type, '')
		print(self.refer_date, self.refer_mag, self.refer_band, self.refer_from_today, self.z, self.done, self.ZTF)
		print(self.data)

	#@classmethod
	#def from_TNS(cls, name):
	#	target = cls(name, discover_date, ra, dec, refer_date, refer_mag, refer_band, z, done=False, ZTF='no_ZTF', Type='ToR')
	#	target.info()
	#	return target

class obs_targets:

	def __init__(self):
		self.targets = {}
		self.target_number = 0
		self.reminder = []

	def set_reminder(self, text):
		if type(text) == type('haha'):
			text = [text]
		self.reminder = text

	def add_reminder(self, text):
		self.reminder.append(text)

	def add_target(self, target):
		target_name = target.name
		if target_name in self.targets.keys():
			print('Already exists!')
		else:
			self.targets[target_name] = target
			self.target_number += 1
			
	def print_high(self, targets_name, filename=None, nanshan=False):
		if type(targets_name) == type('haha'):
			targets_name = [targets_name]
		quene_dict = {}
		for item in targets_name:
			quene_dict[self.targets[item].ra] = self.targets[item].name
		quene_array = np.array(list(quene_dict.keys()))
		quene_array.sort()
		if filename:
			with open(filename,'w') as f:
				print('NAME,RA,DEC,BAND+MAG,ZTF', file=f)
				for item in quene_array:
					target_ = self.targets[quene_dict[item]]
					print(target_.name,',',target_.ra,',',target_.dec,',',target_.refer_band,target_.refer_mag,',',target_.ZTF, file=f)
			high_data = pd.read_csv(filename)[['NAME','RA','DEC','BAND+MAG']]
			high_data.to_excel('myexcel.xlsx')
		else:
			for item in quene_array:
				target_ = self.targets[quene_dict[item]]
				if nanshan == True:
					if len(target_.name) < 11:
						if target_.name[0] == 'J':
							print(target_.name[:5] + '	' + str(target_.ra) + '	' + str(target_.dec))
						else:
							print(target_.name[4:] + '	' + str(target_.ra) + '	' + str(target_.dec))
					else:
						print('J'+target_.name.split('J')[-1][:6] + '	' + str(target_.ra) + '	' + str(target_.dec))
				else:
					if len(target_.name) < 11:
						if target_.name[0] == 'J':
							print(target_.name[:5] + ' ' + str(target_.ra) + ' ' + str(target_.dec))
						else:
							print(target_.name[4:] + ' ' + str(target_.ra) + ' ' + str(target_.dec))
					else:
						print('J'+target_.name.split('J')[-1][:6] + ' ' + str(target_.ra) + ' ' + str(target_.dec))

	def print_80_txt(self, targets_name, filename=None, camara=5):
		if type(targets_name) == type('haha'):
			targets_name = [targets_name]
		quene_dict = {}
		for item in targets_name:
			quene_dict[self.targets[item].ra] = self.targets[item].name
		quene_array = np.array(list(quene_dict.keys()))
		quene_array.sort()
		total_time = 0
		if filename:
			with open(filename,'w') as f:
				for item in quene_array:
					target_ = self.targets[quene_dict[item]]
					print('$,%s,%s,%s,2000,1,;'%(target_.name,target_.ra,target_.dec),file=f)
					mag_ = target_.refer_mag
					'''
					if target_.refer_phase > 15:
						mag_ -= 0.027*target_.refer_from_today
					else:
						mag_ += 0.3333*target_.refer_from_today
						print(target_.name)
					if target_.refer_band not in ['r','o']:
						mag_ += 0.2
					'''
					exptime = np.array([300 for i in range(5)])
					band = ['B','V','g','r','i']
					times = np.array([1 for i in range(5)])
					'''
					if mag_ <= 17.5:
						pass
					elif mag_ > 18.75:
						times = 2*times
					elif mag_ > 18.25:
						exptime[[0,1]] = 420
						exptime[[2,3,4]] = 360
					elif mag_ > 17.75:
						exptime[[0,1]] = 360
					'''
					total_time += np.sum(exptime*times)
					print(np.sum(exptime*times))
					for i in range(5):
						print('0,%d,%s,%d,1,%d,;'%(exptime[i],band[i],camara,times[i]), file=f)
						
		else:
			for item in quene_array:
				target_ = self.targets[quene_dict[item]]
				print('$,%s,%s,%s,2000,1,;'%(target_.name,target_.ra,target_.dec),file=f)
				mag_ = target_.refer_mag
				if target_.refer_phase > 20:
					mag_ -= 0.04*target_.refer_from_today
				else:
					mag_ += 0.3333*target_.refer_from_today
					print(target_.name)
				if target_.band not in ['r','o']:
					mag_ += 0.2
				exptime = np.array([300 for i in range(5)])
				band = ['B','V','g','r','i']
				times = np.array([1 for i in range(5)])
				if mag_ <= 17.5:
					pass
				elif mag_ > 18.75:
					times = 2*times
				elif mag_ > 18.25:
					exptime[[0,1]] = 420
					exptime[[2,3,4]] = 360
				elif mag_ > 17.75:
					exptime[[0,1]] = 360
				total_time += np.sum(exptime*times)
				for i in range(5):
					print('0,%d,%s,%d,1,%d,;'%(exptime[i],band[i],camara,times[i]))
		print('total_time:%d h'%(total_time/3600))

	def save_txt(self, filename, targets_name = None):
		pd.set_option('display.max_rows', None)
		if not targets_name:
			targets_name = list(self.targets.keys())
		'''
		quene_dict = {}
		for item in targets_name:
			quene_dict[self.targets[item].ra] = self.targets[item].name
		quene_array = np.array(list(quene_dict.keys()))
		quene_array.sort()
		'''
		ra_list = []
		for item in targets_name:
			ra_list.append(self.targets[item].ra)
		quene_ = np.argsort(ra_list)
		with open(filename, 'w') as f:
			#for item in quene_array:
			for index in quene_:
				#v = self.targets[quene_dict[item]]
				v = self.targets[targets_name[index]]
				print(v.name, v.ra, v.dec, v.discover_date, v.Type, file=f)
				print(v.refer_date, v.refer_mag, v.refer_band, v.refer_from_today, v.z, v.done, v.ZTF, file=f)
				print(v.data,'\n', file=f)
			print(file=f)
			for item in self.reminder:
				print(item,file=f)

	def save_fits(self, filename):
		hdr = fits.Header()
		hdr['COMMENT'] = 'haha'

		hdu_primary = fits.PrimaryHDU(header=hdr)

		hduList = [hdu_primary]

		for target_name_, target_data_ in self.targets.items():
			hdr = fits.Header()
			hdr['NAME'] = target_name_
			hdr['DIS-DATE']= target_data_.discover_date
			hdr['RA'] = target_data_.ra
			hdr['DEC'] = target_data_.dec
			hdr['Z'] = target_data_.z
			hdr['DONE'] = target_data_.done
			hdr['REF-DATE'] = target_data_.refer_date
			hdr['REF-MAG'] = target_data_.refer_mag
			hdr['REF-BAND'] = target_data_.refer_band
			hdr['ZTF'] = target_data_.ZTF
			hdr['TYPE'] = target_data_.Type

			if target_data_.data.empty:
				obs_date_ = fits.Column(name='obs_date', array=[], format='20A')
				obs_type_ = fits.Column(name='obs_type', array=[], format='20A')
			else:
				obs_date_ = fits.Column(name='obs_date', array=target_data_.data['obs_date'].to_numpy(), format='20A')
				obs_type_ = fits.Column(name='obs_type', array=target_data_.data['obs_type'].to_numpy(), format='20A')
			hdu_target_ = fits.BinTableHDU.from_columns([obs_date_, obs_type_], header=hdr, name=target_name_)
			hduList.append(hdu_target_)
		reminder_ = fits.Column(name='reminder', array=self.reminder, format='256A')
		hdu_reminder = fits.BinTableHDU.from_columns([reminder_], name='reminder')
		hduList.append(hdu_reminder)

		hdul = fits.HDUList(hduList)
		hdul.writeto(filename, overwrite=True)

	@classmethod
	def read_fits(cls, filename):
		targets = cls()
		with fits.open(filename) as hdul_read:
			for i in range(1, len(hdul_read)-1):
				#name, discover_date, ra, dec, obs_date, mag, band, from_today
				hdr_ = hdul_read[i].header
				hdd_ = hdul_read[i].data
				name_ = hdr_['NAME']
				discover_date_ = hdr_['DIS-DATE']
				ra_ = hdr_['RA']
				dec_ = hdr_['DEC']
				z_ = hdr_['Z']
				done_ = hdr_['DONE']
				refer_date_ = hdr_['REF-DATE']
				refer_mag_ = hdr_['REF-MAG']
				refer_band_ = hdr_['REF-BAND']
				ZTF_ = hdr_['ZTF']
				Type_ = hdr_['TYPE']
				target_ = obs_target(name_, discover_date_, ra_, dec_, refer_date_, refer_mag_, refer_band_, z_, ZTF = ZTF_, Type=Type_)
				try:
					obs_date_ = hdd_['obs_date']
					obs_type_ = hdd_['obs_type']
					target_.initial_data(obs_date_, obs_type_)
				except:
					pass
				targets.add_target(target_)
			try:
				targets.set_reminder(hdul_read[-1].data['reminder'].tolist())
			except:
				pass
		return targets

	@classmethod
	def read_txt(cls, filename):
		targets = cls()
		with open(filename, 'r') as txt_read:
			lines = txt_read.readlines()
			len_lines = len(lines)
			target_tail_pos = []
			for i_line in range(len_lines):
				if lines[i_line] == '\n':
					target_tail_pos.append(i_line)
			len_targets = len(target_tail_pos)
			target_start_pos = 0
			for i_target in range(len_targets):
				if target_tail_pos[i_target] - target_start_pos > 2:
					line_split = lines[target_start_pos].split()
					if line_split[0] == 'TMTS' or line_split[0] == 'SN':
						name_ = ' '.join(line_split[:2])
						ra_ = line_split[2]
						dec_ = line_split[3]
						discover_date_ = line_split[4]
						Type_ = line_split[5]
					elif line_split[-2] == 'SN':
						name_ = line_split[0]
						ra_ = line_split[1]
						dec_ = line_split[2]
						discover_date_ = line_split[3]
						Type_ = line_split[5]
					else:
						name_ = line_split[0]
						ra_ = line_split[1]
						dec_ = line_split[2]
						discover_date_ = line_split[3]
						Type_ = line_split[4]
					line_split = lines[target_start_pos+1].split()
					refer_date_ = line_split[0]
					refer_mag_ = line_split[1]
					refer_band_ = line_split[2]
					z_ = line_split[4]
					done_ = line_split[5]
					ZTF_ = line_split[6]
					target_ = obs_target(name_, discover_date_, ra_, dec_, refer_date_, refer_mag_, refer_band_, z_, ZTF = ZTF_, Type=Type_)
					line_split = lines[target_start_pos+2].split()
					if line_split[0] != 'Empty':
						obs_date_ = []
						obs_type_ = []
						for i_obs in range(target_start_pos+3, target_tail_pos[i_target]):
							data_ = lines[i_obs].split()
							obs_date_.append(data_[1])
							obs_type_.append(data_[4])
						target_.initial_data(obs_date_, obs_type_)
					targets.add_target(target_)
					target_start_pos = target_tail_pos[i_target] + 1
		return targets



