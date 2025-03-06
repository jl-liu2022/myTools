import numpy as np
import pandas as pd
import warnings
import os
from myTools.timeTransform import timeToMjd

from .smoothTools import non_uniform_savgol
import scipy.signal as signal

#warnings.filterwarnings('error')

dateNameDict = {'atlas':'MJD', 'ztf':'jd'}
fluxNameDict = {'atlas':'uJy', 'ztf':'forcediffimflux'}
dFluxNameDict = {'atlas':'duJy', 'ztf':'forcediffimfluxunc'}
filterNameDict = {'atlas':'F', 'ztf':'filter'}

def readAtlasData(dataFileName, minDate=None, maxDate=None, correction=False, clipped=True, image=False):
	if minDate:
		minDate = timeToMjd(minDate)
	if maxDate:
		maxDate = timeToMjd(maxDate)
	with open(dataFileName,'r') as f:
		header = f.readline().split()
	header[0] = header[0][3:]
	data = np.loadtxt(dataFileName, dtype='str')
	AtlasData = pd.DataFrame(data, columns=header)
	if image:
		return AtlasData
	AtlasData = toFloat(AtlasData)
	backgroundStd, backgroundMean, maskBackground = backgroundInfo(AtlasData, 'atlas', minDate, maxDate)
	if correction:
		Correction(AtlasData, 'atlas', backgroundMean, maskBackground)
	AtlasData = AtlasData[~maskBackground].reset_index(drop=True)
	if clipped:
		AtlasData = Clipped(AtlasData, 'atlas', backgroundStd=backgroundStd)
	AtlasData = SelectParameters(AtlasData, 'atlas')
	return AtlasData

def readASASSN(dataFileName):
	data = pd.read_csv(dataFileName)
	data = data[['HJD','Filter','mag','mag_err']]
	data.rename(columns={'HJD':'MJD', 'Filter':'filter','mag':'m','mag_err':'dm'}, inplace=True)
	data['MJD'] = data['MJD'] - 24e5 - 0.5
	data['detect'] = True
	data = data[data['dm']<99].reset_index(drop=True)
	data['m'] = data['m'].astype('float')
	for i in range(len(data)):
		data.loc[i,'filter']='ASASSN_'+data.loc[i,'filter']
	return data


def readLickData(dataFileName, tel = 'NICKEL'):
	header = ['MJD','filter','m','dm','detect']
	file_data = np.loadtxt(dataFileName, dtype='str')
	data = []
	bands = ['B', 'V', 'R', 'I']
	for i in range(len(file_data[0])):
		if file_data[0][i] in bands:
			for j in range(1, len(file_data[:,0])):
				if file_data[j][i] != 'NaN':
					data_ = file_data[j][[1,i,i+1]].astype('float').tolist()
					data_.append(True)
					data_.insert(1, tel + '_' + file_data[0][i])
					data.append(data_)
	LickData = pd.DataFrame(data, columns=header)
	LickData = LickData.sort_values(by='MJD').reset_index(drop=True)
	return LickData

def readNOWT(filename):
	file_data = pd.read_csv(filename)
	NOWTData = file_data[['HJD','FILTER','MAG','MAGERR']]
	NOWTData['detect'] = True
	NOWTData['HJD'] = NOWTData['HJD'] - 2400000.5
	NOWTData['FILTER'] = 'nowt_' + NOWTData['FILTER']
	NOWTData.rename(columns={'HJD':'MJD', 'FILTER':'filter', 'MAG':'m', 'MAGERR':'dm'}, inplace=True)
	NOWTData=NOWTData.sort_values(by='MJD').reset_index(drop=True)
	return NOWTData

def readSnova(filename):
	file_data = pd.read_csv(filename)
	SnovaData = file_data[['HJD','FILTER','MAG','MAGERR']]
	SnovaData['detect'] = True
	SnovaData['HJD'] = SnovaData['HJD'] - 2400000.5
	SnovaData['FILTER'] = 'snova_' + SnovaData['FILTER']
	SnovaData.rename(columns={'HJD':'MJD', 'FILTER':'filter', 'MAG':'m', 'MAGERR':'dm'}, inplace=True)
	SnovaData=SnovaData.sort_values(by='MJD').reset_index(drop=True)
	return SnovaData

def readTntData(dataFileName):
	bands_BV = {'U':1, 'B':4, 'V':7, 'R':10, 'I':13}
	bands_gri = {'g':1, 'r':4, 'i':7}
	if 'sloan' in dataFileName:
		bands = bands_gri
	else:
		bands = bands_BV
	header = ['MJD','filter','m','dm','detect']
	file_data = np.loadtxt(dataFileName, dtype='str')
	data = []
	for k, v in bands.items():
		for j in range(len(file_data[:,0])):
			if file_data[j][v] != 'nan':
				data_ = file_data[j][[0,v,v+1]].astype('float').tolist()
				data_[0] = timeToMjd(data_[0])
				data_.append(True)
				data_.insert(1, 'tnt_' + k)
				data.append(data_)
	LickData = pd.DataFrame(data, columns=header)
	LickData = LickData.sort_values(by='MJD').reset_index(drop=True)
	return LickData

def readAst3_3(dataFileName):
	data_ = np.loadtxt(dataFileName, dtype='str')
	Ast3_3Data = pd.DataFrame({'MJD':data_[1:,1].astype('float'),
							'filter':['ast3-3_g' for i in range(len(data_)-1)],
							'm':data_[1:,3].astype('float'),
							'dm':data_[1:,4].astype('float'),
							'detect':[True for i in range(len(data_)-1)]})
	Ast3_3Data = Ast3_3Data.sort_values(by='MJD').reset_index(drop=True)
	return Ast3_3Data

def readBootes(curDir):
	files = os.listdir(curDir)
	BootesData = []
	header = ['MJD', 'filter', 'm', 'dm', 'detect']
	for file in files:
		filePath = curDir + '/' + file
		if '.dat' in file:
			if '_psf_cal' in file:
				band_ = file.split('_')[1]
				data_ = np.loadtxt(filePath)
				data_df = pd.DataFrame({'MJD':data_[:,0].astype('float')-2400000.5,
									   'filter':['bootes_' + band_ for i in range(len(data_))],
									   'm':data_[:,1].astype('float'),
									   'dm':data_[:,2].astype('float'),
									   'detect':[True for i in range(len(data_))]})
				BootesData.append(data_df)
			else:
				band_ = file.split('.')[0][-1]
				data_ = np.loadtxt(filePath, dtype='str')
				data_df = pd.DataFrame({'MJD':data_[:,0].astype('float'),
									   'filter':['bootes_' + band_ for i in range(len(data_))],
									   'm':data_[:,1].astype('float'),
									   'dm':data_[:,2].astype('float'),
									   'detect':[True for i in range(len(data_))]})
				BootesData.append(data_df)
	BootesData = pd.concat(BootesData).sort_values(by='MJD').reset_index(drop=True)
	return BootesData

def readYahpt(curDir):
	files = os.listdir(curDir)
	YahptData = []
	for file in files:
		filePath = curDir + '/' + file
		if '.csv' in file:
			band_ = file.split('.')[0].split('_')[-1][0]
			data_ = pd.read_csv(filePath)
			data_['filter'] = 'yahpt_' + band_
			data_ = data_[['MJD-OBJ','filter','MAG_CAL_NEW','MAGERR_CAL']]
			data_['detect'] = True
			data_.rename(columns={'MJD-OBJ':'MJD', 'MAG_CAL_NEW':'m', 'MAGERR_CAL':'dm'}, inplace=True)
			YahptData.append(data_)
	YahptData = pd.concat(YahptData).sort_values(by='MJD').reset_index(drop=True)
	return YahptData


def readZtfData(dataFileName, minDate=None, maxDate=None, correction=False, clipped=False, from_finished=True, forced=False):
	if from_finished == True:
		header = ['MJD', 'filter', 'm', 'dm', 'detect', 'ujy', 'dujy']
		ZtfData = pd.read_csv(dataFileName, header=0, names=header)
		detect_dict = {'positive':True, 'limit':False}
		for i in range(len(ZtfData)):
			ZtfData.loc[i, 'filter'] = 'ZTF_' + ZtfData.loc[i, 'filter']
			ZtfData.loc[i, 'detect'] = detect_dict[ZtfData.loc[i, 'detect']]
		if forced==False:
			mark_ujy = (~ZtfData['ujy'].isna())*(ZtfData['detect']==False)
		else:
			mark_ujy = ~ZtfData['ujy'].isna()
		ZtfData['m'][mark_ujy] = -2.5*np.log10(ZtfData['ujy'][mark_ujy]) + 23.9
		ZtfData['dm'][mark_ujy] = 1.0857/(ZtfData['ujy'][mark_ujy]/ZtfData['dujy'][mark_ujy])
		ZtfData['detect'][mark_ujy] = True

		ZtfData = ZtfData[ZtfData['detect']].reset_index(drop=True)
		
	else:
		if minDate:
			minDate = timeToMjd(minDate)+24e5+0.5
		if maxDate:
			maxDate = timeToMjd(maxDate)+24e5+0.5
		data = np.loadtxt(dataFileName, dtype='str')
		header = data[0, :]
		for i in range(len(header)):
			header[i] = header[i].replace(',', '')
		data = data[1:, :]
		ZtfData = pd.DataFrame(data, columns=header)
		ZtfData = toFloat(ZtfData)
		backgroundStd, backgroundMean, maskBackground = backgroundInfo(ZtfData, 'ztf', minDate, maxDate)
		if correction:
			Correction(ZtfData, 'ztf', backgroundMean, maskBackground)
		ZtfData = ZtfData[~maskBackground].reset_index(drop=True)
		if clipped:
			ZtfData = Clipped(ZtfData, 'ztf', backgroundStd=backgroundStd)
		ZtfData = SelectParameters(ZtfData, 'ztf')
	ZtfData = ZtfData.sort_values(by='MJD').reset_index(drop=True)
	return ZtfData

def readSwiftData(dataDir, galaxy=True, tel = 'UVOT', json=False):
	if dataDir[-1] != '/' and json == False:
		dataDir = dataDir + '/'
	header = ['MJD','filter','m','dm','detect']
	if json == True:
		SwiftData = pd.read_json(dataDir)
		SwiftData.rename(columns={'mjd':'MJD', 'mag':'m', 'mag_err':'dm', 'upper_limit':'detect'}, inplace=True)
		SwiftData['detect'] = ~SwiftData['detect']
		mark_uvot = (SwiftData['filter'] == 'U') + (SwiftData['filter'] == 'B') + (SwiftData['filter'] == 'V')
		SwiftData['filter'][mark_uvot] = 'UVOT.' + SwiftData['filter'][mark_uvot]
		SwiftData['filter'] = 'UVOT_' + SwiftData['filter']
		SwiftData = SwiftData.sort_values(by='MJD')
		SwiftData = SwiftData[header]
		SwiftData = SwiftData[SwiftData['detect']].drop_duplicates('MJD')
	else:
		data = []
		with open(dataDir+'target_report.txt', 'r') as f:
			line = f.readline()
			while(line):
				if line[0] == 'U':
					bandName = line.split()[3]
					line = f.readline()
					line = f.readline()
					while(line!='\n'):
						line_split = line.split()
						if float(line_split[6]) != 99.00:
							if bandName in ['U','B','V']:
								uvot = 'UVOT.'
							else:
								uvot = ''
							data_ = [float(line_split[2]), tel + '_'+uvot+bandName, float(line_split[6]), float(line_split[7]), True]
							data.append(data_)
						line = f.readline()
				line = f.readline()
		SwiftData = pd.DataFrame(data, columns=header)
		MJD_ref = 59686.797465277836
		t_s_ref = 548458517.2
		SwiftData['MJD'] = MJD_ref + 1/86400*(SwiftData['MJD'] - t_s_ref)
		if galaxy == True:
			galaxy_data = []
			with open(dataDir+'galaxy_report.txt', 'r') as f:
				line = f.readline()
				while(line):
					if line[0] == 'F':
						line = f.readline()
						while(line[0]!='I'):
							if line == '\n':
								pass
							else:
								line_split = line.split()
								galaxy_data_ = [tel + '_'+line_split[0].upper(), float(line_split[3]), float(line_split[4])]
								galaxy_data.append(galaxy_data_)
							line = f.readline()
						break
					line = f.readline()
			for i in range(len(galaxy_data)):
				mark_ = (SwiftData['filter'] == galaxy_data[i][0])
				FT_ = np.power(10, -0.4*SwiftData['m'][mark_].to_numpy())
				FG_ = np.power(10, -0.4*galaxy_data[i][1])
				FT_True = FT_ - FG_
				cof_T = 1/FT_True*FT_*SwiftData['dm'][mark_].to_numpy()
				cof_G = 1/FT_True*FG_*galaxy_data[i][2]
				SwiftData['m'][mark_] = -2.5*np.log10(FT_True)
				SwiftData['dm'][mark_] = np.sqrt(cof_T*cof_T + cof_G*cof_G)
	#snrTot = 1.0857 / SwiftData['dm']
		mark_detect_ = SwiftData['m']<99
		SwiftData = SwiftData[mark_detect_].reset_index(drop=True)
	SwiftData = SwiftData.reset_index(drop=True)
	return SwiftData

def swift_s_to_MJD(t_s):
	MJD_ref = 59686.797465277836
	t_s_ref = 548458517.2
	t_s = MJD_ref + 1/86400*(t_s - t_s_ref)
	return t_s

def toFloat(dataFrame):
	for item in dataFrame.keys():
		for i in range(len(dataFrame)):
			'''
			if dataFrame.loc[i, item] == 'null':
				dataFrame.loc[i, item] = 1e-10
			try:
				dataFrame.loc[i, item] = float(dataFrame.loc[i, item])
			except:
				pass
			'''
			try:
				dataFrame.loc[i, item] = float(dataFrame.loc[i, item])
			except:
				if dataFrame.loc[i, item] == 'null':
					dataFrame.loc[i, item] = 1e-10
				else:
					pass
	return dataFrame

def backgroundInfo(dataFrame, dataName, minDate, maxDate):
	dateName = dateNameDict[dataName]
	fluxName = fluxNameDict[dataName]
	if not minDate:
		minMaskBackground = np.array([False for i in range(len(dataFrame))])
	else:
		minMaskBackground = dataFrame[dateName] < minDate		
	if not maxDate:
		maxMaskBackground = np.array([False for i in range(len(dataFrame))])
	else:
		maxMaskBackground = dataFrame[dateName] > maxDate

	maskBackground = maxMaskBackground + minMaskBackground
	if np.sum(maskBackground) == 0.:
		backgroundStd = None
		backgroundMean = 0.
	else:
		backgroundStd = dataFrame[fluxName][maskBackground].std()
		backgroundMean = dataFrame[fluxName][maskBackground].mean()
	return backgroundStd, backgroundMean, maskBackground

def Correction(dataFrame, dataName, backgroundMean, maskBackground):
	# Baseline
	fluxName = fluxNameDict[dataName]
	dataFrame[fluxName] = dataFrame[fluxName] - backgroundMean
	# FluxUncertainty
	dFluxName = dFluxNameDict[dataName]
	#uncMean = dataFrame[dFluxName].mean()
	#print(uncMean)
	#exit()
	#if uncMean > 1:
	#	dataFrame[dFluxName] = dataFrame[dFluxName] * np.sqrt(uncMean)
	snr = (dataFrame[fluxName]/dataFrame[dFluxName])[maskBackground]
	snr_ = np.percentile(snr, [16, 84])
	snrRms = 0.5*(snr_[1] - snr_[0])
	#print(snrRms)
	#exit()
	if snrRms > 1:
		dataFrame[dFluxName] = dataFrame[dFluxName] * snrRms

def Clipped(dataFrame, dataName, backgroundStd=None, clippedSigma=3, clipped=1000):
	fluxName = fluxNameDict[dataName]
	dFluxName = dFluxNameDict[dataName]
	if not backgroundStd:
		backgroundStd = 50 
	while clipped > 0:
		clipped = 0
		# Work out std for clipping rogue data 
		# abs(dataFrame[dFluxName]
		mask1 = abs(dataFrame[fluxName]) > (backgroundStd + abs(dataFrame[dFluxName]))
		std = dataFrame[fluxName][mask1].std()
		mean = dataFrame[fluxName][mask1].mean()
		mask2 = (dataFrame[fluxName] != 0) * mask1 * \
			(abs(dataFrame[fluxName] - mean) < clippedSigma * std)
		clipped = len(dataFrame) - mask2.sum()
		dataFrame = dataFrame[mask2]
	dataFrame = dataFrame.reset_index(drop=True)
	return dataFrame

def SelectParameters(dataFrame, dataName):
	dateName = dateNameDict[dataName]
	fluxName = fluxNameDict[dataName]
	dFluxName = dFluxNameDict[dataName]
	filterName = filterNameDict[dataName]
	dataFrame.rename(columns={dateName:'MJD', filterName:'filter'}, inplace=True)
	for i in range(len(dataFrame)):
		dataFrame.loc[i, 'filter'] = dataFrame.loc[i, 'filter'].lower()
	snt = 3
	snu = 5
	if dataName == 'atlas':
		dataFrameSelect = dataFrame[['MJD', 'filter']].copy()
		dataFrameSelect['m'] = dataFrame['m'].copy()
		dataFrameSelect['dm'] = dataFrame['dm'].copy()
		dataFrameSelect['detect'] = dataFrame['m']/dataFrame['dm'] > snt
	elif dataName == 'ztf':
		dataFrame['MJD'] = dataFrame['MJD'] - 2400000.5
		data_all = []
		'''
		for filter_ in ['ztf_g','ztf_r']:
			mark_r = (dataFrame['filter'] == filter_)
			dataFrame_ = dataFrame[mark_r].reset_index(drop=True)
			length = len(dataFrame_)
			dataFrame_.loc[length] = dataFrame_.loc[length-1].copy()
			dataFrame_.loc[length,'MJD'] = dataFrame_.loc[length,'MJD']+100
			dataFrame_['stack'] = False
			dataFrame_['forcediffimflux_new'] = 0.
			dataFrame_['forcediffimfluxunc_new'] = 0.
			dataFrame_['w'] = 1.
			head = 0
			dataFrame_.loc[head, 'stack'] = True
			for i in range(0, length+1):
				#if int(dataFrame_.loc[i, 'MJD']) != int(dataFrame_.loc[head, 'MJD']):
				dataFrame_.loc[i, 'forcediffimflux_new'] = dataFrame_.loc[i, 'forcediffimflux']*10**(0.4*(dataFrame_.loc[head, 'zpdiff']
				 - dataFrame_.loc[i, 'zpdiff']))
				#print(i, dataFrame_.loc[i, 'forcediffimflux'], dataFrame_.loc[i, 'forcediffimfluxunc'])
				#print(i, dataFrame_.loc[i, 'forcediffimflux'], 10**(0.4*(dataFrame_.loc[head, 'zpdiff'] - dataFrame_.loc[i, 'zpdiff'])))
				dataFrame_.loc[i, 'forcediffimfluxunc_new'] = dataFrame_.loc[i, 'forcediffimfluxunc']*10**(0.4*(dataFrame_.loc[head, 'zpdiff']
				 - dataFrame_.loc[i, 'zpdiff']))
				dataFrame_.loc[i, 'w'] = 1/dataFrame_.loc[i, 'forcediffimfluxunc_new']**2
				if abs(dataFrame_.loc[i, 'MJD'] - dataFrame_.loc[head, 'MJD']) > 0.5:
					w_sum = dataFrame_.loc[head:(i-1), 'w'].sum()
					print(dataFrame_.loc[head:(i-1), ['forcediffimflux','forcediffimfluxunc',]])
					#print(dataFrame_.loc[head:(i-1), ['forcediffimflux_new','forcediffimfluxunc_new','w']])
					dataFrame_.loc[head, 'MJD'] = (dataFrame_.loc[head:(i-1), 'MJD'] * dataFrame_.loc[head:(i-1), 'w']).sum() / w_sum
					dataFrame_.loc[head, 'forcediffimflux'] = (dataFrame_.loc[head:(i-1), 'forcediffimflux_new'] * dataFrame_.loc[head:(i-1), 'w']).sum() / w_sum
					dataFrame_.loc[head, 'forcediffimfluxunc'] = w_sum**(-0.5)
					print(dataFrame_.loc[head, 'MJD'], dataFrame_.loc[head, 'forcediffimflux'], dataFrame_.loc[head, 'forcediffimfluxunc'])
					#
					#exit()
					#dataFrame_.loc[head, 'MJD'] = dataFrame_.loc[head:(i-1), 'MJD'].sum()/(i-head)
					#dataFrame_.loc[head, 'm'] = dataFrame_.loc[head:(i-1), 'm'].sum()/(i-head)
					#dataFrame_.loc[head, 'dm'] = dataFrame_.loc[head:(i-1), 'dm'].sum()/(i-head)/np.sqrt(i-head)
					#dataFrame_.loc[head, 'dm'] = np.sqrt((dataFrame_.loc[head:(i-1), 'dm']**2).sum())/(i-head)
					head = i
					if head != length:
						dataFrame_.loc[head, 'stack'] = True
			dataFrame_ = dataFrame_[dataFrame_['stack']].reset_index(drop=True)
			data_all.append(dataFrame_)
		dataFrame = pd.concat(data_all).sort_values(by='MJD').reset_index(drop=True)
		'''
		#print(dataFrame[['index','sciinpseeing','zpmaginpsciunc','forcediffimchisq']])
		nearestfflux = np.power(10, 0.4*(dataFrame['zpdiff'] - dataFrame['nearestrefmag']))
		nearestffluxunc = dataFrame['nearestrefmagunc']*nearestfflux/1.0857
		fluxTot = dataFrame['forcediffimflux']+nearestfflux
		mask_t = (dataFrame['forcediffimfluxunc']*dataFrame['forcediffimfluxunc'] - \
				nearestffluxunc*nearestffluxunc).astype('float') > 0
		#mask_t = 0
		fluxuncTot = np.sqrt((dataFrame['forcediffimfluxunc']*dataFrame['forcediffimfluxunc'] - \
				(mask_t*2-1)*nearestffluxunc*nearestffluxunc).astype('float'))
		#print(dataFrame['forcediffimfluxunc'])
		#exit()
		'''
		'''
		snrTot = fluxTot / fluxuncTot
		mag = np.zeros(len(dataFrame))
		magSigma = np.zeros(len(dataFrame))
		maskDetect = snrTot > snt
		mag[maskDetect] = dataFrame['zpdiff'][maskDetect] - 2.5*np.log10(fluxTot[maskDetect].astype('float'))
		mag[~maskDetect] = dataFrame['zpdiff'][~maskDetect] - 2.5*np.log10(snu*fluxuncTot[~maskDetect].astype('float'))
		magSigma = 1.0857 / snrTot
		dataFrameSelect = pd.DataFrame({})
		dataFrameSelect['MJD'] = dataFrame['MJD']
		dataFrameSelect['filter'] = dataFrame['filter']
		dataFrameSelect['m'] = mag
		dataFrameSelect['dm'] = magSigma
		dataFrameSelect['zpdiff'] = dataFrame['zpdiff']
		dataFrameSelect['forcediffimflux'] = dataFrame['forcediffimflux']
		dataFrameSelect['forcediffimfluxunc'] = dataFrame['forcediffimfluxunc']
		dataFrameSelect['nearestffluxunc'] = nearestffluxunc
		dataFrameSelect['flux'] = fluxTot
		dataFrameSelect['dflux'] = fluxuncTot
		dataFrameSelect['detect'] = maskDetect
		# dataFrame.loc[:,'m'] = -2.5*np.log10(dataFrame['m']) + 23.9
		dataFrameSelect.to_csv('ZTF_data.csv',index=False)
	return dataFrameSelect[dataFrameSelect['detect']].reset_index(drop=True)
	