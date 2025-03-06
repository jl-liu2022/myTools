import julian
import datetime

def jdToDate(jd):
	return str(julian.from_jd(jd, fmt='jd')).replace(' ','T')

def mjdToDate(mjd):
	return str(julian.from_jd(mjd, fmt='mjd')).replace(' ','T')

def timeToMjd(time):
	# Check time fmt, number or UT
	time = str(time)
	time = 'T'.join(time.split()) # Change 'XX XX' to 'XXTXX' 
	timeSplit = time.split('T')
	if len(time.split('T')) == 1:
		# JD, MJD and date like 20220418
		time = float(time)
		if time > 1e7:
			day = int(time%1e2)
			tempt = (time - day)/100
			month = int(tempt%1e2)
			year = int((tempt - month)/100)
			dt = datetime.datetime(year, month, day)
			mjd = julian.to_jd(dt, fmt='mjd')
		elif time > 2e6:
			mjd = time - 24e5-0.5
		elif time > 0:
			mjd = time
		else:
			raise valueError('Invalid Time')
	elif len(time.split('T')) > 1:
		# Date like 2022-07-21T04:43:16.04
		YMD = timeSplit[0].split('-') 
		HMS = timeSplit[1].split(':')
		year = int(YMD[0])
		month = int(YMD[1])
		day = int(YMD[2])
		hour = int(HMS[0])
		minute = int(HMS[1])
		second = int(HMS[2].split('.')[0])
		try:
			ms = msTransform(HMS[2].split('.')[1])
		except:
			ms = 0
		dt = datetime.datetime(year, month, day, hour, minute, second, ms)
		mjd = julian.to_jd(dt, fmt='mjd')
	return mjd

def msTransform(ms):
	# '040' - > '4e4'
	while(len(ms)<6):
		ms += '0'
	return int(ms)