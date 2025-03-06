from astropy.io import fits

def readSpectrumFits(filename):
	hdul = fits.open(filename)
	hdr = hdul[0].header
	params = {'DATE-OBS':hdr['DATE-OBS'], 
			  'MJD':timeToMjd(hdr['DATE-OBS']),
			  'TEL':hdr['TELESCOP'],
			  'EXPT':hdr['EXPTIME']}
	for k, v in params:
		print(k, v)
	return params
