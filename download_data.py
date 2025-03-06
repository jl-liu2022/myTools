import wget

def download(SN_names, data_type='lc', out_dir=None):
	if type(SN_names) == type('haha'):
		SN_names = [SN_names]
	if type(data_type) == type('haha'):
		data_type = [data_type]
	for data_type_ in data_type:
		if data_type_ == 'lc':
			download_str = '/photometry/time+magnitude+e_magnitude+band+source?format=csv&sortby=time&complete'
			tail_str = '_lc.csv'
		elif data_type_ == 'basic':
			download_str = '/ra+dec+redshift+maxdate/value?format=csv&first'
			tail_str = '_basic.csv'
		else:
			raise Exception('bad data type')
		for SN_name in SN_names:
			url = 'https://api.astrocats.space/' + SN_name + download_str
			if out_dir is not None:
				if out_dir[-1] != '/':
					out_dir = out_dir + '/'
				out = out_dir + SN_name + tail_str
			else:
				out = '/Users/liujialian/work/SN_catalog/' + SN_name + tail_str
			wget.download(url, out=out)

