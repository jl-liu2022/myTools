def get_pstars(ra, dec, size, output_dir, filters="grizy"):

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


    import numpy as np
    from astropy.io import fits
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    import requests
    import pandas as pd
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

    if url is not None:
        with fits.open(url[0],ignore_missing_end = True,lazy_load_hdus = True, cache=False) as hdu:
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
    else:
        print('template not found')
    return url

#get_pstars('02:42:05.499', '-16:57:22.90', 240, output_dir='/Users/liujialian/work/scripts/myTools/PS_temp', filters="r")