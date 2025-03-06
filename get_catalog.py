from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
from .Photometry import SDSS_to_Johnson
import numpy as np
import pandas as pd

#V/154/sdss16

def sdsstrn(_sdssdfc):
    """
    transformations:
    U - g = a1(u - g) + b1 + a2(g - r) + b2
    B - g = a2(g - r) + b2
    V - g = a3(g - r) + b3
    R - r = a4(r - i) + b4
    I - i = a5(i - z) + b5
    Args:
        _sdssdfc: catalog dataframe of SDSS

    Returns:
        transformed dataframe of SDSS

    """
    if not isinstance(_sdssdfc, pd.DataFrame):
        raise RuntimeError('>>> Input must be data frame!')

    # parameters
    a1, ea1, b1, eb1 = 0.79, 0.02, -0.93, 0.02
    a2, ea2, b2, eb2 = 0.313, 0.003, 0.219, 0.002
    a3, ea3, b3, eb3 = -0.565, 0.001, -0.016, 0.001
    a4, ea4, b4, eb4 = -0.153, 0.003, -0.117, 0.003
    a5, ea5, b5, eb5 = -0.386, 0.004, -0.397, 0.001
    au1, eau1, au2, eau2, bu, ebu = 0.52, 0.06, 0.53, 0.09, -0.82, 0.04

    ugmag = _sdssdfc['umag'].values - _sdssdfc['gmag'].values
    grmag = _sdssdfc['gmag'].values - _sdssdfc['rmag'].values
    rimag = _sdssdfc['rmag'].values - _sdssdfc['imag'].values
    izmag = _sdssdfc['imag'].values - _sdssdfc['zmag'].values

    _sdssdfc['Bmag'] = _sdssdfc['gmag'].values + a2 * grmag + b2
    _sdssdfc['e_Bmag'] = np.sqrt(np.power(_sdssdfc['e_gmag'].values, 2) +
                                 np.power(_sdssdfc['e_rmag'].values, 2))

    # _sdssdfc['Umag'] = _sdssdfc['Bmag'].values + a1 * ugmag + b1
    # _sdssdfc['e_Umag'] = np.sqrt(np.power(_sdssdfc['e_Bmag'].values, 2) +
    #                              np.power(_sdssdfc['e_umag'].values, 2))

    # _sdssdfc['Umag'] = _sdssdfc['Bmag'].values + au1 * ugmag + au2 * grmag + bu
    # _sdssdfc['e_Umag'] = np.sqrt(np.power(_sdssdfc['e_umag'].values, 2) +
    #                              np.power(_sdssdfc['e_gmag'].values, 2) +
    #                              np.power(_sdssdfc['e_rmag'].values, 2))

    # x1 = (-0.3467, 0.016)
    # x2 = (0.3971, 0.022)
    # cb = (-0.5843, 0.032)
    _sdssdfc['Umag'] = _sdssdfc['umag'].values - 0.3467 * ugmag + 0.3971 * grmag - 0.5843
    _sdssdfc['e_Umag'] = np.sqrt(np.power(_sdssdfc['e_gmag'].values, 2) +
                                 np.power(_sdssdfc['e_umag'].values, 2) +
                                 np.power(_sdssdfc['e_rmag'].values, 2))

    _sdssdfc['Vmag'] = _sdssdfc['gmag'].values + a3*grmag + b3
    _sdssdfc['e_Vmag'] = np.sqrt(np.power(_sdssdfc['e_gmag'].values, 2) +
                                 np.power(_sdssdfc['e_rmag'].values, 2))

    _sdssdfc['Rmag'] = _sdssdfc['rmag'].values + a4*rimag + b4
    _sdssdfc['e_Rmag'] = np.sqrt(np.power(_sdssdfc['e_imag'].values, 2) +
                                 np.power(_sdssdfc['e_rmag'].values, 2))

    _sdssdfc['Imag'] = _sdssdfc['imag'].values + a5*izmag + b5
    _sdssdfc['e_Imag'] = np.sqrt(np.power(_sdssdfc['e_imag'].values, 2) +
                                 np.power(_sdssdfc['e_zmag'].values, 2))
    return _sdssdfc


def ps1trn(_ps1dfc):
    """
    transformations:
    U - g = a1(g - r) + b1
    B - g = a2(g - r) + b2
    V - g = a3(g - r) + b3
    R - r = a4(r - i) + b4
    I - i = a5(r - i) + b5
    Args:
        _ps1dfc: catalog from ps1

    Returns:
        transformed catalog in data frame

    """
    if not isinstance(_ps1dfc, pd.DataFrame):
        raise RuntimeError('>>> Input must be data frame!')

    # parameters
    a1, ea1, b1, eb1 = 2.816, 0.007, -0.206, 0.007
    a2, ea2, b2, eb2 = 0.533, 0.002, 0.202, 0.001
    a3, ea3, b3, eb3 = -0.506, 0.001, -0.022, 0.001
    a4, ea4, b4, eb4 = -0.297, 0.001, -0.162, 0.001
    a5, ea5, b5, eb5 = -0.256, 0.001, -0.410, 0.001

    grmag = _ps1dfc['gmag'].values - _ps1dfc['rmag'].values
    rimag = _ps1dfc['rmag'].values - _ps1dfc['imag'].values
    izmag = _ps1dfc['imag'].values - _ps1dfc['zmag'].values
    gimag = _ps1dfc['gmag'].values - _ps1dfc['imag'].values

    _ps1dfc['Umag'] = _ps1dfc['gmag'].values + a1 * grmag + b1
    _ps1dfc['e_Umag'] = np.sqrt(np.power(_ps1dfc['e_gmag'].values, 2) +
                                np.power(_ps1dfc['e_rmag'].values, 2))

    # _ps1dfc['Umag'] = _ps1dfc['gmag'].values + 0.6677 * grmag + 0.5968 * gimag - 0.0252
    # _ps1dfc['e_Umag'] = np.sqrt(np.power(_ps1dfc['e_gmag'].values, 2) +
    #                             np.power(_ps1dfc['e_rmag'].values, 2))

    _ps1dfc['Bmag'] = _ps1dfc['gmag'].values + a2 * grmag + b2
    _ps1dfc['e_Bmag'] = np.sqrt(np.power(_ps1dfc['e_gmag'].values, 2) +
                                np.power(_ps1dfc['e_rmag'].values, 2))
    _ps1dfc['Vmag'] = _ps1dfc['gmag'].values + a3 * grmag + b3
    _ps1dfc['e_Vmag'] = np.sqrt(np.power(_ps1dfc['e_gmag'].values, 2) +
                                np.power(_ps1dfc['e_rmag'].values, 2))
    _ps1dfc['Rmag'] = _ps1dfc['rmag'].values + a4 * rimag + b4
    _ps1dfc['e_Rmag'] = np.sqrt(np.power(_ps1dfc['e_imag'].values, 2) +
                                np.power(_ps1dfc['e_rmag'].values, 2))
    _ps1dfc['Imag'] = _ps1dfc['imag'].values + a5 * rimag + b5
    _ps1dfc['e_Imag'] = np.sqrt(np.power(_ps1dfc['e_rmag'].values, 2) +
                                np.power(_ps1dfc['e_imag'].values, 2))

    return _ps1dfc


def apasstrn(_apassdfc):
    """
    transformations:
    U - g = a1(g - r) + b1
    B - g = a2(g - r) + b2
    V - g = a3(g - r) + b3
    R - r = a4(r - i) + b4
    I - i = a5(r - i) + b5
    Args:
        _apassdfc: catalog from apass in data frame

    Returns:
        transformed catalog in data frame

    """
    if not isinstance(_apassdfc, pd.DataFrame):
        raise RuntimeError('>>> Input must be data frame!')
    # parameters
    a1, ea1, b1, eb1 = 2.640, 0.023, -0.160, 0.015
    a2, ea2, b2, eb2 = 0.400, 0.008, 0.197, 0.005
    a3, ea3, b3, eb3 = -0.566, 0.004, -0.016, 0.003
    a4, ea4, b4, eb4 = -0.245, 0.002, -0.161, 0.002
    a5, ea5, b5, eb5 = -0.225, 0.010, -0.365, 0.003
    au1, eau1, au2, eau2, aub, eaub = -0.2218, 0.199, 1.460, 0.198, -0.3076, -0.048

    grmag = _apassdfc['gmag'].values - _apassdfc['rmag'].values
    gimag = _apassdfc['gmag'].values - _apassdfc['imag'].values
    rimag = _apassdfc['rmag'].values - _apassdfc['imag'].values
    BVmag = _apassdfc['Bmag'].values - _apassdfc['Vmag'].values

    # _apassdfc['Umag'] = _apassdfc['gmag'].values + a1 * grmag + b1
    # _apassdfc['e_Umag'] = np.sqrt(np.power(_apassdfc['e_gmag'].values, 2) +
    #                               np.power(_apassdfc['e_rmag'].values, 2))

    # _apassdfc['Umag'] = _apassdfc['Bmag'].values + au1 * BVmag + au2 * grmag + aub
    # _apassdfc['e_Umag'] = np.sqrt(np.power(_apassdfc['e_gmag'].values, 2) +
    #                               np.power(_apassdfc['e_Bmag'].values, 2))

    _apassdfc['Umag'] = _apassdfc['gmag'].values + 1.4312 * grmag + 0.1426 * gimag - 0.1391
    _apassdfc['e_Umag'] = np.sqrt(np.power(_apassdfc['e_gmag'].values, 2) +
                                  np.power(_apassdfc['e_rmag'].values, 2))

    _apassdfc['Bmag'] = _apassdfc['gmag'].values + a2 * grmag + b2
    _apassdfc['e_Bmag'] = np.sqrt(np.power(_apassdfc['e_gmag'].values, 2) +
                                  np.power(_apassdfc['e_rmag'].values, 2))
    _apassdfc['Vmag'] = _apassdfc['gmag'].values + a3 * grmag + b3
    _apassdfc['e_Vmag'] = np.sqrt(np.power(_apassdfc['e_gmag'].values, 2) +
                                  np.power(_apassdfc['e_rmag'].values, 2))
    _apassdfc['Rmag'] = _apassdfc['rmag'].values + a4 * rimag + b4
    _apassdfc['e_Rmag'] = np.sqrt(np.power(_apassdfc['e_imag'].values, 2) +
                                  np.power(_apassdfc['e_rmag'].values, 2))
    _apassdfc['Imag'] = _apassdfc['imag'].values + a5 * rimag + b5
    _apassdfc['e_Imag'] = np.sqrt(np.power(_apassdfc['e_rmag'].values, 2) +
                                  np.power(_apassdfc['e_imag'].values, 2))
    return _apassdfc

def get_catalog(ra, dec, radius, catalog, save=None, columns=None, mag_limit=None, limit_band=None):
	#catalog=["I/331/apop","I/284/out"],row_limit = -1
    if columns is not None:
        v = Vizier(catalog=catalog,row_limit = -1,columns=columns)
    else:
        v = Vizier(catalog=catalog,row_limit = -1)
    result = v.query_region(SkyCoord(" %s  %s"%(ra, dec),unit= (u.hourangle,u.deg)),radius=radius*u.arcmin)
    table = result[0]
    if 'RA_ICRS' in table.keys():
        table.rename_column('RA_ICRS', 'RA')
        table.rename_column('DE_ICRS', 'DEC')
    else:
        table.rename_column('_RAJ2000', 'RA')
        table.rename_column('_DEJ2000', 'DEC')

    band_name_list = ['U','B','V','R','I','u','g','r','i','z']
    for band_name in band_name_list:
        if band_name + 'mag' in table.keys():
            table.rename_column(band_name+'mag', band_name)
        if 'e_' + band_name + 'mag' in table.keys():
            table.rename_column('e_' + band_name + 'mag', band_name + '_err')  

    if 'sdss' in catalog:
        table = table['RA','DEC','umag','e_umag','gmag','e_gmag','rmag','e_rmag','imag','e_imag','zmag','e_zmag',]
        mark = table['gmag'] < 19
        mark[table['gmag'].mask.nonzero()[0]] = False
        table = table[mark]
        SDSS_to_Johnson(table)
    if 'syntphot' in catalog:
        for band in ['U','B','V','R','I']:
            table[band+'_err'] = 1.0857*table['e_F'+band]/table['F'+band]
        table = table[['RA','DEC','B','B_err','U','U_err','V','V_err','R','R_err','I','I_err']]
        mark = table['B'] < 99
        mark[table['B'].mask.nonzero()[0]] = False
        table = table[mark]
    table.sort(['RA','DEC'])
    len_data = len(table)
    '''
    mark = [True for i in range(len_data)]
    for i in range(len_data-1):
    	seperation = np.sqrt((table['RA'][i] - table['RA'][i+1])**2 + (table['DEC'][i] - table['DEC'][i+1])**2)
    	if seperation < 0.0025:
    		mark[i] = False
    table = table[mark]
    '''
    if mag_limit is not None and limit_band is not None:
        mark = table[limit_band] < mag_limit
        table = table[mark]
    print(table[:10])

    if save is not None:
    	table.write(save + '.csv', format='csv')
    return table