from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from astropy.coordinates import Angle
from astropy import units as u

#import dustmaps.sfd
#dustmaps.sfd.fetch()

def coordsToEbv(ra, dec):

	ra = str(ra)
	dec = str(dec)
	if ':' in ra:
		coords = SkyCoord(ra, dec,  unit='hourangle, deg', frame='icrs')
	else:
		coords = SkyCoord(float(ra), float(dec),  unit='deg', frame='icrs')
	sfd = SFDQuery()
	ebv98 = sfd(coords)
	ebv11 = ebv98*0.86
	return ebv11

# Test
#ra = 190.89306
#dec = 11.57663
#print(coordsToEbv(ra, dec))
