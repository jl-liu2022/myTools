import ephem as ep
import numpy as np

observatory = ep.Observer()
observatory.lon = '117.5750' # longtitude, unit is degree
observatory.lat = '40.3933' # latitude, unit is degree
timezone = +8

ra_target = "12:26:48.846"
dec_target = "+08:26:55.32"
observatory.date = "2023-02-27 19:29:43"
horizon = '-18' # twightlight

observatory.horizon = horizon
twightlight_begin = observatory.next_setting(ep.Sun(), use_center=True)
twightlight_end = observatory.next_rising(ep.Sun(), use_center=True)

eq = ep.Equatorial(ra_target, dec_target)
star = ep.FixedBody()
ra, dec = eq.ra*180./np.pi, eq.dec*180./np.pi

star = ep.FixedBody()
star._ra = ep.hours(eq.ra)
star._dec = ep.degrees(eq.dec)
star.compute(observatory)
moon = ep.Moon()
salt = star.alt*180./np.pi
saz = star.az*180./np.pi
air = 1/np.cos((90.-salt)*np.pi/180.)
airmass = air*(1-0.0012*(air**2.-1))

observatory.elevation = 950 # elevation, unit is meter
moon.compute(observatory)
moon_phase = moon.phase
distance=ep.separation(star,moon)*180./np.pi
print("Moon phase (degree): "+str(round(moon.phase,2)))
print("Moon distance (degree): "+str(round(distance,2)))
print("Twightlight_begin (UT): "+str(twightlight_begin))
print("Twightlight_end (UT): "+str(twightlight_end))
print("Star altitude: "+str(round(salt,3)))
print("Star az: "+str(round(saz,3)))
print("Airmass: "+str(round(airmass,3)))

