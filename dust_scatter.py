import astropy.units as u
import astropy.constants as c
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.random.seed(123456)
radius_generator_c = 1/3
pi = np.pi
c = 3e5
# generate a photon
#outer_radius = 3000 #km/s
#tau = 1
#alpha = tau / outer_radius
def radius_generator(outer_radius):
	random = np.random.rand()
	return random**(radius_generator_c)*outer_radius

def theta_generator():
	random = np.random.rand()
	#return pi*random
	return 2*random - 1

#photon_status = [radius_generator(outer_radius), theta_generator(), False]
# photon travel
def next_scatter_length(alpha):
	#alpha = 1/outer_radius*tau
	random = np.random.rand()
	return -1 * np.log(random) / alpha

def photon_travel(photon_status, alpha, outer_radius, radius_record_=None, theta_record_=None, lam_record_=None, travel_length_record_=None, inner_radius_dust=0):
	while(1):
		travel_length = next_scatter_length(alpha)
		if inner_radius_dust > 0:
			photon_next_radius = np.sqrt(photon_status[0]**2 + travel_length**2 + 2*photon_status[0]*travel_length*photon_status[1])
			# in inner region at first
			if photon_status[0] < inner_radius_dust:
				inner_travel_length = -photon_status[0]*photon_status[1] + np.sqrt(inner_radius_dust**2 - (1-photon_status[1]**2)*photon_status[0]**2)
			# in outer region at first
			elif np.sqrt(1 - (inner_radius_dust**2/photon_status[0]**2)) + photon_status[1] < 0:
				distance_to_inner = -photon_status[0]*photon_status[1] - np.sqrt(inner_radius_dust**2 - (1-photon_status[1]**2)*photon_status[0]**2)
				if travel_length > distance_to_inner:
					inner_travel_length = (-photon_status[0]*photon_status[1] - distance_to_inner)*2
				else:
					inner_travel_length = 0
			else:
				inner_travel_length = 0
			travel_length += inner_travel_length
		if travel_length_record_ is not None:
			travel_length_record_.append(travel_length)
		#photon_next_radius = np.sqrt(photon_status[0]**2 + travel_length**2 + 2*photon_status[0]*travel_length*np.cos(photon_status[1]))
		photon_next_radius = np.sqrt(photon_status[0]**2 + travel_length**2 + 2*photon_status[0]*travel_length*photon_status[1])
		if photon_next_radius > outer_radius:
			photon_status[-1] = 1
			if radius_record_ is not None:
				radius_record_.append(photon_next_radius)
			if theta_record_ is not None:
				theta_record_.append(photon_status[1])
			if lam_record_ is not None:
				lam_record_.append(photon_status[2])
			break
		else:
			theta_1 = 1 / (2 * photon_next_radius * travel_length) *(photon_next_radius**2 + travel_length**2 - photon_status[0]**2)
			#lam_1 = photon_status[2] * (1 + photon_next_radius * np.cos(theta_1) / c)
			lam_1 = photon_status[2] * (1 + photon_next_radius * theta_1 / c)
			theta_2 = theta_generator()
			#lam_2 = lam_1 * (1 - photon_next_radius * np.cos(theta_2) / c)
			lam_2 = lam_1 * (1 - photon_next_radius * theta_2 / c)
			photon_status = [photon_next_radius, theta_2, lam_2, 0]
			if radius_record_ is not None:
				radius_record_.append(photon_next_radius)
			if theta_record_ is not None:
				theta_record_.append(theta_2)
			if lam_record_ is not None:
				lam_record_.append(lam_2)
	return photon_status

#def photon_lam(rest_lam, outer_radius, radius, theta):
#	return rest_lam * (1 + radius * np.cos(theta) / c)

def photons_travel(photon_number, outer_radius_emission, tau, rest_lam, record=True, outer_radius_dust=None, inner_radius_dust=0):
	if outer_radius_dust is None:
		outer_radius_dust = outer_radius_emission
	alpha = tau / (outer_radius_dust - inner_radius_dust)
	photons_status = []
	if record == True:
		travel_length_record = []
		radius_record = []
		theta_record = []
		lam_record = []
	for i in range(photon_number):
		radius_record_ = []
		travel_length_record_ = [0]
		theta_record_ = []
		lam_record_ = []
		radius_ = radius_generator(outer_radius_emission)
		theta_ = theta_generator()
		#lam_ = rest_lam * (1 - radius_ * np.cos(theta_) / c)
		lam_ = rest_lam * (1 - radius_ * theta_ / c)
		photon_status_ = [radius_, theta_, lam_, 0]
		if record == True:
			radius_record_.append(radius_)
			theta_record_.append(theta_)
			lam_record_.append(lam_)
			photon_status_ = photon_travel(photon_status_, alpha, outer_radius_dust, radius_record_, theta_record_, lam_record_, travel_length_record_, inner_radius_dust=inner_radius_dust)
			radius_record.append(radius_record_)
			theta_record.append(theta_record_)
			lam_record.append(lam_record_)
			travel_length_record.append(travel_length_record_)
		else:
			photon_status_ = photon_travel(photon_status_, alpha, outer_radius_dust, inner_radius_dust=inner_radius_dust)
		photons_status.append(photon_status_)
	if record == True:
		return np.array(photons_status), radius_record, travel_length_record, theta_record, lam_record
	else:
		return np.array(photons_status)


def scatter_profile(outer_radius_emission, tau, rest_lam, z=0, photon_number=40000, record=False, save=None, outer_radius_dust=None, inner_radius_dust=0):
	photons_status = photons_travel(photon_number, outer_radius_emission, tau, rest_lam=rest_lam, record=record, outer_radius_dust=outer_radius_dust, inner_radius_dust=inner_radius_dust)
	photons_lam = photons_status[:,2]/(1+z)
	bins = int((np.max(photons_lam) - np.min(photons_lam)) / 5) + 1
	hist_lam, hist_edge = np.histogram(photons_lam, bins=bins)
	wave = hist_edge[1:] - (hist_edge[1] - hist_edge[0])*0.5
	'''
	print(bins)
	plt.hist(photons_lam, bins=hist_edge, density=True)
	plt.plot(wave, hist_lam)
	plt.show()
	exit()
	'''
	hist_lam = hist_lam / np.max(hist_lam)
	wave = (wave - rest_lam) / rest_lam / (outer_radius_emission / c)
	hist_lam = np.concatenate([[0], hist_lam])
	wave = np.concatenate([[-1], wave])
	if save is not None:
		print('save')
		np.savetxt(save, np.array([wave, hist_lam]).T)
	return interp1d(wave, hist_lam, kind='linear', bounds_error=False, fill_value=0)

def scatter_kernel(tau, expanding=False):
	if expanding == False:
		kernel_path = '/Users/liujialian/work/SN_catalog/scatter_database/tau_%.1f.txt'%tau
	else:
		kernel_path = '/Users/liujialian/work/SN_catalog/scatter_database_expanding_dust/tau_%s.txt'%tau
	return np.loadtxt(kernel_path).T

def scatter_f(x, outer_radius_emission, rest_lam, v, kernel):
	kernel_ = (kernel[0] * (outer_radius_emission / c) * rest_lam + rest_lam) * (1 + v/c)
	return interp1d(kernel_, kernel[1], kind='linear', bounds_error=False, fill_value=0)(x)
'''
# expanding
outer_radius_emission = 2000 #km/s
outer_radius_dust = 3000 #km/s
inner_radius_dust = 2000 #km/s
#tau = 3
rest_lam = 10000

tau_list = np.linspace(0.1, 5, 50)
tau_list = [0.5]
for tau_ in tau_list:
	print('tau: %.1f'%tau_)
	save_ = 'scatter_database_expanding_dust/tau_%.1f.txt'%tau_
	scatter = scatter_profile(outer_radius_emission, tau_, rest_lam, z=0, photon_number=5000000, record=False, save=save_, outer_radius_dust=outer_radius_dust, inner_radius_dust=inner_radius_dust)
	wave = np.linspace(-1, 3, 100)
	fontsize_ = 18
	fig, ax = plt.subplots(figsize=(8,6))
	ax.text(0.8, 0.8, '$\\tau=%.1f$'%tau_, transform=ax.transAxes, fontsize=fontsize_)
	ax.set_xlabel('Wavelength [$\\rm \\Delta$$\\lambda/\\lambda_0/(v_{\\rm max}/c)$]', fontsize=fontsize_)
	ax.set_ylabel('Scaled Flux', fontsize=fontsize_)
	ax.tick_params(labelsize=fontsize_)
	ax.plot(wave, scatter(wave))
	plt.show()
exit()

# no expanding
outer_radius = 3000 #km/s
#tau = 3
rest_lam = 10000

tau_list = np.linspace(0.1, 5, 50)
for tau_ in tau_list:
	print('tau: %.1f'%tau_)
	save_ = 'scatter_database/tau_%.1f.txt'%tau_
	scatter = scatter_profile(outer_radius, tau_, rest_lam, z=0, photon_number=5000000, record=False, save=save_)
	#wave = np.linspace(-1, 3, 100)
	#plt.plot(wave, scatter(wave))
	#plt.show()
exit()


scatter = scatter_profile(outer_radius, tau, rest_lam, z=0, photon_number=100000, record=False)

wave = np.linspace(-1, 3, 100)
plt.plot(wave, scatter(wave))
plt.show()
exit()



x_bins = np.linspace(9750, 11000, 101)
plt.hist(photons_lam, bins=x_bins)
plt.show()

exit()

### plot
fontsize_ = 18
taulist = ['0','1','2','0.5']
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('Wavelength [$\\rm \\Delta$$\\lambda/\\lambda_0/(v_{\\rm max}/c)$]', fontsize=fontsize_)
ax.set_ylabel('Scaled Flux', fontsize=fontsize_)
ax.tick_params(labelsize=fontsize_)

otherline = [
	 ('solid', 'solid'),      # Same as (0, ()) or '-'
	 ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot'),  # Same as '-.'
     ('long dash with offset', (5, (10, 3))),
     ('long long dash with offset', (5, (20, 3))),
     ('densely dashed',        (0, (5, 1))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('dotted', 'dotted'),]    # Same as (0, (1, 1)) or ':'

for tau_i, tau in enumerate(taulist):
	if tau != '0.5':
		scatter_profile = scatter_kernel(float(tau))
	else:
		scatter_profile = scatter_kernel(float(tau), expanding=True)
	ax.plot(scatter_profile[0], scatter_profile[1], label='$\\tau=%s$'%tau, linestyle=otherline[tau_i][1])
ax.set_xlim(-1,3)
plt.legend(fontsize=fontsize_)
fig.savefig('scatter_plot.pdf', bbox_inches='tight')
plt.show()
'''
