import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

def quick_mcmc(method, xdata, ydata, ydata_err, guess, bounds, steps=5000, plot=True):
	params,params_covariance=optimize.curve_fit(method, xdata, ydata, sigma=ydata_err, p0=guess, maxfev=500000, bounds=bounds)
	print(params)
	print(np.sum(((method(xdata, *params) - ydata)/ydata_err)**2) / (len(xdata) - len(params)))

	### MCMC
	len_para = len(params)
	np.random.seed(123456789)
	def log_likelihood(theta, xdata, ydata, ydata_err):
		return -0.5*np.sum(((ydata - method(xdata, *theta))/ydata_err)**2)
	def log_prior(theta):
		for item_i, item in enumerate(theta):
			if item < bounds[0][item_i] or item > bounds[1][item_i]:
				return -np.inf
		return 0.0
	def log_probability(theta, xdata, ydata, ydata_err):
		lp = log_prior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + log_likelihood(theta, xdata, ydata, ydata_err)
	rand_start = np.zeros([32,len_para])
	for start_i in range(1,32):
		while(1):
			for para_i in range(len_para):
				rand_start[start_i][para_i] = np.random.randn()*params[para_i]*0.05
			if not np.isinf(log_prior(params + rand_start[start_i])):
				break
	start_MC = params + rand_start
	nwalkers, ndim = start_MC.shape
	AutocorrError = emcee.autocorr.AutocorrError
	while(1):
		sampler = emcee.EnsembleSampler(
			nwalkers, ndim, log_probability, args=(xdata, ydata, ydata_err)
		)
		sampler.run_mcmc(start_MC, steps, progress=True)
		try:
			tau = sampler.get_autocorr_time()
		except AutocorrError:
			steps *= 2
		else:
			break
	samples = sampler.get_chain()
	labels = ['%d'%i for i in range(len_para)]
	if plot:
		fig, axes = plt.subplots(6, figsize=(10, 7), sharex=True)
		for i in range(ndim):
		    ax = axes[i]
		    ax.plot(samples[:, :, i], "k", alpha=0.3)
		    ax.set_xlim(0, len(samples))
		    ax.set_ylabel(labels[i])
		    ax.yaxis.set_label_coords(-0.1, 0.5)

		axes[-1].set_xlabel("step number")
		plt.show()
	flat_samples = sampler.get_chain(discard=int(2.5*np.max(tau)), thin=int(np.max(tau)/2), flat=True)
	log_likelihood_value = []
	for i in range(flat_samples.shape[0]):
		log_likelihood_value.append(log_likelihood(flat_samples[i], xdata, ydata, ydata_err))
	max_log_likelihood = np.max(log_likelihood_value)
	for i in range(flat_samples.shape[0]):
		if log_likelihood_value[i] == max_log_likelihood:
			pos_max = i
	theta = flat_samples[pos_max]
	print(theta)
	print(np.sum(((method(xdata, *theta) - ydata)/ydata_err)**2) / (len(xdata) - len(theta)))
	if plot:
		fig = corner.corner(flat_samples, labels=labels, truths=theta)
		plt.show()
	for i in range(len_para):
		print(labels[i])
		print(np.percentile(flat_samples[:,i], [16, 50, 84]))
	if plot:
		plt.errorbar(xdata, ydata, yerr=ydata_err, capsize=5, fmt='o', linestyle='')
		plt.plot(xdata, method(xdata, *theta))
		plt.show()
	###
	return theta

'''
x = np.linspace(0,10,11)
y = np.linspace(0,10,11) + np.random.randn(11)*0.1
y_err = np.ones(11)*0.2
def method(x, k, b):
	return k*x + b
guess = [1,0]
bounds = [(-np.inf, -np.inf),(np.inf, np.inf)]

quick_mcmc(method, x, y, y_err, guess, bounds, plot=False)
'''


