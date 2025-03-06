import numpy as np
from scipy.stats import pearsonr
from scipy.stats import kstwobign

def twoDKS(sample1, sample2):
	sample1 = np.array(sample1)
	sample2 = np.array(sample2)
	# Check the shapes of the samples
	if sample1.shape[0] != 2 or sample2.shape[0] != 2:
		raise Exception('The shape of the sample should be like (2, n)')

	N1 = sample1.shape[1]
	N2 = sample2.shape[1]

	# Sample1 as origin
	d1 = 0.0
	for i in range(N1):
		count1 = twoDCount(sample1[0][i], sample1[1][i], sample1)
		count2 = twoDCount(sample1[0][i], sample1[1][i], sample2)
		count1 += (count1>count2)*1/N1
		diff = np.abs(count1 - count2)
		d1 = np.max([d1, np.max(diff)])

	# Sample2 as origin
	d2 = 0.0
	for i in range(N2):
		count1 = twoDCount(sample2[0][i], sample2[1][i], sample1)
		count2 = twoDCount(sample2[0][i], sample2[1][i], sample2)
		count2 += (count1<count2)*1/N2
		diff = np.abs(count1 - count2)
		d2 = np.max([d2, np.max(diff)])

	d = 0.5*(d1+d2)

	sqen = np.sqrt(N1*N2/float(N1+N2))
	print(d1, d2, d, sqen)
	r1 = pearsonr(sample1[0],sample1[1])[0]
	r2 = pearsonr(sample2[0],sample2[1])[0]
	rr = np.sqrt(1.0-0.5*(r1*r1+r2*r2))
	prob=1-kstwobign.cdf(d*sqen/(1.0+rr*(0.25-0.75/sqen)))
	return prob, r1, r2, rr


def twoDCount(x, y, sample):
	# l:left r:right b:bottom u:upper
	N = np.size(sample[0])	
	xr = sample[0]>x
	xl = ~xr
	yu = sample[1]>y
	yb = ~yu
	bl = (xl*yb).sum()
	br = (xr*yb).sum()
	ul = (xl*yu).sum()
	ur = (xr*yu).sum()
	if np.array([x,y]) in sample.T:
		bl -= 1
	count = np.array([bl, br, ul, ur]).astype('float')
	fN = 1/N
	count *= fN

	return count

np.random.seed(399991)
# Test
from scipy.stats import kstest
from matplotlib import pyplot as plt
xa = np.random.rand(100)
xb = np.random.rand(300)
a = [xa, xa]
b = [xb, xb]
print(twoDKS(a,b))
print(kstest(xa,xb))
plt.scatter(a[0],a[1])
plt.scatter(b[0],b[1])
plt.show()