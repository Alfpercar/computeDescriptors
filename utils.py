import numpy as np
from scipy.signal import resample, blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift
import math, copy, sys, os


def isPower2(num):
	"""
	Check if num is power of two
	"""
	return ((num & (num - 1)) == 0) and num > 0


def peakDetection(mX, t):
	"""
	Detect spectral peak locations
	mX: magnitude spectrum, t: threshold
	returns ploc: peak locations
	"""

	thresh = np.where(mX[1:-1]>t, mX[1:-1], 0);             # locations above threshold
	next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
	prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
	ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
	ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
	return ploc

def peakInterp(mX, pX, ploc):
	"""
	Interpolate peak values using parabolic interpolation
	mX, pX: magnitude and phase spectrum, ploc: locations of peaks
	returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
	"""

	val = mX[ploc]                                          # magnitude of peak bin
	lval = mX[ploc-1]                                       # magnitude of bin at left
	rval = mX[ploc+1]                                       # magnitude of bin at right
	iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
	ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
	ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks by linear interpolation
	return iploc, ipmag, ipphase

def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
	"""
	Function that wraps the f0 detection function TWM, selecting the possible f0 candidates
	and calling the function TWM with them
	pfreq, pmag: peak frequencies and magnitudes,
	ef0max: maximum error allowed, minf0, maxf0: minimum  and maximum f0
	f0t: f0 of previous frame if stable
	returns f0: fundamental frequency in Hz
	"""
	if (minf0 < 0):                                  # raise exception if minf0 is smaller than 0
		raise ValueError("Minumum fundamental frequency (minf0) smaller than 0")

	if (maxf0 >= 10000):                             # raise exception if maxf0 is bigger than 10000Hz
		raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

	if (pfreq.size < 3) & (f0t == 0):                # return 0 if less than 3 peaks and not previous f0
		return 0

	f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0] # use only peaks within given range
	if (f0c.size == 0):                              # return 0 if no peaks within range
		return 0
	f0cf = pfreq[f0c]                                # frequencies of peak candidates
	f0cm = pmag[f0c]                                 # magnitude of peak candidates

	if f0t>0:                                        # if stable f0 in previous frame
		shortlist = np.argwhere(np.abs(f0cf-f0t)<f0t/2.0)[:,0]   # use only peaks close to it
		maxc = np.argmax(f0cm)
		maxcfd = f0cf[maxc]%f0t
		if maxcfd > f0t/2:
			maxcfd = f0t - maxcfd
		if (maxc not in shortlist) and (maxcfd>(f0t/4)): # or the maximum magnitude peak is not a harmonic
			shortlist = np.append(maxc, shortlist)
		f0cf = f0cf[shortlist]                         # frequencies of candidates

	if (f0cf.size == 0):                             # return 0 if no peak candidates
		return 0

	f0, f0error = UF_C.twm(pfreq, pmag, f0cf)        # call the TWM function with peak candidates

	if (f0>0) and (f0error<ef0max):                  # accept and return f0 if below max error allowed
		return f0
	else:
		return 0

def harmonicDetection(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):
	"""
	Detection of the harmonics of a frame from a set of spectral peaks using f0
	to the ideal harmonic series built on top of a fundamental frequency
	pfreq, pmag, pphase: peak frequencies, magnitudes and phases
	f0: fundamental frequency, nH: number of harmonics,
	hfreqp: harmonic frequencies of previous frame,
	fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
	returns hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases
	"""

	if (f0<=0):                                          # if no f0 return no harmonics
		return np.zeros(nH), np.zeros(nH), np.zeros(nH)
	hfreq = np.zeros(nH)                                 # initialize harmonic frequencies
	hmag = np.zeros(nH)-100                              # initialize harmonic magnitudes
	hphase = np.zeros(nH)                                # initialize harmonic phases
	hf = f0*np.arange(1, nH+1)                           # initialize harmonic frequencies
	hi = 0                                               # initialize harmonic index
	if hfreqp == []:                                     # if no incomming harmonic tracks initialize to harmonic series
		hfreqp = hf
	while (f0>0) and (hi<nH) and (hf[hi]<fs/2):          # find harmonic peaks
		pei = np.argmin(abs(pfreq - hf[hi]))               # closest peak
		dev1 = abs(pfreq[pei] - hf[hi])                    # deviation from perfect harmonic
		dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi]>0 else fs) # deviation from previous frame
		threshold = f0/3 + harmDevSlope * pfreq[pei]
		if ((dev1<threshold) or (dev2<threshold)):         # accept peak if deviation is small
			hfreq[hi] = pfreq[pei]                           # harmonic frequencies
			hmag[hi] = pmag[pei]                             # harmonic magnitudes
			hphase[hi] = pphase[pei]                         # harmonic phases
		hi += 1                                            # increase harmonic index
	return hfreq, hmag, hphase

