# functions that implement analysis and synthesis of sounds using the Harmonic Model
# (for example usage check the models_interface directory)

import numpy as np
import math
import dftModel as DFT
#import utilFunctions as UF
import utils as UF
import sineModel as SM
from scipy.signal import get_window


import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../violinDemoRT'))
from pitchUtils import freq_from_autocorr


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


def harmonicAnalisys(x, fs, windowType='blackman', M=1201, N=2048, t=-90,
					 minSineDur=0.1, nH=100, minf0=130, maxf0=300, f0et=7, harmDevSlope=0.01, H=128):
    # hop size (has to be 1/4 of Ns)
    # H = 128
    # compute analysis window
    w = get_window(windowType, M)

    # detect harmonics of input sound
    hfreq, hmag, hphase , f0stable = harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
    return hfreq, hmag

def harmonicModelAnal(x, fs, window, fft_size, hop_size, min_fft_val, nSines, minf0, maxf0, f0et, harmDevSlope=0.01, minSineDur=.02):
	"""
	Analysis of a sound using the sinusoidal harmonic model
	x: input sound; fs: sampling rate, w: analysis window; N: FFT size (minimum 512); t: threshold in negative dB, 
	nH: maximum number of harmonics;  minf0: minimum f0 frequency in Hz, 
	maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
	harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
	returns xhfreq, xhmag, xhphase: harmonic frequencies, magnitudes and phases
	"""

	if (minSineDur <0):                                     # raise exception if minSineDur is smaller than 0
		raise ValueError("Minimum duration of sine tracks smaller than 0")
		
	#hN = fft_size / 2                                                # size of positive spectrum
	hM1 = int(math.floor((window.size + 1) / 2))                     # half analysis window size by rounding
	hM2 = int(math.floor(window.size / 2))                         # half analysis window size by floor
	x = np.append(np.zeros(hM2), x)                          # add zeros at beginning to center first window at sample 0
	x = np.append(x, np.zeros(hM2))                          # add zeros at the end to analyze last sample
	pin = hM1                                               # init sound pointer in middle of anal window          
	pend = x.size - hM1                                     # last sample to start a frame
	#fftbuffer = np.zeros(fft_size)                                 # initialize buffer for FFT
	window = window / sum(window)                                          # normalize analysis window
	hfreqp = []                                             # initialize harmonic frequencies of previous frame
	f0t = 0                                                 # initialize f0 track
	f0stable = 0                                            # initialize f0 stable

	while pin<=pend:
		#print("pin:", pin, " pend:", pend)
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		#--------- harmonic Analysis frame
		# mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft            
		# ploc = UF.peakDetection(mX, t)                        # detect peak locations   
		# iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
		# ipfreq = fs * iploc/N                                 # convert locations to Hz
		# f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
		# if ((f0stable==0)&(f0t>0)) \
		# 		or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
		# 	f0stable = f0t                                      # consider a stable f0 if it is close to the previous one
		# else:
		# 	f0stable = 0
		# hfreq, hmag, hphase = harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope) # find harmonics
		#-----------
		useTWM=0
		mX, f0stable, f0t, hfreq, hmag, hphase = harmonicModelAnalFrame (x1, window, fft_size, min_fft_val, fs, hfreqp, f0et, minf0, maxf0, nSines, f0stable, harmDevSlope, useTWM)
		hfreqp = hfreq #hfreq(previous)
		if pin == hM1:                                        # first frame
			xhfreq = np.array([hfreq])
			xhmag = np.array([hmag])
			xhphase = np.array([hphase])
		else:                                                 # next frames
			xhfreq = np.vstack((xhfreq,np.array([hfreq])))
			xhmag = np.vstack((xhmag, np.array([hmag])))
			xhphase = np.vstack((xhphase, np.array([hphase])))
		pin += hop_size                                              # advance sound pointer
	xhfreq = SM.cleaningSineTracks(xhfreq, round(fs * minSineDur / hop_size))     # delete tracks shorter than minSineDur
	return xhfreq, xhmag, xhphase, f0stable

def harmonicModelAnalFrame(grain, window, fftSize, minFFTVal, fs, hfreqp, f0et, minf0, maxf0, nH, f0stable, harmDevSlope, useTWM):
	mX, pX = DFT.dftAnal(grain, window, fftSize)                        # compute dft            
	ploc = UF.peakDetection(mX, minFFTVal)                        # detect peak locations   
	iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
	ipfreq = fs * iploc/fftSize                                 # convert locations to Hz
	if(useTWM):
		f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
	else:
		#try:
		f0t=freq_from_autocorr(grain,fs)
		if(f0t<minf0):
			f0t=0
		#except:
		#	f0t = minf0 #freq_from_autocorr(grain, fs)
		#	return mX, f0stable, f0t, np.zeros(nH), np.zeros(nH), np.zeros(nH)
	if ((f0stable == 0) & (f0t > 0)) \
			or ((f0stable > 0) & (np.abs(f0stable - f0t) < f0stable / 5.0)):
		f0stable = f0t  # consider a stable f0 if it is close to the previous one
	else:
		f0stable = 0
	hfreq=[]
	try:
		hfreq, hmag, hphase = harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope) # find harmonics
	except:
		return mX, f0stable, f0t, np.zeros(nH), np.zeros(nH), np.zeros(nH)
	return mX, f0stable, f0t, hfreq, hmag, hphase

# def freq_from_autocorr(sig, fs):
#     """
#     Estimate frequency using autocorrelation
#     """
#     # Calculate autocorrelation (same thing as convolution, but with
#     # one input reversed in time), and throw away the negative lags
#     corr = fftconvolve(sig, sig[::-1], mode='full')
#     corr = corr[len(corr)//2:]
#
#     # Find the first low point
#     d = diff(corr)
#     start = find(d > 0)[0]
#
#     # Find the next peak after the low point (other than 0 lag).  This bit is
#     # not reliable for long signals, due to the desired peak occurring between
#     # samples, and other peaks appearing higher.
#     # Should use a weighting function to de-emphasize the peaks at longer lags.
#     peak = argmax(corr[start:]) + start
#
#     if (peak>=corr.shape[0]-1):
#         peak=corr.shape[0]-2
#     elif (peak == 0):
#         peak = 1
#     px, py = parabolic(corr, peak)
#
#     return fs / px


# def parabolic(f, x):
# 	"""Quadratic interpolation for estimating the true position of an
# 	inter-sample maximum when nearby samples are known.
#
# 	f is a vector and x is an index for that vector.
#
# 	Returns (vx, vy), the coordinates of the vertex of a parabola that goes
# 	through point x and its two neighbors.
#
# 	Example:
# 	Defining a vector f with a local maximum at index 3 (= 6), find local
# 	maximum if points 2, 3, and 4 actually defined a parabola.
#
# 	In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
#
# 	In [4]: parabolic(f, argmax(f))
# 	Out[4]: (3.2142857142857144, 6.1607142857142856)
#
# 	"""
# 	xv=1
# 	yv=0
# 	try:
# 		xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1] + sys.float_info.epsilon) + x
# 	except:
# 		print("kk")
# 	yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
# 	return (xv, yv)