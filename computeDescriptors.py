from matplotlib.mlab import find
import numpy as np
import scipy.interpolate as interp


def computeHarmonicEnvelope(hfreq, hmag, nyqFreq, minFFTVal, fftSize, freqs):
	# compute harmonic envelope as spline or linear interpolation
	#print("hfreq", hfreq[iFrame,:])
	#print("hmag", hmag[iFrame,:])
	idx= find(hfreq>0)
	if len(idx)>3:
	    hfreqSpline=hfreq[idx]
	    hmagSpline=hmag[idx]
	    hfreqSpline=np.hstack([0, hfreqSpline, nyqFreq-1]) #hfreqSpline[-1]+ (hfreqSpline[-1]-hfreqSpline[-2])])
	    hmagSpline=np.hstack([minFFTVal, hmagSpline, minFFTVal]) #TODO: change first parameter 0 by mX[0]
	    harmonicEnvelope=interp.interp1d(hfreqSpline, hmagSpline, kind='linear')(freqs)
	    #energyBand[iFrame,:]=energyInBands(harmonicEnvelope[iFrame,:],bandCentersHz, fs)
	    #plt.plot(energyBand[iFrame,:])
	    #print("energyBand[iFrame,:]:",energyBand[iFrame,:])
	else:
	    harmonicEnvelope=np.ones(fftSize/2+1)*minFFTVal
	return harmonicEnvelope

def energyInBands(fftMag, bandCentersHz, fs, minFFTVal):
    if (bandCentersHz == []):
        bandCentersHz = np.array(
            [103, 171, 245, 326, 413, 508, 611, 722, 843, 975, 1117, 1272, 1439, 1621, 1819, 2033, 2266, 2518, 2792,
             3089, 3412, 3761, 4141, 4553, 5000, 5485, 6011, 6582, 7202, 7874, 8604, 9396, 10255, 11187, 12198, 13296,
             14487, 15779, 17181, 18703])

    l = len(fftMag)
    bandCentersHz = np.hstack((1, bandCentersHz, fs / 2)) - 1
    bandCentersBin = np.asarray(bandCentersHz * l / (fs / 2), dtype=int)
    fftMag = 10 ** (fftMag / 20)
    energyBand_dB = np.zeros(len(bandCentersHz) - 2) * minFFTVal
    for iBand in range(1, len(bandCentersHz) - 1):
        iBandIndx = np.array(range(bandCentersBin[iBand - 1], bandCentersBin[iBand + 1]))
        energyBand_dB[iBand - 1] = 10 * np.log10(
            sum(np.power(fftMag[iBandIndx], 2) + np.finfo(float).eps) / len(iBandIndx))
    #energy_bands = 10 ** (energyBand_dB / 20)
    #rmsEnergy_dB = 20 * np.log10(np.sqrt(np.mean(energy_bands ** 2, 0)))
    #energy_bands_norm = energyBand_dB / rmsEnergy_dB
    #energy_bands_norm = energy_bands_norm / 4

    return energyBand_dB