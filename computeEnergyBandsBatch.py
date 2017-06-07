from tkinter import *

import numpy as np
import matplotlib.pyplot as plt

import os, sys
from scipy.signal import get_window
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import sineModel as SM
import harmonicModel as HM
from computeDescriptors import energyInBands, computeHarmonicEnvelope

import scipy.io


def main():
    doPlot = 0
    fftSize = 2048  # fftSize
    hopSize = 128  # hop Size
    minFFTVal = -120
    bandCentersHz = np.array(
        [103, 171, 245, 326, 413, 508, 611, 722, 843, 975, 1117, 1272, 1439, 1621, 1819, 2033, 2266, 2518, 2792, 3089,
         3412, 3761, 4141, 4553, 5000, 5485, 6011, 6582, 7202, 7874, 8604, 9396, 10255, 11187, 12198, 13296, 14487,
         15779, 17181, 18703])
    # read input files
    data_dir = "/Users/alfonso/recordings/recordingsYamaha/phrases/"
    scoreListFilename = "recording_script.scoreList"
    scoreList = [line.rstrip('\r\n') for line in open(data_dir + scoreListFilename)]
    # data_dir="/Users/alfonso/matlab/IndirectAcquisition/keras/dataforMarius/export"
    # files = [os.path.join(data_dir, file_i) for file_i in os.listdir(data_dir) if file_i.endswith('.mat')]

    for score in scoreList:
        inputFile = data_dir + "tools_phrases_" + score + "/" + score + "-16bit.wav"
        outputFile = data_dir + "tools_phrases_" + score + "/" + score + "-16bit-EnergyBankFilter2.txt"
        # TODO: waveFile is inside tools_ folder
        print("opening:", inputFile)

        (fs, x) = UF.wavread(inputFile)
        NyqFreq = fs / 2
        if (doPlot):
            root = Tk()
            root.title('sms-tools models GUI')
            root.geometry('+0+0')
            # create figure to show plots
            fig = plt.figure(figsize=(12, 9))

            # plot the input sound
            plt.subplot(3, 1, 1)
            plt.plot(np.arange(x.size) / float(fs), x)
            plt.axis([0, x.size / float(fs), min(x), max(x)])
            plt.ylabel('amplitude')
            plt.xlabel('time (sec)')
            plt.title('input sound: x')
            plt.tight_layout()
            plt.ion()
            plt.show()
            fig.set_tight_layout(True)

        hfreq, hmag = HM.harmonicAnalisys(x, fs, H=hopSize, t=minFFTVal)

        freqs = np.array(range(0, fftSize / 2 + 1)) * fs / (fftSize)
        freqs[-1] = NyqFreq - 1

        harmonicEnvelope = np.zeros(shape=(hfreq.shape[0], fftSize / 2 + 1))
        energyBand = np.zeros(shape=(hfreq.shape[0], len(bandCentersHz)))
        for iFrame in range(0, hfreq.shape[0]):
            # # compute harmonic envelope as spline
            harmonicEnvelope[iFrame, :] = computeHarmonicEnvelope(hfreq[iFrame, :], hmag[iFrame, :], NyqFreq, minFFTVal,
                                                                  fftSize, freqs)
            energyBand[iFrame, :] = energyInBands(harmonicEnvelope[iFrame, :], bandCentersHz, fs, minFFTVal)
            if (doPlot):
                plt.subplot(3, 1, 2)
                plt.plot(harmonicEnvelope[iFrame, :])
                plt.subplot(3, 1, 3)
                plt.plot(energyBand[iFrame, :])

        # Save energyBand
        # energyBandFile = open(outputFile, "w")
        # energyBandFile.write
        # energyBandFile.close()
        np.savetxt(outputFile, energyBand, fmt='%.5f', delimiter=' ', newline='\n', header='', footer='',
                   comments='%40 energy bank filter (dB)')  # [source]

        print("End!")
        if (doPlot):
            root.mainloop()





def doPlot(x, fs, hfreq, hmag, hopSize):
    # create figure to show plots
    fig = plt.figure(figsize=(12, 9))

    # frequency range to plot
    maxplotfreq = 5000.0

    # plot the input sound
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(x.size) / float(fs), x)
    plt.axis([0, x.size / float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot the harmonic frequencies
    # print("shape hfreq=", hfreq.shape )
    # plt.subplot(3,1,2)
    # if (hfreq.shape[1] > 0):
    # 	numFrames = hfreq.shape[0]
    # 	frmTime = H*np.arange(numFrames)/float(fs)
    # 	hfreq[hfreq<=0] = np.nan
    # 	plt.plot(frmTime, hfreq)
    # 	plt.axis([0, x.size/float(fs), 0, maxplotfreq])
    # 	plt.title('frequencies of harmonic tracks')

    plt.tight_layout()
    plt.ion()
    plt.show()
    fig.set_tight_layout(True)


if __name__ == "__main__":
    main()
