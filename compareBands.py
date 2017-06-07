import os, sys
#import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def main():
    data_dir = "/Users/alfonso/recordings/recordingsYamaha/phrases/"
    score = "phrase_2_pp"
    inputFile1 = data_dir + "tools_phrases_" + score + "/" + score + "-16bit-EnergyBankFilter.txt"
    inputFile2 = data_dir + "tools_phrases_" + score + "/" + score + "-16bit-EnergyBankFilter2.txt"

    bands1 = np.loadtxt(inputFile1, skiprows=1)
    bands1 = bands1[:,3:43]
    bands2 = np.loadtxt(inputFile2, skiprows=0)

    plt.plot(bands1)
    plt.figure()
    plt.plot(bands2)
    print("The end")
    #lines = text_file.read().split(',')

if __name__ == "__main__":
    main()