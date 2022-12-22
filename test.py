
import matplotlib.pyplot as plt
import numpy as np
import essentia.standard as ess
import essentia

loader  = ess.MonoLoader(filename='/home/vboxuser/PycharmProjects/pythonProject/venv/index.wav')

audio = loader()

#Calculating MFCC's
'''
Step1: Windowing to reduce spectral leakage before FFT

'''

window = ess.Windowing(size=1024,type='hann')
spectrum = ess.Spectrum(size=1024)
mfcc = ess.MFCC()

mfccs = []
mel_bands = []
mel_bands_log = []
logNorm = ess.UnaryOperator(type='log')

for frame in ess.FrameGenerator(audio,startFromZero=True):
    spec = spectrum(window(frame)) #window the frame and calculate spectrum
    mfcc_band,mfcc_coeffs = mfcc(spec)
    mfccs.append(mfcc_coeffs)
    mel_bands.append(mfcc_band)
    mel_bands_log.append(logNorm(mfcc_band))

mfccs = essentia.array(mfccs).T
melbands = essentia.array(mel_bands).T
melbands_log = essentia.array(mel_bands_log).T


mfccs = essentia.array(mfccs).T
print(mfccs)
melbands = essentia.array(melbands).T
melbands_log = essentia.array(melbands_log).T


plt.imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
plt.title("Mel band spectral energies in frames")
plt.savefig("MelBandpec.png")

plt.imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
plt.title("Log-normalized mel band spectral energies in frames")
plt.savefig("MelBandpeclog.png")

plt.imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
plt.title("MFCCs in frames")
plt.savefig("MFCC.png")



