"""
Python for Data Analysis and Scientific Computing 
Annie Robison

In this project, three of my passions intersect‒music, coding, and math. I will 
attempt to answer the question: how can you detect pitch from audio using Python?
I plan to load simple piano audio into Python. I will then take a FFT of the data,
and estimate the frequency using the peak of the FFT. I will then use this 
frequency to determine the note. To visualize the audio data, I will also use 
matplotlib. Depending on how successful the FFT approach proves to be, I may 
also explore other methods for detecting the pitch, such as autocorrelation, 
average magnitude difference, and average squared difference.
"""

import scipy as sp
import scipy.signal
from scipy.io import wavfile
import numpy as np
import pylab as plt
import librosa
import os
import math
import sounddevice as sd


def get_frequency_fft(data,rate):
    """Find the post powerful frequency from a np array of audio data
    Params: np array of audio data
            sample rate"""
    #normalize to [-1,-1]
    d = data / (2**15)
    #take the fft
    f = sp.fft(d)
    
    #only need half since fft is symmetric
    N = len(d)
    unique_points = int(np.ceil((N + 1) / 2.0))
    f = f[0:unique_points]

    #create an array of frequencies
    freq_array = np.arange(0, unique_points, 1.0) * (rate / N)

    #find the frequency with the max power in dB
    maxi = np.argmax(10 * np.log10(f))
    return freq_array[maxi]


def note_from_freq(freq):
    """Return a note from its frequency"""
    #using A4 / 440 as base
    #formula without modulo yields half steps from A4
    #modulo provides easy mapping to note  
    steps = round(12 * np.log2(freq/440)) % 12
    notes = notes = {0:'A',1:'A#',2:'B',3:'C',4:'C#',5:'D',6:'D#',7:'E',8:'F',
                     9:'F#',10:'G',11:'G#'}
    return notes[steps]

    
def trim_audio(filename,data,onsets,rate):
    """Write trimmed audio file based on onsets detected
    Params: filename for trimmed audio, 
            np array of audio data
            np array of onset times
            sample rate"""
    name, ext = os.path.splitext(filename)
    sd.play(data[int(onsets[0] * rate):int(onsets[-1] * rate)],samplerate=rate)
    wavfile.write(name + '_trimmed' + ext,rate,
                  data[int(onsets[0] * rate):int(onsets[-1] * rate)])    


def display_plots(data,rate,raw_onsets,onsets,frequencies,filename):
    """Plot the time/pressure signal with vertical lines at the onsets;
    Plot the power/frequency using the fft
    Params: np array of audio data
            sample rate
            np array of raw onset times
            np array of accepted onset times
            list of frequencies
            name of audio file (for title)"""    
    #get an array of times
    time_array = np.arange(0, len(data), 1)
    time_array = time_array / rate
    time_array = time_array * 1000  #scale to milliseconds
    
    fig = plt.figure()
    fig.canvas.set_window_title(os.path.basename(filename) + ' Plots') 
    
    #plot the time domain data
    plt.subplot(2,1,1)
    plt.title('Amplitude vs time')
    #plot the time domain data
    plt.plot(time_array,data,color='black')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.hold(True)
    #plot each of the onsets (removed as --)
    for i in raw_onsets:
        if i in onsets:
            plt.axvline(x=1000*i,color='magenta')
            plt.hold(True)
        else:
            plt.axvline(x=1000*i,linestyle='--',color='magenta')
            plt.hold(True)

    #plot the power vs frequency
    
    #same next steps as above in get_frequency_fft())
    d = data / (2**15)
    f = sp.fft(d)
    N = len(d)
    unique_points = int(np.ceil((N + 1) / 2.0))
    f = f[0:unique_points]
    freqArray = np.arange(0, unique_points, 1.0) * (rate / N)        
    
    #get magnitude
    f = abs(f)
    #divide f by N so it does not depend on the length
    f = f / float(N)
    #square to get power
    f = f**2
    
    #multiply by two to account for dropping half the data
    #except Nyquist if it exists
    if N % 2 == 0:
        f[1:unique_points-1] = f[1:unique_points - 1] * 2
    else:
        f[1:unique_points] = f[1:unique_points] * 2
        
    plt.subplot(2,1,2)
    plt.title('Power vs frequency')
    plt.plot(freqArray, 10*np.log10(f), color='black')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.xlim(0,math.ceil(max(frequencies) / 500.0) * 500.0)
    
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    plt.show()


filename = input('File name (with path): ')
#filename = 'audio/moon.wav'

#read audio data
rate,data = wavfile.read(filename)
y, sr = librosa.load(filename, sr=rate)

#detect the onsets
onset_frames = librosa.onset.onset_detect(y=y, sr=rate)
times = librosa.frames_to_time(onset_frames, sr=sr)

#drop second channel
data = data[:,0]

#threshold for the validation
threshold = 0.10
edited_times = []
frequencies = []
notes = []
#main loop
for i in range(1,len(times)):
    #get subset of data corresponding to the current offsets
    new_data = data[int(times[i-1] * rate):int(times[i] * rate)]
    #validate the offset
    #if the mean amptlitude for the chunk is less than (threshold * overall mean)
    #we skip this subset
    if np.abs(new_data).mean() < threshold * np.abs(data).mean():
        continue
    #initialize list
    if len(edited_times) == 0:
        edited_times.append(times[i-1])
    edited_times.append(times[i])
    freq = get_frequency_fft(new_data,rate)
    note = note_from_freq(freq)
    frequencies.append(freq)
    notes.append(note)

print(' '.join(notes))
trim_audio(filename,data,edited_times,rate)
display_plots(data,rate,times,edited_times,frequencies,filename)
