Platform: 
Mac/Python 3.5.4 conda

Package dependencies:
scipy
numpy
pylab
librosa
os
math
sounddevice

Execution:
The code can be executed as is. 
It will ask you to enter an audio file name. There are four options in the audio folder. 
I recommend moon.wav as it requires more onset reduction than the others.

Description:
In this project, three of my passions intersect‒music, coding, and math. I attempt
to answer the question: how can you detect pitch from audio using Python?

First, the program asks you to enter a filename (ex. audio/moon.wav) to load some
simple piano audio. It uses librosa to detect onsets, validates these offsets,
then uses the fft to find the most powerful frequency for each note 
(get_frequency_fft). It then uses a formula to determine the note from the 
frequency (note_from_freq). It creates a trimmed audio file, removing data 
outside the valid onsets (trim_audio). This function also plays the trimmed audio. 
To visualize the data, the program plots the amplitude vs time as well as the 
power vs frequency (display_plots). 

