import torch
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
from PIL import Image
import os

SPECTROGRAM_DPI = 90 # image quality of spectrograms
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOPE_LENGHT = 1024

class audio():
    def __init__(self, filepath_, hop_lenght = DEFAULT_HOPE_LENGHT, samples_rate = DEFAULT_SAMPLE_RATE):
        self.hop_lenght = hop_lenght
        self.samples_rate = samples_rate
        self.waveform, self.sample_rate = torchaudio.load(filepath_)

    def plot_spectrogram(self) -> None:
        waveform = self.waveform.numpy()
        _, axes = plt.subplots(1, 1)
        axes.specgram(waveform[0], Fs=self.sample_rate)
        plt.axis('off')
        #plt.show(block=False)
    
    def write_disk_spectrogram(self, path, dpi=SPECTROGRAM_DPI) -> None:
        self.plot_spectrogram()
        plt.savefig(path, dpi=dpi, bbox_inches='tight')



if __name__ =="__main__":
    #sound_path = "./Data/UrbanSound8k/audio/fold1/7061-6-0-0.wav"
    sound_path = "./Data/test/test_drum.wav"
    #output_path = "./Data/Images/fold1/7061-6-0-0.png"
    output_path = "./Data/test_images/test_drum.png"
    sound = audio(sound_path)
    sound.write_disk_spectrogram(output_path, dpi=SPECTROGRAM_DPI)

    # Get the directory path
    #folder_path = "./Data/UrbanSound8k/audio/fold1/"

    # Iterate over all files in the directory
    #for filename in os.listdir(folder_path):
        # Check if the file is a WAV file
        #if filename.endswith(".wav"):
            # Get the full file path
            #file_path = os.path.join(folder_path, filename)
            
            # Generate the output file path with the same name but different extension
            #output_path = os.path.join("./Data/Images/fold1/", os.path.splitext(filename)[0] + ".png")
            
            # Create an instance of the audio class
            #sound = audio(file_path)
            
            # Write the spectrogram to disk
            #sound.write_disk_spectrogram(output_path, dpi=SPECTROGRAM_DPI)
