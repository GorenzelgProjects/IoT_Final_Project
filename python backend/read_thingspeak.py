import urllib.request
import requests
import threading
import json
import os
import random
import numpy as np
from scipy.io.wavfile import write
import pandas as pd
import torch
import torchaudio
import csv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch import nn
from CNN_Clap_our_data import CNNNetwork


# Define the UrbanSoundDataset class for data loading and preprocessing
class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir,transformation,target_sample_rate,num_samples,device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        # signal -> (num_channels,samples) -> (2,16000) -> (1,16000)
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label
    
    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1,50000) -> (1,22050)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1,1,1] -> [1,1,1,0,0]
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2,16000)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


    def _get_audio_sample_path(self, index):
        fold = None
        #print(self.annotations.iloc[index, 0])
        if self.annotations.iloc[index, 1] == "clap":
            fold = 1
        elif self.annotations.iloc[index, 1] == "no clap":
            fold = 2
        fold = f"fold{fold}" #fold1 is correct and fold2 is incorrect
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path

    def _get_audio_sample_label(self, index):
        #print(self.annotations.iloc[index, 1])
        return self.annotations.iloc[index, 1]
    
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

# Define a function that will post on server every 15 Seconds

def thingspeak_post():
    threading.Timer(15,thingspeak_post).start()
    val=random.randint(1,30)
    URl='https://api.thingspeak.com/update?api_key='
    KEY='G5WG42VHHEELJ3I4'
    HEADER='&field1={}&field2={}'.format(val,val)
    NEW_URL=URl+KEY+HEADER
    #print(NEW_URL)
    data=urllib.request.urlopen(NEW_URL)
    #print(data)

def read_data_thingspeak():
    #URL='https://api.thingspeak.com/channels/2511688/fields/1.json?api_key='
    URL='https://api.thingspeak.com/channels/2522707/feeds.json?api_key='
    KEY='8KDFZQ0JZEUZTZ2S'
    HEADER='&results=1'
    NEW_URL=URL+KEY+HEADER
    #print(NEW_URL)

    #print(requests.get(NEW_URL))
    get_data=requests.get(NEW_URL).json()
    #print(get_data)
    channel_id=get_data['channel']['id']

    field_1=get_data['feeds']
    #print(field_1)
    t = str(field_1[-1]['field1'])
    t = t[:-1]

    return t

def process_string(input_string, max_length):
    # Split the string by commas
    elements = input_string.split(',')
    
    # List to store the modified elements
    result = []
    
    # Process each element
    for element in elements:
        # Strip any whitespace
        num = element.strip()
        
        # Check if the element can be converted to an integer
        try:
            value = int(num)
            if value < 0:
                # Zero-padding for negative numbers
                padding_length = abs(value)
                for pad in range(padding_length):
                    result.append(0)
            else:
                # Appending both positive and its negative value
                result.append(value)
                result.append((value*(-1)))
        except ValueError:
            # Handle the case where conversion to integer fails
            print(f"Warning: '{num}' is not a valid integer and will be ignored.")
    
    if len(result) < max_length:
        for pad in range(512-len(result)):
            result.append(0)
    # Join all the processed elements with a comma and return
    result = result[:max_length]

    return result

def data_to_wav(data, output_path, rate=100):
    # Convert the data to a numpy array
    data = np.array(processed_output)

    # Scale the data to int16 range
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)

    # Write the data to a WAV file
    write(output_path, rate, scaled)

def test_model(ANNOTATIONS_FILE= './Data_and_ML/Data/iot_self_made/labeled_dataset.csv',AUDIO_DIR = "./Data_and_ML/Data/iot_self_made/audio",num_test=1):
    #class_mapping = {0: "no clap", 1: "clap"}
    
    #class mapping
    class_mapping = ["no clap", "clap"]

    #ANNOTATIONS_FILE = './Data/iot_self_made/labeled_dataset.csv'
    #AUDIO_DIR = "./Data/iot_self_made/audio"
    ANNOTATIONS_FILE = ANNOTATIONS_FILE
    AUDIO_DIR = AUDIO_DIR

    # Define a function to make an inference
    def predict(model, input, target, class_mapping):
        model.eval()
        with torch.no_grad():
            predictions = model(input)
            # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
            predicted_index = predictions[0].argmax(0)
            predicted = class_mapping[predicted_index]
            #target = torch.tensor([1 if x == "clap" else 0 for x in target])
            expected = target
            #print(expected)
            #expected = class_mapping[target]
        return predicted, expected

    # load the model
    cnn = CNNNetwork()
    state_dict = torch.load("./Data_and_ML/feedforwardnet.pth")
    cnn.load_state_dict(state_dict)

    # load urban sound dataset dataset
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024, #frame size
        hop_length=512,
        n_mels=64
    )

    class MelSpectrogram3Channel(nn.Module):
        def __init__(self, n_channels):
            super().__init__()
            self.mel_spectrogram = mel_spectogram
            self.n_channels = n_channels

        def forward(self, input_data):
            mel_spec = self.mel_spectrogram(input_data)
            mel_spec = mel_spec.unsqueeze(1).repeat(1, self.n_channels, 1, 1)
            return mel_spec

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            MelSpectrogram3Channel(3),
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            "cpu")


    # get a sample from the urban sound dataset for inference
    index = 0
    input, target = usd[index][0], usd[index][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")

    # test the model
    num_test = num_test
    test = []
    for i in range(num_test):
        index = i
        input, target = usd[index][0], usd[index][1] # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)

        # make an inference
        predicted, expected = predict(cnn, input, target,
                                    class_mapping)
        if predicted == expected:
            test.append(1)
        print(f"Predicted: '{predicted}', expected: '{expected}'")
    
    #print accuracy
    print(f"Accuracy: {len(test)/num_test}")

    #return True if clap detected
    return True if test else False

def create_csv(filename, directory):
    # Define the name of the CSV file to create
    csv_filename = 'output.csv'

    # Create the full path for the CSV file
    full_path = os.path.join(directory, csv_filename)

    # Define the header of the CSV file
    header = ['file', 'label']

    # Define the data to be written (you can add more entries as needed)
    data = [
        {'file': filename, 'label': 'clap'}
    ]

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")

    # Open the file in write mode and create a CSV writer object
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)

        # Write the header and data to the CSV file
        writer.writeheader()
        writer.writerows(data)
    
    print(f'CSV file created successfully at: {full_path}')

if __name__ == '__main__':
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050
    max_length = 512
    rate = 100

    # Define the file root and extension
    file_folder = 'samples/'
    sound_folder = 'audio/fold1/'
    file_name = 'current_sample.wav'
    output_path = file_folder + sound_folder+ file_name
    output_path1 = file_folder + file_name

    # check if folder exists and create if not
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    # Read the data from ThingSpeak
    #sound_string = read_data_thingspeak()

    sound_string = '-124,122,114,112,-3,112,-26,125,-36,125,-36,120,-35,115,-39,123,-55,118,-37,121,-42,117'

    # Process the data
    processed_output = process_string(sound_string, max_length)
    data_to_wav(data=processed_output, output_path=output_path, rate=rate)
    print(f"Data written to {output_path}")

    # Create a CSV file
    #create_csv(file_name, output_path1)

    # Test the model with the created data
    clap = test_model(ANNOTATIONS_FILE='./samples/labeled_dataset.csv',AUDIO_DIR='./samples/audio',num_test=1)
    if clap:
        print("Clap detected!")
    else:
        print("No clap detected!")
    # Post data to ThingSpeak
    print(clap)