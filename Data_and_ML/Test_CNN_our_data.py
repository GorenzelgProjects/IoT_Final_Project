import os
import pandas as pd

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch import nn
from CNN_Clap_our_data import *

#class_mapping = {0: "no clap", 1: "clap"}
class_mapping = ["no clap", "clap"]

ANNOTATIONS_FILE = './Data/iot_self_made/labeled_dataset.csv'
AUDIO_DIR = "./Data/iot_self_made/audio"

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


cnn = CNNNetwork()
state_dict = torch.load("./feedforwardnet.pth")
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
index = 10
input, target = usd[index][0], usd[index][1] # [batch size, num_channels, fr, time]
input.unsqueeze_(0)

# make an inference
predicted, expected = predict(cnn, input, target,
                              class_mapping)
print(f"Predicted: '{predicted}', expected: '{expected}'")

num_test = 42
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

print(f"Accuracy: {len(test)/num_test}")