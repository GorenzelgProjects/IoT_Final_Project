import os
import pandas as pd

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch import nn
from torchsummary import summary
from tqdm import tqdm

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device {device}")

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = './Data/iot_rasmus/labels_dataset_random.csv'
AUDIO_DIR = "./Data/iot_rasmus/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

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

# Modify the UrbanSoundDataset class
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

# Modify the CNNNetwork class
class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        input_data = input_data.squeeze(1)
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

# Modify the main code



class UrbanSoundDataset1(Dataset):

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
    
class CNNNetwork1(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 5 * 4, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        target = torch.tensor([1 if x == "clap" else 0 for x in target])
        #print(input.shape)
        #print(type(input))
        #print(target)
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in tqdm(range(epochs)):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

mel_spectogram1 = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024, #frame size
    hop_length=512,
    n_mels=64
)

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f"Using device {device}")

#For single channel use:

"""
if __name__=="__main__":

    usd = UrbanSoundDataset1(ANNOTATIONS_FILE, AUDIO_DIR,mel_spectogram1,SAMPLE_RATE,NUM_SAMPLES,device)

    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    #  construct model and assign it to device
    cnn = CNNNetwork1().to(device)
    #print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")

    
    class_mapping = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music"
    ]

    cnn = CNNNetwork()
    state_dict = torch.load("../trained-cnn-model/cnnnet.pth")
    cnn.load_state_dict(state_dict)




    # get a sample from the urban sound dataset for inference
    input, target = usd[0][0], usd[0][1] # [batch size, num_channels, fr, time]
    input.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")"""

#For RBG channel use:

if __name__=="__main__":

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR,MelSpectrogram3Channel(3),SAMPLE_RATE,NUM_SAMPLES,device)

    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    #  construct model and assign it to device
    cnn = CNNNetwork().to(device)
    #print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")