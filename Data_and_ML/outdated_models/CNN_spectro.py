
import numpy as np
import torch
from torch import nn
import torch.multiprocessing
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torchvision
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer as timer
from CNN_model import neuralNetworkV1
from helper import *

def infer(sound_path: str) -> int:
    model = torch.load("./model_path", map_location=DEVICE)
    # class we declared earlier to turn audio file into spectrogram
    sound = audio(sound_path)
    sound.write_disk_spectrogram(SPECTROGRAM_SAVE_PATH, dpi=90)
    image = Image.open(SPECTROGRAM_SAVE_PATH).convert('RGB')
    with torch.no_grad():
        image_array = np.array(image)
        image_array = np.transpose(image_array, (2, 0, 1))
        image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        predictions = model(image_tensor)
        top_index = torch.argmax(predictions, dim=1).item()
    return predictions[top_index]


if __name__ == "__main__":

    writer_path = 'runs/log_file_tensorboard'
    writer = SummaryWriter(writer_path)

    NUM_WORKERS = 4 # number of worker used when loading data into dataloader
    DATASET_PATH = '../database/images/' # path of our spectrogram dataset
    IMAGE_SIZE = (1024, 1024) # image size
    CHANNEL_COUNT = 3 # 3 channel as an image has 3 color (R,G,B)
    ATTRIBUTION = ["dog", "bird", "car"] # class labels exemple, we'll have 3 class in this exemple
    ACCURACY_THRESHOLD = 90 # accuracy at which to stop

    MAX_EPOCHS = 100
    LEARNING_RATE = 0.01
    GRADIENT_MOMENTUM = 0.90
    loss_func = torch.nn.CrossEntropyLoss()
    
    # Load the dataset
    print(f"Loading images from dataset at {DATASET_PATH}")
    transform=transforms.ToTensor()
    dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

    #transform=transforms.ToTensor()

    # train / test split
    val_ratio = 0.2
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"{train_size} images for training and {val_size} images for validation")
    batch_size = 16

    # Load training dataset into batches
    train_batches = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)
    # Load validation dataset into batches
    val_batches = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size*2,
                                            num_workers=NUM_WORKERS)

    # display 32 (batch_size*2) sample from the first validation batch
    batches_display(val_batches, writer_path=writer_path)


    our_model = neuralNetworkV1()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device {device}")

    print("Model summary : ")
    print(summary(our_model, (CHANNEL_COUNT, IMAGE_SIZE[0], IMAGE_SIZE[1])))

    optimizer = torch.optim.SGD(selected_model.parameters(), lr=LEARNING_RATE, momentum=GRADIENT_MOMENTUM)

    train_time_start_on_gpu = timer()
    model_accuracy = train_neural_net(MAX_EPOCHS, selected_model, loss_func, optimizer, train_batches, val_batches)
    print(f"Training complete : {model_accuracy} %")
    display_training_time(start=train_time_start_on_gpu,
                    end=timer())

    torch.save(selected_model, "./saving_path")
    writer.flush()
    writer.close()


    SPECTROGRAM_SAVE_PATH = './spectrogram.png'
    DEVICE = torch.device('cpu')

