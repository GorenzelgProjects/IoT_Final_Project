import urllib.request
import requests
import threading
import json
import os
import random
import numpy as np
from scipy.io.wavfile import write


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


if __name__ == '__main__':

    max_length = 512
    rate = 100

    # Define the file root and extension
    file_folder = 'samples/'
    file_name = 'current_sample.wav'
    output_path = file_folder + file_name

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
