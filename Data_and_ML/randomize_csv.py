import pandas as pd

# Load the CSV file
df = pd.read_csv('./Data/iot_rasmus/labels_dataset.csv')

# Randomize the rows
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Save the randomized DataFrame to a new CSV file
df_shuffled.to_csv('./Data/iot_rasmus/labels_dataset_random.csv', index=False)