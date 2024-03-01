import pandas as pd
import numpy as np

import os

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array, to_categorical

import cv2

train = pd.DataFrame(columns=["label", "path"])
test = pd.DataFrame(columns=["label", "path"])

for dirname, _, filenames in os.walk('/kaggle/input/grape-disease-dataset-original/Original Data'):
    for filename in filenames:

        paths = dirname.split('/')
        batch = paths[-2]
        label = paths[-1]
        file_path = os.path.join(dirname, filename)

        dic = {'label': label, 'path': file_path}
        if batch == 'test':

            test.loc[len(test)] = dic
        elif batch == 'train':

            train.loc[len(train)] = dic

print(train.shape)
print(test.shape)
display(train.head())
fig, ax = plt.subplots(1,2,figsize=(12,4))

sns.countplot(data=train, x='label', ax=ax[0])
sns.countplot(data=test, x='label', ax=ax[1]);

X_train = []

for i, path in enumerate(train['path']):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    X_train.append(image)

    if i % 500 == 0:
        print(f"[info] processing image number {i}")

print('[info] preprocessing over...')
X_train = np.array(X_train)
print('[info] normalizing array..')
X_train /= 255
y_train = train['label']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

print('[info] train splits ready')
print(X_train.shape)
print(y_train.shape)
disease_list = train.label.unique()

disease_dic = {disease:i for i, disease in enumerate(disease_list)}
decoder = {value:key for key, value in disease_dic.items()}

y_train = np.array([disease_dic[y] for y in y_train])
y_val = np.array([disease_dic[y] for y in y_val])
#y_test = np.array([disease_dic[y] for y in y_test])

y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)
#y_test = to_categorical(y_test, num_classes=4)

bs = 32
lr = 0.0001
epoch = 10
decay = lr / epoch