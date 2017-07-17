import os
import numpy as np
from pydub import AudioSegment
import pandas as pd

# Logging
import logging
l = logging.getLogger("pydub.converter")
l.setLevel(logging.DEBUG)
l.addHandler(logging.StreamHandler())

# Working Directores
root_dir = os.path.dirname(os.path.abspath(__file__))
mp3_parentdir = os.path.join(root_dir, "mp3")

def mp3_to_array(file):

    # MP3からRAWへの変換
    song = AudioSegment.from_mp3(file)

    # RAWからbytestring型への変換
    song_data = song._data

    # bytestringからNumpy配列への変換
    song_arr = np.fromstring(song_data, np.int16)

    return song_arr


nClass = 25
nBatchSize=50
nDatas=3000
nEpoc=50

tags_df = pd.read_csv('misc/annotations_final.csv', delim_whitespace=True)
tags_df = tags_df.sample(frac=1)
tags_df = tags_df[:nDatas] #3000


top50_tags = tags_df.iloc[:, 1:189].sum().sort_values(ascending=False).index[:nClass].tolist()
y =  tags_df[top50_tags].values


files = tags_df.mp3_path.values
X = np.array([ mp3_to_array(os.path.join(mp3_parentdir, file)) for file in files ])
X = X.reshape(X.shape[0], X.shape[1], 1)


#
from sklearn.model_selection import train_test_split
random_state = 42

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Get model
import keras
from keras.models import Model
from keras.layers import Dense,  Flatten, Input
from keras.layers import Conv1D, MaxPooling1D

features = train_X.shape[1]

x_inputs = Input(shape=(features, 1), name='x_inputs') # (特徴量数, チャネル数)
x = Conv1D(128, 256, strides=256,
           padding='valid', activation='relu') (x_inputs)
x = Conv1D(32, 8, activation='relu') (x) # (チャネル数, フィルタの長さ )
x = MaxPooling1D(4) (x) # （フィルタの長さ）
x = Conv1D(32, 8, activation='relu') (x)
x = MaxPooling1D(4) (x)
x = Conv1D(32, 8, activation='relu') (x)
x = MaxPooling1D(4) (x)
x = Conv1D(32, 8, activation='relu') (x)
x = MaxPooling1D(4) (x)
x = Flatten() (x)
x = Dense(100, activation='relu') (x) #（ユニット数）
x_outputs = Dense(nClass, activation='sigmoid', name='x_outputs') (x)

model = Model(inputs=x_inputs, outputs=x_outputs)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
model.fit(train_X, train_y, batch_size=nBatchSize, epochs=nEpoc)


# post
from keras.utils.visualize_util import plot
plot(model, to_file="music_only.png", show_shapes=True)

# test
from sklearn.metrics import roc_auc_score
pred_y_x1 = model.predict(test_X, batch_size=nBatchSize)
print(roc_auc_score(test_y, pred_y_x1)) # => 0.668582599155
