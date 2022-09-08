import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import tensorflow.keras.backend as kb
import re

train_frames = []
test_frames = []
train_landmarks = []
test_landmarks = []
train_emotions = []
test_emotions = []

emotion_dict = {"a": 0, "d": 1, "f": 2, "h": 3, "n": 4, "sa": 5, "su": 6}


def sorted_nicely(strings):
    "Sort strings the way humans are said to expect."
    return sorted(strings, key=natural_sort_key)


def natural_sort_key(key):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', key)]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


audio_frames_path = "res/audio-frames/"
landmarks_path = "dataset/landmarks65/"
audio_frames_path_list = [
    os.path.join(audio_frames_path, path)
    for path in get_immediate_subdirectories(audio_frames_path)
]
landmarks_path_list = [
    os.path.join(landmarks_path, path)
    for path in get_immediate_subdirectories(landmarks_path)
]

test_set_names = [
    "a14", "a15",
    "d14", "d15",
    "f14", "f15",
    "h14", "h15",
    "n27", "n28",
    "n29", "n30",
    "sa14", "sa15",
    "su14", "su15"
]

train_audio_set = []
test_audio_set = []
for x in audio_frames_path_list:
    if x.split("/")[-1].lower() not in test_set_names:
        train_audio_set.append(x)
    else:
        test_audio_set.append(x)

train_landmarks_set = []
test_landmarks_set = []
for x in landmarks_path_list:
    if x.split("/")[-1].lower() not in test_set_names:
        train_landmarks_set.append(x)
    else:
        test_landmarks_set.append(x)

# print(train_landmarks_set, "\n")
# print(test_landmarks_set, "\n")

print("Train Set")
for i in range(len(train_audio_set)):
    types = '*.txt'
    current_audio_frames = train_audio_set[i]
    current_landmark_frames = train_landmarks_set[i]
    current_emotion = re.sub('[0-9]', "", train_audio_set[i].split("/")[-1])
    current_emotion = emotion_dict[current_emotion]

    audio_frames_list = []
    for files in types:
        audio_frames_list.extend(
            glob.glob(os.path.join(current_audio_frames, files)))

    audio_frames_list = sorted_nicely(audio_frames_list)

    landmarks_list = []
    for files in types:
        landmarks_list.extend(
            glob.glob(os.path.join(current_landmark_frames, files)))

    landmarks_list = sorted_nicely(landmarks_list)

    l_f = len(train_frames)
    for _, text in enumerate(audio_frames_list):
        if (text[-1] == "."):
            continue
        file = open(text)
        lines = file.readlines()
        frame = []
        for line in lines:
            x = [float(y) for y in line.split(" ")]
            frame.append(x)

        train_frames.append(frame)
        train_emotions.append(current_emotion)

    l_l = len(train_landmarks)
    for _, landmarks_p in enumerate(landmarks_list):
        if _ % 2 == 0:
            if landmarks_p[-1] == ".":
                continue
            file = open(landmarks_p)
            frame = []

            lines = file.readlines()
            for line in lines:
                frame.append(float(line))

            train_landmarks.append(frame)

    if len(train_frames) > len(train_landmarks):
        train_frames = train_frames[:len(train_landmarks)]
    elif len(train_frames) < len(train_landmarks):
        train_landmarks = train_landmarks[:len(train_frames)]

    if len(train_emotions) > len(train_landmarks):
        train_emotions = train_emotions[:len(train_landmarks)]

    print(current_audio_frames, len(train_frames) - l_f)
    print(current_landmark_frames, len(train_landmarks) - l_l)
    print("No of emotions :", len(train_emotions) - l_l)
    print("")

print("Test Set")
for i in range(len(test_audio_set)):
    types = '*.txt'
    current_audio_frames = test_audio_set[i]
    current_landmark_frames = test_landmarks_set[i]
    current_emotion = re.sub('[0-9]', "", test_audio_set[i].split("/")[-1])
    current_emotion = emotion_dict[current_emotion]

    audio_frames_list = []
    for files in types:
        audio_frames_list.extend(
            glob.glob(os.path.join(current_audio_frames, files)))

    audio_frames_list = sorted_nicely(audio_frames_list)

    landmarks_list = []
    for files in types:
        landmarks_list.extend(
            glob.glob(os.path.join(current_landmark_frames, files)))

    landmarks_list = sorted_nicely(landmarks_list)

    l_f = len(test_frames)
    for _, text in enumerate(audio_frames_list):
        if text[-1] == ".":
            continue
        file = open(text)
        lines = file.readlines()
        frame = []
        for line in lines:
            x = [float(y) for y in line.split(" ")]
            frame.append(x)

        test_frames.append(frame)
        test_emotions.append(current_emotion)

    l_l = len(test_landmarks)
    for _, landmarks_p in enumerate(landmarks_list):
        if _ % 2 == 0:
            if landmarks_p[-1] == ".":
                continue
            file = open(landmarks_p)
            frame = []

            lines = file.readlines()
            for line in lines:
                frame.append(float(line))

            test_landmarks.append(frame)

    if len(test_frames) > len(test_landmarks):
        test_frames = test_frames[:len(test_landmarks)]
    elif len(test_frames) < len(test_landmarks):
        test_landmarks = test_landmarks[:len(test_frames)]

    if len(test_emotions) > len(test_landmarks):
        test_emotions = test_emotions[:len(test_landmarks)]

    print(current_audio_frames, len(test_frames) - l_f)
    print(current_landmark_frames, len(test_landmarks) - l_l)
    print("No of emotions :", len(test_emotions) - l_l)
    print("")

print("")
print("No of train frames :", len(train_frames))
print("No of test frames :", len(test_frames))
print("No of train landmarks :", len(train_landmarks))
print("No of test landmarks :", len(test_landmarks))
print("No of train emotions :", len(train_emotions))
print("No of test emotions", len(test_emotions))
print("")

train_frames = np.asarray(train_frames)
test_frames = np.asarray(test_frames)
train_landmarks - np.asarray(train_landmarks)
test_landmarks = np.asarray(test_landmarks)
train_emotions = np.reshape(train_emotions, -1)
train_emotions = np.eye(7)[train_emotions]
x = train_emotions
train_emotions = []
for a in x:
    y = []
    for _ in range(64):
        y.append(a)
    train_emotions.append(np.asarray(y))

train_emotions = np.asarray(train_emotions)
train_emotions = np.transpose(train_emotions, axes=(0, 2, 1))
train_emotions = np.expand_dims(train_emotions, 3)
train_emotions_32 = train_emotions[:, :, :32]
train_emotions_16 = train_emotions[:, :, :16]
train_emotions_8 = train_emotions[:, :, :8]
train_emotions_4 = train_emotions[:, :, :4]
print(train_emotions_32.shape)


test_emotions = np.reshape(test_emotions, -1)
test_emotions = np.eye(7)[test_emotions]
x = test_emotions
test_emotions = []
for a in x:
    y = []
    for _ in range(64):
        y.append(a)
    test_emotions.append(np.asarray(y))
test_emotions = np.asarray(test_emotions)
test_emotions = np.transpose(test_emotions, axes=(0, 2, 1))
test_emotions = np.expand_dims(test_emotions, 3)
test_emotions_32 = test_emotions[:, :, :32]
test_emotions_16 = test_emotions[:, :, :16]
test_emotions_8 = test_emotions[:, :, :8]
test_emotions_4 = test_emotions[:, :, :4]
print(test_emotions_32.shape)


print(np.asarray(train_frames).shape, np.asarray(test_frames).shape, np.asarray(train_landmarks).shape,
      np.asarray(test_landmarks).shape)

train_frames = np.expand_dims(train_frames, axis=1)
test_frames = np.expand_dims(test_frames, axis=1)
print(train_frames.shape, test_frames.shape, train_emotions.shape)


def custom_loss_function(y_actual, y_predicted):
    pos_loss = kb.mean(kb.sum(kb.square(y_actual - y_predicted), axis=1)) / 2

    y_act_roll = tf.roll(y_actual, -1, 0)
    y_pred_roll = tf.roll(y_predicted, -1, 0)

    y_act_diff = y_actual[slice(None, -1, 2)] - y_act_roll[slice(None, -1, 2)]
    y_pred_diff = y_predicted[slice(
        None, -1, 2)] - y_pred_roll[slice(None, -1, 2)]

    motion_loss = kb.mean(kb.sum(kb.square(y_act_diff - y_pred_diff), axis=1))

    return pos_loss + motion_loss


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

models = tf.keras.models
layers = tf.keras.layers

frames = tf.keras.Input(shape=(1, 64, 32))
emotions_64 = tf.keras.Input(shape=(7, 64, 1))
emotions_32 = tf.keras.Input(shape=(7, 32, 1))
emotions_16 = tf.keras.Input(shape=(7, 16, 1))
emotions_8 = tf.keras.Input(shape=(7, 8, 1))
emotions_4 = tf.keras.Input(shape=(7, 4, 1))

formants = layers.Conv2D(input_shape=(1, 64, 32), filters=72, kernel_size=[1, 3], strides=[1, 2],
                         padding="SAME",
                         kernel_initializer='RandomUniform', data_format="channels_first")(frames)
formants = layers.BatchNormalization(epsilon=1e-5)(formants)
formants = layers.Activation('relu')(formants)

formants = layers.Conv2D(filters=108, kernel_size=[1, 3], strides=[1, 2],
                         padding="SAME",
                         kernel_initializer='RandomUniform', data_format="channels_first")(formants)
formants = layers.BatchNormalization(epsilon=1e-5)(formants)
formants = layers.Activation('relu')(formants)

formants = layers.Conv2D(filters=162, kernel_size=[1, 3], strides=[1, 2],
                         padding="SAME",
                         kernel_initializer='RandomUniform', data_format="channels_first")(formants)
formants = layers.BatchNormalization(epsilon=1e-5)(formants)
formants = layers.Activation('relu')(formants)

formants = layers.Conv2D(filters=243, kernel_size=[1, 3], strides=[1, 2],
                         padding="SAME",
                         kernel_initializer='RandomUniform', data_format="channels_first")(formants)
formants = layers.BatchNormalization(epsilon=1e-5)(formants)
formants = layers.Activation('relu')(formants)

formants = layers.Conv2D(filters=256, kernel_size=[1, 3], strides=[1, 2],
                         padding="SAME",
                         kernel_initializer='RandomUniform', data_format="channels_first")(formants)
formants = layers.BatchNormalization(epsilon=1e-5)(formants)
formants = layers.Activation('relu')(formants)

for_emotion_concat = tf.keras.layers.Concatenate(
    axis=1)(inputs=[formants, emotions_64])

landmarks = layers.Conv2D(filters=256 + 7, kernel_size=[3, 1], strides=[2, 1],
                          padding="SAME",
                          kernel_initializer='RandomUniform', data_format="channels_first")(for_emotion_concat)

landmarks = layers.BatchNormalization(epsilon=1e-5)(landmarks)
landmarks = layers.Activation('relu')(landmarks)

for_emotion_concat = tf.keras.layers.Concatenate(
    axis=1)(inputs=[landmarks, emotions_32])

landmarks = layers.Conv2D(filters=256 + 7, kernel_size=[3, 1], strides=[2, 1],
                          padding="SAME",
                          kernel_initializer='RandomUniform', data_format="channels_first")(for_emotion_concat)
landmarks = layers.BatchNormalization(epsilon=1e-5)(landmarks)
landmarks = layers.Activation('relu')(landmarks)

for_emotion_concat = tf.keras.layers.Concatenate(
    axis=1)(inputs=[landmarks, emotions_16])
landmarks = layers.Conv2D(filters=256 + 7, kernel_size=[3, 1], strides=[2, 1],
                          padding="SAME",
                          kernel_initializer='RandomUniform', data_format="channels_first")(for_emotion_concat)
landmarks = layers.BatchNormalization(epsilon=1e-5)(landmarks)
landmarks = layers.Activation('relu')(landmarks)

for_emotion_concat = tf.keras.layers.Concatenate(
    axis=1)(inputs=[landmarks, emotions_8])
landmarks = layers.Conv2D(filters=256 + 7, kernel_size=[3, 1], strides=[2, 1],
                          padding="SAME",
                          kernel_initializer='RandomUniform', data_format="channels_first")(for_emotion_concat)
landmarks = layers.BatchNormalization(epsilon=1e-5)(landmarks)
landmarks = layers.Activation('relu')(landmarks)

for_emotion_concat = tf.keras.layers.Concatenate(
    axis=1)(inputs=[landmarks, emotions_4])
landmarks = layers.Conv2D(filters=256 + 7, kernel_size=[4, 1], strides=[4, 1],
                          padding="SAME",
                          kernel_initializer='RandomUniform', data_format="channels_first")(for_emotion_concat)
landmarks = layers.BatchNormalization(epsilon=1e-5)(landmarks)
landmarks = layers.Activation('relu')(landmarks)

landmarks = layers.Flatten()(landmarks)
landmarks = layers.Dense(65, activation='linear')(landmarks)
outputs = layers.Dense(130, activation='linear')(landmarks)

model = tf.keras.Model(inputs=[frames, emotions_64, emotions_32,
                               emotions_16, emotions_8, emotions_4], outputs=outputs)
model.compile(optimizer='adam', loss=custom_loss_function,
              metrics=['accuracy'])

# model.summary()

checkpoint_path = "training_emotion_all/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest is None:
    model.save_weights(checkpoint_path.format(epoch=0))
else:
    model.load_weights(latest)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, period=1)

history = model.fit([np.asarray(train_frames), train_emotions, train_emotions_32, train_emotions_16, train_emotions_8,
                     train_emotions_4], np.asarray(train_landmarks), initial_epoch=20,
                    epochs=40, validation_data=(
    [np.asarray(test_frames), test_emotions, test_emotions_32,
     test_emotions_16, test_emotions_8, test_emotions_4],
    np.asarray(test_landmarks)),
    shuffle=False, callbacks=[cp_callback])

model.summary()

# initial_epoch=0,
test_frames = np.asarray(test_frames)
# print(test_frames.shape)
test_landmarks = np.asarray(test_landmarks)
kpt = model.predict([test_frames, test_emotions, test_emotions_32,
                     test_emotions_16, test_emotions_8, test_emotions_4])
kpt_xy = []
# print(kpt.shape)
cnt = 0
for a in kpt:
    np.savetxt(os.path.join("res/pred_emotion_all_40",
                            "frame{}.txt".format(str(cnt))), np.asarray(a))
    cnt += 1

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
# plt.savefig("/res/eval_1.jpg")

test_loss, test_acc = model.evaluate([test_frames, test_emotions, test_emotions_32,
                                      test_emotions_16, test_emotions_8, test_emotions_4], test_landmarks, verbose=2)
print(test_loss, test_acc)


#
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            (None, 1, 64, 32)    0
# __________________________________________________________________________________________________
# conv2d (Conv2D)                 (None, 72, 64, 16)   288         input_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1 (BatchNo (None, 72, 64, 16)   64          conv2d[0][0]
# __________________________________________________________________________________________________
# activation (Activation)         (None, 72, 64, 16)   0           batch_normalization_v1[0][0]
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 108, 64, 8)   23436       activation[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_1 (Batch (None, 108, 64, 8)   32          conv2d_1[0][0]
# __________________________________________________________________________________________________
# activation_1 (Activation)       (None, 108, 64, 8)   0           batch_normalization_v1_1[0][0]
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 162, 64, 4)   52650       activation_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_2 (Batch (None, 162, 64, 4)   16          conv2d_2[0][0]
# __________________________________________________________________________________________________
# activation_2 (Activation)       (None, 162, 64, 4)   0           batch_normalization_v1_2[0][0]
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 243, 64, 2)   118341      activation_2[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_3 (Batch (None, 243, 64, 2)   8           conv2d_3[0][0]
# __________________________________________________________________________________________________
# activation_3 (Activation)       (None, 243, 64, 2)   0           batch_normalization_v1_3[0][0]
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 256, 64, 1)   186880      activation_3[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_4 (Batch (None, 256, 64, 1)   4           conv2d_4[0][0]
# __________________________________________________________________________________________________
# activation_4 (Activation)       (None, 256, 64, 1)   0           batch_normalization_v1_4[0][0]
# __________________________________________________________________________________________________
# input_2 (InputLayer)            (None, 7, 64, 1)     0
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 263, 64, 1)   0           activation_4[0][0]
#                                                                  input_2[0][0]
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 263, 32, 1)   207770      concatenate[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_5 (Batch (None, 263, 32, 1)   4           conv2d_5[0][0]
# __________________________________________________________________________________________________
# activation_5 (Activation)       (None, 263, 32, 1)   0           batch_normalization_v1_5[0][0]
# __________________________________________________________________________________________________
# input_3 (InputLayer)            (None, 7, 32, 1)     0
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 270, 32, 1)   0           activation_5[0][0]
#                                                                  input_3[0][0]
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 263, 16, 1)   213293      concatenate_1[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_6 (Batch (None, 263, 16, 1)   4           conv2d_6[0][0]
# __________________________________________________________________________________________________
# activation_6 (Activation)       (None, 263, 16, 1)   0           batch_normalization_v1_6[0][0]
# __________________________________________________________________________________________________
# input_4 (InputLayer)            (None, 7, 16, 1)     0
# __________________________________________________________________________________________________
# concatenate_2 (Concatenate)     (None, 270, 16, 1)   0           activation_6[0][0]
#                                                                  input_4[0][0]
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 263, 8, 1)    213293      concatenate_2[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_7 (Batch (None, 263, 8, 1)    4           conv2d_7[0][0]
# __________________________________________________________________________________________________
# activation_7 (Activation)       (None, 263, 8, 1)    0           batch_normalization_v1_7[0][0]
# __________________________________________________________________________________________________
# input_5 (InputLayer)            (None, 7, 8, 1)      0
# __________________________________________________________________________________________________
# concatenate_3 (Concatenate)     (None, 270, 8, 1)    0           activation_7[0][0]
#                                                                  input_5[0][0]
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 263, 4, 1)    213293      concatenate_3[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_8 (Batch (None, 263, 4, 1)    4           conv2d_8[0][0]
# __________________________________________________________________________________________________
# activation_8 (Activation)       (None, 263, 4, 1)    0           batch_normalization_v1_8[0][0]
# __________________________________________________________________________________________________
# input_6 (InputLayer)            (None, 7, 4, 1)      0
# __________________________________________________________________________________________________
# concatenate_4 (Concatenate)     (None, 270, 4, 1)    0           activation_8[0][0]
#                                                                  input_6[0][0]
# __________________________________________________________________________________________________
# conv2d_9 (Conv2D)               (None, 263, 1, 1)    284303      concatenate_4[0][0]
# __________________________________________________________________________________________________
# batch_normalization_v1_9 (Batch (None, 263, 1, 1)    4           conv2d_9[0][0]
# __________________________________________________________________________________________________
# activation_9 (Activation)       (None, 263, 1, 1)    0           batch_normalization_v1_9[0][0]
# __________________________________________________________________________________________________
# flatten (Flatten)               (None, 263)          0           activation_9[0][0]
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 65)           17160       flatten[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 130)          8580        dense[0][0]
# ==================================================================================================
# Total params: 1,539,431
# Trainable params: 1,539,359
# Non-trainable params: 72
# __________________________________________________________________________________________________
#
