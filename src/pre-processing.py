import os
from glob import glob

import cv2
import dlib
import matplotlib.pyplot as plt
import moviepy.editor as mp
import numpy as np
from imutils import face_utils
from scipy.fftpack import fft
from scipy.io import wavfile as wav

dataset_path = "dataset"
extracted_audio_save_path = "res/audio"
preprocessed_audio_frames_path = "res/audio-frames"
preprocessed_facial_landmarks_path = "res/landmarks"

frame_centre = [360, 360]

if not os.path.exists(preprocessed_audio_frames_path):
    os.mkdir(preprocessed_audio_frames_path)

if not os.path.exists(extracted_audio_save_path):
    os.mkdir(extracted_audio_save_path)

if not os.path.exists(preprocessed_facial_landmarks_path):
    os.mkdir(preprocessed_facial_landmarks_path)

supported_video_formats = ("*.mp4")
video_paths = []

for file_path in supported_video_formats:
    video_paths.extend(glob(os.path.join(dataset_path, file_path)))

num_videos = len(video_paths)

face_detector = dlib.get_frontal_face_detector()
facial_landmarks_predictor = dlib.shape_predictor(
    "models/shape_predictor/shape_predictor_68_face_landmarks.dat")

for idx, file_path in enumerate(video_paths):
    if str(file_path) == "{}\.".format(dataset_path):
        continue

    print("Generating video frames and landmarks for {}".format(file_path))
    stripped_file_name = file_path[(len(dataset_path)+1):-4]

    if not os.path.exists(preprocessed_facial_landmarks_path + "/{}".format(stripped_file_name)):
        os.mkdir(preprocessed_facial_landmarks_path +
                 "/{}".format(stripped_file_name))
    video_capturer = cv2.VideoCapture(file_path)

    success, frame = video_capturer.read()
    frame_counter = 0
    while success:
        # cv2.imwrite(video_frames_path + "/{}/frame%d.jpg".format(file_name) % count, image)  # save frame as JPEG file
        # image = imutils.resize(image, width=400)
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        detected_face_bounding_boxes = face_detector(grayscaled_frame, 0)

        for face_bounding_box in detected_face_bounding_boxes:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            facial_landmarks_coordinates = face_utils.shape_to_np(
                facial_landmarks_predictor(grayscaled_frame, face_bounding_box))

            # loop over the (x, y)-coordinates for the facial landmarks
            # and save them
            kpt = []
            for [x, y] in facial_landmarks_coordinates:
                kpt.append([x, y])

            # Transformations to keep pt 30 at centre of frame
            # Find x shift and y shift from centre and translate all points accordingly
            pt_30_x = kpt[29][0]
            pt_30_y = kpt[29][1]
            transform_x = frame_centre[0] - pt_30_x
            transform_y = frame_centre[1] - pt_30_y

            kpt_transformed = []
            for [x, y] in kpt:
                x += transform_x
                y += transform_y
                kpt_transformed.append([x, y])

            np.savetxt(os.path.join(preprocessed_facial_landmarks_path + "/{}".format(stripped_file_name),
                                    "frame{}.txt".format(str(frame_counter))), kpt_transformed)

        success, frame = video_capturer.read()
        frame_counter += 1

    print("Extracting audio clips from {}".format(file_path))
    video_clip = mp.VideoFileClip(file_path)
    video_clip.audio.write_audiofile(
        extracted_audio_save_path + "/{}.wav".format(stripped_file_name))
    video_clip.reader.close()
    video_clip.audio.reader.close_proc()

    print("Generating audio frames for {}".format(file_path))

    if not os.path.exists(preprocessed_audio_frames_path + "/{}".format(stripped_file_name)):
        os.mkdir(preprocessed_audio_frames_path + "/{}".format(stripped_file_name))

    sampling_rate, data_points = wav.read(extracted_audio_save_path + "/{}.wav".format(stripped_file_name))
    fft_out = fft(data_points)
    ff = np.array(np.real(fft_out))
    ff = list(ff)
    frame_points_array = []
    step = int(44100 / 60)
    ub = int(len(ff)/step)-1
    print(len(ff), ub, step)
    ub = ub*step + 1
    for x in range(0, ub, step):
        frame_points_array.append(ff[x: x+step])

    print("Total FFT points : {}".format(len(ff)))
    # frame_points_array = np.array_split(ff,count)
    print("Frame points : {}".format(len(frame_points_array[0])))

    frame_counter = 0
    for audio_frame in frame_points_array:
        plt.ylim(-55000, 55000)
        plt.plot(audio_frame)
        plt.savefig(preprocessed_audio_frames_path +
                    "/{}/frame{}.jpg".format(stripped_file_name, frame_counter))
        frame = cv2.imread(preprocessed_audio_frames_path +
                           "/{}/frame{}.jpg".format(stripped_file_name, frame_counter))
        # cv2.imshow(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("LOL", image)
        cv2.imwrite(preprocessed_audio_frames_path +
                    "/{}/frame{}.jpg".format(stripped_file_name, frame_counter), frame)
        plt.clf()
        frame_counter += 1

    print("Done!\n")
