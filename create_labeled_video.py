import cv2     # for capturing videos
import math   # for mathematical operations
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from Classifier import recreate_model, predict_animal
# import matplotlib.pyplot as plt    # for plotting the images
# # %matplotlib inline
# import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
# import numpy as np    # for mathematical operations
# from keras.utils import np_utils
# from skimage.transform import resize   # for resizing images


def convert_to_frames():
	if not cv2.os.path.exists("images"):
		cv2.os.makedirs("images")
	count = 0
	video_file = "national_dog_show.mp4"
	cap = cv2.VideoCapture(video_file)  # capturing the video from the given path
	frame_rate = cap.get(5)  # frame rate
	while cap.isOpened():
		frame_id = cap.get(1)  # current frame number
		ret, frame = cap.read()
		if ret is not True:
			break
		if frame_id % math.floor(frame_rate) == 0:
			filename = "%d.jpg" % count
			count += 1
			cv2.imwrite("images/"+filename, frame)
	cap.release()
	print("Done!")
	return count


def label_frame(model, frame, i):
	full_path = "images/"+frame
	img = Image.open(full_path)
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype("TIMESS.ttf", 100)
	predicted_animal = predict_animal(full_path, model)
	draw.text((150, 150), predicted_animal, (255, 255, 255), font=font)
	img.save('labels/'+str(i)+'.jpg')


def predict_frames():
	dog_frames = cv2.os.listdir("images")
	dog_frames.sort()
	model = recreate_model()
	if not cv2.os.path.exists("labels"):
		cv2.os.makedirs("labels")
	for i, frame in enumerate(dog_frames):
		label_frame(model, frame, i)

def create_video():
	image_folder = 'labels'
	video_name = 'video.avi'

	images = [img for img in cv2.os.listdir(image_folder) if img.endswith(".jpg")]
	frame = cv2.imread(cv2.os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, 0, 1, (width, height))
	images.sort()
	for image in images:
		video.write(cv2.imread(cv2.os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()

if __name__ == '__main__':
	convert_to_frames()