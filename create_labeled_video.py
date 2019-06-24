import cv2     # for capturing videos
import math   # for mathematical operations
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
# import matplotlib.pyplot as plt    # for plotting the images
# # %matplotlib inline
# import pandas as pd
# from keras.preprocessing import image   # for preprocessing the images
# import numpy as np    # for mathematical operations
# from keras.utils import np_utils
# from skimage.transform import resize   # for resizing images


def convert_to_frames():
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
			filename = "frame%d.jpg" % count
			count += 1
			cv2.imwrite(filename, frame)
	cap.release()
	print("Done!")
	return count

convert_to_frames()
def label_frame(frame, i):
	img = Image.open(frame)
	draw = ImageDraw.Draw(img)
	# font = ImageFont.truetype(<font-file>, <font-size>)
	font = ImageFont.truetype("TIMESS.ttf", 100)
	# draw.text((x, y),"Sample Text",(r,g,b))
	draw.text((150, 150), "Labrador", (255, 255, 255), font=font)
	img.save('labels/labeled_frame'+str(i)+'.jpg')


def predict_frames(num_frames):
	for i in range(num_frames):
		label_frame("frame"+str(i)+'.jpg', i)

def create_video():
	image_folder = 'labels'
	video_name = 'video.avi'

	images = [img for img in cv2.os.listdir(image_folder) if img.endswith(".jpg")]
	frame = cv2.imread(cv2.os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape

	video = cv2.VideoWriter(video_name, 0, 1, (width, height))

	for image in images:
		video.write(cv2.imread(cv2.os.path.join(image_folder, image)))

	cv2.destroyAllWindows()
	video.release()
