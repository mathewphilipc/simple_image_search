from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np

import os
"""
Classify images with pre-trained resnet50 courtesy of Keras
"""
def classify_single_image(image_file_path, top_num = 5):
	model = ResNet50(weights='imagenet')
	input_image = image.load_img(image_file_path, target_size=(224, 224))

	processed_image = image.img_to_array(input_image)
	processed_image = np.expand_dims(processed_image, axis=0)
	processed_image = preprocess_input(processed_image)

	predictions = model.predict(processed_image)
	return decode_predictions(predictions, top=top_num)

"""
Classify all images in a folder (need to speed this up later)
"""
def classify_folder(dir_path, top_num = 5):
	output = {}
	for filename in os.listdir(dir_path):
		print("new file = {}".format(filename))
		if filename.endswith(".jpg"):
			new_output = classify_single_image(dir_path + "/" + filename)
	return output

test_image_path = "test_image_set/img0.jpg"

output = classify_single_image(test_image_path)
print(type(output))
print(len(output))

print(output)

test_dir_path = "test_image_set"
x = classify_folder(test_dir_path)