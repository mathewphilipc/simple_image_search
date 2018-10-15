from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
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

#test_image_path = "test_image_set/img0.jpg"

#output = classify_single_image(test_image_path)
#print(type(output))
#print(len(output))

#print(output)