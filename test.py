import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('./better_brain_tumor_classifier.h5')
class_labels = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Loading and preprocessing an image
img_path = './Testing/glioma/Te-gl_0197.jpg' # Any image I just used a random one from the testing directory.
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalizing

# Predicting
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
predicted_label = class_labels[predicted_index]

print("Predicted tumor type:", predicted_label)

