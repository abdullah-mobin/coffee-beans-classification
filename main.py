from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from PIL import Image
import sys
import os
from dotenv import load_dotenv

env = load_dotenv()

model = os.getenv('MODEL_SAVE_COFFEENET')
model = tf.keras.models.load_model(model)

# Define the class labels
class_labels = ['espresso', 'french', 'green', 'light']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0  
    return img_array

def plot_images(image_path, class_name, confidence):
    img = Image.open(image_path)
    plt.figure(figsize=(6, 4))  
    plt.imshow(img)  
    plt.title(f'Name: {class_name}\n Accuracy: {confidence:.4f}')
    plt.show() 

def predict_class(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])  
    return class_labels[predicted_class], predictions[0][predicted_class] 

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    
    if not os.path.exists(img_path):
        print(f"Error: The file {img_path} does not exist.")
        sys.exit(1)

    predicted_class, confidence = predict_class(img_path)
    plot_images(img_path,predicted_class,confidence)
    print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")


"""
python main.py <path_to_image>
python3 main.py <path_to_image>

"""