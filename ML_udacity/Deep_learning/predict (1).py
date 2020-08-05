
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image 
import json


# parser arguments
parser= argparse.ArgumentParser()

parser.add_argument('--chosen_image', default='./test_images/cautleya_spicata.jpg')
parser.add_argument('--fitting_model', default='classifier_image_model.h5')
parser.add_argument('--top_related_image', default=5)
parser.add_argument('--category_names', default='label_map.json')


# value of arguments
argument= parser.parse_args()

chosen_image = argument.chosen_image
fitting_model = argument.fitting_model
top_related_image = argument.top_related_image
category_names = argument.category_names

# reading json file
with open('label_map.json', 'r') as names:
    class_names = json.load(names)
    
# load the model   
fitting_model = tf.keras.models.load_model(fitting_model,custom_objects={'KerasLayer':hub.KerasLayer})

# reshape the image to fit the model 
def process_image(image):
    image= tf.convert_to_tensor(image)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    image = image.numpy()
    return image

# TODO: Create the predict function
def predict(image, model,top_k ):
    new_image = Image.open(image)
    test_image = np.asarray(new_image)
    processed_test_image = process_image(test_image)
    edit_image=np.expand_dims(processed_test_image ,axis=0)
    predicted_model= model.predict(edit_image)
    prob_pred = predicted_model.tolist()

# get the topk 
    values, indices= tf.math.top_k(predicted_model, k=top_k)
    
# convert topk values to list of probability
    probability=values.numpy().tolist()[0]
    
# convert topk indices to classes then labels
    classes=indices.numpy().tolist()[0]
    label_names = [class_names[str(id+1)] for id in classes]
    
    print('the probabilites:', probability)
    print('\n the classes:',label_names)

    
    return probability, classes, label_names


if __name__ == '__main__':
    predict(chosen_image,fitting_model,top_related_image)
    

