import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from PIL import Image

def load_model(model_name: str):
    with open("saved_models/" + model_name + "_model_architecture.json", "r") as f:
        model = models.model_from_json(f.read())
    model.load_weights("saved_models/" + model_name + "_model_weights.h5")
    return model
  
  
model = load_model('cnn')
  
def extract_landmarks(y_pred, img_size_x, img_size_y):
    landmarks = []
    for i in range(0, len(y_pred), 2):
        landmark_x, landmark_y = y_pred[i] * img_size_x, y_pred[i+1] * img_size_y
        landmarks.append((landmark_x, landmark_y))
    return landmarks


def save_img_with_landmarks(img, landmarks, plot_name, gray_scale=False):
    if gray_scale:
        plt.imshow(np.squeeze(img), cmap=plt.get_cmap("gray"))
    else:
        plt.imshow(np.squeeze(img))
    for landmark in landmarks:
        plt.plot(landmark[0], landmark[1], "go")
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()
    
def load_model(model_name: str):
    with open("saved_models/" + model_name + "_model_architecture.json", "r") as f:
        model = models.model_from_json(f.read())
    model.load_weights("saved_models/" + model_name + "_model_weights.h5")
    return model

def sunglasses_filter(input_image_name:str, filter_name:str='sunglasses'):
    # Load original image
    # face_img_path = "input/" + input_image_name + ".png"
    orig_img = cv2.imread(input_image_name)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_size_x, orig_size_y = orig_img.shape[0], orig_img.shape[1]

    # Prepare input image
    img = cv2.imread(input_image_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, axis=2)
    img = img.astype("float32") / 255
    #img = img.astype("float32")

    # Predict landmarks
    
   
    y_pred = model.predict(np.expand_dims(img, axis=0))[0]
    landmarks = extract_landmarks(y_pred, orig_size_x, orig_size_y)

    # Save original image with landmarks on top
    save_img_with_landmarks(orig_img, landmarks, "landmarks"+ input_image_name + ".png")

    # Extract x and y values from landmarks of interest
    # eye position
    left_eye_center_x = int(landmarks[0][0])
    left_eye_center_y = int(landmarks[0][1])
    right_eye_center_x = int(landmarks[1][0])
    right_eye_center_y = int(landmarks[1][1])
    left_eye_outer_x = int(landmarks[3][0])
    right_eye_outer_x = int(landmarks[5][0])

    # Load images using PIL
    # PIL has better functions for rotating and pasting compared to cv2
    face_img = Image.open(input_image_name)
    sunglasses_img = Image.open("input/" + filter_name + ".png")

    # Resize sunglasses
    sunglasses_width = int((left_eye_outer_x - right_eye_outer_x) * 1.4)
    sunglasses_height = int(sunglasses_img.size[1] * (sunglasses_width / sunglasses_img.size[0]))
    sunglasses_resized = sunglasses_img.resize((sunglasses_width, sunglasses_height))

    # Rotate sunglasses
    eye_angle_radians = np.arctan((right_eye_center_y - left_eye_center_y) / (left_eye_center_x - right_eye_center_x))
    sunglasses_rotated = sunglasses_resized.rotate(np.degrees(eye_angle_radians), expand=True, resample=Image.BICUBIC)

    # Compute positions such that the center of the sunglasses is
    # positioned at the center point between the eyes
    x_offset = int(sunglasses_width * 0.5)
    y_offset = int(sunglasses_height * 0.5)
    pos_x = int((left_eye_center_x + right_eye_center_x) / 2) - x_offset
    pos_y = int((left_eye_center_y + right_eye_center_y) / 2) - y_offset

    # Paste sunglasses on face image
    face_img.paste(sunglasses_rotated, (pos_x, pos_y), sunglasses_rotated)
    output_path= "output/" + input_image_name +"_"+ filter_name + ".png"
    face_img.save("output/" + input_image_name +"_"+ filter_name + ".png")
    
    # landmarks_img = Image.open("output/landmarks"+ input_image_name + ".png")
    # sunglasses_img = Image.open("output/"+ input_image_name +"_"+ filter_name + ".png")

    return output_path
