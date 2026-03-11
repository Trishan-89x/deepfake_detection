import torch
import cv2
import numpy as np
import os

from model import FrequencyCNN
from utils.face_detection import detect_face

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FrequencyCNN().to(device)

model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))

model.eval()

def preprocess(img):

    face = detect_face(img)

    if face is None:
        h,w,_ = img.shape
        size = min(h,w)
        start_x = w//2 - size//2
        start_y = h//2 - size//2
        face = img[start_y:start_y+size, start_x:start_x+size]

    face = cv2.resize(face,(256,256))

    gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

    gray = gray/255.0

    tensor = torch.tensor(gray).float().unsqueeze(0).unsqueeze(0)

    return tensor


test_folder = "test_images"

for file in os.listdir(test_folder):

    path = os.path.join(test_folder,file)

    img = cv2.imread(path)

    input_tensor = preprocess(img).to(device)

    with torch.no_grad():

        output = model(input_tensor)

        prob = torch.softmax(output,1)

        pred = torch.argmax(prob,1).item()

        confidence = prob[0][pred].item()

    label = "FAKE" if pred==1 else "REAL"

    print(file,"->",label,"(",round(confidence*100,2),"%)")