import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from .face_detection import detect_face
from .fft_transform import fft_image

class DeepfakeDataset(Dataset):

    def __init__(self, root_dir):

        self.images = []
        self.labels = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        for img in os.listdir(real_dir):
            self.images.append(os.path.join(real_dir,img))
            self.labels.append(0)

        for img in os.listdir(fake_dir):
            self.images.append(os.path.join(fake_dir,img))
            self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        path = self.images[idx]

        img = cv2.imread(path)

        face = detect_face(img)

        if face is None:
            h, w, _ = img.shape
            size = min(h, w)
            start_x = w//2 - size//2
            start_y = h//2 - size//2
            face = img[start_y:start_y+size, start_x:start_x+size]

        face = cv2.resize(face,(256,256))


        # data augmentation
        if np.random.rand() > 0.5:
            face = cv2.flip(face,1)

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = gray / 255.0
        freq_img = gray

        freq_img = torch.tensor(freq_img, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return freq_img, label