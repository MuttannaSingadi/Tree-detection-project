import cv2
import torch
import numpy as np
from train import UNet

# Load model
model = UNet()
model.load_state_dict(torch.load("tree_model.pth", map_location="cpu"))
model.eval()

# Load test image
image = cv2.imread("test.jpg")
orig = image.copy()

image = cv2.resize(image, (256, 256)) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

image = torch.tensor(image, dtype=torch.float32)

# Predict
with torch.no_grad():
    output = model(image)
    output = torch.sigmoid(output)   
    output = output[0][0].numpy()

# Threshold
mask = (output > 0.6).astype(np.uint8) * 255

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

result = cv2.bitwise_and(orig, orig, mask=mask)

cv2.imshow("Mask", mask)
cv2.imshow("Trees ", result)
cv2.waitKey(0)
cv2.destroyAllWindows()