
# asking IA from Bing "python inference for a fastrcnn_fpn mode"
# and adapted and slightly modified by Alfonso Blanco



import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO


import matplotlib.pyplot as plt
import os
import re

# PARAMETERS
CONF_THRESHOLD=0.9
#imgpath="Brain Tumors Detection.v2-v2.coco\\test\\" # images test folder
#imgpath="Test1" # images test folder
imgpath="Test2" # images test folder
LabelClass="drone "

# asking IA Bing PIL draw rectangle width changes with rectangle width python
def draw_scaled_rectangle(draw, xy, outline, scale_factor=0.05):
    x0, y0, x1, y1 = xy
    rect_width = abs(x1 - x0)
    rect_height = abs(y1 - y0)
    stroke_width = max(1, int(min(rect_width, rect_height) * scale_factor))
    draw.rectangle(xy, outline=outline, width=stroke_width)


# -----------------------------
# 1. Load Pretrained Model
# -----------------------------

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Path to your saved weights file (.pth or .pt)
weights_path = "fasterrcnn_fpn.pth"

# 1️⃣ Create the model architecture (no pretrained weights)
model = fasterrcnn_resnet50_fpn(weights=None)  # For torchvision >= 0.13
# For older versions:
#model = fasterrcnn_resnet50_fpn(pretrained=False)

in_features = model.roi_heads.box_predictor.cls_score.in_features
# Replace head for custom classes
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

# 2️⃣ Load the saved state dictionary
try:
    state_dict = torch.load(weights_path, map_location="cpu")  # or "cuda" if GPU
    model.load_state_dict(state_dict)
    print("✅ Weights loaded successfully from", weights_path)
except FileNotFoundError:
    print(f"❌ File not found: {weights_path}")
except RuntimeError as e:
    print(f"❌ Error loading weights: {e}")

model.eval()  # Set to evaluation mode


# -----------------------------
# 2. Load and Preprocess Images
# -----------------------------
    
     
for root, dirnames, filenames in os.walk(imgpath):

 
 
 for filename in filenames:
     
     if re.search("\.(jpg|JPEG|jpeg|png|bmp|tiff)$", filename):
         
         
        filepath = os.path.join(root, filename)
       
        img = Image.open(filepath).convert("RGB")

        # Transform to tensor
        img_tensor = F.to_tensor(img)  # Converts to [C,H,W] and scales to [0,1]

        # -----------------------------
        # 3. Run Inference
        # -----------------------------
        with torch.no_grad():
            predictions = model([img_tensor])  # Model expects a list of tensors

        # Extract predictions for the first image
        pred = predictions[0]
        boxes = pred["boxes"]
        labels = pred["labels"]
        scores = pred["scores"]

        # -----------------------------
        # 4. Filter by Confidence
        # -----------------------------
        threshold = CONF_THRESHOLD
        print(scores)
        keep = scores >= threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]

        # -----------------------------
        # 5. Draw Results
        # -----------------------------
        draw = ImageDraw.Draw(img)
        try:
            #font = ImageFont.truetype("arial.ttf", 16) # MOD
            font_size = 50
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            print("ERROR FONT")
            font = ImageFont.load_default()

        # Get COCO labels
        #coco_labels = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

        Cont=0
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            #draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=6)
            draw_scaled_rectangle(draw, (x1, y1, x2, y2), "red") # MOD
            #text = f"{coco_labels[label]}: {score:.2f}"  # MOD         
            text = f"{LabelClass }: {score:.2f}"  
            
            text_size=draw.textbbox((0,0),text=text, font=font)
            
            draw.rectangle(((x1, y1 - text_size[1]), (x1 + text_size[0], y1)), fill="red", width=6)

            
            draw.text((x1, y1 - text_size[1]), text, fill="white", font=font, width=6)

            Cont=Cont+1
            

        # -----------------------------
        # 6. Show Image
        # -----------------------------
        #img.show()
        print(f"{LabelClass }" + "s detected = " + str(Cont))
        # Trick to show and wait
        plt.imshow(img)
        #plt.axis('off')  # Hide axes for a cleaner look
        plt.title(filename)
        plt.show()
        
       
