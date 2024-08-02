from ultralytics import YOLO
import cv2
import torch
import numpy as np
import os
import glob

from ultralytics import YOLO
from pathlib import Path



def draw_mask(image, mask_generated):
    """
    Extracted from https://inside-machinelearning.com/en/plot-segmentation-mask/
    """
    masked_image = image.copy()
    h, w, _ = masked_image.shape
    mask_generated = cv2.resize(mask_generated, (w, h))

    masked_image = np.where(
        mask_generated.astype(int), np.array([0, 255, 0], dtype="uint8"), masked_image
    )

    masked_image = masked_image.astype(np.uint8)

    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)


def yolo_segmentation_demo():
    model = YOLO("yolov8n-seg.pt", task="segment")
    should_use_pretraining = True
    
    epochs = int(input("Input how many epochs should be run (empty = 100): ") or "100")

    # for now if you use segmentaiton the augment option breaks the code, that's why we use it as false
    model.train(data="./config.yml", epochs=epochs, pretrained=should_use_pretraining, augment=False)

    save_path = "./segmentation_ant_roads"
    Path(save_path).mkdir(parents=True, exist_ok=True)

    images = []
    for file in glob.glob(f"./dataset/test/images/*.jpg"):
        images.append(file)

    temporal_file = f"{save_path}/temporal_mask.jpg"
    for image_path in images:
        img = cv2.imread(image_path)
        image_name = Path(image_path).name
        is_new_mask = False
        for result in model.predict(img):
            if result.masks:
                rmasks = result.masks.data
                ants_mask = torch.any(rmasks, dim=0).int() * 255
                is_new_mask = True
        if is_new_mask:
            cv2.imwrite(temporal_file, ants_mask.cpu().numpy())
            mask = cv2.imread(temporal_file)
            masked_image = draw_mask(img, mask)
        else:
            masked_image = img
        cv2.imwrite(f"{save_path}/{image_name}.jpg", masked_image)
    os.remove(temporal_file)

  
if __name__ == "__main__":
    yolo_segmentation_demo()