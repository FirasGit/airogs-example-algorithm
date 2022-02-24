import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import albumentations as A
from networks.detection.utils import save_one_box
import pytorch_lightning as pl
import timm
import cv2 as cv
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

# TODO: Output looks random as of right now
# TODO: Output doesn't match expected output


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
SHAPE = [224, 224]


class MinMaxBrightness(ImageOnlyTransform):
    def apply(self, img_rgb, **params):
        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = self.min_max_scaler(
            img_hsv[:, :, 2], min=0, max=255)
        img_rgb_scaled = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)
        return img_rgb_scaled

    def min_max_scaler(self, img, min=0, max=255):
        img_std = (img - img.min()) / (img.max() - img.min())
        img_scaled = img_std * (max - min) + min
        return img_scaled


class LightningClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('tf_efficientnetv2_m', pretrained=False, num_classes=1, in_chans=3)
        self.model.eval()
    def forward(self, image):
        return self.model(image)


class LightningClassifierInferer(nn.Module):
    def __init__(self):
        super().__init__()
        checkpoint_path_fold_0 = './networks/classification/epoch=66_Val_epoch_loss=0.434_Val_partial_auc=0.930.ckpt'
        checkpoint_path_fold_1 = './networks/classification/epoch=81_Val_epoch_loss=0.438_Val_partial_auc=0.925.ckpt'
        checkpoint_path_fold_2 = './networks/classification/epoch=76_Val_epoch_loss=0.454_Val_partial_auc=0.930.ckpt'
        checkpoint_path_fold_3 = './networks/classification/epoch=41_Val_epoch_loss=0.417_Val_partial_auc=0.926.ckpt'
        checkpoint_path_fold_4 = './networks/classification/epoch=69_Val_epoch_loss=0.523_Val_partial_auc=0.919.ckpt'
        torch.set_grad_enabled(False) # Don't generate a computational graph

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used: ", self.device)

        model_fold_0 = LightningClassifier().load_from_checkpoint(checkpoint_path=checkpoint_path_fold_0)
        model_fold_0.eval()
        model_fold_0.to(self.device)
        
        model_fold_1 = LightningClassifier().load_from_checkpoint(checkpoint_path=checkpoint_path_fold_1)
        model_fold_1.eval()
        model_fold_1.to(self.device)


        model_fold_2 = LightningClassifier().load_from_checkpoint(checkpoint_path=checkpoint_path_fold_2)
        model_fold_2.eval()
        model_fold_2.to(self.device)

        model_fold_3 = LightningClassifier().load_from_checkpoint(checkpoint_path=checkpoint_path_fold_3)
        model_fold_3.eval()
        model_fold_3.to(self.device)

        model_fold_4 = LightningClassifier().load_from_checkpoint(checkpoint_path=checkpoint_path_fold_4)
        model_fold_4.eval()
        model_fold_4.to(self.device)


        self.model_folds = nn.ModuleList([model_fold_0, model_fold_1, model_fold_2, model_fold_3, model_fold_4])

        self.preprocessing = A.Compose([
            MinMaxBrightness(always_apply=True),
            A.Equalize(by_channels=False, always_apply=True),
            A.Normalize(mean=MEAN, std=STD, always_apply=True),
            A.Resize(height=SHAPE[0], width=SHAPE[1], always_apply=True),
        ])

        self.tta_transforms = A.Compose([
            A.Rotate(limit=10, p=1),
            A.Flip(p=0.5),
            ToTensorV2()
        ])


		# TODO: Find a better cut-off point, e.g. Youden's Index?
        self.prediction_cut_off = 0.63
    
    def preprocess(self, image):
        return self.preprocessing(image=image)['image']
    
    def transform(self, image):
        return self.tta_transforms(image=image)['image']
    
    def convert_to_tensor(self, image):
        """Needed if no transforms are applied"""
        return ToTensorV2()(image=image)['image']
    
    def get_prediction_proba(self, prediction):
        return torch.sigmoid(prediction)
    
    def get_prediction(self, model, image) -> float:
        image_batch = image[None] # Add batch size
        image_batch = image_batch.to(self.device)
        prediction_logits = model(image_batch)
        prediction = self.get_prediction_proba(prediction_logits).item()
        return prediction
    
    def get_label(self, prediction) -> int:
        if prediction <= self.prediction_cut_off:
            return False
        else:
            return True
    
    def forward(self, image, tta=True, num_tta_transforms=5):
        prep_image = self.preprocess(image)
        predictions_all_folds = []
        predictions_all_folds_all_tta = []
        for fold in self.model_folds:
            if tta:
                predictions_tta = []
                for _ in range(num_tta_transforms):
                    tta_prep_image = self.transform(prep_image)
                    prediction_tta = self.get_prediction(model=fold, image=tta_prep_image)
                    predictions_tta.append(prediction_tta)
                predictions_all_folds_all_tta.extend(predictions_tta)
                prediction_fold = np.mean(predictions_tta)
            else:
                prep_image_tensor = self.convert_to_tensor(prep_image)
                prediction_fold = self.get_prediction(model=fold, image=prep_image_tensor)
                predictions_all_folds_all_tta.append(prediction_fold)
            predictions_all_folds.append(prediction_fold)
        prediction = np.mean(predictions_all_folds)
        label = self.get_label(prediction)
        variance = np.var(predictions_all_folds_all_tta)
        return prediction, label, variance


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        path_to_repo = './networks/detection/yolov5'
        path_to_model = './networks/detection/yolov5/runs/train/exp4/weights/best.pt'
        self.model = torch.hub.load(path_to_repo, 'custom', path=path_to_model, source='local')
        self.model.eval()
        self.model.max_det = 1 # Take only most confident score
        self.model.conf = 0 # Min confidence score is 0

        self.preprocessing = A.Compose([A.Equalize(always_apply=True)])
    
    def preprocess(self, image):    
        return self.preprocessing(image=image)['image']

    def forward(self, image):
        prep_image = self.preprocess(image)
        results = self.model(prep_image, augment=True)
        crop_dict = results.crop(save=False)
        box_idx = -1
        if not crop_dict:
            # No bounding box was found
            cropped_image = None
            confidence_score = 0
            return cropped_image, confidence_score
        confidence_score = crop_dict[box_idx]['conf'].item()
        cropped_image, _ = save_one_box(
                xyxy=crop_dict[box_idx]['box'],
                im=image,
                path=None, 
                BGR=True,
                save=False,
                square=True,
            )
        return cropped_image, confidence_score


# Test everything again own validation set. Have to achieve same results as best AUC
def do_inference(input_image_array: np.ndarray, Detector: Detector, Classifier: LightningClassifier):
    """Do inference on the input image

    Args:
        input_image_array (np.ndarray): Shape (w, h, c)
        Detector: Model that finds the ROI around the optic disk
        Classifier: Model that uses the ROI to detect presence of RG/ NRG
    """
    cropped_img, confidence_score = Detector(image=input_image_array)
    if cropped_img is not None:
        rg_likelihood, rg_binary, rg_variance = Classifier(cropped_img)
    else:
        # No prediction possible, however it's more likely to be NRG (due to more cases)
        # TODO: When using variance, make sure that when no cropped image is available it cannot be used (as it is
        # not created)
        rg_likelihood, rg_binary, rg_variance = 0, False, 1e9

    # TODO: Find some better metric
    ungradability_score = 1 - confidence_score

    # TODO: Find an appropriate cut-off
    cut_off = 0.9
    if ungradability_score <= cut_off:
        ungradability_binary = False
    else: 
        ungradability_binary = True
    
    return rg_likelihood, rg_binary, ungradability_score, ungradability_binary, rg_variance

    