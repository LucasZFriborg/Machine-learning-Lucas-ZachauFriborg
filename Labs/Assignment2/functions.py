import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask


def load_model():
    weights = get_model_weights('resnet18').DEFAULT
    model = get_model('resnet18', weights=weights).eval()
    preprocess = weights.transforms()
     
    return model, preprocess

def get_classes(base_path):
    classes = []
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if os.path.isdir(full_path):
            classes.append(name)
    return sorted(classes)

def load_images(class_path):
    positive_path = os.path.join(class_path, 'positive.jpg')
    negative_path = os.path.join(class_path, 'negative.jpg')
    
    img_positive = decode_image(positive_path)
    img_negative = decode_image(negative_path)

    return {
        'positive': img_positive,
        'negative': img_negative
    }

def run_inference(img, model, preprocess):
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    output = model(input_batch)
    prediction = output.squeeze(0).softmax(0)

    return output, prediction

def predict_class(prediction, class_index_path):
    with open(class_index_path, 'r') as f:
        class_index = json.load(f)
    
    top_idx = int(prediction.argmax())
    synset_id, class_name = class_index[str(top_idx)]

    return {
        'class_index': top_idx,
        'class_id': synset_id,
        'class_name': class_name,
        'confidence': float(prediction[top_idx])
    }

def generate_cam(model, output):
    with LayerCAM(model) as cam_extractor:
        activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
    return activation_map

def display_cam(activation_map, title):
    plt.imshow(activation_map[0].squeeze(0).numpy())
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_overlay(img, activation_map, title):
    result = overlay_mask(
        to_pil_image(img),
        to_pil_image(activation_map[0].squeeze(0), mode='F'), 
        alpha=0.5
    )
    plt.imshow(result)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()