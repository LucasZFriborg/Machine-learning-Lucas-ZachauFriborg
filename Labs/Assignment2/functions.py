import os
import json
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

    return input_batch, output, prediction

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

def generate_cam(model, input_batch):
    with LayerCAM(model) as cam_extractor:
        output = model(input_batch)  # forward sker här!
        class_idx = output.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_idx, output)
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

def analyze_class(model, preprocess, class_path, class_index_path):
    images = load_images(class_path)

    results = {}

    for label, img in images.items():
        input_batch, output, prediction = run_inference(img, model, preprocess)
        pred_info = predict_class(prediction, class_index_path)
        cam = generate_cam(model, input_batch)

        results[label] = {
            "image": img,
            "prediction": pred_info,
            "cam": cam
        }

    return results["positive"], results["negative"]

def plot_class_results(pos, neg, class_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Positive
    pos_overlay = overlay_mask(
        to_pil_image(pos["image"]),
        to_pil_image(pos["cam"][0].squeeze(0), mode='F'),
        alpha=0.5
    )

    axes[0].imshow(pos_overlay)
    axes[0].set_title(
        f"Positive\nPred: {pos['prediction']['class_name']} ({pos['prediction']['confidence']*100:.1f}%)"
    )
    axes[0].axis('off')

    # Negative
    neg_overlay = overlay_mask(
        to_pil_image(neg["image"]),
        to_pil_image(neg["cam"][0].squeeze(0), mode='F'),
        alpha=0.5
    )

    axes[1].imshow(neg_overlay)
    axes[1].set_title(
        f"Negative\nPred: {neg['prediction']['class_name']} ({neg['prediction']['confidence']*100:.1f}%)"
    )

    axes[1].axis('off')
    plt.suptitle(f"Class: {class_name}")
    plt.tight_layout()
    plt.show()