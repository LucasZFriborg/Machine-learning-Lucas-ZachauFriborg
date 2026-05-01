import os
import json
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask

def load_model():
    """
    Loads a pretrained ResNet18 together with its default preprocessing.
    The model is set to evalutation mode since no training is performed.
    """
    weights = get_model_weights('resnet18').DEFAULT
    model = get_model('resnet18', weights=weights).eval()
    preprocess = weights.transforms()
     
    return model, preprocess

def load_images(class_path):
    """
    Loads the positive and negative example images from a given class folder.
    Each class is assumed to contain one image of each type.
    """
    positive_path = os.path.join(class_path, 'positive.jpg')
    negative_path = os.path.join(class_path, 'negative.jpg')
    
    img_positive = decode_image(positive_path)
    img_negative = decode_image(negative_path)

    return {
        'positive': img_positive,
        'negative': img_negative
    }

def run_inference(img, model, preprocess):
    """
    Preprocesses the input image and performs a forward pass through the model.
    The output is converted to probabilities using softmax.
    """
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    output = model(input_batch)
    prediction = output.squeeze(0).softmax(0)

    return input_batch, prediction

def predict_class(prediction, class_index_path):
    """
    Extracts the predicted class by selecting the highest probability
    and mapping it to a human-readable label using the ImageNet index.
    """
    with open(class_index_path, 'r') as f:
        class_index = json.load(f)
    
    top_idx = int(prediction.argmax())
    synset_id, class_name = class_index[str(top_idx)]

    return {
        'class_index': top_idx,
        'class_id': synset_id,
        'class_name': class_name,
        'confidence': float(prediction[top_idx].detach())
    }

def generate_cam(model, input_batch):
    """
    Generates a class activation map using LayerCAM.
    The map highlights which regions of the image contributed most to the predicted class.
    """
    with LayerCAM(model, target_layer='layer4') as cam_extractor:
        output = model(input_batch)
        class_idx = output.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_idx, output)
    return activation_map

def analyze_class(model, preprocess, class_path, class_index_path):
    """
    Runs the full analysis for a class by preprocessing both positive and negative images.
    For each image, prediction and activation maps are generated.
    """
    images = load_images(class_path)

    results = {}

    for label, img in images.items():
        input_batch, prediction = run_inference(img, model, preprocess)
        pred_info = predict_class(prediction, class_index_path)
        cam = generate_cam(model, input_batch)

        results[label] = {
            "image": img,
            "prediction": pred_info,
            "cam": cam
        }

    return results["positive"], results["negative"]

def plot_class_results(pos, neg, class_name):
    """
    Visualizes the results by overlaying the activation maps on the original images.
    Both positive and negative examples are shown side by side for comparison.
    """
    _, axes = plt.subplots(1, 2, figsize=(10, 5))

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