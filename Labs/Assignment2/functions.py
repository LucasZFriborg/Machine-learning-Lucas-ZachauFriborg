import os
import json
import torch
import matplotlib.pyplot as plt
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask


weights = get_model_weights('resnet18').DEFAULT
model = get_model('resnet18', weights=weights).eval()

preprocess = weights.transforms()

base_path = 'Labs/Assignment2/Images'

classes = os.listdir(base_path)

def predict_class(output_tensor: torch.Tensor, class_index_path: str) -> dict:
    with open(class_index_path, 'r') as f:
        class_index = json.load(f)

    probs = output_tensor.squeeze()
    if probs.ndim != 1 or probs.shape[0] != 1000:
        raise ValueError(
            f'Expected a tensor of 1000 values, got shape {tuple(output_tensor.shape)}'
        )
    
    top_idx = int(probs.argmax())
    synset_id, class_name = class_index[str(top_idx)]

    return {
        'class index': top_idx,
        'class_id': synset_id,
        'class_name': class_name,
        'confidence': float(probs[top_idx])
    }

for cls in classes:
    class_path = os.path.join(base_path, cls)
    print(class_path)

    positive_path = os.path.join(class_path, 'positive.jpg')
    negative_path = os.path.join(class_path, 'negative.jpg')

    img_positive = decode_image(positive_path)
    img_negative = decode_image(negative_path)

    for label, img in [('positive', img_positive), ('negative', img_negative)]:
        input_tensor = preprocess(img)

        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch)

        prediction = output.squeeze(0).softmax(0)

        prediction_result = predict_class(prediction.detach(), 'Labs/Assignment2/imagenet_class_index.json')

        print(f'Class: {cls} | Type: {label}')
        print(prediction_result)

        with LayerCAM(model) as cam_extractor:
            activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)

        plt.imshow(activation_map[0].squeeze(0).numpy())
        plt.title(f'{cls} - {label}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        overlay_result = overlay_mask(
            to_pil_image(img), 
            to_pil_image(activation_map[0].squeeze(0), mode='F'),
            alpha=0.5
        )
        plt.imshow(overlay_result)
        plt.title(f'{cls} - {label} (overlay)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()