import numpy as np
import cv2
from resnet.feature_extractor import BBResNet18


def extract_image_patch(image, bbox, patch_shape=None):
    """Extract image patch from bounding box."""
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    
    # return None if the bounding box is empty
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    
    # extract the image patch
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    if patch_shape is not None:
        image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


def create_box_encoder(batch_size=32):
    """Create a function that generates feature vectors for given
    image and bounding boxes."""
    image_encoder = BBResNet18(batch_size=batch_size)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        # extract the image patches
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        
        # prepare the image patches for the feature extractor
        image_patches = np.asarray(image_patches).transpose(0, 3, 1, 2)
        image_patches = image_patches.astype(np.float32) / 255.

        # extract the features
        features = image_encoder.feature_extraction(image_patches)
        return features

    return encoder