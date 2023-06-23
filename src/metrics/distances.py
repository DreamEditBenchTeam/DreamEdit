import numpy as np


def compute_cosine_distance(image_features, image_features2):
    # normalized features
    image_features = image_features / np.linalg.norm(np.float32(image_features), ord=2)
    image_features2 = image_features2 / np.linalg.norm(np.float32(image_features2), ord=2)
    return np.dot(image_features, image_features2)


def compute_l2_distance(image_features, image_features2):
    return np.linalg.norm(np.float32(image_features - image_features2))
