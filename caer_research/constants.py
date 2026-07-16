"""Shared CAER-S constants."""

CLASS_NAMES = ("Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {index: name for name, index in LABEL_TO_INDEX.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
