import logging
import numpy as np


class OneShotPredictor:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, test_image, list_support_images):
        list_images_1 = [test_image]*len(list_support_images)
        list_images_2 = list_support_images
        probs = self.predictor.predict_prob(list_images_1, list_images_2)
        logging.info('Prob: %s', probs)
        return np.argmax(probs)
