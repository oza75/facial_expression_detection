import os.path

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file
from pipelines import DetectMiniXceptionFER, MiniXceptionFER


class DetectVGG16FER(DetectMiniXceptionFER):
    def __init__(self, offsets=[0, 0], colors=None):
        super().__init__(offsets=offsets, colors=colors)
        self.classify = VGG16FER()


class VGG16FER(MiniXceptionFER):
    def __init__(self):
        super().__init__()
        self.classifier = VGG16((48, 48, 1), 7, weights='FER')


def VGG16(input_shape, num_classes, weights=None):
    filename = os.path.dirname(__file__) + '/weights/model_optimal.h5'
    # path = get_file(filename, filename, cache_subdir='paz/models')
    model = load_model(filename)
    model._name = 'MINI-XCEPTION'
    return model
