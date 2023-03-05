import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def FastRCNN_model():
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    # Faster R-CNN is a model that predicts both bounding boxes and class scores for potential objects in the image.
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)