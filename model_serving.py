import os.path
import pickle
import cv2
# from detectron2.engine.defaults import DefaultPredictor
# from detectron2.data.transforms import ResizeTransform
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.utils.visualizer import Visualizer, ColorMode


def predict_img_svm(img_path, new_img_name, dest_folder):
    # load the model file
    with open('models/svm_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    # read the image
    img = cv2.imread(img_path)
    # Use orb object to detect and compute descriptors and keypoints
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nlevels=1)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    # use the descriptors as the features to predict the label of keypoints
    preds = loaded_model.predict(descriptors)
    # draw keypoints that are predicted to be 1 (in a bbox, or on wheat)
    for kpt, pred in zip(keypoints, preds):
        if pred == 1:
            img = cv2.drawKeypoints(img, [kpt], None, color=(0, 255, 0))
    cv2.imwrite(os.path.join(dest_folder, new_img_name), img)


# def predict_img_dl(img_path):
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.WEIGHTS = 'model_final.pth'
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#     MetadataCatalog.get("dataset").thing_classes = ['wheat']
#
#     model = DefaultPredictor(cfg)
#
#     def evaluate(image_path):
#         img = cv2.imread(image_path)
#         resize = ResizeTransform(new_h=256, new_w=256, h=1024, w=1024)
#         img = resize.apply_image(img)
#         return model(img)
#
#     outputs = evaluate(img_path)
#
#     v = Visualizer(cv2.imread('53f253011.jpg')[:, :, ::-1], metadata=MetadataCatalog.get("dataset"), scale=0.5, instance_mode=ColorMode.IMAGE_BW)
#
#     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     cv2.imwrite(f"{img_path.split('.')[0]}-with-bbox.jpg", v.get_image()[:, :, ::-1])


# predict_img_dl('51f1be19e.jpg')
