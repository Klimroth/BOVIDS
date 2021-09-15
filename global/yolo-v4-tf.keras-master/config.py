yolo_config = {
    # Basic
    'img_size': (416,416, 3), # 416
    'anchors': [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401], #459, 401 statt 50, 170
    'strides': [8, 16, 32],
    'xyscale': [1.2, 1.1, 1.05],

    # Training
    'iou_loss_thresh': 0.5,
    'batch_size': 2,
    'num_gpu': 1,
    # Inference
    'max_boxes': 1,
    'iou_threshold':0.5,
    'score_threshold':0.0000000001,
}