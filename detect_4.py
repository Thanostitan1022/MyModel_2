import matplotlib.pyplot as plt
import torchvision
import cv2
import numpy as np


def decode_predictions(preds, conf_thresh=0.5):
    boxes = []
    for bbox, cls in preds:
        B, _, H, W = cls.shape
        cls = torch.sigmoid(cls)
        for b in range(B):
            for i in range(H):
                for j in range(W):
                    score = cls[b, 0, i, j]
                    if score > conf_thresh:
                        dx, dy, dw, dh = bbox[b, :, i, j]
                        x_center = j + dx.item()
                        y_center = i + dy.item()
                        w = torch.exp(dw).item()
                        h = torch.exp(dh).item()
                        x1 = x_center - w / 2
                        y1 = y_center - h / 2
                        x2 = x_center + w / 2
                        y2 = y_center + h / 2
                        boxes.append([x1, y1, x2, y2, score.item()])
    return boxes


def draw_boxes(image, boxes):
    img = image.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for box in boxes:
        x1, y1, x2, y2, score = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.imshow(img)
    plt.axis('off')
    plt.title("Detected Stroke Lesions")
    plt.show()

# def decode_predictions(preds, conf_thresh=0.5):
#     boxes_all = []
#
#     for bbox_pred, cls_pred in preds:
#         B, _, H, W = cls_pred.shape
#         cls_pred = torch.sigmoid(cls_pred)
#
#         for b in range(B):
#             for i in range(H):
#                 for j in range(W):
#                     score = cls_pred[b, 0, i, j]
#                     if score > conf_thresh:
#                         # 将预测框回归值解码为 [x, y, w, h]
#                         dx, dy, dw, dh = bbox_pred[b, :, i, j]
#                         x_center = j + dx
#                         y_center = i + dy
#                         w = torch.exp(dw)
#                         h = torch.exp(dh)
#                         x1 = x_center - w / 2
#                         y1 = y_center - h / 2
#                         x2 = x_center + w / 2
#                         y2 = y_center + h / 2
#                         boxes_all.append([x1.item(), y1.item(), x2.item(), y2.item(), score.item()])
#     return boxes_all
#
#
# def draw_boxes(image, boxes):
#     img = image.squeeze().cpu().numpy()
#     img = (img * 255).astype(np.uint8)
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#     for box in boxes:
#         x1, y1, x2, y2, score = box
#         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#         cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()
