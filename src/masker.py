import cv2
import torch
import numpy as np

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import time
import matplotlib.pyplot as plt
# from fastsam import FastSAM, FastSAMPrompt

import supervision as sv

import time
class Masker():

    def __init__(self): #, config

        sam2_checkpoint = "pretrained/sam2.1_hiera_large.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.devide = "cuda" if torch.cuda.is_available() else "cpu"

        sam2_model = build_sam2(sam2_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        self.mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side = 32, # 32,
        points_per_batch = 64,
        pred_iou_thresh = 0.8, # 0.8
        stability_score_thresh = 0.85, # 0.9, # 0.95,
        stability_score_offset = 1.0,
        mask_threshold = 0.0,
        box_nms_thresh = 0.7,
        crop_n_layers = 0,
        crop_nms_thresh = 0.7,
        crop_overlap_ratio = 512 / 1500,
        crop_n_points_downscale_factor = 1,
        point_grids = None,
        min_mask_region_area = 0,
        output_mod = "binary_mask",
        use_m2m = False,
        multimask_output = True,
        )

        self.save_dir = "test"

    def get_instances_from_uncer_mask(self, uncer_mask, image, sample_ratio=0.1):
        H, W = uncer_mask.shape[:2]
        step_H = int(H / 480 * 10)
        step_W = int(W / 640 * 10)
        y_coords = np.arange(0, H, step_H)
        x_coords = np.arange(0, W, step_W)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.column_stack([xx.ravel(), yy.ravel()])

        uncer_mask_HW3 = (uncer_mask[:, :, np.newaxis].astype(np.uint8)*255).repeat(3, axis=2)

        start = time.time()
        anns = self.mask_generator.generate(uncer_mask_HW3)
        print(f"self.mask_generator.generate: {time.time() - start}")
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        
        instance_masks = []
        if len(sorted_anns) > 1:
            for i, ann in enumerate(sorted_anns):

                img_bin = ann['segmentation']
                tmp_mask = uncer_mask[img_bin]
                if np.count_nonzero(tmp_mask)/len(tmp_mask) < 0.8:
                    continue

                valids = img_bin[yy.ravel(), xx.ravel()]
                point_coords = points[valids]
                point_labels = np.ones(len(point_coords))

                instance_mask, boxes = masker.get_obj_with_center(image, point_coords, "test", point_labels=point_labels)
                
                if (np.count_nonzero(instance_mask & img_bin) / np.count_nonzero(img_bin)) > 0.8:
                    instance_masks.append(instance_mask)

        return instance_masks
    
    def get_obj_with_center(self, image, input_coords, label, point_labels=None):
        self.sam2_predictor.set_image(image)
        masks, scores, logits = self.sam2_predictor.predict(  # SAM2
            point_coords=input_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        class_ids = np.array([0])

        labels_vis = [
            f"{label}"
        ]

        """
        Visualize image with supervision useful API
        """
        mask = masks[0]
        input_box = np.zeros((1, 4), dtype=int)

        foreground_coords = np.argwhere(mask)
        top_left = foreground_coords.min(axis=0)
        bottom_right = foreground_coords.max(axis=0)

        input_box[0, 0] = top_left[1]
        input_box[0, 1] = top_left[0]
        input_box[0, 2] = bottom_right[1]
        input_box[0, 3] = bottom_right[0]

        detections = sv.Detections(
            xyxy=np.array(input_box),  # (n, 4)
            mask=np.array([mask], dtype=bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_vis)
        
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        

        annotated_frame[input_coords[:,1], input_coords[:,0]] = np.array([255, 0, 0])
        cv2.imwrite("test.png", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        return mask, input_box[0]

    
if __name__ == '__main__':
    masker = Masker()

    mask = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED).astype(bool)
    img = cv2.imread("image.png")

    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 
            
        ax.imshow(img)
        plt.axis('off')
        plt.savefig("test.png", dpi=150, bbox_inches='tight', pad_inches=0)

    start = time.time()
    masks = masker.mask_generator.generate(img, mask)
    print(f"gen: {time.time() - start}")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    show_anns(masks)
