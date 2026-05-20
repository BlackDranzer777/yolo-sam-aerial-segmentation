"""
find_good_images.py — Scan candidate images and collect those with detections.

Tries up to CANDIDATES images, stops once we have TARGET with at least one detection.
Prints a summary so we can pick the final 10 for the expanded study.
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.pipeline import SegmentationPipeline

# Already used in the original 12-image study
USED = {'001','004','011','022','075','150','165','235','238','411','531','568'}

# Candidates to try (not in USED)
CANDIDATES = [
    '002','003','005','006','013','014','015','016','018','019',
    '021','023','026','028','031','035','038','040','041','042',
    '043','044','045','047','049','051','052','053','055','056',
    '057','058','059','060','062','063','065','068','070','071',
    '073','074','077','078','079','080','081','083','086','088',
]

TARGET = 10   # we want this many with detections

def main():
    print('Loading pipeline...')
    pipeline = SegmentationPipeline()
    print('Ready.\n')

    with_detections = []
    no_detections   = []

    for img_id in CANDIDATES:
        if len(with_detections) >= TARGET:
            break

        if img_id in USED:
            continue

        img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, f'{img_id}.jpg')
        gt_path  = os.path.join(config.LABEL_MASKS_DIR,     f'{img_id}.png')

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            print(f'[{img_id}] missing file — skip')
            continue

        try:
            result = pipeline.run(image_path=img_path, image_id=img_id)
            n = len(result['detections'])
            classes = list({d['class_name'] for d in result['detections']})
            if n > 0:
                print(f'[{img_id}] detections={n}  classes={classes}  *** KEEP ***')
                with_detections.append(img_id)
            else:
                print(f'[{img_id}] detections=0  skip')
                no_detections.append(img_id)
        except Exception as e:
            print(f'[{img_id}] ERROR: {e}')

    print('\n' + '='*50)
    print(f'Images WITH detections ({len(with_detections)}): {with_detections}')
    print(f'Images with NO detections ({len(no_detections)}): {no_detections}')

if __name__ == '__main__':
    main()
