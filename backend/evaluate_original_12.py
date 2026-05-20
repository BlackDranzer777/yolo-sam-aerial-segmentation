"""
evaluate_original_12.py — Re-evaluate the original 12-image study images.

Saves per-image metrics to evaluation_original_12.json for building
the combined 22-image results table in the thesis.

Run from the backend/ directory:
    python evaluate_original_12.py
"""
import sys, os, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from modules.pipeline import SegmentationPipeline
from modules.evaluator import evaluate_results

ORIGINAL_IDS = ['001', '004', '011', '022', '075', '150', '165', '235', '238', '411', '531', '568']
OUTPUT_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation_original_12.json')


def run_batch():
    print('Loading pipeline (YOLO + SAM)...')
    pipeline = SegmentationPipeline()
    print('Pipeline ready.\n')

    all_results = []

    for img_id in ORIGINAL_IDS:
        img_path = os.path.join(config.ORIGINAL_IMAGES_DIR, f'{img_id}.jpg')
        gt_path  = os.path.join(config.LABEL_MASKS_DIR,     f'{img_id}.png')

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            print(f'[{img_id}] missing file — skip')
            all_results.append({'image_id': img_id, 'error': 'file not found'})
            continue

        print(f'[{img_id}] Running pipeline...')
        try:
            result     = pipeline.run(image_path=img_path, image_id=img_id)
            seg_results = result['seg_results']
            detections  = result['detections']

            print(f'  detections={len(detections)}  masks={len(seg_results)}')

            if not seg_results:
                print(f'  No detections')
                all_results.append({
                    'image_id':   img_id,
                    'detections': 0,
                    'evaluation': None,
                    'note':       'no detections',
                })
                continue

            eval_out = evaluate_results(seg_results, img_id)
            overall  = eval_out['overall']

            print(f'  Overall  IoU={overall["mean_iou"]:.4f}  '
                  f'P={overall["mean_precision"]:.4f}  R={overall["mean_recall"]:.4f}')
            for obj in eval_out['per_object']:
                if obj['iou'] is not None:
                    print(f'    {obj["class_name"]:10s}  IoU={obj["iou"]:.4f}  '
                          f'P={obj["precision"]:.4f}  R={obj["recall"]:.4f}  ({obj["note"]})')
                else:
                    print(f'    {obj["class_name"]:10s}  IoU=None  ({obj.get("note","")})')

            all_results.append({
                'image_id':   img_id,
                'detections': len(detections),
                'evaluation': eval_out,
            })

        except Exception as exc:
            print(f'  ERROR: {exc}')
            all_results.append({'image_id': img_id, 'error': str(exc)})

        print()

    valid = [r for r in all_results
             if r.get('evaluation') and r['evaluation']['overall']['evaluated_count'] > 0]

    if valid:
        mean_iou = np.mean([r['evaluation']['overall']['mean_iou'] for r in valid])
        mean_pre = np.mean([r['evaluation']['overall']['mean_precision'] for r in valid])
        mean_rec = np.mean([r['evaluation']['overall']['mean_recall'] for r in valid])
        print('─' * 60)
        print(f'Summary over {len(valid)} evaluated images:')
        print(f'  Mean IoU:       {mean_iou:.4f}')
        print(f'  Mean Precision: {mean_pre:.4f}')
        print(f'  Mean Recall:    {mean_rec:.4f}')
        summary = {
            'evaluated_images': len(valid),
            'mean_iou':         round(float(mean_iou), 4),
            'mean_precision':   round(float(mean_pre), 4),
            'mean_recall':      round(float(mean_rec), 4),
        }
    else:
        summary = {'evaluated_images': 0}

    output = {'summary': summary, 'images': all_results}

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=lambda x: None if isinstance(x, np.ndarray) else x)

    print(f'\nResults saved to: {OUTPUT_FILE}')


if __name__ == '__main__':
    run_batch()
