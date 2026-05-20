"""
build_combined_table.py — Merge original 12 + new 10 results into one table.

Prints a thesis-ready ASCII table and combined statistics.
Run from backend/: python build_combined_table.py
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')

ORIG_FILE = 'evaluation_original_12.json'
NEW_FILE  = 'evaluation_new_10.json'

def load_rows(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    rows = []
    for img in data['images']:
        img_id = img['image_id']
        if img.get('error') or not img.get('evaluation'):
            rows.append({
                'image_id': img_id,
                'classes': '—',
                'iou': None,
                'precision': None,
                'recall': None,
                'note': img.get('note', img.get('error', 'no detections')),
            })
            continue
        ev = img['evaluation']
        overall = ev['overall']
        classes = ', '.join(sorted({o['class_name'] for o in ev['per_object'] if o['iou'] is not None}))
        rows.append({
            'image_id':  img_id,
            'classes':   classes if classes else '—',
            'iou':       overall['mean_iou'],
            'precision': overall['mean_precision'],
            'recall':    overall['mean_recall'],
            'note':      '',
        })
    return rows

orig_rows = load_rows(ORIG_FILE)
new_rows  = load_rows(NEW_FILE)

# Print combined table
header = f"{'ID':>4}  {'Classes':<20}  {'IoU':>7}  {'Precision':>9}  {'Recall':>7}  Note"
sep    = '─' * 72

print('\nTable: YOLO+SAM per-image evaluation across 22 ICG images')
print(sep)
print('── Original 12-image study ──')
print(header)
print(sep)

valid_orig = []
for r in orig_rows:
    iou_s = f'{r["iou"]:.4f}' if r['iou'] is not None else '    —   '
    pre_s = f'{r["precision"]:.4f}' if r['precision'] is not None else '    —    '
    rec_s = f'{r["recall"]:.4f}'    if r['recall']    is not None else '    —   '
    print(f'{r["image_id"]:>4}  {r["classes"]:<20}  {iou_s:>7}  {pre_s:>9}  {rec_s:>7}  {r["note"]}')
    if r['iou'] is not None:
        valid_orig.append(r)

print(sep)
if valid_orig:
    m_iou = sum(r['iou'] for r in valid_orig) / len(valid_orig)
    m_pre = sum(r['precision'] for r in valid_orig) / len(valid_orig)
    m_rec = sum(r['recall'] for r in valid_orig) / len(valid_orig)
    print(f'Mean ({len(valid_orig)} images with detections):  IoU={m_iou:.4f}  Precision={m_pre:.4f}  Recall={m_rec:.4f}')

print()
print('── Extended 10-image validation ──')
print(header)
print(sep)

valid_new = []
for r in new_rows:
    iou_s = f'{r["iou"]:.4f}' if r['iou'] is not None else '    —   '
    pre_s = f'{r["precision"]:.4f}' if r['precision'] is not None else '    —    '
    rec_s = f'{r["recall"]:.4f}'    if r['recall']    is not None else '    —   '
    print(f'{r["image_id"]:>4}  {r["classes"]:<20}  {iou_s:>7}  {pre_s:>9}  {rec_s:>7}  {r["note"]}')
    if r['iou'] is not None:
        valid_new.append(r)

print(sep)
if valid_new:
    m_iou = sum(r['iou'] for r in valid_new) / len(valid_new)
    m_pre = sum(r['precision'] for r in valid_new) / len(valid_new)
    m_rec = sum(r['recall'] for r in valid_new) / len(valid_new)
    print(f'Mean ({len(valid_new)} images with detections):  IoU={m_iou:.4f}  Precision={m_pre:.4f}  Recall={m_rec:.4f}')

# Combined
print()
all_valid = valid_orig + valid_new
if all_valid:
    m_iou_all = sum(r['iou'] for r in all_valid) / len(all_valid)
    m_pre_all = sum(r['precision'] for r in all_valid) / len(all_valid)
    m_rec_all = sum(r['recall'] for r in all_valid) / len(all_valid)
    print(sep)
    print(f'COMBINED ({len(all_valid)} images with detections out of 22):')
    print(f'  Mean IoU:       {m_iou_all:.4f}')
    print(f'  Mean Precision: {m_pre_all:.4f}')
    print(f'  Mean Recall:    {m_rec_all:.4f}')
    print(sep)
