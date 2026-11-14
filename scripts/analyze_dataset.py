import argparse
import csv
import cv2
import glob
import hashlib
import numpy as np
import os
import random
from collections import Counter, defaultdict

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def dhash(gray, hash_size=8):
    """Difference hash: 64-bit perceptual hash, robust to small changes."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view('>u8')[0])


def analyze_image(img_path, args):
    with open(img_path, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {'path': img_path, 'error': 'unreadable'}

    channels = 1 if img.ndim == 2 else img.shape[2]
    dtype = str(img.dtype)
    if img.dtype != np.uint8:
        img = (img / np.iinfo(img.dtype).max * 255).astype(np.uint8)
    if channels == 1:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif channels == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        bgr = img
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # background (glass / empty slide) = bright and unsaturated pixels
    background = (hsv[..., 1] < args.bg_sat) & (hsv[..., 2] > args.bg_val)
    mean_b, mean_g, mean_r = bgr.reshape(-1, 3).mean(0)

    return {
        'path': img_path,
        'error': '',
        'h': img.shape[0],
        'w': img.shape[1],
        'channels': channels,
        'dtype': dtype,
        # variance of Laplacian: low value -> blurry / out-of-focus
        'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'bg_frac': float(background.mean()),
        'mean_r': float(mean_r),
        'mean_g': float(mean_g),
        'mean_b': float(mean_b),
        'mean_sat': float(hsv[..., 1].mean()),
        'md5': md5,
        'dhash': dhash(gray),
    }


def find_near_duplicates(records, max_dist):
    """Pairs of images whose dHash differs in <= max_dist bits."""
    hashes = np.array([r['dhash'] for r in records], dtype=np.uint64)
    popcount = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    pairs = []
    for i in range(len(hashes) - 1):
        xor = np.bitwise_xor(hashes[i + 1:], hashes[i])
        dist = popcount[xor.view(np.uint8)].reshape(-1, 8).sum(1)
        for j in np.nonzero(dist <= max_dist)[0]:
            pairs.append((records[i]['path'], records[i + 1 + j]['path'], int(dist[j])))
    return pairs


def make_contact_sheet(records, save_path, title, tile=128, cols=8):
    if not records:
        return
    rows = (len(records) + cols - 1) // cols
    sheet = np.full((rows * (tile + 20) + 30, cols * tile, 3), 255, np.uint8)
    cv2.putText(sheet, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    for idx, rec in enumerate(records):
        img = cv2.imread(rec['path'], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        y, x = 30 + (idx // cols) * (tile + 20), (idx % cols) * tile
        sheet[y:y + tile, x:x + tile] = img
        label = f'{idx}: s={rec["sharpness"]:.0f} bg={rec["bg_frac"]:.2f}'
        cv2.putText(sheet, label, (x + 2, y + tile + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, sheet)
    # index -> file name, so that images on the sheet can be found
    with open(os.path.splitext(save_path)[0] + '.txt', 'w') as f:
        for idx, rec in enumerate(records):
            f.write(f'{idx}\t{rec["path"]}\n')


def describe(values):
    p = np.percentile(values, [0, 5, 25, 50, 75, 95, 100])
    return ' '.join(f'{name}={v:.1f}' for name, v in zip(['min', 'p5', 'p25', 'med', 'p75', 'p95', 'max'], p))


def main(args):
    img_paths = sorted(
        p for p in glob.glob(os.path.join(args.input, '**', '*'), recursive=True) if p.lower().endswith(IMG_EXTS))
    if args.max_images:
        random.Random(args.seed).shuffle(img_paths)
        img_paths = sorted(img_paths[:args.max_images])
    print(f'Found {len(img_paths)} images in {args.input}')

    records, errors = [], []
    for i, img_path in enumerate(img_paths):
        rec = analyze_image(img_path, args)
        (errors if rec['error'] else records).append(rec)
        if (i + 1) % 500 == 0:
            print(f'  processed {i + 1}/{len(img_paths)}')
    for rec in errors:
        rec['path'] = os.path.relpath(rec['path'], args.input)
    for rec in records:
        rec['group'] = os.path.dirname(os.path.relpath(rec['path'], args.input))
        rec['path'] = os.path.relpath(rec['path'], args.input)
    abs_path = {rec['path']: os.path.join(args.input, rec['path']) for rec in records}

    # ---------------- per-image table ----------------
    os.makedirs(args.output, exist_ok=True)
    fields = ['path', 'group', 'h', 'w', 'channels', 'dtype', 'sharpness', 'bg_frac', 'mean_r', 'mean_g', 'mean_b',
              'mean_sat', 'md5', 'dhash']
    with open(os.path.join(args.output, 'stats.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    # ---------------- summary ----------------
    sharp = np.array([r['sharpness'] for r in records])
    blur_thr = np.percentile(sharp, args.blur_percentile)
    blurry = [r for r in records if r['sharpness'] <= blur_thr]
    empty = [r for r in records if r['bg_frac'] >= args.max_bg_frac]

    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r['md5']].append(r['path'])
    exact_dups = [paths for paths in by_md5.values() if len(paths) > 1]
    near_dups = find_near_duplicates(records, args.dhash_dist)

    lines = [
        f'Images: {len(records)} readable, {len(errors)} unreadable',
        f'Sizes (HxW): {dict(Counter((r["h"], r["w"]) for r in records))}',
        f'Channels: {dict(Counter(r["channels"] for r in records))}',
        f'Dtypes: {dict(Counter(r["dtype"] for r in records))}',
        '',
        f'Sharpness (Laplacian var): {describe(sharp)}',
        f'Background fraction:       {describe([r["bg_frac"] * 100 for r in records])} (%)',
        f'Mean saturation:           {describe([r["mean_sat"] for r in records])}',
        '',
        f'Blurry (sharpness <= p{args.blur_percentile} = {blur_thr:.1f}): {len(blurry)}',
        f'Mostly background (bg_frac >= {args.max_bg_frac}): {len(empty)}',
        f'Exact duplicates (md5): {len(exact_dups)} groups',
        f'Near duplicates (dHash dist <= {args.dhash_dist}): {len(near_dups)} pairs',
        '',
        'Per group:',
    ]
    groups = defaultdict(list)
    for r in records:
        groups[r['group']].append(r)
    for name, recs in sorted(groups.items()):
        rgb = np.mean([[r['mean_r'], r['mean_g'], r['mean_b']] for r in recs], 0)
        lines.append(f'  {name or "."}: n={len(recs)}  sharp_med={np.median([r["sharpness"] for r in recs]):.1f}  '
                     f'bg_med={np.median([r["bg_frac"] for r in recs]):.3f}  '
                     f'meanRGB=({rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f})')
    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(os.path.join(args.output, 'summary.txt'), 'w') as f:
        f.write(summary + '\n')

    # ---------------- lists of suspicious files ----------------
    with open(os.path.join(args.output, 'errors.txt'), 'w') as f:
        f.writelines(f'{r["path"]}\n' for r in errors)
    with open(os.path.join(args.output, 'duplicates.txt'), 'w') as f:
        for paths in exact_dups:
            f.write('exact\t' + '\t'.join(paths) + '\n')
        for a, b, d in near_dups:
            f.write(f'near(d={d})\t{a}\t{b}\n')
    with open(os.path.join(args.output, 'suspicious.txt'), 'w') as f:
        for r in sorted(blurry, key=lambda r: r['sharpness']):
            f.write(f'blurry\t{r["sharpness"]:.1f}\t{r["path"]}\n')
        for r in sorted(empty, key=lambda r: -r['bg_frac']):
            f.write(f'background\t{r["bg_frac"]:.3f}\t{r["path"]}\n')

    # ---------------- contact sheets ----------------
    def with_abs(recs):
        return [dict(r, path=abs_path[r['path']]) for r in recs]

    n = args.sheet_size
    rng = random.Random(args.seed)
    make_contact_sheet(
        with_abs(rng.sample(records, min(n, len(records)))), os.path.join(args.output, 'sheet_random.png'),
        'random')
    by_sharp = sorted(records, key=lambda r: r['sharpness'])
    make_contact_sheet(with_abs(by_sharp[:n]), os.path.join(args.output, 'sheet_blurriest.png'), 'blurriest')
    make_contact_sheet(with_abs(by_sharp[-n:][::-1]), os.path.join(args.output, 'sheet_sharpest.png'), 'sharpest')
    make_contact_sheet(
        with_abs(sorted(records, key=lambda r: -r['bg_frac'])[:n]), os.path.join(args.output, 'sheet_background.png'),
        'most background')
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    """Analyze an image dataset before training: sizes, channels, sharpness, background fraction,
    color statistics and duplicates. Saves a summary, per-image csv and contact sheets.

    Example:
        python scripts/analyze_dataset.py --input datasets/bc_multiset/raw/BC-MultiSet-50 \
            --output datasets/bc_multiset/analysis
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Dataset folder (searched recursively)')
    parser.add_argument('--output', type=str, required=True, help='Folder for the analysis results')
    parser.add_argument('--max_images', type=int, default=0, help='Analyze a random subset (0: all images)')
    parser.add_argument('--blur_percentile', type=float, default=5, help='Flag images below this sharpness percentile')
    parser.add_argument('--max_bg_frac', type=float, default=0.5, help='Flag images with more background than this')
    parser.add_argument('--bg_sat', type=int, default=20, help='Background pixel: HSV saturation below this')
    parser.add_argument('--bg_val', type=int, default=200, help='Background pixel: HSV value above this')
    parser.add_argument('--dhash_dist', type=int, default=2, help='Max dHash Hamming distance for near duplicates')
    parser.add_argument('--sheet_size', type=int, default=48, help='Number of images per contact sheet')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    main(args)
