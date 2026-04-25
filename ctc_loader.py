import argparse, glob, os
import numpy as np
import tifffile
from pipeline import run_pipeline


def load_ctc_sequence(folder, max_frames=None):
    tif_paths = sorted(glob.glob(os.path.join(folder, 't*.tif')))
    if not tif_paths:
        tif_paths = sorted(glob.glob(os.path.join(folder, '*.tif')))
    if not tif_paths:
        raise FileNotFoundError(f'No .tif files found in {folder}')

    if max_frames:
        tif_paths = tif_paths[:max_frames]

    frames = []
    for p in tif_paths:
        img = tifffile.imread(p)
        if img.ndim == 3:
            img = img[0] if img.shape[0] < img.shape[1] else img[:, :, 0]
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
        frames.append(img.astype(np.uint8))

    print(f'Loaded {len(frames)} frames from {folder}')
    return frames


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTC pipeline runner')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CTC image sequence folder')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max number of frames to process')
    parser.add_argument('--out_dir', type=str, default='results_ctc',
                        help='Output directory')
    args = parser.parse_args()

    frames = load_ctc_sequence(args.data_dir, max_frames=args.max_frames)
    run_pipeline(frames=frames, out_dir=args.out_dir)
