import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import (
    filters, measure, morphology, segmentation,
    exposure, feature, draw
)
from skimage.segmentation import watershed
import os, warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATOR
# ──────────────────────────────────────────────

def generate_synthetic_sequence(n_frames=8, height=256, width=256,
                                 n_cells=12, seed=42):
    rng = np.random.default_rng(seed)
    frames = []

    centres = rng.uniform(30, min(height, width) - 30, size=(n_cells, 2))
    radii   = rng.uniform(10, 20, size=n_cells)
    vels    = rng.uniform(-2.5, 2.5, size=(n_cells, 2))   
    growth  = rng.uniform(0.0, 0.4, size=n_cells)          
    intensities = rng.uniform(180, 255, size=n_cells)

    for t in range(n_frames):
        img = np.zeros((height, width), dtype=np.float32)

        for i in range(len(centres)):
            cy, cx = centres[i]
            r  = radii[i] + growth[i] * t
            Y, X = np.ogrid[:height, :width]
            sigma = r / 2.2
            blob  = intensities[i] * np.exp(-((Y - cy)**2 + (X - cx)**2) / (2 * sigma**2))
            img  += blob

        img   = np.clip(img, 0, 255)
        noise = rng.poisson(img + 5).astype(np.float32)
        bg    = rng.normal(10, 3, size=(height, width)).astype(np.float32)
        frame = np.clip(noise + bg, 0, 255).astype(np.uint8)
        frames.append(frame)

        centres += vels
        for i in range(len(centres)):
            for ax in range(2):
                limit = height if ax == 0 else width
                if centres[i, ax] < 20 or centres[i, ax] > limit - 20:
                    vels[i, ax] *= -1

    return frames, centres, radii


# ──────────────────────────────────────────────
# 2. DETECTION & SEGMENTATION
# ──────────────────────────────────────────────

def segment_frame(frame, min_cell_area=80):
    enhanced = exposure.equalize_adapthist(frame, clip_limit=0.03)
    smooth = filters.gaussian(enhanced, sigma=2)
    thresh  = filters.threshold_otsu(smooth)
    binary  = smooth > thresh
    cleaned = morphology.remove_small_objects(binary, min_size=min_cell_area)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(3))
    cleaned = ndimage.binary_fill_holes(cleaned)
    distance = ndimage.distance_transform_edt(cleaned)
    smoothed_dist = filters.gaussian(distance, sigma=2)

    local_max = feature.peak_local_max(
        smoothed_dist, min_distance=12,
        labels=cleaned, exclude_border=False
    )
    marker_image = np.zeros_like(cleaned, dtype=bool)
    marker_image[tuple(local_max.T)] = True
    markers, _ = ndimage.label(marker_image)

    labels = watershed(-smoothed_dist, markers, mask=cleaned)
    return labels


def extract_cell_props(labels, frame_idx, intensity_image=None):
    props = measure.regionprops(labels, intensity_image=intensity_image)
    records = []
    for p in props:
        if p.area < 50:
            continue
        rec = {
            'frame':      frame_idx,
            'cell_id':    p.label,
            'centroid_y': p.centroid[0],
            'centroid_x': p.centroid[1],
            'area':       p.area,
            'perimeter':  p.perimeter,
            'eccentricity': p.eccentricity,
        }
        if intensity_image is not None:
            rec['mean_intensity'] = p.mean_intensity
        else:
            rec['mean_intensity'] = 0.0
        records.append(rec)
    return records


# ──────────────────────────────────────────────
# 3. TRACKING  
# ──────────────────────────────────────────────

def track_cells(all_props, max_dist=35):
    df = pd.DataFrame(all_props)
    df['track_id'] = -1

    next_track = 1
    prev_frame_data = None   

    for t in sorted(df['frame'].unique()):
        curr = df[df['frame'] == t].copy()
        curr_idx = curr.index.tolist()

        if prev_frame_data is None or len(prev_frame_data) == 0:
            for idx in curr_idx:
                df.at[idx, 'track_id'] = next_track
                next_track += 1
        else:
            prev_ids   = list(prev_frame_data.keys())
            prev_cents = np.array([prev_frame_data[tid] for tid in prev_ids])
            curr_cents = curr[['centroid_y', 'centroid_x']].values

            if len(prev_cents) and len(curr_cents):
                D = cdist(curr_cents, prev_cents)
                used_prev = set()
                assigned  = {}

                order = np.argsort(D.min(axis=1))
                for ci in order:
                    pi = np.argmin(D[ci])
                    if D[ci, pi] < max_dist and pi not in used_prev:
                        assigned[curr_idx[ci]] = prev_ids[pi]
                        used_prev.add(pi)

                for idx in curr_idx:
                    if idx in assigned:
                        df.at[idx, 'track_id'] = assigned[idx]
                    else:
                        df.at[idx, 'track_id'] = next_track
                        next_track += 1
            else:
                for idx in curr_idx:
                    df.at[idx, 'track_id'] = next_track
                    next_track += 1

        prev_frame_data = {
            int(df.at[idx, 'track_id']): (df.at[idx, 'centroid_y'], df.at[idx, 'centroid_x'])
            for idx in curr_idx
        }

    return df


# ──────────────────────────────────────────────
# 4. FEATURE EXTRACTION
# ──────────────────────────────────────────────

def extract_track_features(df):
    records = []
    for tid, grp in df.groupby('track_id'):
        grp = grp.sort_values('frame')
        if len(grp) < 2:
            continue

        areas = grp['area'].values
        frames_arr = grp['frame'].values
        cy = grp['centroid_y'].values
        cx = grp['centroid_x'].values

        mean_size = float(np.mean(areas))

        if len(frames_arr) >= 2:
            coeffs = np.polyfit(frames_arr, areas, 1)
            growth_rate = float(coeffs[0])   
        else:
            growth_rate = 0.0

        dy = np.diff(cy)
        dx = np.diff(cx)
        displacements = np.sqrt(dy**2 + dx**2)
        mean_speed    = float(np.mean(displacements))
        total_path    = float(np.sum(displacements))
        net_disp      = float(np.sqrt((cy[-1]-cy[0])**2 + (cx[-1]-cx[0])**2))

        records.append({
            'track_id':       tid,
            'n_frames':       len(grp),
            'mean_area_px':   round(mean_size, 1),
            'growth_rate_px2_per_frame': round(growth_rate, 2),
            'mean_speed_px_per_frame':  round(mean_speed, 2),
            'total_path_px':  round(total_path, 1),
            'net_displacement_px': round(net_disp, 1),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# 5. VISUALISATION
# ──────────────────────────────────────────────

def make_color_lut(n=512):
    rng = np.random.default_rng(7)
    colors = rng.uniform(0.2, 1.0, size=(n, 3))
    colors[0] = [0, 0, 0]
    return ListedColormap(colors)


COLOR_LUT = make_color_lut()


def visualize_pipeline(frames, all_labels, track_df, features_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    n = min(len(frames), 8)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                             facecolor='#111')
    axes = axes.ravel()
    for i in range(n):
        ax = axes[i]
        ax.imshow(frames[i], cmap='gray', vmin=0, vmax=255)
        ax.imshow(all_labels[i], cmap=COLOR_LUT, alpha=0.45,
                  vmin=0, vmax=511, interpolation='nearest')
        ax.set_title(f'Frame {i}', color='white', fontsize=9)
        ax.axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Cell Segmentation – All Frames', color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'A_segmentation_grid.png'),
                dpi=130, bbox_inches='tight', facecolor='#111')
    plt.close(fig)
    print('  Saved A_segmentation_grid.png')

    t_last = max(track_df['frame'])
    last_frame = frames[t_last]
    fig, ax = plt.subplots(figsize=(7, 7), facecolor='#111')
    ax.imshow(last_frame, cmap='gray', vmin=0, vmax=255)
    ax.set_facecolor('#111')

    track_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for tid, grp in track_df.groupby('track_id'):
        grp = grp.sort_values('frame')
        xs, ys = grp['centroid_x'].values, grp['centroid_y'].values
        c = track_colors[int(tid) % 20]
        ax.plot(xs, ys, '-o', color=c, linewidth=1.2, markersize=3, alpha=0.85)
        ax.text(xs[-1]+2, ys[-1]-2, str(int(tid)), color=c,
                fontsize=6, fontweight='bold')

    ax.set_title('Cell Tracks (trajectories)', color='white', fontsize=12)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'B_cell_tracks.png'),
                dpi=130, bbox_inches='tight', facecolor='#111')
    plt.close(fig)
    print('  Saved B_cell_tracks.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor='#1a1a2e')
    feature_defs = [
        ('mean_area_px',             'Cell Size (px²)',              '#4ecdc4'),
        ('growth_rate_px2_per_frame','Growth Rate (px²/frame)',      '#ff6b6b'),
        ('mean_speed_px_per_frame',  'Mean Speed (px/frame)',        '#ffe66d'),
    ]
    for ax, (col, title, color) in zip(axes, feature_defs):
        ax.set_facecolor('#16213e')
        vals = features_df[col].dropna()
        ax.hist(vals, bins=max(5, len(vals)//2), color=color, edgecolor='#fff',
                linewidth=0.6, alpha=0.9)
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xlabel(col, color='#aaa', fontsize=8)
        ax.set_ylabel('# Tracks', color='#aaa', fontsize=8)
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
    fig.suptitle('Extracted Cell Features', color='white', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'C_feature_histograms.png'),
                dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print('  Saved C_feature_histograms.png')

    top_tracks = (features_df.nlargest(8, 'n_frames')['track_id'].tolist())
    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#111')
    ax.set_facecolor('#1a1a1a')
    for tid in top_tracks:
        grp = track_df[track_df['track_id'] == tid].sort_values('frame')
        c = track_colors[int(tid) % 20]
        ax.plot(grp['frame'], grp['area'], '-o', label=f'Track {int(tid)}',
                color=c, linewidth=1.6, markersize=4)
    ax.set_title('Cell Area over Time (top tracks)', color='white', fontsize=12)
    ax.set_xlabel('Frame', color='#aaa')
    ax.set_ylabel('Area (px²)', color='#aaa')
    ax.tick_params(colors='#aaa')
    ax.legend(fontsize=7, facecolor='#333', labelcolor='white',
              ncol=2, framealpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'D_area_over_time.png'),
                dpi=130, bbox_inches='tight', facecolor='#111')
    plt.close(fig)
    print('  Saved D_area_over_time.png')

    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#111')
    ax.set_facecolor('#1a1a1a')
    sc = ax.scatter(features_df['mean_speed_px_per_frame'],
                    features_df['net_displacement_px'],
                    c=features_df['mean_area_px'],
                    cmap='plasma', s=60, edgecolors='white', linewidths=0.4)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Mean Cell Size (px²)', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    ax.set_xlabel('Mean Speed (px/frame)', color='#aaa')
    ax.set_ylabel('Net Displacement (px)', color='#aaa')
    ax.set_title('Speed vs Net Displacement', color='white', fontsize=12)
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'E_speed_vs_displacement.png'),
                dpi=130, bbox_inches='tight', facecolor='#111')
    plt.close(fig)
    print('  Saved E_speed_vs_displacement.png')


# ──────────────────────────────────────────────
# 6. Running 
# ──────────────────────────────────────────────

def run_pipeline(frames=None, out_dir='results'):
    print('\n=== Cell Tracking Pipeline ===\n')

    if frames is None:
        print('[1/4] Generating synthetic microscopy sequence …')
        frames, _, _ = generate_synthetic_sequence(n_frames=8, n_cells=14)
    else:
        print(f'[1/4] Using provided sequence ({len(frames)} frames)')

    print('[2/4] Segmenting cells …')
    all_labels = []
    all_props  = []
    for t, frame in enumerate(frames):
        labels = segment_frame(frame)
        all_labels.append(labels)
        props  = extract_cell_props(labels, frame_idx=t, intensity_image=frame.astype(float))
        all_props.extend(props)
        n_cells = len(props)
        print(f'   Frame {t:02d}: {n_cells} cells detected')

    print('[3/4] Tracking cells across frames …')
    track_df = track_cells(all_props)
    n_tracks = track_df['track_id'].nunique()
    print(f'   {n_tracks} unique tracks identified')

    print('[4/4] Extracting per-track features …')
    features_df = extract_track_features(track_df)
    print(f'   Features extracted for {len(features_df)} tracks\n')

    print('─── Feature Summary ───')
    print(features_df.describe().round(2).to_string())

    print('\n─── Generating Visualisations ───')
    visualize_pipeline(frames, all_labels, track_df, features_df, out_dir)

    track_df.to_csv(os.path.join(out_dir, 'track_data.csv'), index=False)
    features_df.to_csv(os.path.join(out_dir, 'cell_features.csv'), index=False)
    print(f'\nCSV results saved to {out_dir}/')

    return track_df, features_df


if __name__ == '__main__':
    # Write your path here
    run_pipeline(out_dir='/path/results')
