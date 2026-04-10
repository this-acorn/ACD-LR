import os
import gc
import glob
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from monai.networks.nets import UNet, AttentionUnet

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODEL_NAMES = {
    "3D U-Net": "best_unet.pth",
    "3D Residual U-Net": "best_resunet.pth",
    "3D Attention U-Net": "best_attn_unet.pth",
}

# ── Corruption parameters (same as training notebook) ──────────────
SLICE_FRAC = (0.15, 0.35)
SHIFT_PX = (20, 35)
BLUR_KERNEL = (3, 9)
NOISE_STD = (30.0, 80.0)
HU_MIN, HU_MAX = -1000, 400
DOWNSAMPLE_RANGE = (0.3, 0.6)

CORRUPTION_OPTIONS = {
    "Anatomical Cropping": "crop",
    "Motion Blur": "blur",
    "Rigid Shift": "shift",
    "Slice Removal": "slicing",
    "Gaussian Noise": "noise",
    "Downsampling": "downsample",
}

CORRUPTION_DESC = {
    "clean": "None (clean scan)",
    "noise": "Gaussian noise",
    "crop": "Anatomical cropping",
    "blur": "Motion blur",
    "shift": "Rigid shift",
    "slicing": "Slice removal",
    "downsample": "Downsampling",
    "combo_2": "2-corruption combo",
    "combo_3": "3-corruption combo",
}


# ── Corruption functions ───────────────────────────────────────────
def resize_vol(vol, target, order=1):
    return zoom(vol, [t / s for t, s in zip(target, vol.shape)], order=order).astype(np.float32)


def corrupt_crop(vol, mask, rng=None):
    rng = rng or np.random.default_rng()
    d = vol.shape[0]
    out = vol.copy()
    n = max(1, int(d * rng.uniform(*SLICE_FRAC)))
    mode = rng.choice(["top", "bottom", "both", "center", "scattered"])
    if mode == "top":
        out[:n] = 0
    elif mode == "bottom":
        out[-n:] = 0
    elif mode == "both":
        t, b = n // 2, n - n // 2
        out[:t] = 0; out[-b:] = 0
    elif mode == "center":
        lo = int(d * 0.25)
        hi = max(lo, int(d * 0.75) - n)
        s = rng.integers(lo, hi + 1)
        out[s:s + n] = 0
    else:
        idx = rng.choice(d, size=n, replace=False)
        out[idx] = 0
    cmap = (np.abs(vol - out) > 1e-6).astype(np.float32)
    return out, cmap


def corrupt_blur(vol, mask, rng=None):
    rng = rng or np.random.default_rng()
    ks = rng.choice(range(BLUR_KERNEL[0], BLUR_KERNEL[1] + 1, 2))
    axis = rng.choice(["axial", "coronal", "sagittal"])
    sigma = {"axial": (ks / 6, 0.5, 0.5), "coronal": (0.5, ks / 6, 0.5), "sagittal": (0.5, 0.5, ks / 6)}[axis]
    blurred = gaussian_filter(vol, sigma=sigma, mode="nearest").astype(np.float32)
    diff = np.abs(vol - blurred)
    cmap = (diff > diff.mean() + diff.std()).astype(np.float32)
    return blurred, cmap


def corrupt_shift(vol, mask, rng=None):
    rng = rng or np.random.default_rng()
    dx = int(rng.integers(*SHIFT_PX)) * rng.choice([-1, 1])
    dy = int(rng.integers(*SHIFT_PX)) * rng.choice([-1, 1])
    shifted = np.stack([np.roll(vol[z], (dy, dx), axis=(0, 1)) for z in range(vol.shape[0])])
    diff = np.abs(vol - shifted)
    cmap = (diff > diff.mean() + diff.std()).astype(np.float32)
    return shifted, cmap


def corrupt_slicing(vol, mask, rng=None):
    rng = rng or np.random.default_rng()
    d = vol.shape[0]
    n = max(1, min(int(d * rng.uniform(*SLICE_FRAC)), d - 1))
    mode = rng.choice(["top", "bottom", "both", "middle", "scattered"])
    keep = np.ones(d, dtype=bool)
    if mode == "top":
        keep[:n] = False
    elif mode == "bottom":
        keep[-n:] = False
    elif mode == "both":
        t = n // 2; keep[:t] = False; keep[-(n - t):] = False
    elif mode == "middle":
        lo = int(d * 0.25)
        hi = max(lo, int(d * 0.75) - n)
        s = rng.integers(lo, hi + 1)
        keep[s:s + n] = False
    else:
        keep[rng.choice(d, size=n, replace=False)] = False
    restored = resize_vol(vol[keep], vol.shape, order=1)
    diff = np.abs(vol - restored)
    cmap = (diff > diff.mean() + diff.std()).astype(np.float32)
    return restored, cmap


def corrupt_noise(vol, mask, rng=None):
    rng = rng or np.random.default_rng()
    std = rng.uniform(*NOISE_STD) / (HU_MAX - HU_MIN)
    noisy = np.clip(vol + rng.normal(0, std, vol.shape).astype(np.float32), 0, 1)
    diff = np.abs(vol - noisy)
    cmap = (diff > diff.mean() + diff.std()).astype(np.float32)
    return noisy, cmap


def corrupt_downsample(vol, mask, rng=None):
    rng = rng or np.random.default_rng()
    f = rng.uniform(*DOWNSAMPLE_RANGE)
    small_shape = tuple(max(1, int(s * f)) for s in vol.shape)
    small = zoom(vol, [s / o for s, o in zip(small_shape, vol.shape)], order=1)
    back = zoom(small, [o / s for o, s in zip(vol.shape, small.shape)], order=1)
    if back.shape != vol.shape:
        back = back[tuple(slice(0, s) for s in vol.shape)]
    back = back.astype(np.float32)
    diff = np.abs(vol - back)
    cmap = (diff > diff.mean() + diff.std()).astype(np.float32)
    return back, cmap


DISPATCH = {
    "crop": corrupt_crop,
    "blur": corrupt_blur,
    "shift": corrupt_shift,
    "slicing": corrupt_slicing,
    "noise": corrupt_noise,
    "downsample": corrupt_downsample,
}


def apply_corruptions(vol, corruption_keys, seed=None):
    rng = np.random.default_rng(seed)
    mask = np.ones_like(vol)
    combined = np.zeros_like(vol, dtype=np.float32)
    corruption_keys = sorted(corruption_keys, key=lambda x: 0 if x in ("crop", "slicing") else 1)
    for key in corruption_keys:
        vol, cmap = DISPATCH[key](vol, mask, rng=rng)
        combined = np.maximum(combined, cmap)
    return vol, combined


# ── Model functions ────────────────────────────────────────────────
def make_model(name):
    if name == "3D U-Net":
        return UNet(spatial_dims=3, in_channels=1, out_channels=1,
                    channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2)
    elif name == "3D Residual U-Net":
        return UNet(spatial_dims=3, in_channels=1, out_channels=1,
                    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=3)
    elif name == "3D Attention U-Net":
        return AttentionUnet(spatial_dims=3, in_channels=1, out_channels=1,
                             channels=(32, 64, 128, 256), strides=(2, 2, 2))


def load_model(name):
    model = make_model(name)
    weights_path = os.path.join(MODEL_DIR, MODEL_NAMES[name])
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            f"Download them from the Google Drive link in the README and place "
            f"the .pth files into the src/models/ directory."
        )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_scan_list():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))


def load_scan(path):
    d = np.load(path, allow_pickle=True)
    vol = d["vol"].astype(np.float32)
    ctype = str(d["ctype"])
    gt_mask = d["cmap"].astype(np.float32) if "cmap" in d else None
    return vol, ctype, gt_mask


def run_inference(model, volume):
    inp = torch.from_numpy(volume[None, None]).float()
    with torch.no_grad():
        out = torch.sigmoid(model(inp))
    pred = out.cpu().numpy()[0, 0]
    del inp, out
    return (pred >= 0.5).astype(np.float32)


def analyze_scan(vol, mask):
    """Analyze predicted corruption mask and generate a quality report."""
    total_voxels = mask.size
    corrupted_voxels = mask.sum()
    pct = (corrupted_voxels / total_voxels) * 100

    # Status and confidence
    if pct < 0.5:
        status = "CLEAN"
        confidence = 100.0 - pct
    else:
        status = "CORRUPTED"
        confidence = min(99.9, 50 + pct)

    # ── Location analysis ──────────────────────────────────────────
    per_slice = mask.mean(axis=(1, 2))
    depth = len(per_slice)
    top_third = per_slice[:depth // 3].mean()
    mid_third = per_slice[depth // 3: 2 * depth // 3].mean()
    bot_third = per_slice[2 * depth // 3:].mean()

    regions = {"Upper slices (superior)": top_third,
               "Middle slices": mid_third,
               "Lower slices (inferior)": bot_third}

    if max(regions.values()) < 0.01:
        location = "No significant corruption detected"
    elif max(regions.values()) / max(sum(regions.values()), 1e-9) > 0.6:
        location = max(regions, key=regions.get)
    else:
        location = "Spread across the full volume"

    # ── Corruption type estimation ─────────────────────────────────
    # Signal 1: Zero slices in volume → crop
    zero_slices = int(np.sum(vol.max(axis=(1, 2)) < 1e-6))

    # Signal 2: Edge vs center mask density → shift
    h, w = mask.shape[1], mask.shape[2]
    edge_band = max(1, h // 6)
    edge_region = np.zeros((h, w), dtype=bool)
    edge_region[:edge_band, :] = True
    edge_region[-edge_band:, :] = True
    edge_region[:, :edge_band] = True
    edge_region[:, -edge_band:] = True
    edge_density = mask[:, edge_region].mean()
    center_density = mask[:, ~edge_region].mean()

    # Signal 3: Per-slice variance
    slice_std = per_slice.std()

    # Signal 4: Mask spatial granularity (noise = fine-grained, blur = smooth blobs)
    # Count transitions in mask (corrupted↔clean) per slice as a proxy
    h_transitions = np.abs(np.diff(mask, axis=2)).mean()  # horizontal transitions
    v_transitions = np.abs(np.diff(mask, axis=1)).mean()  # vertical transitions
    granularity = h_transitions + v_transitions

    # Signal 5: Slice-to-slice abrupt changes → slicing
    slice_diffs = np.abs(np.diff(per_slice))

    # ── Build corruption signatures ────────────────────────────────
    signatures = []

    # Crop: zeroed-out slices
    if zero_slices > depth * 0.05:
        signatures.append(("Anatomical cropping",
                           "Re-scan with extended coverage to include missing anatomy"))

    # Shift: corruption concentrated at image borders
    if pct > 0.5 and edge_density > 0.01 and edge_density > center_density * 1.8:
        signatures.append(("Rigid shift / positioning error",
                           "Re-scan with proper patient positioning and table alignment"))

    # Slicing: abrupt per-slice density jumps without zero slices
    if zero_slices <= depth * 0.05 and slice_diffs.max() > 0.25:
        signatures.append(("Slice removal / data loss",
                           "Re-scan or retrieve original data without lossy compression"))

    # Noise: uniform spread, fine-grained mask (many small transitions)
    if pct > 0.3 and granularity > 0.08 and slice_std < 0.05:
        signatures.append(("Gaussian noise",
                           "Re-scan with calibrated detector or apply denoising filter"))

    # Blur: uniform spread, smooth/blob-like mask (fewer transitions than noise)
    if pct > 1.0 and granularity <= 0.08 and slice_std < 0.08 and edge_density <= center_density * 1.8:
        signatures.append(("Motion blur",
                           "Re-scan with patient immobilization to reduce motion artifacts"))

    # Downsample: similar to blur but typically higher affected %
    if pct > 5.0 and granularity <= 0.06 and slice_std < 0.05 and edge_density <= center_density * 1.5:
        signatures.append(("Downsampling / low resolution",
                           "Re-scan with higher resolution acquisition settings"))

    if not signatures:
        if pct < 0.5:
            suspected = "None detected"
            recommendation = "Scan appears acceptable for clinical use"
        else:
            suspected = "Unknown corruption pattern"
            recommendation = "Manual review recommended; consider re-acquisition"
    elif len(signatures) == 1:
        suspected, recommendation = signatures[0]
    else:
        suspected = " + ".join(s[0] for s in signatures)
        recommendation = signatures[0][1]

    return {
        "status": status,
        "confidence": confidence,
        "affected_pct": pct,
        "location": location,
        "suspected_type": suspected,
        "recommendation": recommendation,
    }


def render_slices(vol_slice, mask_slice, gt_slice=None):
    ncols = 3 if gt_slice is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(vol_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT Slice")
    axes[0].axis("off")

    axes[1].imshow(vol_slice, cmap="gray", vmin=0, vmax=1)
    overlay = np.ma.masked_where(mask_slice < 0.5, mask_slice)
    axes[1].imshow(overlay, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    if gt_slice is not None:
        axes[2].imshow(vol_slice, cmap="gray", vmin=0, vmax=1)
        gt_overlay = np.ma.masked_where(gt_slice < 0.5, gt_slice)
        axes[2].imshow(gt_overlay, cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

    plt.tight_layout()
    return fig


# ── Streamlit UI ───────────────────────────────────────────────────
st.set_page_config(page_title="ACD-LR: CT Corruption Segmentation", layout="wide")

st.title("ACD-LR: CT Corruption Segmentation")
st.markdown("Segment synthetic corruptions in 3D CT scans using deep learning models trained on the LUNA16 dataset.")

st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox("Select Model", list(MODEL_NAMES.keys()))

mode = st.sidebar.radio("Mode", ["Sample Scans", "Custom Corruption"])

if mode == "Sample Scans":
    scan_files = get_scan_list()
    if not scan_files:
        st.error(f"No .npz scan files found in {DATA_DIR}.")
        st.stop()
    scan_names = [os.path.basename(f) for f in scan_files]
    scan_pick = st.sidebar.selectbox("Select Scan", scan_names)
else:
    # Find clean scans for custom corruption
    scan_files = get_scan_list()
    clean_files = [f for f in scan_files if "clean" in os.path.basename(f)]
    if not clean_files:
        clean_files = scan_files
    clean_names = [os.path.basename(f) for f in clean_files]
    scan_pick = st.sidebar.selectbox("Select Clean Scan", clean_names)

    st.sidebar.markdown("**Select Corruptions:**")
    selected_corruptions = []
    for label, key in CORRUPTION_OPTIONS.items():
        if st.sidebar.checkbox(label, key=f"chk_{key}"):
            selected_corruptions.append(key)

    random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)

run_button = st.sidebar.button("Run Inference")

# Session state
for key in ["pred_mask", "current_vol", "last_scan", "last_model", "corruption_type", "gt_mask", "run_mode", "known_corruptions"]:
    if key not in st.session_state:
        st.session_state[key] = None

if run_button:
    if mode == "Sample Scans":
        scan_path = scan_files[scan_names.index(scan_pick)]
        vol, ctype, gt_mask = load_scan(scan_path)
    else:
        if not selected_corruptions:
            st.warning("Select at least one corruption type.")
            st.stop()
        scan_path = clean_files[clean_names.index(scan_pick)]
        clean_vol, _, _ = load_scan(scan_path)
        with st.spinner("Applying corruptions..."):
            vol, gt_mask = apply_corruptions(clean_vol.copy(), selected_corruptions, seed=random_seed)
        ctype = " + ".join(CORRUPTION_DESC.get(c, c) for c in selected_corruptions)

    try:
        if "loaded_model_name" in st.session_state and st.session_state.loaded_model_name != model_choice:
            del st.session_state.loaded_model
            gc.collect()

        if "loaded_model" not in st.session_state or st.session_state.loaded_model_name != model_choice:
            with st.spinner(f"Loading {model_choice}..."):
                st.session_state.loaded_model = load_model(model_choice)
                st.session_state.loaded_model_name = model_choice

        model = st.session_state.loaded_model

        with st.spinner("Running inference..."):
            mask = run_inference(model, vol)

        st.session_state.pred_mask = mask
        st.session_state.current_vol = vol
        st.session_state.last_scan = scan_pick
        st.session_state.last_model = model_choice
        st.session_state.corruption_type = ctype
        st.session_state.gt_mask = gt_mask
        st.session_state.run_mode = mode
        st.session_state.known_corruptions = selected_corruptions if mode == "Custom Corruption" else None
    except FileNotFoundError as e:
        st.error(str(e))

if st.session_state.current_vol is not None and st.session_state.pred_mask is not None:
    vol = st.session_state.current_vol
    mask = st.session_state.pred_mask
    depth = vol.shape[0]

    ctype = st.session_state.corruption_type
    ctype_label = CORRUPTION_DESC.get(ctype, ctype)
    st.markdown(
        f"**Scan:** {st.session_state.last_scan} &nbsp; | &nbsp; "
        f"**Model:** {st.session_state.last_model} &nbsp; | &nbsp; "
        f"**Corruption:** {ctype_label}"
    )

    left_col, right_col = st.columns([3, 1])

    with left_col:
        z = st.slider("Z-axis slice", 0, depth - 1, depth // 2)

        gt = st.session_state.gt_mask
        fig = render_slices(vol[z], mask[z], gt[z] if gt is not None else None)
        st.pyplot(fig)
        plt.close(fig)

    with right_col:
        st.subheader("Scan Quality Report")

        report = analyze_scan(vol, mask)

        status_color = "red" if report["status"] == "CORRUPTED" else "green"
        st.markdown(f"**Status:** :{status_color}[{report['status']}]")
        st.markdown(f"**Confidence:** {report['confidence']:.1f}%")
        st.markdown(f"**Affected Region:** {report['affected_pct']:.1f}%")
        st.markdown(f"**Location:** {report['location']}")
        st.markdown(f"**Detected Type:** {report['suspected_type']}")
        st.markdown(f"**Recommendation:** {report['recommendation']}")

        if st.session_state.run_mode == "Custom Corruption" and st.session_state.known_corruptions:
            st.markdown("---")
            applied = " + ".join(CORRUPTION_DESC.get(c, c) for c in st.session_state.known_corruptions)
            st.markdown(f"**Applied:** {applied}")
            st.markdown(f"**Detected:** {report['suspected_type']} *(rule-based pattern analysis)*")

    pct = (mask.sum() / mask.size) * 100
    st.sidebar.metric("Corrupted Voxels", f"{pct:.2f}%")
    gt = st.session_state.gt_mask
    if gt is not None:
        gt_pct = (gt.sum() / gt.size) * 100
        st.sidebar.metric("Ground Truth", f"{gt_pct:.2f}%")
else:
    st.info("Select a model and scan from the sidebar, then click **Run Inference** to get started.")
