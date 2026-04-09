import os
import gc
import glob
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from monai.networks.nets import UNet, AttentionUnet

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

MODEL_NAMES = {
    "3D U-Net": "best_unet.pth",
    "3D Residual U-Net": "best_resunet.pth",
    "3D Attention U-Net": "best_attn_unet.pth",
}

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
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    return files


def load_scan(path):
    d = np.load(path, allow_pickle=True)
    vol = d["vol"].astype(np.float32)
    ctype = str(d["ctype"])
    return vol, ctype


def run_inference(model, volume):
    inp = torch.from_numpy(volume[None, None]).float()
    with torch.no_grad():
        out = torch.sigmoid(model(inp))
    pred = out.cpu().numpy()[0, 0]
    del inp, out
    mask = (pred >= 0.5).astype(np.float32)
    return mask


def render_slices(vol_slice, mask_slice):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(vol_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("CT Slice")
    axes[0].axis("off")

    axes[1].imshow(vol_slice, cmap="gray", vmin=0, vmax=1)
    overlay = np.ma.masked_where(mask_slice < 0.5, mask_slice)
    axes[1].imshow(overlay, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Predicted Corruption Mask")
    axes[1].axis("off")

    plt.tight_layout()
    return fig


st.set_page_config(page_title="ACD-LR: CT Corruption Segmentation", layout="wide")

st.title("ACD-LR: CT Corruption Segmentation")
st.markdown("Segment synthetic corruptions in 3D CT scans using deep learning models trained on the LUNA16 dataset.")

st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox("Select Model", list(MODEL_NAMES.keys()))

scan_files = get_scan_list()
if not scan_files:
    st.error(f"No .npz scan files found in {DATA_DIR}. Check that the data/ directory contains sample scans.")
    st.stop()
scan_names = [os.path.basename(f) for f in scan_files]
scan_pick = st.sidebar.selectbox("Select Scan", scan_names)

run_button = st.sidebar.button("Run Inference")

if "pred_mask" not in st.session_state:
    st.session_state.pred_mask = None
if "current_vol" not in st.session_state:
    st.session_state.current_vol = None
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None
if "corruption_type" not in st.session_state:
    st.session_state.corruption_type = None

if run_button:
    scan_path = scan_files[scan_names.index(scan_pick)]
    vol, ctype = load_scan(scan_path)

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
    except FileNotFoundError as e:
        st.error(str(e))

if st.session_state.current_vol is not None and st.session_state.pred_mask is not None:
    vol = st.session_state.current_vol
    mask = st.session_state.pred_mask
    depth = vol.shape[0]

    st.markdown(
        f"**Scan:** {st.session_state.last_scan} &nbsp; | &nbsp; "
        f"**Model:** {st.session_state.last_model} &nbsp; | &nbsp; "
        f"**Corruption:** {st.session_state.corruption_type}"
    )

    z = st.slider("Z-axis slice", 0, depth - 1, depth // 2)

    fig = render_slices(vol[z], mask[z])
    st.pyplot(fig)
    plt.close(fig)

    pct = (mask.sum() / mask.size) * 100
    st.sidebar.metric("Corrupted Voxels", f"{pct:.2f}%")
else:
    st.info("Select a model and scan from the sidebar, then click **Run Inference** to get started.")
