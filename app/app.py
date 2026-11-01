import io, os
from pathlib import Path
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import load_student   # same architecture as train.py

st.set_page_config(page_title="Waste Classifier", page_icon="♻️", layout="centered")

# Student checkpoints (class names are stored INSIDE each file).
# Relative paths so it works on Streamlit Cloud, Hugging Face and locally.
WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS = {
    "GD (10 classes)": str(WEIGHTS_DIR / "student_gd_s42.pth"),
    "RealWaste (9 classes)": str(WEIGHTS_DIR / "student_realwaste_s42.pth"),
    "HGCD (12 classes)": str(WEIGHTS_DIR / "student_hgcd_s42.pth"),
}
IMG_SIZE = 224

@st.cache_resource(show_spinner="Loading model...")
def get_model(path):
    if not os.path.exists(path):
        return None, None
    return load_student(path, "cpu")

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@torch.no_grad()
def predict(model, img):
    x = tf(img.convert("RGB")).unsqueeze(0)
    return F.softmax(model(x), dim=1)[0]

st.title("♻️ Waste Classifier")
ds = st.selectbox("Dataset / model", list(WEIGHTS))
model, classes = get_model(WEIGHTS[ds])
if model is None:
    st.error(f"Model file not found: {WEIGHTS[ds]}")
    st.stop()

tab1, tab2 = st.tabs(["📁 Upload", "📷 Camera"])
img = None
with tab1:
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if f:
        img = Image.open(io.BytesIO(f.read()))
with tab2:
    c = st.camera_input("Take a photo")
    if c:
        img = Image.open(c)

if img is not None:
    st.image(img, use_container_width=True)
    if st.button("🔍 Classify", type="primary", use_container_width=True):
        with st.spinner("Classifying..."):
            probs = predict(model, img)
        top = torch.topk(probs, k=min(3, len(classes)))
        st.subheader(f"Prediction: {classes[top.indices[0]]}  ({top.values[0]*100:.1f}%)")
        for p, i in zip(top.values, top.indices):
            st.progress(float(p), text=f"{classes[i]}: {p*100:.1f}%")
        if top.values[0] < 0.5:
            st.warning("Low confidence. Try a clearer photo with a plain background.")
else:
    st.info("Upload or capture an image, then click Classify.")