"""Real or AI? A Streamlit app for the CIFAKE detectors (Small CNN and MobileNetV2).

Run from the folder that holds models_tf/ and results_tf/:
    streamlit run app_tensorflow.py

Optional: put a few example images in samples/REAL and samples/FAKE so visitors can try the
app without their own images. If a cifake/test folder is present, it is used too.
"""
import io
import json
import random
from pathlib import Path

import altair as alt
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

# ---------------------------------------------------------------- settings you may edit

GITHUB_URL = "https://github.com/ajibola-debug/ai-playground/tensorflow/real-or-ai"
AUTHOR = "Ajibola Akinbode"
AUTHOR_URL = "https://github.com/ajibola-debug"

# Every path is relative to this file, so the app works wherever it is started from
# (including Streamlit Cloud, and repos where the app sits in a subfolder).
HERE = Path(__file__).resolve().parent

IMG = 32         # the models were trained on 32x32 images
DISPLAY = 320    # small images are shown enlarged to this size
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
MODELS = {
    "Small CNN": {"key": "cnn", "path": HERE / "models_tf" / "best_cnn.keras",
                  "about": "Trained from scratch on 32×32 images. Fast, and the baseline."},
    "MobileNetV2": {"key": "mobilenet", "path": HERE / "models_tf" / "best_mobilenet.keras",
                    "about": "Pretrained on ImageNet, then fine-tuned in two phases "
                             "(head first, then the whole network). Usually more accurate."},
}
RESULTS = HERE / "results_tf"
SAMPLE_DIRS = [HERE / "samples", HERE / "cifake" / "test"]

st.set_page_config(page_title="Real or AI?", page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- styling

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero h1 {font-size: 2.4rem; margin-bottom: 0.2rem; letter-spacing: -0.02em;}
.hero p {font-size: 1.05rem; opacity: 0.8; margin-top: 0;}
.pill {display: inline-block; padding: 0.2rem 0.7rem; margin: 0.15rem 0.3rem 0.15rem 0;
       border-radius: 999px; font-size: 0.85rem; background: #EEF2FF; color: #3730A3;
       border: 1px solid #C7D2FE;}
.verdict {border-radius: 14px; padding: 1.2rem 1.4rem; margin: 0.5rem 0 1rem 0;
          border-left: 8px solid; }
.verdict.ai {background: #FEF2F2; border-color: #DC2626; color: #450A0A;}
.verdict.real {background: #F0FDF4; border-color: #16A34A; color: #052E16;}
.verdict .label {font-size: 2rem; font-weight: 700; line-height: 1.1;}
.verdict .sub {font-size: 1rem; margin-top: 0.3rem;}
.verdict .note {font-size: 0.95rem; margin-top: 0.6rem; opacity: 0.9;}
.meter {position: relative; height: 12px; border-radius: 999px; margin-top: 0.8rem;
        background: linear-gradient(90deg, #16A34A 0%, #FACC15 50%, #DC2626 100%);}
.meter .dot {position: absolute; top: -5px; width: 22px; height: 22px; border-radius: 50%;
             background: #FFFFFF; border: 3px solid #111827; transform: translateX(-50%);}
.meter .tick {position: absolute; top: -6px; width: 2px; height: 24px; background: #111827;
              opacity: 0.55;}
.meter-labels {display: flex; justify-content: space-between; font-size: 0.8rem;
               margin-top: 0.35rem; opacity: 0.8;}
.caption-tag {font-size: 0.8rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 6px;}
.caption-tag.ai {background: #FEE2E2; color: #991B1B;}
.caption-tag.real {background: #DCFCE7; color: #166534;}
.footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid rgba(128,128,128,0.3);
         font-size: 0.85rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model helpers

@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    return keras.models.load_model(path)


@st.cache_resource(show_spinner=False)
def backbone_with_features(path):
    """For MobileNetV2: the nested backbone, also returning its last conv feature map."""
    model = load_model(path)
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            try:
                features = layer.get_layer("out_relu").output
            except ValueError:
                features = [l for l in layer.layers if len(l.output.shape) == 4][-1].output
            return keras.Model(layer.inputs, [features, layer.output])
    return None


def open_image(source):
    """Open an upload or file path as RGB, respecting phone-camera rotation. None if unreadable."""
    try:
        image = Image.open(source)
        return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None


def prepare(image, jpeg_quality=None):
    """Optionally JPEG-compress, then shrink to 32x32 like the training data."""
    if jpeg_quality is not None:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        image = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    small = image.resize((IMG, IMG), Image.BICUBIC)
    return small, np.asarray(small, dtype=np.float32)[np.newaxis]  # pixels 0-255, as in training


def prob_ai(model, batch):
    return float(model(batch, training=False)[0, 0])


def gradcam(path, batch, explain_ai):
    """Grad-CAM heatmap (0-1) for the predicted class, or None if it can't be computed."""
    model = load_model(path)
    convs = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]
    last_conv = convs[-1] if convs else None
    x = tf.convert_to_tensor(batch)
    features = None
    with tf.GradientTape() as tape:
        for layer in model.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            if isinstance(layer, keras.Model):
                features, x = backbone_with_features(path)(x, training=False)
            else:
                x = layer(x, training=False)
                if layer is last_conv:
                    features = x
        score = x[:, 0] if explain_ai else 1.0 - x[:, 0]
    if features is None:
        return None
    grads = tape.gradient(score, features)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))[0].numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def overlay(small, cam, alpha):
    base = np.asarray(small.resize((DISPLAY, DISPLAY), Image.NEAREST), dtype=np.float32) / 255
    heat = Image.fromarray(np.uint8(cam * 255)).resize((DISPLAY, DISPLAY), Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    return Image.fromarray(np.uint8(np.clip((1 - alpha) * base + alpha * colored, 0, 1) * 255))


def enlarge(small):
    return small.resize((DISPLAY, DISPLAY), Image.NEAREST)


def verdict(p, threshold):
    is_ai = p >= threshold
    confidence = p if is_ai else 1 - p
    return is_ai, ("AI-generated" if is_ai else "Real"), confidence


def strength(p, threshold):
    gap = abs(p - threshold)
    if gap >= 0.35:
        return "Strong signal. The model is clear about this one."
    if gap >= 0.15:
        return "Moderate signal. Leaning one way, but not certain."
    return "Uncertain. This is close to the decision threshold, so treat the result with caution."


def verdict_card(p, threshold):
    is_ai, label, conf = verdict(p, threshold)
    css = "ai" if is_ai else "real"
    icon = "🤖" if is_ai else "📷"
    st.markdown(f"""
<div class="verdict {css}">
  <div class="label">{icon} {label}</div>
  <div class="sub"><b>{conf:.0%} confident</b> &nbsp;·&nbsp; probability AI-generated: {p:.1%}</div>
  <div class="meter"><div class="tick" style="left:{threshold*100:.1f}%"></div>
       <div class="dot" style="left:{p*100:.1f}%"></div></div>
  <div class="meter-labels"><span>Real</span><span>threshold {threshold:.2f}</span><span>AI-generated</span></div>
  <div class="note">{strength(p, threshold)}</div>
</div>""", unsafe_allow_html=True)


def tag(p, threshold):
    is_ai, label, _ = verdict(p, threshold)
    return f'<span class="caption-tag {"ai" if is_ai else "real"}">{label} · {p:.0%}</span>'


@st.cache_data(show_spinner=False)
def list_samples():
    """Up to 8 example images per class from samples/ or cifake/test/."""
    found = {"REAL": [], "FAKE": []}
    for folder in SAMPLE_DIRS:
        for cls in found:
            if (folder / cls).is_dir() and len(found[cls]) < 8:
                files = sorted(p for p in (folder / cls).iterdir()
                               if p.suffix.lower().lstrip(".") in IMAGE_TYPES)
                random.Random(7).shuffle(files)
                found[cls] += [str(p) for p in files[:8 - len(found[cls])]]
    return found


def load_metrics(key):
    f = RESULTS / f"metrics_{key}.json"
    return json.loads(f.read_text()) if f.exists() else None


# ---------------------------------------------------------------- sidebar

available = {name: info for name, info in MODELS.items() if info["path"].exists()}
if not available:
    found = sorted(p.name for p in (HERE / "models_tf").glob("*")) if (HERE / "models_tf").exists() else None
    st.error("No trained model found. The app looks for **models_tf/best_cnn.keras** "
             "(or best_mobilenet.keras) in the same folder as app_tensorflow.py.")
    st.markdown(f"- App folder: `{HERE}`\n"
                f"- models_tf folder: {'found, containing ' + str(found) if found is not None else '**missing**'}\n\n"
                "Train with `python cifake_tensorflow.py --data-dir cifake`, then make sure the "
                "models_tf folder is uploaded next to app_tensorflow.py.")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_name = st.selectbox("Model", list(available))
    st.caption(available[model_name]["about"])
    model_path = str(available[model_name]["path"])
    model_key = available[model_name]["key"]
    model = load_model(model_path)

    threshold = st.slider(
        "Decision threshold", 0.05, 0.95, 0.50, 0.05,
        help="An image is called AI-generated when its probability is at or above this. "
             "Raise it to flag fewer images (fewer false alarms); lower it to catch more fakes.")
    show_cam = st.toggle("Show heatmap (Grad-CAM)", value=True,
                         help="Highlights the parts of the image that pushed the model towards its answer.")
    cam_alpha = st.slider("Heatmap strength", 0.1, 0.9, 0.45, 0.05, disabled=not show_cam)

    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "Trained on **CIFAKE**: 60,000 real photos and 60,000 images made with Stable Diffusion 1.4, "
        "all 32×32 pixels. Your image is shrunk to that size before the model sees it.")
    st.markdown(f"[Source code on GitHub]({GITHUB_URL})")

# ---------------------------------------------------------------- header

metrics = load_metrics(model_key)
pills = [f'<span class="pill">Model: {model_name}</span>']
if metrics:
    m = metrics["original"]
    pills += [f'<span class="pill">Test accuracy {m["accuracy"]:.1%}</span>',
              f'<span class="pill">ROC-AUC {m["roc_auc"]:.3f}</span>',
              f'<span class="pill">Tested on {m["n_images"]:,} unseen images</span>']
st.markdown(f"""
<div class="hero">
  <h1>🔍 Real or AI?</h1>
  <p>Upload a photo and a neural network will judge whether it's a real photograph or an image
  generated by AI, and show you where it looked.</p>
  {''.join(pills)}
</div>""", unsafe_allow_html=True)

tab_analyse, tab_batch, tab_compare, tab_stress, tab_card = st.tabs(
    ["🔍 Analyse", "🗂️ Batch", "⚖️ Compare models", "🧪 Stress test", "📊 Model card"])


def show_analysis(image, truth=None):
    small, batch = prepare(image)
    with st.spinner("Analysing…"):
        p = prob_ai(model, batch)
        is_ai, label, _ = verdict(p, threshold)
        cam = gradcam(model_path, batch, is_ai) if show_cam else None

    verdict_card(p, threshold)
    if truth is not None:
        correct = (truth == "FAKE") == is_ai
        truth_label = "AI-generated" if truth == "FAKE" else "Real"
        (st.success if correct else st.error)(
            f"{'✅ Correct' if correct else '❌ Wrong'}: this example is actually **{truth_label}**.")

    cols = st.columns(3 if cam is not None else 2)
    cols[0].image(image, caption="Your image", use_container_width=True)
    cols[1].image(enlarge(small), caption="What the model sees (32×32)", use_container_width=True)
    if cam is not None:
        cols[2].image(overlay(small, cam, cam_alpha),
                      caption=f"Heatmap: red areas pushed it towards “{label}”", use_container_width=True)

    with st.expander("How to read this result"):
        st.markdown(
            f"- **The meter** shows the model's probability that the image is AI-generated. "
            f"The black line is your threshold ({threshold:.2f}); anything right of it is called AI-generated.\n"
            "- **What the model sees**: the model was trained on tiny 32×32 images, so yours is shrunk first. "
            "Fine detail is lost, which is why large, sharp photos are harder for it.\n"
            "- **The heatmap** (Grad-CAM) colours the regions that most influenced the answer: red is strong, "
            "blue is weak. It shows where the model looked, not proof that it's right.\n"
            "- This model only learned what **Stable Diffusion 1.4** images look like. Images from newer tools "
            "(Gemini, Midjourney, DALL·E 3) can fool it.")


# ---------------------------------------------------------------- Analyse

with tab_analyse:
    samples = list_samples()
    has_samples = bool(samples["REAL"] or samples["FAKE"])
    options = ["📁 Upload", "📸 Camera"] + (["🎲 Try an example"] if has_samples else [])
    source = st.radio("Choose an image", options, horizontal=True, label_visibility="collapsed")

    image, truth = None, None
    if source == "📁 Upload":
        file = st.file_uploader("Drag and drop an image, or browse", type=IMAGE_TYPES, key="single")
        if file:
            image = open_image(file)
            if image is None:
                st.error("That file couldn't be opened as an image. Try a JPG or PNG.")
    elif source == "📸 Camera":
        shot = st.camera_input("Take a photo")
        if shot:
            image = open_image(shot)
    else:
        st.caption("Examples from the CIFAKE test set, which the model never saw during training. "
                   "Pick one and see if the model gets it right.")
        pool = [(p, "REAL") for p in samples["REAL"]] + [(p, "FAKE") for p in samples["FAKE"]]
        random.Random(1).shuffle(pool)
        grid = st.columns(4)
        for i, (path, cls) in enumerate(pool[:8]):
            with grid[i % 4]:
                thumb = open_image(path)
                if thumb is not None:
                    st.image(enlarge(thumb.resize((IMG, IMG))), use_container_width=True)
                    if st.button("Analyse this", key=f"sample_{i}", use_container_width=True):
                        st.session_state.sample = (path, cls)
        if "sample" in st.session_state:
            path, truth = st.session_state.sample
            image = open_image(path)
            st.divider()

    if image is not None:
        show_analysis(image, truth)
    elif source != "🎲 Try an example":
        st.info("👆 Add an image to get started. JPG, PNG, WEBP and BMP all work. "
                "Nothing you upload is stored.")

# ---------------------------------------------------------------- Batch

with tab_batch:
    st.markdown("Analyse many images at once and download the results.")
    files = st.file_uploader("Upload several images", type=IMAGE_TYPES, accept_multiple_files=True,
                             key="batch")
    if files:
        rows, thumbs, skipped = [], [], []
        progress = st.progress(0.0, text="Analysing…")
        for i, f in enumerate(files):
            img = open_image(f)
            if img is None:
                skipped.append(f.name)
            else:
                small, batch = prepare(img)
                p = prob_ai(model, batch)
                _, label, conf = verdict(p, threshold)
                rows.append({"File": f.name, "Verdict": label, "P(AI-generated)": p, "Confidence": conf})
                thumbs.append((small, f.name, p))
            progress.progress((i + 1) / len(files), text=f"Analysed {i + 1} of {len(files)}")
        progress.empty()
        if skipped:
            st.warning(f"Skipped {len(skipped)} file(s) that couldn't be opened: {', '.join(skipped)}")

        if rows:
            df = pd.DataFrame(rows).sort_values("P(AI-generated)", ascending=False)
            n_ai = int((df.Verdict == "AI-generated").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Images analysed", len(df))
            c2.metric("Flagged as AI-generated", n_ai)
            c3.metric("Judged real", len(df) - n_ai)

            show = st.segmented_control("Show", ["All", "AI-generated", "Real"], default="All") \
                if hasattr(st, "segmented_control") else st.radio("Show", ["All", "AI-generated", "Real"],
                                                                  horizontal=True)
            view = df if show in (None, "All") else df[df.Verdict == show]
            st.dataframe(
                view, use_container_width=True, hide_index=True,
                column_config={
                    "P(AI-generated)": st.column_config.ProgressColumn(
                        "P(AI-generated)", format="%.2f", min_value=0.0, max_value=1.0),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                })
            st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False).encode(),
                               "real_or_ai_results.csv", "text/csv")

            st.markdown("##### Gallery")
            wanted = set(view.File)
            grid = st.columns(5)
            shown = [t for t in sorted(thumbs, key=lambda t: -t[2]) if t[1] in wanted]
            for i, (small, name, p) in enumerate(shown):
                with grid[i % 5]:
                    st.image(enlarge(small), use_container_width=True)
                    st.markdown(f"{tag(p, threshold)}<br><small>{name}</small>", unsafe_allow_html=True)
    else:
        st.info("👆 Select several images at once (Ctrl/Cmd-click), or drag a group of files in.")

# ---------------------------------------------------------------- Compare

with tab_compare:
    if len(available) < 2:
        st.info("Train the second model to compare them side by side:\n\n"
                "`python train_mobilenet.py --data-dir cifake`")
    else:
        st.markdown("Run both models on the same image and see whether they agree.")
        file = st.file_uploader("Image to compare on", type=IMAGE_TYPES, key="compare")
        if file and (image := open_image(file)) is not None:
            small, batch = prepare(image)
            results = {}
            cols = st.columns(len(available))
            for col, (name, info) in zip(cols, available.items()):
                path = str(info["path"])
                p = prob_ai(load_model(path), batch)
                results[name] = p
                with col:
                    st.markdown(f"#### {name}")
                    verdict_card(p, threshold)
                    cam = gradcam(path, batch, p >= threshold) if show_cam else None
                    st.image(overlay(small, cam, cam_alpha) if cam is not None else enlarge(small),
                             use_container_width=True)
            verdicts = {verdict(p, threshold)[1] for p in results.values()}
            if len(verdicts) == 1:
                st.success(f"Both models agree: **{verdicts.pop()}**.")
            else:
                st.warning("The models disagree. That usually means the image is a hard case.")
        elif file:
            st.error("That file couldn't be opened as an image.")

# ---------------------------------------------------------------- Stress test

with tab_stress:
    st.markdown("Images shared online are usually re-compressed. This tab compresses your image step by "
                "step and shows how the model's answer changes. A trustworthy detector shouldn't flip its "
                "verdict just because an image was saved as a lower-quality JPEG.")
    file = st.file_uploader("Image to stress-test", type=IMAGE_TYPES, key="stress")
    if file and (image := open_image(file)) is not None:
        qualities = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        with st.spinner("Compressing and re-testing…"):
            probs = [prob_ai(model, prepare(image, q)[1]) for q in qualities]
        df = pd.DataFrame({"JPEG quality": qualities, "P(AI-generated)": probs})
        line = alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("JPEG quality:Q", scale=alt.Scale(reverse=True),
                    title="JPEG quality (right = more compressed)"),
            y=alt.Y("P(AI-generated):Q", scale=alt.Scale(domain=[0, 1])),
            tooltip=["JPEG quality", alt.Tooltip("P(AI-generated):Q", format=".1%")])
        rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[6, 4]).encode(y="t:Q")
        st.altair_chart(line + rule, use_container_width=True)
        st.caption("The dashed line is the decision threshold.")

        flips = [q for q, p in zip(qualities, probs) if (p >= threshold) != (probs[0] >= threshold)]
        if flips:
            st.warning(f"The verdict flips at JPEG quality **{flips[0]}** and below.")
        else:
            st.success("The verdict stays the same at every compression level.")

        q = st.select_slider("Preview at quality", options=qualities, value=50)
        small, batch = prepare(image, q)
        c1, c2 = st.columns(2)
        c1.image(enlarge(small), caption=f"JPEG quality {q}, as the model sees it", use_container_width=True)
        with c2:
            verdict_card(prob_ai(model, batch), threshold)
    elif file:
        st.error("That file couldn't be opened as an image.")
    else:
        st.info("👆 Upload an image to stress-test it.")

# ---------------------------------------------------------------- Model card

with tab_card:
    if metrics:
        m = metrics["original"]
        st.markdown(f"#### {model_name} on {m['n_images']:,} test images it never saw during training")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{m['accuracy']:.1%}", help="Share of all test images classified correctly.")
        c2.metric("F1 score", f"{m['f1_fake']:.3f}", help="Balance of precision and recall for the AI class.")
        c3.metric("ROC-AUC", f"{m['roc_auc']:.3f}",
                  help="How well the model ranks AI images above real ones, across every threshold. 1.0 is perfect, 0.5 is guessing.")
        c4.metric("AI images caught", f"{m['recall_fake']:.1%}", help="Recall: share of AI images flagged as AI.")

        st.markdown("##### Robustness to JPEG compression")
        names = {"original": "Original images", "jpeg_q90": "JPEG quality 90",
                 "jpeg_q70": "JPEG quality 70", "jpeg_q50": "JPEG quality 50"}
        table = pd.DataFrame(metrics).T.rename(index=names)[
            ["accuracy", "precision_fake", "recall_fake", "f1_fake", "roc_auc"]]
        table.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    else:
        st.info(f"No test results for {model_name} yet. They appear here after `cifake_tensorflow.py` finishes.")

    c1, c2 = st.columns(2)
    cm_file = RESULTS / f"confusion_matrix_{model_key}.png"
    if cm_file.exists():
        with c1:
            st.markdown("##### Confusion matrix")
            st.image(str(cm_file), use_container_width=True)
    hist_file = RESULTS / f"history_{model_key}.json"
    if hist_file.exists():
        hist = pd.DataFrame(json.loads(hist_file.read_text()))
        hist["epoch"] = range(1, len(hist) + 1)
        with c2:
            st.markdown("##### Training curves")
            for metric, title in [("accuracy", "Accuracy"), ("loss", "Loss")]:
                if metric in hist and f"val_{metric}" in hist:
                    long = hist.melt("epoch", [metric, f"val_{metric}"], var_name="set", value_name=title)
                    long["set"] = long["set"].map({metric: "Training", f"val_{metric}": "Validation"})
                    st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                        x=alt.X("epoch:O", title="Epoch"), y=alt.Y(f"{title}:Q"),
                        color=alt.Color("set:N", title=None)).properties(height=200),
                        use_container_width=True)

    st.markdown("##### Limitations")
    st.markdown(
        "- Trained only on **32×32** images; larger photos lose most of their detail before the model sees them.\n"
        "- Every AI image in the training data came from **Stable Diffusion 1.4**. Images from newer generators "
        "can be called real.\n"
        "- High accuracy on CIFAKE doesn't mean it detects AI images in general.\n"
        "- Use it as a demo of how such detectors work, not as proof that an image is real or fake.")

# ---------------------------------------------------------------- footer

st.markdown(f"""
<div class="footer">
Built by <a href="{AUTHOR_URL}" target="_blank">{AUTHOR}</a> with TensorFlow, Keras and Streamlit ·
<a href="{GITHUB_URL}" target="_blank">Source code</a> · Images you upload are processed in memory and not stored.
</div>""", unsafe_allow_html=True)
