import numpy as np
from PIL import Image
from pathlib import Path

PROCESSED_DIR = Path("Artifacts/Data_ingestion")
MEAN_STD_FILE = PROCESSED_DIR / "mean_std.npz"

CLASS_NAMES = [
    'Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions',
    'Dermatofibroma', 'Melanocytic nevi', 'Melanoma', 'Vascular lesions'
]

def load_mean_std():
    if not MEAN_STD_FILE.exists():
        raise FileNotFoundError("Mean/std file not found. Run data ingestion first.")
    data = np.load(MEAN_STD_FILE)
    return float(data['mean']), float(data['std'])

def preprocess_upload_image_pil(pil_image, target_size=(120,160)):
    """
    Accepts a PIL Image (already opened), resizes, normalizes according to training mean/std.
    Returns array shape (1,h,w,3)
    """
    h, w = target_size
    img = pil_image.convert('RGB').resize((w, h), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    mean, std = load_mean_std()
    arr = (arr - mean) / (std + 1e-8)
    return arr.reshape(1, h, w, 3)

def ensemble_vote(pred_arrays, voting='avg'):
    """
    pred_arrays: list of arrays (n_models x 1 x n_classes) OR (n_models x n_classes)
    voting: 'avg' -> average probabilities; 'majority' -> majority class voting using argmax per model
    returns: (label_index, probability, class_probs)
    """
    import numpy as np
    probs = np.vstack([p.reshape(-1) for p in pred_arrays])
    if voting == 'avg':
        avg = probs.mean(axis=0)
        idx = int(avg.argmax())
        return idx, float(avg[idx]), avg
    elif voting == 'majority':
        idxs = probs.argmax(axis=1)
        # majority vote
        vals, counts = np.unique(idxs, return_counts=True)
        winner = vals[counts.argmax()]
        # compute average probability for winner
        avg = probs.mean(axis=0)
        return int(winner), float(avg[winner]), avg
    else:
        raise ValueError("Unsupported voting method")
