import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NON_FRUIT = {"background", "unknown"}


class SigLIPClassifier:
    def __init__(self, models_dir=None, threads=4):
        self.models_dir = Path(models_dir) if models_dir else ROOT / "models"
        self.ready = False
        self.thr = 0.9
        try:
            self._load(threads)
            self.ready = True
            print("[classifier] SigLIP ONNX loaded.")
        except Exception as e:  # missing files, bad export, etc.
            print(f"[classifier] MOCK mode ({type(e).__name__}: {e})")

    # ------------------------------------------------------------ loading
    def _load(self, threads):
        import onnxruntime as ort

        d = self.models_dir
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(d / "siglip_vision.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

        self.classes = json.load(open(d / "classes.json"))
        self.concept_class = np.array(json.load(open(d / "concept_class.json")))
        self.text = np.load(d / "text_emb.npy").astype(np.float32)
        meta = json.load(open(d / "siglip_meta.json"))
        self.scale, self.bias, self.pre = meta["logit_scale"], meta["logit_bias"], meta["preprocess"]
        try:
            self.thr = json.load(open(d / "threshold.json"))["threshold"]
        except FileNotFoundError:
            pass

        from PIL import Image
        self.masks = [self.concept_class == k for k in range(len(self.classes))]
        self.mean = np.array(self.pre["mean"], np.float32)
        self.std = np.array(self.pre["std"], np.float32)
        self.resample = Image.Resampling(self.pre["resample"])
        self.predict(np.zeros((self.pre["height"], self.pre["width"], 3), np.uint8))  # warm-up

    def _prep(self, bgr):
        import cv2
        from PIL import Image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((self.pre["width"], self.pre["height"]), self.resample)
        x = (np.asarray(im, np.float32) / 255.0 - self.mean) / self.std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    # ------------------------------------------------------------ inference
    def predict(self, image):
        """image: BGR numpy array (an OpenCV crop). Returns a dict with label + confidence."""
        if not self.ready:
            return {"label": "apple", "confidence": 0.95, "is_fruit": True,
                    "uncertain": False, "top3": [("apple", 0.95)], "mock": True}

        emb = self.sess.run(None, {self.in_name: self._prep(image)})[0][0]
        emb = emb / np.linalg.norm(emb)
        logits = (self.text @ emb) * self.scale + self.bias           # one score per prompt
        cls = np.array([logits[m].max() for m in self.masks])         # best prompt per class
        p = np.exp(cls - cls.max())
        p /= p.sum()
        order = np.argsort(-p)[:3]
        label, conf = self.classes[int(order[0])], float(p[order[0]])
        return {"label": label, "confidence": conf, "is_fruit": label not in NON_FRUIT,
                "uncertain": conf < self.thr,
                "top3": [(self.classes[int(i)], float(p[i])) for i in order], "mock": False}


if __name__ == "__main__":            # python -m backend.classifier some.jpg
    import cv2
    print(SigLIPClassifier().predict(cv2.imread(sys.argv[1])))
