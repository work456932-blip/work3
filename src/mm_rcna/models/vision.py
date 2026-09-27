from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import numpy as np
from PIL import Image

from mm_rcna.schemas import VisionOutput


def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SharedVisionBackbone:
    """
    OpenI-oriented multiview CXR vision wrapper.

    Goals:
    1) keep original high-resolution image information for view/quality/multiview fusion logic
    2) use TorchXRayVision for lesion scoring
    3) use CXR Foundation for image embeddings when available
    4) keep backward-compatible interface:
       VisionOutput(feature_vector, lesion_scores, region_scores, quality_flags)
    """

    def __init__(
        self,
        lesion_labels,
        lung_regions,
        dim: int = 64,
        image_size: int = 512,
        image_encoder_name: str = "cxr_foundation",
        vision_backbone_name: str = "torchxrayvision_densenet121",
        device: str = "cpu",
    ) -> None:
        self.lesion_labels = list(lesion_labels)
        self.lung_regions = list(lung_regions)
        self.dim = int(dim)
        self.image_size = int(image_size)
        self.image_encoder_name = image_encoder_name
        self.vision_backbone_name = vision_backbone_name
        self.device = device

        self.lesion_bias = {k: 0.0 for k in self.lesion_labels}
        self.region_bias = {k: 0.0 for k in self.lung_regions}

        self._vision_model = None
        self._vision_pathologies = []
        self._cxr_processor = None
        self._cxr_model = None

        self._init_models()

    def load_checkpoint(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        self.lesion_bias.update(obj.get("lesion_bias", {}))
        self.region_bias.update(obj.get("region_bias", {}))

    def _init_models(self) -> None:
        # TorchXRayVision classifier for lesion scoring
        if self.vision_backbone_name == "torchxrayvision_densenet121":
            try:
                import torch
                import torchxrayvision as xrv

                model = xrv.models.DenseNet(weights="densenet121-res224-all")
                model = model.to(self.device)
                model.eval()
                self._vision_model = model
                self._vision_pathologies = list(model.pathologies)
            except Exception:
                self._vision_model = None
                self._vision_pathologies = []

        # CXR Foundation embedding model
        if self.image_encoder_name == "cxr_foundation":
            try:
                import torch
                from transformers import AutoModel, AutoImageProcessor

                model_id = "google/cxr-foundation"
                self._cxr_processor = AutoImageProcessor.from_pretrained(model_id)
                self._cxr_model = AutoModel.from_pretrained(model_id).to(self.device)
                self._cxr_model.eval()
            except Exception:
                self._cxr_processor = None
                self._cxr_model = None

    def _open_gray_original(self, path: str) -> np.ndarray:
        if not path or not Path(path).exists():
            return np.zeros((self.image_size, self.image_size), dtype=np.float32)
        img = Image.open(path).convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    def _open_rgb_pil(self, path: str, target: int = 224) -> Image.Image:
        if not path or not Path(path).exists():
            return Image.fromarray(np.zeros((target, target, 3), dtype=np.uint8))
        return Image.open(path).convert("RGB").resize((target, target))

    def _resize_pad(self, arr, target=512):
        """
        arr: HxW float numpy array in [0,1]
        output: target x target float numpy array in [0,1]
        """
        if arr is None:
            return np.zeros((target, target), dtype=np.float32)

        arr = np.asarray(arr, dtype=np.float32)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((target, target), dtype=np.float32)

        scale = min(target / h, target / w)
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))

        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
        img = img.resize((nw, nh), Image.BILINEAR)

        canvas = np.zeros((target, target), dtype=np.float32)
        top = (target - nh) // 2
        left = (target - nw) // 2
        canvas[top:top + nh, left:left + nw] = np.asarray(img, dtype=np.float32) / 255.0
        return canvas

    def _infer_view_position_from_path(self, path: str) -> str:
        name = Path(path).name.lower()
        if "lateral" in name or "_l" in name or "lat" in name:
            return "lateral"
        if "pa" in name or "ap" in name or "frontal" in name:
            return "frontal"
        return "unknown"

    def _quality_flags(self, arrs: List[np.ndarray], view_positions: List[str]) -> List[str]:
        if not arrs:
            return ["no_image"]

        flags = []
        resized = [self._resize_pad(a, 512) for a in arrs]
        stacked = np.stack(resized, axis=0)

        mean_v = float(stacked.mean())
        std_v = float(stacked.std())

        if std_v < 0.10:
            flags.append("low_contrast")
        if mean_v < 0.15:
            flags.append("underexposed")
        if mean_v > 0.85:
            flags.append("overexposed")
        if len(arrs) == 1:
            flags.append("single_view_only")
        if not any("frontal" in (vp or "") for vp in view_positions):
            flags.append("no_frontal_view")
        return sorted(set(flags))

    def _predict_with_xrv_per_image(self, path: str) -> Dict[str, float]:
        if self._vision_model is None or not Path(path).exists():
            return {}

        try:
            import torch
            import torchxrayvision as xrv

            img = Image.open(path).convert("L").resize((224, 224))
            arr = np.asarray(img, dtype=np.float32)
            arr = xrv.datasets.normalize(arr, 255)
            arr = arr[None, :, :]  # [1, H, W]
            tens = torch.from_numpy(arr).float().unsqueeze(0).to(self.device)  # [B, 1, H, W]

            with torch.no_grad():
                out = self._vision_model(tens)
                out = _safe_sigmoid(out.detach().cpu().numpy())[0]

            patho_map = {name: float(out[i]) for i, name in enumerate(self._vision_pathologies)}

            def pool(keys: List[str], default: float = 0.0) -> float:
                vals = [patho_map[k] for k in keys if k in patho_map]
                return float(max(vals)) if vals else default

            lesion_scores = {}
            for lesion in self.lesion_labels:
                ll = lesion.lower()
                if ll == "opacity":
                    score = pool(["Lung Opacity", "Consolidation", "Infiltration"])
                elif ll == "effusion":
                    score = pool(["Effusion", "Pleural Effusion"])
                elif ll == "pneumothorax":
                    score = pool(["Pneumothorax"])
                elif ll == "edema":
                    score = pool(["Edema"])
                elif ll == "atelectasis":
                    score = pool(["Atelectasis"])
                elif ll == "cardiomegaly":
                    score = pool(["Cardiomegaly"])
                elif ll == "device":
                    score = pool(["Support Devices", "Tube", "Line"])
                else:
                    score = 0.0

                lesion_scores[lesion] = float(np.clip(score + self.lesion_bias.get(lesion, 0.0), 0.0, 1.0))

            return lesion_scores
        except Exception:
            return {}

    def _fuse_multiview_lesions(
        self,
        per_image_scores: List[Dict[str, float]],
        view_positions: List[str],
    ) -> Dict[str, float]:
        if not per_image_scores:
            return {}

        frontal_scores = []
        lateral_scores = []
        unknown_scores = []

        for score_dict, vp in zip(per_image_scores, view_positions):
            vp = (vp or "").lower()
            if "frontal" in vp or "pa" in vp or "ap" in vp:
                frontal_scores.append(score_dict)
            elif "lateral" in vp:
                lateral_scores.append(score_dict)
            else:
                unknown_scores.append(score_dict)

        def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
            if not dicts:
                return {}
            out = {}
            for lesion in self.lesion_labels:
                vals = [float(d.get(lesion, 0.0)) for d in dicts]
                out[lesion] = float(np.mean(vals)) if vals else 0.0
            return out

        frontal_mean = mean_dict(frontal_scores)
        lateral_mean = mean_dict(lateral_scores)
        unknown_mean = mean_dict(unknown_scores)
        all_mean = mean_dict(per_image_scores)

        fused = {}
        for lesion in self.lesion_labels:
            f = frontal_mean.get(lesion, 0.0)
            l = lateral_mean.get(lesion, 0.0)
            u = unknown_mean.get(lesion, 0.0)
            a = all_mean.get(lesion, 0.0)

            if frontal_scores and lateral_scores:
                # frontal主导，lateral辅助
                score = 0.68 * f + 0.24 * l + 0.08 * a
            elif frontal_scores:
                score = 0.82 * f + 0.18 * a
            elif unknown_scores:
                score = 0.75 * u + 0.25 * a
            else:
                score = a

            fused[lesion] = float(np.clip(score, 0.0, 1.0))
        return fused

    def _embed_with_cxr_foundation(self, image_paths: List[str], view_positions: List[str]) -> Optional[np.ndarray]:
        if self._cxr_model is None or self._cxr_processor is None or not image_paths:
            return None

        try:
            import torch

            frontal_imgs = []
            lateral_imgs = []
            unknown_imgs = []

            for p, vp in zip(image_paths, view_positions):
                if not Path(p).exists():
                    continue
                img = self._open_rgb_pil(p, target=224)
                vp = (vp or "").lower()
                if "frontal" in vp or "pa" in vp or "ap" in vp:
                    frontal_imgs.append(img)
                elif "lateral" in vp:
                    lateral_imgs.append(img)
                else:
                    unknown_imgs.append(img)

            ordered_imgs = frontal_imgs + lateral_imgs + unknown_imgs
            if not ordered_imgs:
                return None

            with torch.no_grad():
                inputs = self._cxr_processor(images=ordered_imgs, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._cxr_model(**inputs)

                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    vecs = outputs.pooler_output
                elif hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    vecs = outputs.last_hidden_state.mean(dim=1)
                else:
                    return None

                vecs = vecs.detach().cpu().numpy().astype(np.float32)

            # 多图融合：frontal优先
            n_f = len(frontal_imgs)
            n_l = len(lateral_imgs)
            n_u = len(unknown_imgs)

            parts = []
            offset = 0
            if n_f > 0:
                parts.append((0.68, vecs[offset:offset + n_f].mean(axis=0)))
                offset += n_f
            if n_l > 0:
                parts.append((0.24, vecs[offset:offset + n_l].mean(axis=0)))
                offset += n_l
            if n_u > 0:
                parts.append((0.08, vecs[offset:offset + n_u].mean(axis=0)))
                offset += n_u

            if not parts:
                return None

            weights = np.array([w for w, _ in parts], dtype=np.float32)
            weights = weights / weights.sum()
            fused = np.zeros_like(parts[0][1], dtype=np.float32)
            for (w, v), nw in zip(parts, weights):
                fused += nw * v

            if fused.shape[0] < self.dim:
                fused = np.concatenate([fused, np.zeros(self.dim - fused.shape[0], dtype=np.float32)], axis=0)
            else:
                fused = fused[: self.dim]

            norm = np.linalg.norm(fused)
            if norm > 0:
                fused = fused / norm
            return fused.astype(np.float32)
        except Exception:
            return None

    def _rule_fallback_feature(self, arrs: List[np.ndarray], view_positions: List[str]) -> np.ndarray:
        if not arrs:
            return np.zeros(self.dim, dtype=np.float32)

        norm_arrs = [self._resize_pad(a, 512) for a in arrs]

        frontal_arrs = []
        lateral_arrs = []
        unknown_arrs = []

        for a, vp in zip(norm_arrs, view_positions):
            vp = (vp or "").lower()
            if "frontal" in vp or "pa" in vp or "ap" in vp:
                frontal_arrs.append(a)
            elif "lateral" in vp:
                lateral_arrs.append(a)
            else:
                unknown_arrs.append(a)

        ref = None
        if frontal_arrs:
            ref = np.mean(np.stack(frontal_arrs, axis=0), axis=0)
        elif unknown_arrs:
            ref = np.mean(np.stack(unknown_arrs, axis=0), axis=0)
        else:
            ref = np.mean(np.stack(norm_arrs, axis=0), axis=0)

        flat = ref.astype(np.float32).flatten()
        hist, _ = np.histogram(flat, bins=16, range=(0.0, 1.0), density=True)
        mean = np.array(
            [
                float(ref.mean()),
                float(ref.std()),
                float(np.percentile(ref, 10)),
                float(np.percentile(ref, 50)),
                float(np.percentile(ref, 90)),
            ],
            dtype=np.float32,
        )

        h, w = ref.shape
        q1 = ref[: h // 2, : w // 2].mean()
        q2 = ref[: h // 2, w // 2 :].mean()
        q3 = ref[h // 2 :, : w // 2].mean()
        q4 = ref[h // 2 :, w // 2 :].mean()
        aux = np.array([q1, q2, q3, q4], dtype=np.float32)

        feat = np.concatenate([mean, hist.astype(np.float32), aux], axis=0)
        if feat.shape[0] < self.dim:
            feat = np.concatenate([feat, np.zeros(self.dim - feat.shape[0], dtype=np.float32)], axis=0)
        else:
            feat = feat[: self.dim]

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat.astype(np.float32)

    def _region_scores(self, arrs, view_positions):
        """
        arrs: list[np.ndarray], each is HxW grayscale float array in [0,1]
        view_positions: list[str], e.g. frontal/lateral/unknown
        """
        if not arrs:
            return {k: 0.0 for k in self.lung_regions}

        # 先统一所有图像尺寸，避免 np.stack 报 shape 错
        norm_arrs = [self._resize_pad(a, 512) for a in arrs]

        frontal_arrs = []
        lateral_arrs = []
        unknown_arrs = []

        for a, vp in zip(norm_arrs, view_positions):
            vp = (vp or "").lower()
            if "frontal" in vp or "pa" in vp or "ap" in vp:
                frontal_arrs.append(a)
            elif "lateral" in vp:
                lateral_arrs.append(a)
            else:
                unknown_arrs.append(a)

        # 优先用 frontal；没有 frontal 就退化到 unknown；再不行就全部图
        if frontal_arrs:
            ref = np.mean(np.stack(frontal_arrs, axis=0), axis=0)
        elif unknown_arrs:
            ref = np.mean(np.stack(unknown_arrs, axis=0), axis=0)
        else:
            ref = np.mean(np.stack(norm_arrs, axis=0), axis=0)

        h, w = ref.shape
        mapping = {
            "left_upper": float(ref[: h // 2, : w // 2].mean()),
            "right_upper": float(ref[: h // 2, w // 2 :].mean()),
            "left_lower": float(ref[h // 2 :, : w // 2].mean()),
            "right_lower": float(ref[h // 2 :, w // 2 :].mean()),
        }

        vals = np.array(list(mapping.values()), dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        denom = max(hi - lo, 1e-6)

        out = {}
        for k in self.lung_regions:
            raw = mapping.get(k, 0.0)
            score = (raw - lo) / denom
            out[k] = float(np.clip(score + self.region_bias.get(k, 0.0), 0.0, 1.0))
        return out

    def run(self, image_paths: List[str]) -> VisionOutput:
        valid_paths = [p for p in image_paths if Path(p).exists()]
        arrs = [self._open_gray_original(p) for p in valid_paths]
        view_positions = [self._infer_view_position_from_path(p) for p in valid_paths]

        quality_flags = self._quality_flags(arrs, view_positions)

        # lesion scores: per-image then multiview fuse
        per_image_scores = [self._predict_with_xrv_per_image(p) for p in valid_paths]
        lesion_scores = self._fuse_multiview_lesions(per_image_scores, view_positions)
        if not lesion_scores:
            # fallback
            lesion_scores = {k: 0.0 for k in self.lesion_labels}

        # region scores: robust multiview region fusion
        region_scores = self._region_scores(arrs, view_positions)

        # image embedding: prefer CXR Foundation
        feat = self._embed_with_cxr_foundation(valid_paths, view_positions)
        if feat is None:
            feat = self._rule_fallback_feature(arrs, view_positions)

        return VisionOutput(
            feature_vector=feat.tolist(),
            lesion_scores=lesion_scores,
            region_scores=region_scores,
            quality_flags=quality_flags,
        )


class VisionToolRunner:
    def __init__(
        self,
        image_encoder_name,
        vision_backbone_name,
        lesion_labels,
        lung_regions,
        image_size=512,
        device="cpu",
        checkpoint_path=None,
    ):
        self.image_encoder_name = image_encoder_name
        self.vision_backbone_name = vision_backbone_name
        self.image_size = image_size
        self.device = device

        self.backbone = SharedVisionBackbone(
            lesion_labels=lesion_labels,
            lung_regions=lung_regions,
            dim=64,
            image_size=image_size,
            image_encoder_name=image_encoder_name,
            vision_backbone_name=vision_backbone_name,
            device=device,
        )
        if checkpoint_path:
            self.backbone.load_checkpoint(checkpoint_path)

    def run(self, image_paths):
        return self.backbone.run(image_paths)