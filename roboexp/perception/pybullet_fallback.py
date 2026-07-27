"""
PyBullet-specific perception fallback.

RoboEXP's default perception stack (GroundingDINO + SAM) is trained on real-world
images and frequently fails on low-polygon, texture-less PyBullet renderings of
simple geometric primitives.  This module provides a lightweight, simulation-only
fallback that segments objects by their rendered color.

It is intentionally separate from the generic RoboPercept pipeline: in a real
robot setting you would replace this with a different fallback (e.g., SAM
automatic masks + CLIP, or a domain-adapted detector), but for kitchen-worlds
synthetic objects with known colors this is enough to keep the memory pipeline
fed.
"""
import numpy as np
import cv2


class PyBulletFallbackDetector:
    """
    Color-based fallback detector for synthetic PyBullet objects.

    .. note::
        This is a **simulation-only, PyBullet-specific** fallback. It relies on
        rendered colors / segmentation IDs that only exist in PyBullet. Do not
        use it on a real robot; replace it with a domain-appropriate fallback
        (e.g. SAM automatic masks + CLIP, or a fine-tuned detector).

    Example config:
        config = {
            "cup": {
                "lower": [0.25, 0.00, 0.00],   # RGB lower bound
                "upper": [1.00, 0.20, 0.15],   # RGB upper bound
                "morph_kernel": 3,
            },
        }
    """

    def __init__(self, color_config=None, feat_dim=512, seed=42):
        self.color_config = color_config or {}
        self.feat_dim = feat_dim
        self.rng = np.random.RandomState(seed)

    def _label_feature(self, label, dim=None):
        """Return a deterministic but label-specific feature vector."""
        if dim is None:
            dim = self.feat_dim
        state = np.random.RandomState(hash(label) % (2 ** 31))
        feat = state.randn(dim).astype(np.float32)
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat /= norm
        return feat * 0.1 + 1e-4  # keep non-zero to avoid cosine division-by-zero

    def _color_mask(self, rgb, lower, upper, morph_kernel=3):
        """
        rgb: (H, W, 3) float array in [0, 1].
        Returns a bool mask of pixels whose RGB lies inside the given bounds.
        """
        lower = np.array(lower, dtype=np.float32).reshape(1, 1, 3)
        upper = np.array(upper, dtype=np.float32).reshape(1, 1, 3)
        mask = np.all((rgb >= lower) & (rgb <= upper), axis=-1)
        if morph_kernel > 0:
            kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
            mask = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel
            ).astype(bool)
        else:
            mask = mask.astype(bool)
        return mask

    def detect(self, rgb, labels, feat_dim=None):
        """
        For each label in `labels`, produce a fallback detection if a color
        region matching the configured thresholds is found.

        Returns a list of dicts with keys:
            label, phrase, box, mask, feature
        """
        detections = []
        for label in labels:
            cfg = self.color_config.get(label)
            if cfg is None:
                continue
            mask = self._color_mask(
                rgb,
                cfg["lower"],
                cfg["upper"],
                cfg.get("morph_kernel", 3),
            )
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            detections.append({
                "label": label,
                "phrase": f"{label}:1.0",
                "box": np.array([[x1, y1, x2, y2]], dtype=np.float32),
                "mask": mask.astype(bool),
                "feature": self._label_feature(label, dim=feat_dim),
            })
        return detections

    def fill_missing_labels(
        self, observations, observation_attributes, target_labels,
        replace_existing=False
    ):
        """
        In-place fill missing labels in `observation_attributes` using color
        fallback.  `observation_attributes` follows the same format returned by
        RoboPercept.

        For each camera and each target label, if the label is not already
        present in the DINO+SAM detections, a fallback detection is appended.
        If `replace_existing` is True, any existing detection for the target
        labels is replaced by the fallback mask (useful when DINO gives a
        false-positive for a PyBullet synthetic object).
        """
        for name, obs in observations.items():
            if name not in observation_attributes:
                continue
            attrs = observation_attributes[name]
            existing_labels = set()
            if attrs.get("pred_phrases") is not None:
                for phrase in attrs["pred_phrases"]:
                    existing_labels.add(str(phrase).split(":")[0])

            if replace_existing:
                labels_to_add = list(target_labels)
            else:
                labels_to_add = [l for l in target_labels if l not in existing_labels]
            feat_dim = None
            if attrs.get("mask_feats") is not None and len(attrs["mask_feats"]) > 0:
                feat_dim = attrs["mask_feats"].shape[1]
            fallback_dets = self.detect(obs["rgb"], labels_to_add, feat_dim=feat_dim)
            if not fallback_dets:
                continue

            # Build filtered lists of existing detections, optionally dropping
            # target labels that will be replaced by fallback.
            existing_boxes = []
            existing_masks = []
            existing_phrases = []
            existing_feats = []
            if len(attrs.get("pred_boxes", [])):
                for j, phrase in enumerate(attrs["pred_phrases"]):
                    label = str(phrase).split(":")[0]
                    if replace_existing and label in target_labels:
                        continue
                    existing_boxes.append(attrs["pred_boxes"][j][None, :])
                    existing_masks.append(attrs["pred_masks"][j][None, ...].astype(bool))
                    existing_phrases.append(phrase)
                    existing_feats.append(attrs["mask_feats"][j][None, :])

            boxes = existing_boxes
            masks = existing_masks
            phrases = list(existing_phrases)
            feats = existing_feats

            for det in fallback_dets:
                boxes.append(det["box"])
                masks.append(det["mask"][None, ...].astype(bool))
                phrases.append(det["phrase"])
                feats.append(det["feature"][None, ...])

            if boxes:
                attrs["pred_boxes"] = np.concatenate(boxes, axis=0)
                attrs["pred_masks"] = np.concatenate(masks, axis=0)
                attrs["pred_phrases"] = phrases
                attrs["mask_feats"] = np.concatenate(feats, axis=0)
            else:
                attrs["pred_boxes"] = np.zeros((0, 4), dtype=np.float32)
                attrs["pred_masks"] = np.zeros((0, *obs["rgb"].shape[:2]), dtype=bool)
                attrs["pred_phrases"] = []
                attrs["mask_feats"] = np.zeros((0, feat_dim or self.feat_dim), dtype=np.float32)
        return observation_attributes
