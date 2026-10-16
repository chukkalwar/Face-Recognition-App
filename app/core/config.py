"""
Every tunable setting, in one place.

Settings resolve in three layers, each overriding the one before:

    built-in DEFAULTS  <  config.toml  <  environment variables

Only the settings in ENV_OVERRIDES can come from the environment, so a
container can change them without editing a read-only config mount.

    from app.core.config import config
    config.attendance.min_liveness

Unknown keys in config.toml are logged, so a typo is not silently ignored.
Point at another file with FR_CONFIG=/path/to/file.toml.
"""
import logging
import os
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = os.environ.get("FR_CONFIG", "config.toml")

# Used when config.toml is absent or silent. Also the schema: a key not listed
# here is reported as unknown. config.toml documents every key.
DEFAULTS = {
    "runtime": {
        "gpu": False,
        "threads": 0,
        "spin_wait": False,
    },
    "paths": {
        "database": "data/faces.db",
        "spoof_models": "models",
    },
    "detection": {
        "det_size": 480,
        "rec_thresh": 0.45,
        "min_det_score": 0.75,
        "min_face_ratio": 0.12,
        "max_face_ratio": 0.35,
        "min_context": 0.80,
        "min_frame_px": 480,
        "min_blur_var": 20.0,
        "min_brightness": 50.0, #added this line to set minimum brightness for the face to be detected
    },
    "antispoof": {
        "threshold": 0.48,
        "min_liveness": 0.98,
    },
    "enrollment": {
        "samples_per_pose": 3,
        "max_seconds": 90,
        "min_det_score": 0.6,
        "min_liveness": 0.48,
        "min_face_ratio": 0.15,
        "max_face_ratio": 0.60,
        "min_context": 0.0,
        "block_duplicates": True,
        "duplicate_sim": 0.50,
        "duplicate_check_samples": 3,
        "dedup_sim": 0.99,
        "sample_cooldown_s": 0.25,
        "image_max_side": 160,
        "image_jpeg_quality": 80,
    },
    "attendance": {
        "min_similarity": 0.45,
        "frames_required": 5,
        "min_span_s": 1.0,
        "challenge": True,
        "challenge_sides_required": 1,
        "turn_max_face_ratio": 0.45,
        "challenge_min_yaw": 0.20,
        "max_spoof_frames": 2,
        "max_seconds": 30,
        "cooldown_s": 300,
        # add below lines skip part( values above 1 mean "never skip" )
        "skip_challenge_liveness": 0.80,
        "skip_challenge_similarity": 0.60
    },
    "group_attendance": {
        "min_face_ratio": 0.08,
        "max_face_ratio": 0.35,
        "min_context": 0.75,
        "min_face_px": 80,
        "max_faces": 7, #changed 10 ot 7
        "frames_required": 2, #changed value 5 to 2
        "min_span_s": 0.3 ,
        "max_seconds": 60,
        "min_similarity": 0.50, #added this line to set minimum similarity for group attendance ::: min_similarity is the match strength needed in a group, where faces are smaller.
        "min_margin": 0.08, #added this line to set minimum margin for group attendance ::: min_margin is how far the best match must beat the nearest other employee, which prevents look-alike mix-ups.
    },
    "api": {
        "max_sessions": 100,
        "max_image_mb": 5,
        "session_ttl_s": 300,
    },
}


def _as_bool(raw):
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# (section, key) -> (env var, parser)
ENV_OVERRIDES = {
    ("runtime", "gpu"): ("USE_GPU", _as_bool),
    ("runtime", "threads"): ("ORT_THREADS", int),
    ("antispoof", "threshold"): ("SPOOF_THRESH", float),
    ("paths", "spoof_models"): ("SPOOF_MODEL_DIR", str),
    ("paths", "database"): ("DB_PATH", str),
}


class Section(dict):
    """`config.runtime.gpu` and `config["runtime"]["gpu"]` both work. A missing
    key raises, so a typo fails where it is used."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"No config key '{key}' in this section. "
                f"Known keys: {', '.join(sorted(self))}") from None


class Config(Section):
    source = "built-in defaults"

    def providers(self):
        if self.runtime.gpu:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def det_size_wh(self):
        n = int(self.detection.det_size)
        return (n, n)


def _merge(defaults, loaded, source, path=()):
    out = {}
    for key, fallback in defaults.items():
        if isinstance(fallback, dict):
            out[key] = _merge(fallback, loaded.get(key) or {}, source, path + (key,))
        else:
            out[key] = loaded[key] if key in loaded else fallback
    for key in loaded:
        if key not in defaults:
            logger.warning(f"Unknown setting '{'.'.join(path + (key,))}' in {source} "
                           "- ignored. See app/core/config.py for valid keys.")
    return out


def _apply_env(data):
    for (section, key), (var, parse) in ENV_OVERRIDES.items():
        raw = os.environ.get(var)
        if raw:
            try:
                data[section][key] = parse(raw)
            except (TypeError, ValueError):
                logger.warning(f"Ignoring {var}={raw!r}: not a valid "
                               f"{parse.__name__} for {section}.{key}")
    return data


def load(path=None):
    path = Path(path or CONFIG_PATH)
    loaded = {}
    if path.is_file():
        with open(path, "rb") as f:
            loaded = tomllib.load(f)
    data = _apply_env(_merge(DEFAULTS, loaded, path))
    cfg = Config({k: Section(v) for k, v in data.items()})
    cfg.source = str(path) if path.is_file() else f"built-in defaults ({path} not found)"
    overridden = sorted(var for var, _ in ENV_OVERRIDES.values() if os.environ.get(var))
    if overridden:
        cfg.source += f", overridden by {', '.join(overridden)}"
    return cfg


config = load()
