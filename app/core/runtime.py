"""
Thread budget for ONNX Runtime and OpenCV.

Four ONNX sessions run in this process: the detector and recognizer inside
insightface, plus the two anti-spoofing models. By default each sizes its
thread pool from the logical CPU count, which oversubscribes the machine.
These models stop scaling at the physical core count, so that is the default.
Override with [runtime] threads in config.toml.

insightface silently ignores a `sess_options` kwarg, so
insightface_session_options() patches the one session class it builds through.
"""
import contextlib
import os

import cv2
import onnxruntime as ort

from app.core.config import config


def physical_cores():
    """Physical cores, not hyperthreads. Falls back to the logical count."""
    try:
        pairs, phys, core = set(), None, None
        with open("/proc/cpuinfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                key = key.strip()
                if key == "physical id":
                    phys = value.strip()
                elif key == "core id":
                    core = value.strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
        if phys is not None and core is not None:
            pairs.add((phys, core))
        if pairs:
            return len(pairs)
    except OSError:
        pass
    return os.cpu_count() or 1


def thread_budget():
    configured = int(config.runtime.threads or 0)
    return configured if configured > 0 else physical_cores()


def session_options(threads=None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads or thread_budget()
    so.inter_op_num_threads = 1          # each graph is one serial chain
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if not config.runtime.spin_wait:
        # requests arrive in bursts; spinning threads burn CPU between them
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


@contextlib.contextmanager
def insightface_session_options(threads=None):
    """Make FaceAnalysis build its sessions with our thread budget."""
    from insightface.model_zoo import model_zoo

    so = session_options(threads)
    original = model_zoo.PickableInferenceSession

    class _PinnedSession(original):
        def __init__(self, model_path, **kwargs):
            kwargs.setdefault("sess_options", so)
            super().__init__(model_path, **kwargs)

    model_zoo.PickableInferenceSession = _PinnedSession
    try:
        yield so
    finally:
        model_zoo.PickableInferenceSession = original


def configure_opencv(threads=None):
    cv2.setNumThreads(threads or thread_budget())
