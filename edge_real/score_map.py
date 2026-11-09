"""Lossless, versioned pixel labels in template coordinates."""
import base64
import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=8)
def decode_map(encoded, digest):
    raw = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Bản đồ điểm không khớp mã kiểm tra')
    labels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None or labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError('Bản đồ điểm phải là ảnh xám PNG 8-bit')
    if not np.isin(labels, [0, 5, 6, 7, 8, 9, 10]).all():
        raise ValueError('Bản đồ điểm chỉ được chứa 0, 5, 6, 7, 8, 9, 10')
    labels.setflags(write=False)
    return labels


def get_map(config):
    data = config['score_map']
    labels = decode_map(data['png_base64'], data['sha256'])
    if list(labels.shape[::-1]) != config['image_size']:
        raise ValueError('Kích thước bản đồ điểm khác ảnh mẫu')
    return labels


def attach_map(config, template_path):
    path = Path(template_path).resolve().parents[1] / 'Image_Scored' / 'Score.png'
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    previous = config.get('score_map', {}).get('sha256')
    config['score_map'] = {'png_base64': base64.b64encode(raw).decode('ascii'),
                           'sha256': digest, 'source': 'Image_Scored/Score.png'}
    get_map(config)
    if previous != digest:
        config.pop('confirmed_at', None)
    config['confirmed'] = True
    config['confirmation_method'] = 'validated_score_map'
    return config


def pixel_score(point, labels):
    x, y = map(float, point)
    h, w = labels.shape
    if not np.isfinite([x, y]).all():
        raise ValueError('Tọa độ không hợp lệ')
    if not (0 <= x <= w - 1 and 0 <= y <= h - 1):
        return 0
    return int(labels[int(np.floor(y + .5)), int(np.floor(x + .5))])


def near_boundary(point, labels, tolerance):
    x, y = map(float, point)
    h, w = labels.shape
    left, right = max(0, int(np.floor(x-tolerance))), min(w, int(np.ceil(x+tolerance))+1)
    top, bottom = max(0, int(np.floor(y-tolerance))), min(h, int(np.ceil(y+tolerance))+1)
    if left >= right or top >= bottom:
        return False
    return bool(np.any(labels[top:bottom, left:right] != pixel_score(point, labels)))
