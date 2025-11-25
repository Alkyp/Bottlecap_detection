import os
import shutil
import yaml
import pytest

from src.bsort.utils import (
    load_yaml,
    ensure_dir,
    read_yolo_label,
    write_yolo_label,
)


def test_load_yaml(tmp_path):
    """Test membaca file YAML."""
    config_path = tmp_path / "config.yaml"
    data = {"key": "value"}

    with open(config_path, "w") as f:
        yaml.dump(data, f)

    loaded = load_yaml(str(config_path))
    assert loaded["key"] == "value"


def test_ensure_dir(tmp_path):
    """Test pembuatan folder jika belum ada."""
    dir_path = tmp_path / "new_folder"
    ensure_dir(str(dir_path))
    assert dir_path.exists() and dir_path.is_dir()


def test_yolo_read_write(tmp_path):
    """Test baca & tulis label YOLO format."""
    label_path = tmp_path / "label.txt"
    boxes = [
        [0, 0.5, 0.5, 0.4, 0.3],
        [1, 0.1, 0.2, 0.3, 0.4],
    ]

    # Write
    write_yolo_label(str(label_path), boxes)

    assert label_path.exists()

    # Read
    loaded_boxes = read_yolo_label(str(label_path))

    assert len(loaded_boxes) == 2
    assert loaded_boxes[0] == boxes[0]
    assert loaded_boxes[1] == boxes[1]
    for original, loaded in zip(boxes, loaded_boxes):
        assert all(abs(o - l) < 1e-6 for o, l in zip(original, loaded))