"""Checks for student code; the instructor tests/ directory is untouched."""
import json
import pytest

from lab.data import yolo_to_box, objects_to_answer, subset_train, collate_examples, encode_example
from lab.reward import iou_reward


def test_coordinate_roundtrip_and_sort():
    assert yolo_to_box("pothole", [.5, .5, .2, .2])["bbox_2d"] == [400, 400, 600, 600]
    assert yolo_to_box("pothole", [.5, .5, .2, .2], "pixel")["bbox_2d"] == [256, 144, 384, 216]
    objects = [{"label": "crack", "yolo": [.5, .8, .2, .2]},
               {"label": "pothole", "yolo": [.5, .2, .2, .2]}]
    assert json.loads(objects_to_answer(objects))[0]["label"] == "pothole"
    assert objects_to_answer([]) == "[]"


def test_thin_positive_box_survives_rounding_and_edges():
    assert yolo_to_box('crack', [.639, .786, .004, .0001])['bbox_2d'] == [637, 786, 641, 787]
    assert yolo_to_box('crack', [1, 1, .0001, .0001])['bbox_2d'] == [999, 999, 1000, 1000]
    assert yolo_to_box('crack', [.639062, .786111, .003125, 0])['bbox_2d'] == [637, 786, 641, 787]
    with pytest.raises(ValueError, match='Negative'):
        yolo_to_box('crack', [.5, .5, .01, -.01])


def test_subset_is_repeatable_without_replacement():
    rows = list(range(100))
    a = subset_train(rows, .1, 42)
    assert a == subset_train(rows, .1, 42)
    assert len(a) == len(set(a)) == 10
    assert set(subset_train(rows, 1, 42)) == set(rows)


def test_padding_masks_only_added_tokens_and_concatenates_image_patches():
    torch = pytest.importorskip("torch")
    a = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
         "labels": torch.tensor([[-100, 2, 3]]), "mm_token_type_ids": torch.tensor([[1, 0, 0]]),
         "pixel_values": torch.ones(5, 4), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    b = {"input_ids": torch.tensor([[1, 4]]), "attention_mask": torch.ones(1, 2, dtype=torch.long),
         "labels": torch.tensor([[-100, 4]]), "mm_token_type_ids": torch.tensor([[1, 0]]),
         "pixel_values": torch.ones(7, 4), "image_grid_thw": torch.tensor([[1, 4, 2]])}
    batch = collate_examples([a, b], 0)
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 4, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["pixel_values"].shape == (12, 4)
    assert batch["image_grid_thw"].shape == (2, 3)


def test_mask_prefix_and_fail_on_wrong_chat_template():
    torch = pytest.importorskip("torch")
    class Processor:
        bad = False
        def apply_chat_template(self, messages, **kw):
            ids = [8, 9] if len(messages) == 1 else ([7, 9, 10, 11] if self.bad else [8, 9, 10, 11])
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
    processor = Processor()
    entry = {"objects": [], "width": 640, "height": 360}
    assert encode_example(processor, None, entry)["labels"].tolist() == [[-100, -100, 10, 11]]
    processor.bad = True
    with pytest.raises(ValueError, match="prefix"):
        encode_example(processor, None, entry)


def test_reward_perfect_invalid_wrong_class_extra_and_empty():
    gt = [{"label": "pothole", "bbox": [400, 400, 600, 600]}]
    pred = {"label": "pothole", "bbox_2d": [400, 400, 600, 600]}
    assert iou_reward(json.dumps([pred]), gt) == 1
    assert iou_reward("invalid", gt) == 0
    assert iou_reward(json.dumps([{**pred, "label": "crack"}]), gt) == .2
    assert iou_reward(json.dumps([pred, pred]), gt) == pytest.approx(.6)
    assert iou_reward("[]", []) == 1
    assert iou_reward("[]", gt) == .2
