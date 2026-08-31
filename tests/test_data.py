"""Tests for the data pipeline and the cross-validation protocol."""

import numpy as np
import pytest

from interhandnet.data import (
    CLASS_NAMES,
    WHO_STEP_NAMES,
    HandWashingSkeletonDataset,
    class_names,
    resample_sequence,
    temporal_kfold,
)
from interhandnet.data.skeleton_extraction import sliding_windows
from interhandnet.data.splits import camera_wise_folds, group_indices_by_camera, weighted_average
from interhandnet.data.transforms import center_hands, missing_joint_mask, to_model_layout
from interhandnet.graph import NUM_JOINTS, NUM_JOINTS_PER_HAND

NUM_SAMPLES, SOURCE_FRAMES = 24, 40


@pytest.fixture
def archive(tmp_path):
    """A small archive with two camera settings and monotonic time indices."""
    rng = np.random.default_rng(0)
    path = tmp_path / "skeletons.npz"
    np.savez(
        path,
        skeletons=rng.normal(size=(NUM_SAMPLES, SOURCE_FRAMES, NUM_JOINTS, 3)).astype(
            np.float32
        ),
        labels=np.arange(NUM_SAMPLES) % 6,
        cameras=np.array(["100"] * (NUM_SAMPLES // 2) + ["101"] * (NUM_SAMPLES // 2)),
        start_frames=np.arange(NUM_SAMPLES) * 30,
    )
    return path


class TestClassNames:
    def test_seven_classes_start_with_the_other_class(self):
        """The dataset's own label set: 0 = other, 1..6 = WHO steps."""
        names = class_names(7)
        assert names == CLASS_NAMES
        assert names[0].startswith("0:")
        assert names[1:] == WHO_STEP_NAMES

    def test_six_classes_are_the_who_steps_alone(self):
        assert class_names(6) == WHO_STEP_NAMES

    def test_unknown_label_set_falls_back_to_indices(self):
        assert class_names(3) == ("class 0", "class 1", "class 2")

    def test_a_name_exists_for_every_class(self):
        for num_classes in (3, 6, 7):
            assert len(class_names(num_classes)) == num_classes


class TestResample:
    def test_resamples_to_the_target_length(self):
        sequence = np.random.randn(SOURCE_FRAMES, NUM_JOINTS, 3).astype(np.float32)
        assert resample_sequence(sequence, 30).shape == (30, NUM_JOINTS, 3)

    def test_identity_when_lengths_match(self):
        sequence = np.random.randn(30, NUM_JOINTS, 3).astype(np.float32)
        assert np.allclose(resample_sequence(sequence, 30), sequence)

    def test_endpoints_are_preserved(self):
        sequence = np.arange(10, dtype=np.float32).reshape(10, 1, 1).repeat(3, axis=2)
        resampled = resample_sequence(sequence, 30)
        assert resampled[0, 0, 0] == pytest.approx(0.0)
        assert resampled[-1, 0, 0] == pytest.approx(9.0)

    def test_upsampling_is_supported(self):
        sequence = np.random.randn(7, NUM_JOINTS, 3).astype(np.float32)
        assert resample_sequence(sequence, 30).shape == (30, NUM_JOINTS, 3)

    def test_empty_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            resample_sequence(np.zeros((0, NUM_JOINTS, 3), dtype=np.float32), 30)


class TestTransforms:
    def test_model_layout_is_channels_first(self):
        sequence = np.random.randn(30, NUM_JOINTS, 3).astype(np.float32)
        assert to_model_layout(sequence).shape == (3, 30, NUM_JOINTS)

    def test_missing_joint_mask_flags_zeroed_joints(self):
        sequence = np.ones((4, NUM_JOINTS, 3), dtype=np.float32)
        sequence[1, :NUM_JOINTS_PER_HAND] = 0.0
        mask = missing_joint_mask(sequence)
        assert mask.shape == (4, NUM_JOINTS)
        assert mask[1, :NUM_JOINTS_PER_HAND].all()
        assert not mask[1, NUM_JOINTS_PER_HAND:].any()

    def test_center_hands_moves_the_palm_to_the_origin(self):
        sequence = np.random.randn(4, NUM_JOINTS, 3).astype(np.float32)
        centered = center_hands(sequence)
        assert np.allclose(centered[:, 0], 0.0, atol=1e-6)
        assert np.allclose(centered[:, NUM_JOINTS_PER_HAND], 0.0, atol=1e-6)

    def test_center_hands_leaves_undetected_hands_at_zero(self):
        sequence = np.random.randn(2, NUM_JOINTS, 3).astype(np.float32)
        sequence[0, :NUM_JOINTS_PER_HAND] = 0.0
        centered = center_hands(sequence)
        assert np.allclose(centered[0, :NUM_JOINTS_PER_HAND], 0.0)


class TestDataset:
    def test_length_and_item_shape(self, archive):
        dataset = HandWashingSkeletonDataset(archive, window_size=30)
        assert len(dataset) == NUM_SAMPLES
        skeleton, label = dataset[0]
        assert tuple(skeleton.shape) == (3, 30, NUM_JOINTS)
        assert label.item() in range(6)

    def test_subset_is_a_view(self, archive):
        dataset = HandWashingSkeletonDataset(archive)
        subset = dataset.subset([0, 5, 9])
        assert len(subset) == 3
        assert np.allclose(subset[1][0].numpy(), dataset[5][0].numpy())

    def test_class_counts_sum_to_the_subset_size(self, archive):
        dataset = HandWashingSkeletonDataset(archive).subset(range(10))
        assert dataset.class_counts().sum() == 10

    def test_missing_archive_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HandWashingSkeletonDataset(tmp_path / "nope.npz")


class TestTemporalKFold:
    def test_folds_partition_the_samples(self):
        folds = temporal_kfold(range(20), num_folds=5)
        assert len(folds) == 5
        validation = np.concatenate([validation for _, validation in folds])
        assert sorted(validation) == list(range(20))

    def test_validation_blocks_are_contiguous_in_time(self):
        for _, validation in temporal_kfold(range(20), num_folds=5):
            assert np.all(np.diff(validation) == 1)

    def test_gap_removes_neighbouring_training_windows(self):
        train, validation = temporal_kfold(range(20), num_folds=5, gap=2)[2]
        assert not set(train) & set(validation)
        # The two windows on each side of the validation block are dropped.
        assert validation[0] - 1 not in set(train)
        assert validation[-1] + 1 not in set(train)
        assert len(train) == 20 - len(validation) - 4

    def test_train_and_validation_never_overlap(self):
        for train, validation in temporal_kfold(range(23), num_folds=5, gap=1):
            assert not set(train) & set(validation)

    def test_too_few_samples_is_rejected(self):
        with pytest.raises(ValueError, match="cannot build"):
            temporal_kfold(range(3), num_folds=5)

    def test_single_fold_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            temporal_kfold(range(10), num_folds=1)


class TestCameraGrouping:
    def test_groups_are_sorted_by_time(self):
        cameras = np.array(["a", "b", "a", "b"])
        start_frames = np.array([30, 10, 0, 40])
        groups = group_indices_by_camera(cameras, start_frames)
        assert list(groups["a"]) == [2, 0]
        assert list(groups["b"]) == [1, 3]

    def test_missing_camera_metadata_forms_one_group(self):
        groups = group_indices_by_camera(None, np.arange(5))
        assert len(groups) == 1

    def test_camera_wise_folds_cover_every_camera(self, archive):
        dataset = HandWashingSkeletonDataset(archive)
        folds = list(
            camera_wise_folds(dataset.cameras, dataset.start_frames, num_folds=3, gap=0)
        )
        assert len(folds) == 6  # two cameras x three folds
        assert {str(camera) for camera, _, _, _ in folds} == {"100", "101"}

    def test_folds_never_mix_cameras(self, archive):
        dataset = HandWashingSkeletonDataset(archive)
        cameras = np.asarray(dataset.cameras)
        for camera, _, train, validation in camera_wise_folds(
            dataset.cameras, dataset.start_frames, num_folds=3
        ):
            assert set(cameras[train]) == {camera}
            assert set(cameras[validation]) == {camera}


class TestWeightedAverage:
    def test_matches_the_manual_computation(self):
        assert weighted_average([1.0, 0.0], [3, 1]) == pytest.approx(0.75)

    def test_zero_weights_are_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            weighted_average([1.0, 2.0], [0, 0])


class TestSlidingWindows:
    def test_non_overlapping_windows(self):
        sequence = np.random.randn(95, NUM_JOINTS, 3).astype(np.float32)
        windows, starts = sliding_windows(sequence, window_size=30)
        assert windows.shape == (3, 30, NUM_JOINTS, 3)
        assert list(starts) == [0, 30, 60]

    def test_stride_controls_the_overlap(self):
        sequence = np.random.randn(60, NUM_JOINTS, 3).astype(np.float32)
        windows, starts = sliding_windows(sequence, window_size=30, stride=15)
        assert len(windows) == len(starts) == 2
        assert list(starts) == [0, 15]

    def test_short_sequence_can_be_padded(self):
        sequence = np.ones((10, NUM_JOINTS, 3), dtype=np.float32)
        windows, starts = sliding_windows(sequence, window_size=30, drop_last=False)
        assert windows.shape == (1, 30, NUM_JOINTS, 3)
        assert np.allclose(windows[0, 10:], 0.0)
        assert list(starts) == [0]
