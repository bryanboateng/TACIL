import numpy as np

TOLERANCE = 1e-9


def calculate_area(
    *,
    corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
) -> np.ndarray:  # (batch_size,)
    return np.abs(_calculate_signed_area(corner_batch=corner_batch))


def calculate_intersection_area(
    *,
    first_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
    second_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
) -> np.ndarray:  # (batch_size,)
    # Pairs whose enclosing circles stay apart cannot overlap. Dropping them
    # first keeps the costly part off most of the batch.
    overlap_batch = _find_possible_overlaps(
        first_corner_batch=first_corner_batch, second_corner_batch=second_corner_batch
    )

    area_batch = np.zeros(shape=len(first_corner_batch))
    if overlap_batch.any():
        corner_batch, validity_batch = _collect_corners(
            first_corner_batch=first_corner_batch[overlap_batch],
            second_corner_batch=second_corner_batch[overlap_batch],
        )
        area_batch[overlap_batch] = _calculate_area(
            corner_batch=corner_batch, validity_batch=validity_batch
        )

    return area_batch


# ---- Private --------------------------------------------


def _find_possible_overlaps(
    *,
    first_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
    second_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
) -> np.ndarray:  # (batch_size,)
    first_center_batch = first_corner_batch.mean(axis=1)  # (batch_size, 2)
    second_center_batch = second_corner_batch.mean(axis=1)  # (batch_size, 2)

    return np.linalg.norm(
        first_center_batch - second_center_batch, axis=-1
    ) <= _calculate_radius(
        corner_batch=first_corner_batch, center_batch=first_center_batch
    ) + _calculate_radius(
        corner_batch=second_corner_batch, center_batch=second_center_batch
    )


def _calculate_radius(
    *,
    corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
    center_batch: np.ndarray,  # (batch_size, 2)
) -> np.ndarray:  # (batch_size,)
    return np.linalg.norm(corner_batch - center_batch[:, None, :], axis=-1).max(axis=1)


def _collect_corners(
    *,
    first_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
    second_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
) -> tuple[np.ndarray, np.ndarray]:
    # (batch_size, candidate_count, 2), (batch_size, candidate_count)
    crossing_batch, crossing_validity_batch = _calculate_crossings(
        first_corner_batch=first_corner_batch, second_corner_batch=second_corner_batch
    )

    return (
        np.concatenate(
            [first_corner_batch, second_corner_batch, crossing_batch], axis=1
        ),
        np.concatenate(
            [
                _calculate_containment(
                    point_batch=first_corner_batch, corner_batch=second_corner_batch
                ),
                _calculate_containment(
                    point_batch=second_corner_batch, corner_batch=first_corner_batch
                ),
                crossing_validity_batch,
            ],
            axis=1,
        ),
    )


def _calculate_containment(
    *,
    point_batch: np.ndarray,  # (batch_size, point_count, 2)
    corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
) -> np.ndarray:  # (batch_size, point_count)
    edge_batch = np.roll(a=corner_batch, shift=-1, axis=1) - corner_batch

    relative_batch = (
        point_batch[:, :, None, :] - corner_batch[:, None, :, :]
    )  # (batch_size, point_count, corner_count, 2)

    side_batch = (
        edge_batch[:, None, :, 0] * relative_batch[..., 1]
        - edge_batch[:, None, :, 1] * relative_batch[..., 0]
    )  # (batch_size, point_count, corner_count)

    # A polygon may be wound either way, so the side that counts as inside is
    # taken from the polygon itself.
    winding_batch = np.sign(_calculate_signed_area(corner_batch=corner_batch))

    return np.all(side_batch * winding_batch[:, None, None] >= -TOLERANCE, axis=-1)


def _calculate_crossings(
    *,
    first_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
    second_corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
) -> tuple[np.ndarray, np.ndarray]:
    # (batch_size, corner_count ** 2, 2), (batch_size, corner_count ** 2)
    first_edge_batch = np.roll(a=first_corner_batch, shift=-1, axis=1) - (
        first_corner_batch
    )
    second_edge_batch = np.roll(a=second_corner_batch, shift=-1, axis=1) - (
        second_corner_batch
    )

    denominator_batch = (
        first_edge_batch[:, :, None, 0] * second_edge_batch[:, None, :, 1]
        - first_edge_batch[:, :, None, 1] * second_edge_batch[:, None, :, 0]
    )  # (batch_size, corner_count, corner_count)

    offset_batch = (
        second_corner_batch[:, None, :, :] - first_corner_batch[:, :, None, :]
    )  # (batch_size, corner_count, corner_count, 2)

    is_crossing_batch = np.abs(denominator_batch) > TOLERANCE
    safe_denominator_batch = np.where(is_crossing_batch, denominator_batch, 1.0)

    first_fraction_batch = (
        offset_batch[..., 0] * second_edge_batch[:, None, :, 1]
        - offset_batch[..., 1] * second_edge_batch[:, None, :, 0]
    ) / safe_denominator_batch
    second_fraction_batch = (
        offset_batch[..., 0] * first_edge_batch[:, :, None, 1]
        - offset_batch[..., 1] * first_edge_batch[:, :, None, 0]
    ) / safe_denominator_batch

    validity_batch = (
        is_crossing_batch
        & (first_fraction_batch >= 0.0)
        & (first_fraction_batch <= 1.0)
        & (second_fraction_batch >= 0.0)
        & (second_fraction_batch <= 1.0)
    )  # (batch_size, corner_count, corner_count)

    crossing_batch = (
        first_corner_batch[:, :, None, :]
        + first_fraction_batch[..., None] * first_edge_batch[:, :, None, :]
    )  # (batch_size, corner_count, corner_count, 2)

    batch_size = len(first_corner_batch)
    return (
        crossing_batch.reshape(batch_size, -1, 2),
        validity_batch.reshape(batch_size, -1),
    )


def _calculate_area(
    *,
    corner_batch: np.ndarray,  # (batch_size, candidate_count, 2)
    validity_batch: np.ndarray,  # (batch_size, candidate_count)
) -> np.ndarray:  # (batch_size,)
    valid_count_batch = validity_batch.sum(axis=1)  # (batch_size,)

    center_batch = (
        np.sum(corner_batch * validity_batch[..., None], axis=1)
        / np.maximum(valid_count_batch, 1)[:, None]
    )  # (batch_size, 2)

    relative_batch = corner_batch - center_batch[:, None, :]
    angle_batch = np.where(
        validity_batch,
        np.arctan2(relative_batch[..., 1], relative_batch[..., 0]),
        np.inf,
    )  # (batch_size, candidate_count)

    order_batch = np.argsort(a=angle_batch, axis=1)
    sorted_corner_batch = np.take_along_axis(
        corner_batch, order_batch[..., None], axis=1
    )
    sorted_validity_batch = np.take_along_axis(validity_batch, order_batch, axis=1)

    # Invalid corners sort to the end. Repeating the first corner there closes
    # the ring early and adds no area.
    ring_batch = np.where(
        sorted_validity_batch[..., None],
        sorted_corner_batch,
        sorted_corner_batch[:, :1, :],
    )

    return np.where(
        valid_count_batch >= 3,
        np.abs(_calculate_signed_area(corner_batch=ring_batch)),
        0.0,
    )


def _calculate_signed_area(
    *,
    corner_batch: np.ndarray,  # (batch_size, corner_count, 2)
) -> np.ndarray:  # (batch_size,)
    next_corner_batch = np.roll(a=corner_batch, shift=-1, axis=1)
    return 0.5 * np.sum(
        corner_batch[..., 0] * next_corner_batch[..., 1]
        - next_corner_batch[..., 0] * corner_batch[..., 1],
        axis=1,
    )
