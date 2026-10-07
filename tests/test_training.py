import numpy as np
import pytest

from neural_network_lab.training import iterate_minibatches


def test_iterate_minibatches_preserves_order_and_final_partial_batch():
    inputs = np.arange(20).reshape(10, 2)
    targets = np.arange(10)
    batch_size = 4
    shuffle = False

    batches = list(iterate_minibatches(inputs, targets, batch_size, shuffle))

    assert len(batches) == 3

    batch_sizes = [
        batch_inputs.shape[0]
        for batch_inputs, _ in batches
    ]
    assert batch_sizes == [4, 4, 2]

    reconstructed_inputs = np.concatenate([
        batch_inputs
        for batch_inputs, _ in batches
    ])
    reconstructed_targets = np.concatenate([
        batch_targets
        for _, batch_targets in batches
    ])

    np.testing.assert_array_equal(
        reconstructed_inputs,
        inputs,
    )
    np.testing.assert_array_equal(
        reconstructed_targets,
        targets,
    )


def test_iterate_minibatches_shuffles_reproducibly_and_preserves_pairs():
    inputs = np.arange(20).reshape(10, 2)
    targets = np.arange(10)

    first_batches = list(
        iterate_minibatches(
            inputs,
            targets,
            batch_size=4,
            shuffle=True,
            seed=42,
        )
    )
    second_batches = list(
        iterate_minibatches(
            inputs,
            targets,
            batch_size=4,
            shuffle=True,
            seed=42,
        )
    )

    first_inputs = np.concatenate([
        batch_inputs
        for batch_inputs, _ in first_batches
    ])
    first_targets = np.concatenate([
        batch_targets
        for _, batch_targets in first_batches
    ])

    second_inputs = np.concatenate([
        batch_inputs
        for batch_inputs, _ in second_batches
    ])
    second_targets = np.concatenate([
        batch_targets
        for _, batch_targets in second_batches
    ])

    np.testing.assert_array_equal(
        first_inputs,
        second_inputs,
    )
    np.testing.assert_array_equal(
        first_targets,
        second_targets,
    )
    np.testing.assert_array_equal(
        first_inputs,
        inputs[first_targets],
    )


@pytest.mark.parametrize(
    "invalid_batch_size",
    [
        0,
        -1,
        1.5,
        True,
    ],
)
def test_iterate_minibatches_rejects_invalid_batch_sizes(
    invalid_batch_size,
):
    inputs = np.arange(12).reshape(6, 2)
    targets = np.arange(6)

    with pytest.raises(ValueError):
        list(
            iterate_minibatches(
                inputs,
                targets,
                batch_size=invalid_batch_size,
            )
        )


@pytest.mark.parametrize(
    ("inputs", "targets"),
    [
        (1.0, [0]),
        (np.ones((3, 2)), np.arange(2)),
        (np.empty((0, 2)), np.array([])),
    ],
)
def test_iterate_minibatches_rejects_invalid_datasets(
    inputs,
    targets,
):
    with pytest.raises(ValueError):
        list(
            iterate_minibatches(
                inputs,
                targets,
                batch_size=2,
            )
        )


@pytest.mark.parametrize(
    "invalid_shuffle",
    [
        0,
        1,
        "yes",
        None,
    ],
)
def test_iterate_minibatches_rejects_non_boolean_shuffle(
    invalid_shuffle,
):
    inputs = np.arange(12).reshape(6, 2)
    targets = np.arange(6)

    with pytest.raises(ValueError):
        list(
            iterate_minibatches(
                inputs,
                targets,
                batch_size=2,
                shuffle=invalid_shuffle,
            )
        )