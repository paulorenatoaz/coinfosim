"""Transfer sampler: synthetic training, real fixed test set.

:class:`SyntheticTrainRealTestSampler` composes an existing training sampler
(for example a :class:`~coinfosim.samplers.gaussian.GaussianClassConditionalSampler`)
with a fixed real evaluation :class:`~coinfosim.samplers.dataset.Dataset`.

Training samples are drawn from the wrapped ``train_sampler`` (synthetic data),
while :meth:`sample_test` always returns the fixed real test dataset. This
implements the "synthetic training, real evaluation" transfer setting used by
the Occupancy ``single_gaussian_to_real`` arm, and is generic enough to later
support other synthetic training distributions (e.g. Gaussian mixtures) without
changing the Monte Carlo simulator.
"""

from __future__ import annotations

from typing import Optional

from coinfosim.samplers.dataset import Dataset


class SyntheticTrainRealTestSampler:
    """Sampler that trains on synthetic data and tests on a fixed real set.

    Parameters
    ----------
    train_sampler:
        Any sampler exposing ``model`` and
        ``sample_train(n_per_class, replication_id)``. Its training samples are
        used unchanged for every sample size and replication.
    test_dataset:
        The fixed real evaluation :class:`Dataset` returned by
        :meth:`sample_test`.
    name:
        Optional label for metadata/reporting.
    """

    def __init__(
        self,
        train_sampler,
        test_dataset: Dataset,
        name: Optional[str] = None,
    ) -> None:
        train_model = getattr(train_sampler, "model", None)
        train_d = getattr(train_model, "d", None)
        if train_d is not None and int(train_d) != int(test_dataset.d):
            raise ValueError(
                "train_sampler feature dimension and test_dataset feature "
                f"dimension must match; got train d={int(train_d)} and test "
                f"d={int(test_dataset.d)}"
            )

        self._train_sampler = train_sampler
        self._test_dataset = test_dataset
        self.name = name or "synthetic_train_real_test"

    @property
    def model(self):
        """Expose the training model (used by the simulator for d/labels)."""
        return self._train_sampler.model

    @property
    def base_seed(self) -> int:
        return int(getattr(self._train_sampler, "base_seed", 0))

    @property
    def train_sampler(self):
        return self._train_sampler

    @property
    def test_dataset(self) -> Dataset:
        return self._test_dataset

    def sample_train(self, n_per_class: int, replication_id: int) -> Dataset:
        """Draw a training :class:`Dataset` from the wrapped synthetic sampler."""
        return self._train_sampler.sample_train(
            n_per_class=n_per_class,
            replication_id=replication_id,
        )

    def sample_test(self) -> Dataset:
        """Return the fixed real evaluation :class:`Dataset`."""
        return self._test_dataset
