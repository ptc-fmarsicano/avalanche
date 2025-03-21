import warnings
from typing import (
    Optional,
    Sequence,
    Callable,
    Any,
    Dict,
    Tuple,
    Union,
    List,
    Collection,
)

from avalanche.benchmarks.utils import SupportedDataset
from avalanche.benchmarks.utils.classification_dataset import (
    TTargetType,
    classification_subset,
    make_tensor_classification_dataset,
    concat_classification_datasets,
)
from avalanche.benchmarks.utils.transform_groups import XTransform, YTransform


def AvalanceDataset():
    warnings.warn(
        "AvalancheDataset has been deprecated and it will be removed in 0.4. "
        "Use `avalanche.benchmarks.ClassificationDataset` instead.`",
        DeprecationWarning,
    )


def AvalancheSubset(
    dataset: SupportedDataset,
    indices: Optional[Sequence[int]] = None,
    *,
    class_mapping: Optional[Sequence[int]] = None,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Dict[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Sequence[TTargetType]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None,
):
    warnings.warn(
        "AvalancheDataset has been deprecated and it will be removed in 0.4. "
        "Please use `AvalancheDataset` `subset` method to create subsets.`",
        DeprecationWarning,
    )
    return classification_subset(
        dataset,
        indices,
        class_mapping=class_mapping,
        transform=transform,
        target_transform=target_transform,
        transform_groups=transform_groups,
        initial_transform_group=initial_transform_group,
        task_labels=task_labels,
        targets=targets,
        collate_fn=collate_fn,
    )


def AvalancheTensorDataset(
    *dataset_tensors: Sequence,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Dict[str, Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = "train",
    task_labels: Optional[Union[int, Sequence[int]]] = None,
    targets: Optional[Union[Sequence[TTargetType], int]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None,
):
    warnings.warn(
        "AvalancheDataset has been deprecated and it will be removed in 0.4. "
        "Please use `avalanche.benchmarks.make_tensor_classification_dataset` "
        "instead.`",
        DeprecationWarning,
    )
    return make_tensor_classification_dataset(
        dataset_tensors,
        transform=transform,
        target_transform=target_transform,
        transform_groups=transform_groups,
        initial_transform_group=initial_transform_group,
        task_labels=task_labels,
        targets=targets,
        collate_fn=collate_fn,
    )


def AvalancheConcatDataset(
    datasets: Collection[SupportedDataset],
    *,
    transform: Optional[XTransform] = None,
    target_transform: Optional[YTransform] = None,
    transform_groups: Optional[Dict[str, 
                                    Tuple[XTransform, YTransform]]] = None,
    initial_transform_group: Optional[str] = None,
    task_labels: Optional[Union[int, 
                                Sequence[int],
                                Sequence[Sequence[int]]]] = None,
    targets: Optional[Union[
        Sequence[TTargetType], Sequence[Sequence[TTargetType]]
    ]] = None,
    collate_fn: Optional[Callable[[List], Any]] = None,
):
    warnings.warn(
        "AvalancheDataset has been deprecated and it will be removed in 0.4. "
        "Please use `AvalancheDataset` `concat` method to concatenate "
        "datasets.`",
        DeprecationWarning,
    )
    return concat_classification_datasets(
        list(datasets),
        transform=transform,
        target_transform=target_transform,
        transform_groups=transform_groups,
        initial_transform_group=initial_transform_group,
        task_labels=task_labels,
        targets=targets,
        collate_fn=collate_fn,
    )


__all__ = [
    "AvalancheSubset",
    "AvalancheTensorDataset",
    "AvalancheConcatDataset",
]
