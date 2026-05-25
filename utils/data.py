from datasets import load_dataset, Dataset, concatenate_datasets
from config.train_sft_model import DatasetConfig


def load_datasets(
    cfg: DatasetConfig, seed: int, skip_first_train_examples: int = 0
) -> Dataset:
    """
    Load training and validation datasets based on the configuration.
    Args:
        cfg (DatasetConfig): Configuration containing dataset information.
        seed (int): Random seed for reproducibility.
        skip_first_train_examples (int): Number of initial shuffled train examples
            to skip before selecting max_train_examples.
    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        
    If no datasets are provided or an error occurs, returns None for the respective dataset.
    """
    train_datasets, val_datasets = [], []

    try:
        for dataset in cfg.train_datasets:
            dataset = load_dataset(dataset.name_or_path, split=dataset.split)
            train_datasets.append(dataset)

        # We sample based on max_examples and ratios.
        if cfg.max_train_examples is not None:
            ratios = [dataset.ratio for dataset in cfg.train_datasets]
            total_ratio = sum(ratios)
            num_samples = cfg.max_train_examples
            skip_samples = max(0, int(skip_first_train_examples))

            # Sample based on ratios
            if num_samples == -1:
                for i, dataset in enumerate(train_datasets):
                    shuffled = dataset.shuffle(seed=seed)
                    if skip_samples > 0:
                        start = min(skip_samples, len(shuffled))
                        train_datasets[i] = shuffled.select(range(start, len(shuffled)))
                    else:
                        train_datasets[i] = shuffled
            else:
                samples_per_dataset = [
                    int(num_samples * ratio / total_ratio) for ratio in ratios
                ]
                skip_per_dataset = [
                    int(skip_samples * ratio / total_ratio) for ratio in ratios
                ]
                for i, dataset in enumerate(train_datasets):
                    shuffled = dataset.shuffle(seed=seed)
                    start = min(skip_per_dataset[i], len(shuffled))
                    end = min(start + samples_per_dataset[i], len(shuffled))
                    train_datasets[i] = shuffled.select(range(start, end))
        # Concatenate the datasets
        train_datasets = concatenate_datasets(train_datasets)
        train_datasets = train_datasets.shuffle(seed=seed)
    except Exception as e:
        train_datasets = None
        print("No training datasets provided or an error occurred while loading them.")
        print(e)

    try:
        for dataset in cfg.eval_datasets:
            dataset = load_dataset(dataset.name_or_path, split=dataset.split)
            val_datasets.append(dataset)

        if len(val_datasets) == 0:
            return train_datasets, None

        if cfg.max_val_examples is not None:
            ratios = [dataset.ratio for dataset in cfg.eval_datasets]
            total_ratio = sum(ratios)
            num_samples = cfg.max_val_examples

            # Sample based on ratios
            if num_samples == -1:
                for i, dataset in enumerate(val_datasets):
                    val_datasets[i] = dataset.shuffle(seed=seed)
            else:
                samples_per_dataset = [
                    int(num_samples * ratio / total_ratio) for ratio in ratios
                ]
                for i, dataset in enumerate(val_datasets):
                    val_datasets[i] = dataset.shuffle(seed=seed).select(
                        range(samples_per_dataset[i])
                    )

        val_datasets = concatenate_datasets(val_datasets)
        val_datasets = val_datasets.shuffle(seed=seed)
    except Exception as e:
        val_datasets = None
        print(
            "No validation datasets provided or an error occurred while loading them."
        )
        print(e)

    return train_datasets, val_datasets

def load_whole_datasets(cfg: DatasetConfig, seed: int) -> Dataset:
    
    train_datasets, val_datasets = [], []

    try:
        for dataset in cfg.train_datasets:
            dataset = load_dataset(dataset.name_or_path, split=dataset.split)
            train_datasets.append(dataset)

        # We sample based on max_examples and ratios.
        if cfg.max_train_examples is not None:
            ratios = [dataset.ratio for dataset in cfg.train_datasets]
            total_ratio = sum(ratios)

            for i, dataset in enumerate(train_datasets):
                train_datasets[i] = dataset.shuffle(seed=seed)

        # Concatenate the datasets
        train_datasets = concatenate_datasets(train_datasets)
        train_datasets = train_datasets.shuffle(seed=seed)
    except Exception as e:
        train_datasets = None
        print("No training datasets provided or an error occurred while loading them.")
        print(e)

    try:
        for dataset in cfg.eval_datasets:
            dataset = load_dataset(dataset.name_or_path, split=dataset.split)
            val_datasets.append(dataset)

        if len(val_datasets) == 0:
            return train_datasets, None

        if cfg.max_val_examples is not None:
            ratios = [dataset.ratio for dataset in cfg.eval_datasets]
            total_ratio = sum(ratios)

            for i, dataset in enumerate(val_datasets):
                val_datasets[i] = dataset.shuffle(seed=seed)

        val_datasets = concatenate_datasets(val_datasets)
        val_datasets = val_datasets.shuffle(seed=seed)
    except Exception as e:
        val_datasets = None
        print(
            "No validation datasets provided or an error occurred while loading them."
        )
        print(e)

    return train_datasets, val_datasets
