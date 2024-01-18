import re
from collections import defaultdict
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


def get_data(set_type: Literal["cross", "intra"], test_set_type: Literal["val", "test", "test1", "test2", "test3"], split_participants=False):
    train_datasets = get_datasets(set_type, "train", split_participants)
    train_X = standardize(np.stack([d for d, _ in train_datasets]))
    train_y = np.array([label for _, label in train_datasets])

    test_datasets = get_datasets(set_type, test_set_type, split_participants)
    test_X = standardize(np.stack([d for d, _ in test_datasets]))
    test_y = np.array([label for _, label in test_datasets])

    return train_X, train_y, test_X, test_y


def get_datasets(set_type: Literal["cross", "intra"], data_set: Literal["train", "val", "test", "test1", "test2", "test3"], split_participants=False)\
        -> list[tuple[list[np.ndarray], str]]:
    set_type = set_type.capitalize()

    directories = []
    if data_set == "train":
        directories.append(Path("data") / set_type / "train")
    elif data_set == "val":
        directories.append(Path("data") / set_type / "test1")
        directories.append(Path("data") / set_type / "test2")
    elif data_set == "test":
        test_folder_name = "test3" if set_type == "Cross" else "test"
        directories.append(Path("data") / set_type / test_folder_name)
    elif data_set == "test1":
        directories.append(Path("data") / set_type / "test1")
    elif data_set == "test2":
        directories.append(Path("data") / set_type / "test2")
    elif data_set == "test3":
        directories.append(Path("data") / set_type / "test3")

    data = defaultdict(list)
    labels = defaultdict(list)
    for directory in directories:
        for file in sorted(directory.glob("*.h5")):
            with h5py.File(file, 'r') as f:
                participant_id = re.search(r"\d+", file.stem).group(0)
                label = file.stem.split(participant_id)[0][:-1]

                data[participant_id].append(np.array(f[f"{label}_{participant_id}"]))
                labels[participant_id].append(label)

    if split_participants:
        data = [data[participant_id] for participant_id in data]
        labels = [labels[participant_id] for participant_id in labels]
    else:
        data = [d for participant_id in data for d in data[participant_id]]
        labels = [label for participant_id in labels for label in labels[participant_id]]

    return list(zip(data, labels))


def standardize(data: np.ndarray) -> np.ndarray:
    reshaped = data.reshape(-1, data.shape[-1])

    scaler = StandardScaler()
    standardized = scaler.fit_transform(reshaped)

    return standardized.reshape(data.shape)


def plot_activations(activations: np.ndarray):
    per_sensor_data = activations.transpose(1, 0, 2).reshape(248, -1)

    example = activations[0]

    fig, axs = plt.subplots(2, 1, figsize=(8, 7))
    axs[0].bar(np.arange(example.shape[0]), np.mean(example, axis=1), yerr=np.std(example, axis=1))
    axs[0].set_ylabel("Mean activation (standardised)")
    axs[0].set_title("Mean activation of each sensor for 1 example")

    axs[1].bar(np.arange(per_sensor_data.shape[0]), np.mean(per_sensor_data, axis=1),
               yerr=np.std(per_sensor_data, axis=1))
    axs[1].set_xlabel("Sensor #")
    axs[1].set_ylabel("Mean activation (standardised)")
    axs[1].set_title("Mean activation of each sensor across all examples")

    xticks = list(axs[0].get_xticks())
    xticks.remove(250)
    xticks.append(236)

    axs[0].set_xticks(xticks)
    axs[1].set_xticks(xticks)

    axs[0].set_xlim(0, 250)
    axs[1].set_xlim(0, 250)

    plt.show()

    # Big outlier! Might be worth removing
    outlier_mean = 1.5
    print(f"Sensors with absolute mean >{outlier_mean}:",
          np.where(abs(np.mean(per_sensor_data, axis=1)) > outlier_mean))


if __name__ == '__main__':
    get_data("cross", "val", split_participants=True)