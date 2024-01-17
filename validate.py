import h5py
import numpy as np
import tqdm

import pandas as pd

from pathlib import Path
import torch


OLD_FREQ = 2034


NEW_FREQ = OLD_FREQ // 203
BATCH_SIZE = 12
RUN_PATH = "./checkpoints/djotx6he_4.pt"
print(f"{NEW_FREQ=}, {RUN_PATH=}, {BATCH_SIZE=}, ")


def load_labels(path: Path) -> np.ndarray:
    *task, subject_identifier, chunk = path.stem.split("_")
    if "rest" in task:
        y = 0
    elif "math" in task:
        y = 1
    elif "working" in task:
        y = 2
    elif "motor" in task:
        y = 3
    else:
        assert False, "unknown task"
    return np.array([y, int(subject_identifier), int(chunk)])


def downsample(data, old_freq, new_freq):
    # Calculate the downsampling factor
    downsample_factor = old_freq // new_freq

    # Reshape the data to prepare for downsampling
    reshaped_data = data[
        :, : data.shape[1] // downsample_factor * downsample_factor
    ].reshape(data.shape[0], -1, downsample_factor)

    # Perform the downsampling by taking the mean along the last axis
    downsampled_data = reshaped_data.mean(axis=-1)

    return downsampled_data


def load_h5(path: Path) -> np.ndarray:
    with h5py.File(path) as f:
        keys = f.keys()
        assert len(keys) == 1, f"Only one key per file, right? {path}"
        matrix = f.get(next(iter(keys)))[()]
    return downsample(matrix, OLD_FREQ, NEW_FREQ)
    return matrix


def reshape_X(X: np.ndarray) -> np.ndarray:
    m_flat = "0 0 0 0 0 0 0 0 0 0  121  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  122   90   89   120   152  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  123   91   62   61   88   119   151  0 0 0 0 0 0 0 0 0 0 0 0 0  124   92   63   38   37   60   87   118   150  0 0 0 0 0 0 0 0 0 0  177   153   93   64   39   20   19   36   59   86   117   149   176   195  0 0 0  229   212   178   154   126   94   65   40   21   6   5   18   35   58   85   116   148   175   194   228   248   230   213   179   155   127   95   66   41   22   7   4   17   34   57   84   115   147   174   193   227   247  0  231   196   156   128   96   67   42   23   8   3   16   33   56   83   114   146   173   211   246  0 0  232   197   157   129   97   68   43   24   9   2   15   32   55   82   113   145   172   210   245  0 0  233   198   158   130   98   69   44   25   10   1   14   31   54   81   112   144   171   209   244  0 0 0  214   180   131   99   70   45   26   11   12   13   30   53   80   111   143   192   226  0 0 0 0 0 0  159   132   100   71   46   27   28   29   52   79   110   142   170  0 0 0 0 0 0 0  181   160   133   101   72   47   48   49   50   51   78   109   141   169   191  0 0 0 0 0  215   199   182   161   134   102   73   74   75   76   77   108   140   168   190   208   225  0 0 0 0  234   216   200   183   162   135   103   104   105   106   107   139   167   189   207   224   243  0 0 0 0 0 0  235   217   201   184   163   136   137   138   166   188   206   223   242  0 0 0 0 0 0 0 0 0 0  236   218   202   185   164   165   187   205   222   241  0 0 0 0 0 0 0 0 0 0 0 0 0  219   203   186   204   221  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  237   220   240  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  238   239  0 0 0 0 0 0 0 0 0 "
    m_t = np.array(m_flat.split(), dtype=int).reshape(20, 21)
    # n, 248, t -> n, 21, 20, t
    reshaped_data = X[:, m_t - 1]
    reshaped_data[:, m_t == 0] = 0
    # n, 21, 20, t -> n, t, 21, 21

    vitshaped_data = np.pad(
        reshaped_data, ((0, 0), (1, 0), (0, 0), (0, 0)), constant_values=0
    ).reshape(
        (
            reshaped_data.shape[0],
            reshaped_data.shape[-1],
            1,
            21,
            21,
        )
    )

    return vitshaped_data


class Scaler:
    def __init__(self):
        self.mean, self.std = None, None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            self.mean = X.mean(0, keepdims=True)
            self.std = X.std(0, keepdims=True)
        return (X - self.mean) / self.std


def load_X_y(path: Path, scaler=Scaler()):
    X = reshape_X(scaler(np.stack(list(map(load_h5, tqdm.tqdm(path))))))
    y = np.stack(list(map(load_labels, path)))[:, 0]
    return torch.tensor(X).float().to("cuda"), torch.tensor(y).long()

from transformers import VivitConfig, VivitForVideoClassification

data_dir = Path("./data/")
assert data_dir.is_dir()
intra_dir = data_dir / "Intra"
cross_dir = data_dir / "Cross"
intra_train_glob = list((intra_dir / "train").glob("*.h5"))
intra_test_glob = list((intra_dir / "test").glob("*.h5"))

cross_dir_glob = (
    list((cross_dir / "train").glob("*.h5")),
    list((cross_dir / "test1").glob("*.h5")),
    list((cross_dir / "test2").glob("*.h5")),
    list((cross_dir / "test3").glob("*.h5")),
)
(X_train, y_train), (X_test1, y_test1), (X_test2, y_test2), (X_test3, y_test3) = map(load_X_y, cross_dir_glob)

train_dataset = torch.utils.data.dataset.TensorDataset(X_train, y_train)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

num_frames = X_train.shape[1]


conf = VivitConfig(
    image_size=21, num_frames=num_frames, num_channels=1, num_labels=4
)
state_dict = torch.load(RUN_PATH)
vivit_model = VivitForVideoClassification(conf).to("cuda")
vivit_model.load_state_dict(state_dict)
vivit_model.eval()

train_logits = list()
with torch.no_grad():
    for X_batch, y_batch in train_loader:
        train_logits.append(vivit_model(
            pixel_values=X_batch.to("cuda"), labels=y_batch.to("cuda")
        ).logits.argmax(-1).to("cpu").numpy())
        
    test1_out = vivit_model(
        pixel_values=X_test1.to("cuda"), labels=y_test1.to("cuda")
    ).logits.argmax(-1).to("cpu").numpy()
    test2_out = vivit_model(
        pixel_values=X_test2.to("cuda"), labels=y_test2.to("cuda")
    ).logits.argmax(-1).to("cpu").numpy()
    test3_out = vivit_model(
        pixel_values=X_test3.to("cuda"), labels=y_test3.to("cuda")
    ).logits.argmax(-1).to("cpu").numpy()
    
    



# plot_results(train_losses, test_losses, train_accs, test_accs)

