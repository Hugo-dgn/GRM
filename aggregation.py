import numpy as np

from skimage import color

def mean_agg(x):
    return np.mean(x, axis=0)

def mean_var_agg(x):
    mu = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    return np.concatenate(mu, var)

def lab_mean_var_agg(x):
    x = x / 255.0

    x_lab = color.rgb2lab(x.reshape(-1, 1, 3)).reshape(-1, 3)

    mean = np.mean(x_lab, axis=0)
    std = np.std(x_lab, axis=0)

    return np.concatenate([mean, std])

def lab_mean_var_hist_agg(x, bins=8):
    
    x = x / 255.0

    x_lab = color.rgb2lab(x.reshape(-1, 1, 3)).reshape(-1, 3)

    mean = np.mean(x_lab, axis=0)
    std = np.std(x_lab, axis=0)

    # Fixed ranges for Lab
    ranges = [
        (0, 100),      # L
        (-128, 127),   # a
        (-128, 127)    # b
    ]

    hist_features = []
    for i in range(3):
        hist, _ = np.histogram(
            x_lab[:, i],
            bins=bins,
            range=ranges[i],
            density=True
        )
        hist_features.append(hist)

    hist_features = np.concatenate(hist_features)

    return np.concatenate([mean, std, hist_features])