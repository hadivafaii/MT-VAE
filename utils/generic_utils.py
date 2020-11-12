import numpy as np
from datetime import datetime
from sklearn.metrics import mutual_info_score


def compute_mi(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi, c_xy


def to_np(x):
    if isinstance(x, np.ndarray):
        return x
    return x.data.cpu().numpy()


def convert_time(time_in_secs):

    d = time_in_secs // 86400
    h = (time_in_secs - d * 86400) // 3600
    m = (time_in_secs - d * 86400 - h * 3600) // 60
    s = time_in_secs - d * 86400 - h * 3600 - m * 60

    print("\nd / hh:mm:ss   --->   %d / %d:%d:%d\n" % (d, h, m, s))


def now(exclude_hour_min: bool = True):
    if exclude_hour_min:
        return datetime.now().strftime("(%Y_%m_%d)")
    else:
        return datetime.now().strftime("(%Y_%m_%d_%H-%M)")
