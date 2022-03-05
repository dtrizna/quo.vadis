import numpy as np

def pad_array(arr, length=150, drop="middle"):
    """Function that takes arbitrary length array and returns either padded or truncated array.

    Args:
        arr (np.ndarray): Arbitrary length numpy array.
        length (int, optional): Length of returned array. Defaults to 150.
        how (str): choice ["last", "middle"] - specifies how to perform truncation.

    Returns:
        [np.ndarray]: Fixed length array.
    """
    if arr.shape[0] < length:
        # pad with 0 at the end
        return np.pad(arr, [0, length-arr.shape[0]], mode="constant", constant_values=0)
    else:
        if drop == "middle":
            # take only "length" characters, but drop middle ones
            return np.hstack([arr[:int(length/2)],arr[-round(length/2):]])
        elif drop == "first":
            # take only last "length" characters
            return arr[-length:]
        elif drop == "last":
            return arr[:length]
        else:
            raise NotImplementedError


def api_filter(rawseq, apimap):
    """Function that takes array and replaces all the bytes not in keep_bytes by 1.

    Args:
        arr (list): list containing API calls from a single emulation report
        apimap (dict): dictionary with API call keys and values of integers

    Returns:
        np.array: Filtered array with API calls as integers from apimap + 1 value for rare calls
    """
    seq = [apimap[x] if x in apimap.keys() else 1 for x in rawseq]
    return np.array(seq, dtype=int)
