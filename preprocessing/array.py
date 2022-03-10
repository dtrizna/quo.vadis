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


def rawseq2array(rawseq, apimap, padding_length):
    """Function that transforms raw API sequence list to encoded array of fixed length."""
    filtered_seq = api_filter(rawseq, apimap)
    v = pad_array(filtered_seq, padding_length, drop="middle")
    return v


def byte_filter(arr, keep_bytes):
    """Function that takes array and replaces all the bytes not in keep_bytes by 1.

    Args:
        arr (np.ndarray): array where to replace rare bytes (not int keep_bytes)
        keep_bytes (list, np.ndarray): iterable that contains the bytes to keep

    Returns:
        np.ndarray: Filtered array with only keep_bytes.uniq() +1 characters
    """
    mask = np.isin(arr, keep_bytes)
    arr[~mask] = 1
    return arr


def remap(input_array, mapping):
    """Function takes input array and mapping dictionary to efficiently replace dictionary keys with values in the array.

    Args:
        input_array (np.array): Array to perform mapping
        mapping (dict]): Dictionary containing values to remap, e.g. {1: 3, 2:100} will change all 1 to 3 and all 2 to 100

    Returns:
        np.array: Array with substituted values
    """
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    mapping_ar = np.zeros(k.max()+1,dtype=v.dtype) #k,v from approach #1
    mapping_ar[k] = v
    return mapping_ar[input_array]