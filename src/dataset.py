import os
import requests
import sys
import gzip
import struct
import numpy as np

# for debug purposes
# from matplotlib import pyplot as plt

# URLs for the MNIST dataset
DOWNLOAD_URLS = {
    "train_data": 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',  # 60000 images
    "train_labels": 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    "test_data": 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',  # 10000 images
    "test_labels": 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
}


def load_train_data():
    """
    loads the training data
    :return:
        data : 2D numpy array with the image data (one image per row)
        labels : 1D numpy array with the label for each corresponding image
    """
    return __load_data('train')


def load_test_data():
    """
    loads the test data
    :return:
        data : 2D numpy array with the image data (one image per row)
        labels : 1D numpy array with the label for each corresponding image
    """
    return __load_data('test')


def __load_data(which: str = "train"):
    """
    loads data, checks if the files exist and downloads them if needed
    :return:
        data : 2D numpy array with the image data (one image per row)
        labels : 1D numpy array with the label for each corresponding image
    """
    if not __check_downloaded(which):
        print(f"No local {which} dataset found.")
        data, labels = __download_data(which)
    else:
        data = np.load(f'{which}_data.npy')
        labels = np.load(f'{which}_labels.npy')
    return data, labels


def __check_downloaded(which: str = "train"):
    """
    Checks if the data has already been downloaded
    :param which: which dataset to check, 'train' or 'test'
    :return: True if the dataset is downloaded, False otherwise
    """
    if which not in ["train", "test"]:
        raise ValueError("download check error, please enter 'train' or 'test'")
    fp = which + "_data.npy"
    fp2 = which + "_labels.npy"
    return os.path.exists(fp) and os.path.exists(fp2)


def __download_data(which: str = "train"):
    """
    Downloads the data and parses it
    :param which: which dataset to download (train or test)
    :return:
        data : 2D numpy array containing the image data (one image per row)
        labels : 1D numpy array with image labels corresponding to the data
    """
    if which not in ["train", "test"]:
        raise ValueError("download check error, please enter 'train' or 'test'")

    # prepare urls and tmp_files
    data_url = DOWNLOAD_URLS[which + "_data"]
    label_url = DOWNLOAD_URLS[which + "_labels"]
    data_fp = f"tmp_data_{which}.gz"
    label_fp = f"tmp_label_{which}.gz"

    print(f"Downloading {which} data")
    __download_show_progress_bar(data_url, data_fp)
    print(f"Downloading {which} labels")
    __download_show_progress_bar(label_url, label_fp)

    # data is downloaded, parse it
    dims, data, labels = __parse_downloaded(which)

    # save to disk as numpy arrays
    np.save(f"{which}_data", data)
    np.save(f"{which}_labels", labels)
    np.save(f"{which}_dims", dims)
    return data, labels


def __download_show_progress_bar(url, fp):
    """
    Download utility with progress bar for more aesthetic view of the download
    :param url: the url to download
    :param fp: the file path to write it to
    """

    with open(fp, "wb") as f:
        resp = requests.get(url, stream=True)
        size = resp.headers.get("content-length")
        if size is None:
            f.write(resp.content)
            f.close()
            return

        dl_chunks = 0
        chars = 50
        size = int(size)
        for data in resp.iter_content(chunk_size=4096):
            dl_chunks += len(data)
            f.write(data)
            progress = int(chars*dl_chunks / size)
            sys.stdout.write('\r[%a%s%a]' % ('='*(progress-1), '>', ' '*(chars-progress)))
            sys.stdout.flush()
        f.close()
        print()


def __parse_downloaded(which: str = "train"):
    """
    Parses the downloaded gzip files downloaded from the __download_data() function, explained on http://yann.lecun.com/exdb/mnist/
    :param which: which datasets to parse, should be either "train" or "test"
    :return:
        dims : a tuple of the dimensions of each pixel images (row, col)
        data : 2D numpy array containing the pixel data for all images. Each row is one image raveled row by row
        labels: 1D numpy array containing the labels for the images in the same order as the data
    """
    data_fp = f"tmp_data_{which}.gz"
    label_fp = f"tmp_label_{which}.gz"
    if not os.path.exists(data_fp) or not os.path.exists(label_fp):
        raise FileNotFoundError(f"Couldn't find the downloaded {which} files! try downloading them again?")

    # unzip data and label files
    with gzip.open(data_fp, 'rb') as f:
        data = f.read()
        f.close()
    with gzip.open(label_fp, "rb") as f:
        labels = f.read()
        f.close()
    # remove temp files
    os.unlink(data_fp)
    os.unlink(label_fp)

    # Decode the data file, explained on http://yann.lecun.com/exdb/mnist/
    num_images, rows, cols = struct.unpack_from('>iii', data, struct.calcsize('4x'))
    data = np.frombuffer(data, dtype='>B', count=num_images*rows*cols,offset=16)
    # reshape so that each row corresponds to one image
    data = data.reshape((num_images, rows*cols))

    # debug
    # plt.imshow(data[1000].reshape((cols, rows)), cmap="gray")
    # plt.show()

    labels = np.frombuffer(labels, dtype='>B', count=num_images, offset=8)
    return (rows, cols), data, labels

