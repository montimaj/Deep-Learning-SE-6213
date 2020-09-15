import os
import struct
import numpy as np


def load_images_with_labels(type_of_data = "training", path = "./MNIST_data"):
    """
    Extracts the imagesa and corresponding labels from the MNIST data set.
    """
    #Read the filenames
    if type_of_data is "training":
        images = os.path.join(path, 'train-images-idx3-ubyte')
        labels = os.path.join(path, 'train-labels-idx1-ubyte')
    elif type_of_data is "testing":
        images = os.path.join(path, 't10k-images-idx3-ubyte')
        labels = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("type_of_data must be 'testing' or 'training'")

    # Load labels
    with open(labels, 'rb') as label:
        _, num = struct.unpack(">II", label.read(8))
        lbl = np.fromfile(label, dtype=np.int8)
        
    # Load images
    with open(images, 'rb') as image:
        _, _, rows, cols = struct.unpack(">IIII", image.read(16))
        img = np.fromfile(image, dtype=np.uint8).reshape(len(lbl), rows, cols)
        
    # Return the images with corresponding labels
    return (img,lbl)
    
    
def pre_process(data,N=60000,d=784):
    """
    Converts each image into a row vector and normalizes it between 0 and 1
    """
    out = data.reshape(N,d)
    out = np.float32(out)
    out /= np.max(out,axis=1).reshape(-1,1)
    return (out)
    