import pathlib

import numpy as np
from PIL import Image


def load_from_png(
    filename: str,
    labels: np.ndarray,
    invert: bool,
    target_resolution: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Loads a PNG file containing a MNIST-like dataset with N columns and K rows,
    where each element is an image of a digit from 0 to 9. The labels are
    specified in the labels parameter.

    Parameters
    ----------
    filename : str
        Path to the PNG file.
    labels : numpy.ndarray
        2d array with the labels of the images.
    invert : bool
        Whether to invert the colors of the image.
    target_resolution : tuple[int, int], optional
        Target resolution of the image, by default (32, 32)
    """

    # Determine the number of rows and columns
    n_cols = labels.shape[1]
    n_rows = labels.shape[0]

    # Load the image
    img = Image.open(filename)

    # Determine the resolution of the image
    img_width, img_height = img.size

    # Resize the image
    total_width = n_cols * target_resolution[0]
    total_height = n_rows * target_resolution[1]
    img = img.resize((total_width, total_height))

    # Split the image into a list of images
    img_list = [
        img.crop(
            (
                j * target_resolution[0],
                i * target_resolution[1],
                (j + 1) * target_resolution[0],
                (i + 1) * target_resolution[1],
            )
        )
        for i in range(n_rows)
        for j in range(n_cols)
    ]

    # Convert the images to numpy arrays
    img_list = [np.array(img) for img in img_list]

    # Invert the colors if necessary
    if invert:
        img_list = [255 - img for img in img_list]

    # Convert the images to grayscale
    img_list = [np.mean(img, axis=2) for img in img_list]

    # Convert the images to float32
    img_list = [img.astype(np.float32) for img in img_list]

    # Convert the images to 1 channel
    img_list = [img[:, :, np.newaxis] for img in img_list]

    # Convert the images to 0-1 range
    img_list = [img / 255 for img in img_list]

    # Concatenate the images into a single array
    img_array = np.concatenate(img_list, axis=2)

    # Make the first dimension the channel dimension
    img_array = np.transpose(img_array, (2, 0, 1))

    return img_array


def load_from_png_flatten_labels(
    filename: str,
    labels: np.ndarray,
    invert: bool,
    target_resolution: tuple[int, int] = (32, 32),
) -> tuple[np.ndarray, np.ndarray]:
    # Load the images
    images = load_from_png(
        filename, labels, invert, target_resolution=target_resolution
    )

    # Flatten the labels
    labels = labels.flatten()

    return images, labels


def load_labels_from_file(filename: str) -> np.ndarray:
    """Loads the labels from a file.

    Parameters
    ----------
    filename : str
        Path to the file containing the labels.

    Returns
    -------
    numpy.ndarray
        Array containing the labels.
    """

    # Make the path absolute
    filename = str(pathlib.Path(__file__).parent / filename)

    # Load the labels from the file
    with open(filename, "r") as f:
        text = f.read().split(",")
        labels = np.array([int(part.strip()) for part in text])

    return labels


CONFIGURATIONS = {
    "jpossaz.png": {
        "labels": np.array([list(range(10)) for _ in range(10)]),
        "invert": False,
    },
    "jpossaz2.png": {
        "labels": np.array([list(range(10)) for _ in range(10)]),
        "invert": False,
    },
    "JuanRengifo.png": {
        "labels": np.array([list(range(10)) for _ in range(10)]).transpose(),
        "invert": True,
    },
    "cossio.png": {
        "labels": load_labels_from_file("cossio.csv").reshape((31, 10)),
        "invert": True,
    },
}


def load_from_configuration(
    filename: str,
    target_resolution: tuple[int, int] = (32, 32),
) -> tuple[np.ndarray, np.ndarray]:
    """Loads the images and labels from a configuration file.

    Parameters
    ----------
    filename : str
        Path to the configuration file.
    target_resolution : tuple[int, int], optional
        Target resolution of the image, by default (32, 32)\
        
    Returns
    ------- 
    tuple[np.ndarray, np.ndarray]
        Tuple containing the images and labels.
    """
    # Load the configuration
    config = CONFIGURATIONS[filename]

    # Map name to absolute path considering current file location
    filename = str(pathlib.Path(__file__).parent / filename)

    # Load the images and labels
    images, labels = load_from_png_flatten_labels(
        filename,
        config["labels"],
        config["invert"],
        target_resolution=target_resolution,
    )

    return images, labels


def load_all(
    target_resolution: tuple[int, int] = (32, 32),
) -> tuple[np.ndarray, np.ndarray]:
    """Loads all the images and labels from all the configuration files.

    Parameters
    ----------
    target_resolution : tuple[int, int], optional
        Target resolution of the image, by default (32, 32)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the images and labels.
    """
    # Load the images and labels from all the configuration files
    images_list = []
    labels_list = []
    for filename in CONFIGURATIONS:
        images, labels = load_from_configuration(
            filename, target_resolution=target_resolution
        )
        images_list.append(images)
        labels_list.append(labels)

    # Concatenate the images and labels
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return images, labels


if __name__ == "__main__":
    # Load the images
    images, labels = load_from_configuration("cossio.png")

    print(images.shape)

    # Show the images
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(10, 10, figsize=(10, 10))

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.tight_layout()

    for i in range(10):
        for j in range(10):
            ax[i, j].imshow(images[i * 10 + j, :, :], cmap="gray")  # type: ignore
            ax[i, j].set_title(labels[i * 10 + j])  # type: ignore
            ax[i, j].axis("off")  # type: ignore

    plt.show()

    # Load all the images
    images, labels = load_all()

    print(images.shape)
    print(labels.shape)
