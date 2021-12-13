"""
Functions for standardizing and preprocessing images for deep learning.

Alex Angus 3/18/21

Gunnar Thorsteinsson 4/18/21
"""

from .image_scrape import PAINTER_DICT
from PIL import Image as Image_PIL
from IPython.display import Image
import os
import numpy as np
import random


def reduce_resolution(original_path, standardized_path, new_resolution):
    """
    Reduces the resolution of an image.

    Params:
        image_filename: string specifying the file name of the image
        new_resolution: tuple specifying the new resolution of the image

    Returns:
        a reduced resolution version of image
    """
    image = Image_PIL.open(original_path)                                       # generate PIL image
    resized = image.resize(new_resolution)                                      # resize image
    resized.save(standardized_path)                                             # save resized image in path of old image


def show_image(image_filename):
    """
    Display image at image_filename in inline Jupyter
    """
    display(Image(filename=image_filename))


def show_array(array):
    """
    Display array as an image
    """
    image = Image_PIL.fromarray(array)
    image.show()


def pad_image(original_path, standardized_path):
    """
    Pad image such that it becomes square.

    params:
        original_path: path of the original image
        standardized_path: path where standardized image will be saved
    """
    image = Image_PIL.open(original_path)                                       # generate PIL image
    dim = max(image.size)                                                       # get largest dimension
    new_image = Image_PIL.new('RGB', (dim, dim))                                # generate empty square PIL image with shape of largest dimension
    offset = ((dim - image.size[0]) // 2, (dim - image.size[1]) // 2)           # calculate position of old image on blank image
    new_image.paste(image, offset)                                              # insert old image into blank image
    new_image.save(standardized_path)                                           # save padded image


def standardize_images(resolution=(256, 256), artists=PAINTER_DICT.keys(),
                       test_set=False):
    """
    Combines reduce_resolution() and pad_image().

    params:
        resolution: tuple (width, height) of new dimensions
        artists: list of artists names
        test_set: boolean standardize the test set
    """
    for artist in artists:                                                      # for each artist
        standardized_path = artist + '_standardized'                            # define standardized images path

        if not test_set:                                                        # if not the test set
            if standardized_path not in os.listdir('data/images'):              # generate artist's standardized images path
                os.mkdir('data/images/{}'.format(standardized_path))
            artist_images = os.listdir('data/images/{}'.format(artist))         # get artist's images path
        else:
            if standardized_path not in os.listdir('data/test_set'):            # if test set
                os.mkdir('data/test_set/{}'.format(standardized_path))          # define standardized test set path for each artist
            artist_images = os.listdir('data/test_set/{}'.format(artist))       # generate standardized test set path

        print("Standardizing {}'s images".format(artist))

        for image in artist_images:                                                                 # for each image
            if '.jpg' in image:                                                                     # if file is jpg
                if not test_set:                                                                    # if not test set
                    standardized_path = 'data/images/{}/{}'.format(standardized_path, image)        # define standardized image path
                    original_path = 'data/images/{}/{}'.format(artist, image)                       ## define standardized image path
                else:                                                                               # if test set
                    standardized_path = 'data/test_set/{}/{}'.format(standardized_path, image)      # define standardized image path
                    original_path = 'data/test_set/{}/{}'.format(artist, image)                     # define standardized image path
                pad_image(original_path, standardized_path)                                         # pad image
                reduce_resolution(standardized_path, standardized_path, resolution)                 # reduce resolution of image


def apply_dropout(images, dropout_rate):
    """
    Apply dropout to image (set random pixels to zero)

    params:
        images: array of images

    returns:
        images: array of images with dropout applied
    """
    for image in images:                                                        # for each image
        for row in range(image.shape[0]):                                       # for each row in image
            image_row = image[row]
            random_row_pixels = np.random.choice(np.arange(len(image_row)),     # generate indicies for dropout
                                        replace=False,
                                        size=int(len(image_row) * dropout_rate))
            for channel in range(3):                                            # for each channel
                image[row, :, channel][random_row_pixels] = 0                   # set values at dropout indices to zero
    return images                                                               # return dropout image


def get_images(artist, num_imgs, img_res=(256, 256)):
    """
    Load an artist's images into memory as np arrays. This function is for
    Gunnar's data files structure and is used by preprocess_images().

    get_images2() is another version of this function designed for Alex's file
    structure and corresponds to preprocess_images2(). All other functions in
    the data retreival process are designed to work for Alex's file structure
    along with get_images2() and preprocess_images2().

    Params:
        artist (str): Artist name, corresponding to picture folder.
                      Example: 'pablo-picasso'
        num_imgs (int): Number of images to load from folder
        img_res (tuple): resolution of images

    Returns:
        artist_images (array): preprecessed images of shape (num images, width, height, 3)
    """

    artist_standardized_images_path = f'data/images/standardized/{artist}'      # define standardized folder
    k = 0                                                                       # Counter
    dir_list = os.listdir(artist_standardized_images_path)                      # get list of standardized images
    artist_images = np.zeros((num_imgs, img_res, img_res, 3))                   # Preallocate processed images
    random.shuffle(dir_list)                                                    # randomly shuffle images
    for i, image in enumerate(dir_list):                                        # for each image
        if k >= num_imgs:                                                       # if number of images is not exceeded
            break
        if '.jpg' in image:                                                     # if file is jpg
            path = os.path.join(artist_standardized_images_path, image)         # define image path
            image = Image_PIL.open(path)                                        # open image with PIL
            artist_images[i:(i + num_imgs), :, :, :] = np.array(image)          # insert numpy version of image into preprocessed images array
            k += 1                                                              # increment count

    return artist_images


def get_images2(artist, num_images, test_set=False):
    """
    Load an artist's images into memory as np arrays. This function is for
    Alex's data files structure and is used by preprocess_images2().

    get_images() is another version of this function designed for Gunnar's file
    structure and corresponds to preprocess_images(). All other functions in
    the data retreival process are designed to work for Alex's file structure
    along with get_images2() and preprocess_images2().

    Params:
        artist (str): Artist name, corresponding to picture folder.
                      Example: 'pablo-picasso'
        num_images (int): Number of images to load from folder
        test_set (bool): for retreival of test set

    Returns:
        artist_images (array): preprecessed images of shape (num images, width, height, 3)
    """
    if test_set:
        artist_standardized_images_path = 'data/test_set/{}'.format(artist + '_standardized')   # define test set path
    else:
        artist_standardized_images_path = 'data/images/{}'.format(artist + '_standardized')     # define images path

    artist_images = []                                                          # initialize images array list
    count = 0                                                                   # initialize image count
    dir_list = os.listdir(artist_standardized_images_path)                      # get images directory

    random.shuffle(dir_list)                                                    # shuffle images

    for image in dir_list:                                                      # for each image
        if count >= num_images:                                                 # if number of images is not exceeded
            break
        if '.jpg' in image:                                                     # if file is jpg
            image = Image_PIL.open(os.path.join(artist_standardized_images_path,# load image file
                                                image))
            im_array = np.array(image)                                          # convert to np array
            if count == 0:
                last_shape = im_array.shape                                     # record prev shape to ensure uniform shapes
            if im_array.shape == last_shape:                                    # if shape is not different
                artist_images.append(im_array)                                  # append to images list and increment
                count += 1

    return np.array(artist_images)                                              # return array of images


def preprocess_images(artists=None, n_imgs=None, img_res=(256, 256), dropout_rate=None):
    """
    Concatenate images for a list of artists to one numpy array, ready for training

    Params:
        artists (list): Optional. Name of artist(s), corresponding to picture folder. Example 'pablo-picasso'. Defaults to all artists
        n_imgs (int): Number of images per artist. Defaults to all images in folder

    Returns:
        X (np.array): Training data, shape (n_artists * n_images, img_res, img_res, 3)
        y (np.array): Training labels, shape (n_artists * n_images,)
    """
    if artists == None:                                                         # Get all artist names if none is specified
        try:
            dir = 'data/images/standardized/'
            artists = os.listdir(dir)
        except:
            print('You need to call standardize_images() before preprocessing.')
            return np.empty(1), np.empty(1)

    if n_imgs == None:                                                          # Get all images if none is specified
        n_imgs = len(os.listdir(f'{dir}{artists[0]}'))

    n_total_imgs = len(artists) * n_imgs
    shape = (n_total_imgs, img_res[0], img_res[1], 3)
    X = np.zeros(shape, dtype='float32')                                        # dtype must be specified to curtail memory usage
    y = np.zeros(n_total_imgs)
    k = 0                                                                       # Counter

    for i, artist in enumerate(artists):                                        # Concatenate all images for every artist into one numpy array
        images = get_images(artist, n_imgs, img_res)                            # Get all images for one artist
        X[k:(k + n_imgs), :, :, :] = images
        y[k:(k + n_imgs)] = i                                                   # Note this step, creating a numerical classifier for each artist
        k += n_imgs

    for j in range(3):                                                          # Normalize pixel color values
        X[:, :, :, j] /= 255

    if dropout_rate is not None:
        X = apply_dropout(X, dropout_rate)

    return X, y


def preprocess_images2(artists=None, n_imgs=None, img_res=(256, 256),
                       dropout_rate=None, normalize=False, test_set=False):
    """
    Concatenate images for a list of artists to one numpy array, ready for
    training. This function is identical to preprocess_images(), but designed
    for Alex's data file structure.

    Params:
        artists (list): Optional. Name of artist(s), corresponding to picture
                        folder. Example 'pablo-picasso'. Defaults to all artists
        n_imgs (int): Number of images per artist. Defaults to all images in folder
        img_res (tuple): images resolution (width, height)
        dropout_rate (float): apply dropout at the specific rate (0 - 1)
        normalize (bool): normalize RGB values between zero and one
        test_set (bool): for retreival of test set

    Returns:
        X (np.array): Training data, shape (n_artists * n_images, img_res, img_res, 3)
        y (np.array): Training labels, shape (n_artists * n_images,)
    """
    artists_images = []                                                         # initialize images list
    for artist in artists:                                                      # add each artist's images to list
        artists_images.append(get_images2(artist, n_imgs, test_set=test_set))
    X = []                                                                      # initialize image array list
    y = []                                                                      # initialize lables list
    for i, artist_images in enumerate(artists_images):                          # for each artist
        for image in artist_images:                                             # for each image
            X.append(image)                                                     # add array to data list
            y.append(i)                                                         # add label to label list

    X, y = np.array(X), np.array(y)                                             # convert lists to arrays

    if normalize:                                                               # Normalize pixel color values across all channels
        for j in range(3):
            X[:, :, :, j] /= 255

    if dropout_rate is not None:                                                # apply dropout
        X = apply_dropout(X, dropout_rate)

    return X, y
