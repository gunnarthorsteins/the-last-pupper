"""
Functions for evaluation of CAE and AE models.

Alex Angus

4/22/21
"""
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from PIL import Image as Image_PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from .image_formatting import *


def compare_output(model, example, dims=(256, 256)):
    """
    Displays an original image and the output of the Autoencoder inline Jupyter.

    model: the autoencoder model
    example: an array representing an image
    """
    original_image = Image_PIL.fromarray(example)                               # generate PIL image from original example array
    original_image_path = 'visualizations/original_image.png'                   # define original example path
    original_image.save(original_image_path)                                    # save original image to visualizations folder

    expanded_array = np.expand_dims(example, 0)                                 # expand example array to pass through autoencoder
    autoencoder_output = model(expanded_array).numpy().reshape((dims[0],
                                                                dims[1],
                                                                3))
    # restandardize output
    autoencoder_output = restandardize_image(autoencoder_output)                # restandardize output of autoencoder

    autoencoded_image = Image_PIL.fromarray(autoencoder_output)                 # generate PIL image of ae output
    autoencoded_image_path = 'visualizations/autoencoded_image.png'             # define output example path
    autoencoded_image.save(autoencoded_image_path)                              # save output to visualizations

    show_image(original_image_path)                                             # display original and output inline Jupyter
    show_image(autoencoded_image_path)


def restandardize_image(ae_output):
    """
    Restandardize output of autoencoder to RGB image specifications.
    (3 channels with integer values ranging from 0 to 255)

    rgb image = int(output - min(output) / (max(output) - min(output)) * 255)
    """
    output_max = np.max(autoencoder_output)                                     # calculate max and min output values
    output_min = np.min(autoencoder_output)
    ae_output = (ae_output - output_min) / (output_max - output_min) * 255      # normalize between 0 and 255
    return ae_output.astype(np.uint8)                                           # convert values to int


def make_confusion_matrix(predictions, labels, title, timestamp, artists,
                          classifier_name, display_matrix=True,
                          save_matrix=True):
    """
    Generates a confusion matrix and a classification report given a set of
    predictions and labels.

    params:
        predictions: array predicted labels of input images
        labels: array ground truth labels of input images
        title: string title of confusion matrix plot
        timestamp: string time of evaluation
        artists: list of artist names, used for axis labels
        classifier_name: string name of classifier
        display_matrix: boolean display matrix inline
        save_matrix: boolean save matrix plot to figures folder

    returns:
        report: string outlining evaluation of model with various metrics
        report_dict: dictionary version of report
    """
    ax = plt.subplot()                                                          # generate axes
    sns.heatmap(confusion_matrix(labels, predictions), annot=True)              # make confusion matrix and plot as heat map
    ax.set_xlabel('Predictions')                                                # label axes and set title
    ax.set_ylabel('Ground Truth')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(artists)
    ax.yaxis.set_ticklabels(artists)
    if save_matrix:
        plt.savefig('figures/{}_confusion_matrix_{}.png'.format(classifier_name,# save confusion matrix to figures folder
                                                                timestamp))
    if display_matrix:                                                          # display confusion matrix inline
        plt.show()
    plt.clf()                                                                   # clear axes
    report = [classification_report(labels, predictions)]                       # generate classification report
    report_dict = classification_report(labels, predictions, output_dict=True)  # generate dictionary version of classification report
    np.savetxt('figures/{}_report_{}.txt'.format(classifier_name, timestamp),   # save classification report
                                                 report, fmt='%s')
    return report[0], report_dict


def make_training_plot(training_history, type='loss', metric='MSE', filename=None):
    """
    Generates training plots given the specific training history, plot type, and
    metric.

    params:
        training_history: dictionary of training history metrics
        type: string indicating the plot type ('loss' or 'accuracy')
        metric: string indicating the evaluation metric ('MSE' or 'accuracy')
        filename: string filename of saved plot. Plot will not be saved if None
    """
    if type == 'loss':                                                          # plot training and validation losses
        plt.plot(training_history.history['loss'],
                 label='Training Loss')
        plt.plot(training_history.history['val_loss'],
                 label='Validation Loss')

    elif type == 'accuracy':                                                    # plot training and validation accuracies
        plt.plot(training_history.history['accuracy'],
                 label='Training Accuracy')
        plt.plot(training_history.history['val_accuracy'],
                 label='Validation Accuracy')

    plt.title('{} {}'.format(metric, type))                                     # label and title plot
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend(loc='best')

    if filename is not None:                                                    # save plot to figures folder
        plt.savefig('figures/{}'.format(filename))

    plt.show()                                                                  # display plot


def make_k_fold_results(results):
    """
    Calculates average scores accross k folds

    results: report dictionary generated by classification_report(output_dict=True)
    """
    precision = 0                                                               # calculate average metrics across k folds
    recall = 0
    f1_score = 0
    for result in results:
        precision += result['weighted avg']['precision']
        recall += result['weighted avg']['recall']
        f1_score += result['weighted avg']['f1-score']

    print("Precision: ", round(precision / len(results), 3))                    # print rounded average metrics
    print("Recall: ", round(recall / len(results), 3))
    print("F1 Score: ", round(f1_score / len(results), 3))
