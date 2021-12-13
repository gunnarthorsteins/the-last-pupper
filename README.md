# The Last Pupper

This project is in most part based on a previous project project titled *Deep Painter* in the class *Deep Learning and Neural Networks* at Columbia University, with some later improvements.

Contributors:

Alex Angus

Gunnar Thorsteinsson


# Abstract

We employ a Convolutional Autoencoder (CAE) to classify paintings and drawings, that is, identify who painted a given painting from an open-source dataset. Images of art are well suited to be classified by a CAE as they can detect, without manual feature extraction, the nuances of a specific artist’s style present in their work. CAE are particularly powerful at getting good results from a limited dataset, but even the most prolific artists have at most on the order of few thousands of paintings.



# Organization


```
./
├── 3PainterCAE.ipynb
├── Autoencoder.ipynb
├── BinaryClassification.ipynb
├── CAETesting.ipynb
├── CaeClassifier.ipynb
├── e4040.2021spring.AAGT.report.ala2197.gt2447.pdf
├── Original Paper.ipynb
├── README.md
├── RetrievingData.ipynb
├── data
├── utils
│   ├── evaluate.py
│   ├── get_model.py
│   ├── models.py
│   ├── image_formatting.py
│   └── image_scrape.py
└── visualizations
    ├── autoencoded_image.png
    └── original_image.png

```
