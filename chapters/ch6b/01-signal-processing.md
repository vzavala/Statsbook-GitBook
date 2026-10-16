# Signal Processing

* Signal processing (SP) is field of mathematics that provides tools for analyzing, modifying, and synthesizing data.
* Data often appears as temporal signals (e.g., temperature, vibration, sound) or spatial signals (e.g., images, tomography, micrographs).
* We look into a SP task known as filtering (removes unwanted features from data) with goal of extracting dominant features (patterns).
* Filtering is a powerful way to preprocess raw data and can be beneficial in performing estimation, modeling, data analysis.
* *Convolution* is fundamental operation behind filtering. This is a  *weighted sum of data points* over local subdomain (window) of signal.
* *Fourier transform* is another fundamental operation of SP that transforms data to frequency domain; this facilitates convolution and allows us to characterize features in terms of  frequency content.
* Features extracted from convolutions can be used to classify images; this is basic principle behind machine learning techniques such as convolutional neural nets.
