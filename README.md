# Accoustic based anomally detection of industrial machines (Clustering)

## Description
This project is a second part of the is a part of the `Accoustic based anomally detection of industrial machines` learning project at Becode.org AI Bootcamp programme. The goal of this particular project is to investigate the possible use of unsupervised learning based on clustering for anomalous sound detection of faulty industrial machinery for a fictional company Acme Corporation. The first project dealt with classification of labeled data and can be consulted [here](https://github.com/mokegg/machine-monitoring-conditions).

Data samples of normal and abnormal sounds of for kinds of machines, valves, pumps, fans and sliders are downloaded from [Machine Condition Monitoring](https://zenodo.org/record/3384388#.YbIcwZHMJH5).

Duration of the project: 2 weeks

## Learning Objectives

  * Be able to work and process data from audio format
  * Find insights from data, build hypothesis and define conclusions
  * Build machine learning models for predictive classification and Clustering  and select the right performance metrics for the model
  * Evaluate the distribution of data points and evaluate its influence in the model
  * Be able to identify underfitting or overfitting that might exist on the model
  * Tuning parameters of the model for better performance
  * Select the model with better performance and following your customer's requirements
  * Define the strengths and limitations of the model

## Installation
The environment at which the code is developed is provided in the `requirements.txt` file. To run the code, the necessary libraries should be installed  based on that environment. Important libraries are, among others:

  *  Numpy
  *  Pandas
  *  Sklearn
  *  Librosa
  *  Soundfile
  *  IPython
  
 
## Usage
Sound files are downloaded and preprocessed using the same codes developed during the first part of the project. This part focuses on the implementation of the clustering techniques. This is implemented by a new code `Clustering.ipynb `

 `Clustering.ipynb ` 
  
  * Imports the preprocessed labeled data, removes the labels to make it suitable for unsupervised learning. 
  * Extracts principal components (main dimensions of variation) using PCA
  * Calculates distortion (Inertia) to estimate the range of the number of clusters
  * Uses two clustering algorithms (KMeans and DBSCAN) to make cluster
  * Compares the clusters formed using unlabeld data with the given labeld data for verification


## Visualizations and Validation
Evaluation the performance of the clustering algorithm was done by using the human-labels that are included in the original data. using barplots, the clusters are shown using expected Targets(labels). Some visuals are shown below.

![Elbow method to estimate the range of the number of clusters](valve_pics/elbow.png)


Valves of a single model (6dB and ID_06) two clusters (KMeans)             |  Comparison with labeld data
:-------------------------:|:-------------------------:
![](valve_pics/clusters_valves_id06.png) |  ![](valve_pics/barplot_valve_Id06_6dB_2_2f.png)



Valves of a single model (6dB and ID_06) three clusters (KMeans)             |  Comparison with labeld data
:-------------------------:|:-------------------------:
![](valve_pics/clusters_valves_id06_3.png) |  ![](valve_pics/barplot_valve_Id06_6dB_3_2f.png)

##Conclusion
From what has been observed during limited investigation, it seems that clustering could help to detect anomalous sounds of faulty machines. However, there are a lot of overlaps between the clusters and more investigation with other algorithms. 

## Further Development
  * The audio features used for clustering are mfccs, same features used in supervised learning. An incomplete attempt was done to include other features to investigate the effects of using different features. Hopefully this will be implemented in the feature.
  * Performance is evaluated only visualy.Performance metrics should be implemented

