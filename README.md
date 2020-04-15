# IrrigationAI

http://people.ischool.berkeley.edu/~kaiqi/IrrigationAI/index.html

## Introduction
This repository contains a Dockerfile which allows for the download of MODIS data between the specific time range, downsamples the tiff image for pre-trainer model inference and outputs a predicted image containing irrigation/no-irrigation classes.

## Data
The 16-day Terra satellite 1kmx1km resolution [MODIS](https://modis.gsfc.nasa.gov) data is downloaded from the NASA Earthdata portal for 2008-2009 and used for training+validation together with the MIRCA2008 labeled dataset of irrigated and rainfed crops of the same time period. The 26 classes of crops in each irrigated and rainfed categories is combined to reflect a binary classification of irrigated and not irrigation.

The following [bands](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/modis/) are used for training:
* Normalized Difference Vegetation Index (NDVI)
* Enhanced Vegetation Index (EVI)
* Vegetation Index (VI) quality
* Red reflectance (Surface infrared temperature)

## Pipeline
A dedicated pipeline has been created to accomplish the following over the prescribed time:
1) Download MODIS updates or specific satellite product(upcoming feature) for the user-specified start date and end date.
2) Downsample the satellite image to 4800x4800 pixels to enable rapid predictions using the GDAL libraries.
3) Predict irrigation-classified images from pre-trained model.
4) Store images locally or the cloud(highly recommended).

### Data download
The OCTVI and GDAL libraries are used in conjunction to download the data from Earthdata login over the specified time range, satellite product and bands. Individual patches of data are converted to array form to process the pixels over rastering and masking algorithms.

### Data processing
Custom algorithms also calculate the NDVI, EVI and other required bands in the event these are missing in the specified satellite product. Masking also takes place to remove clouds, shadows and water.

The segmented data patches are geospatially stitched into a single mosaic per satellite sweep of the earth.


## Model
A 10-layer deep-learning [U-Net](https://arxiv.org/abs/1505.04597) fully-connected CNN model with binary cross-entropy loss function has been trained over global 2008-2009 labeled images of irrigation and non-irrigation. 

Image augmentation is performed to increase the sample size for coping with underfitting while dropout(0.5) and batch normalization are performed prior to the convolutional cells for generalization to avoid overfitting. In addition, upsampling and downsampling is performed to standardize the resolution of the training & predictions sets.

## Output
The predicted output is a binary classification of irrigated and non-irrigated areas identified on a global map over time. We believe the identified areas together with the pipeline will serve to benefit the environmental sciences community and others alike.

## How-To
Clone the repository and edit the following line in the docker file: 
CMD [ "python", "./test.py", **startdate**, **enddate**, **product**, **bands**]

The **startdate** and **enddate** specified must be in the following format: *"YYYY-MM-DD"*. The **product** specified can be *"MODIS"* or *"LANDSAT"* for this version of the pipeline. 
The bands can be *"ALL"* or individual bands such as *"NDVI,EVI,VI,Red"*.

Build the docker container using: "docker build -t irrigationai .". Ensure sufficient disk space is allocated to docker. Run the docker container via "docker run -it --storage-opt size=500GB irrigationai" or the disk space allocated of the user's choice.
For example, the combined file size of 1kmx1km resolution MODIS product of 4 layers(NDVI, EVI, VI, Red) for one sweep(16 days) is approximately 8GB.

The running container will automatically:
1) download the images (*test.py*, *url.py*)
2) post-process them (*extract.py*, *arr.py*)
3) stitch them together into a mosaic (*init.py*)
4) downsample them for predictions (*downsampletif.py*)
5) predict irrigated areas using a pre-trained deep learning U-Net model to a resulting tif image (*predict.py*)

Depending on the product and number of bands specified, the size of the tiles can be require a fairly large disk space. 

![alt text](sample_output.png)

