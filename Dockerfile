# Dockerfile
FROM python:3.6-slim
#FROM w251/hw06:x86-64

# Update base container install
RUN apt-get update
RUN apt-get upgrade -y

# Install GDAL dependencies
#RUN apt-get install -y wget python3-pip libgdal-dev locales gdal-bin g++ nfs-common openjdk-11-jdk:amd64
RUN apt-get install -y wget python3-pip libgdal-dev locales gdal-bin g++

# Ensure locales configured correctly
RUN locale-gen en_US.UTF-8
ENV LC_ALL='en_US.utf8'

# Set python aliases for python3
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc
RUN python -V

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir numpy boto3 s3fs botocore awscli pandas Keras tensorflow

# This will install latest version of GDAL
#RUN pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
RUN pip install GDAL==1.10.0 --global-option=build_ext --global-option="-I/usr/include/gdal"
RUN gdalinfo --version

RUN pip install --no-cache-dir octvi

# Set the working directory to custom folder
WORKDIR /irrigationai

# Copy the current directory contents into the container at /custom folder
COPY . .

#RUN tar -xvf hegLNX64v2.15.tar
#RUN tar -xvf hegLNX64v2.15/heg.tar
#ENV MRTDATADIR = /irrigationai/hegLNX64v2.15/heg/data
#ENV PGSHOME = /irrigationai/hegLNX64v2.15/heg/TOOLKIT_MTD
#RUN ./hegLNX64v2.15/heg/bin/HEG

#Mount volume
#RUN mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-dc79e476.efs.us-west-2.amazonaws.com:/app
#RUN mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 172.31.58.254:/app

#run the python file
#CMD [ "python", "./test.py", "2019-01", "2020-03"]
CMD python downsampletif.py
CMD predict.py







