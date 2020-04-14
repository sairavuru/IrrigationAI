import numpy as np
import pandas as pd
import os, sys, glob
import datetime
import boto3
import gdal
import subprocess

s3 = boto3.client('s3')
res = boto3.resource('s3')

files = glob.glob("*.tif")

for f in files:
    out_path = f.split('.')[0]+'_downsampled.tif'
    print(out_path)

    process_text = ("gdalwarp -r average -ts 4800 4800 -wm 2048 -multi -wo NUM_THREADS=ALL_CPUS -co BLOCKXSIZE=256 -co BLOCKYSIZE=256 -co NUM_THREADS=ALL_CPUS -co COMPRESS=LZW " + f + " " + out_path).split(' ')
    print(process_text)
    subprocess.call(process_text)

    #res.Bucket('irrigationai-data').upload_file(out_path,'MODIS/MOD13Q1-multiband-2008-2009/2019/' + out_path)