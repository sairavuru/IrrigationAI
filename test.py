import numpy as np
import pandas as pd
import os, sys
import datetime
import boto3
import gdal
import octvi
import init, extract, url
import glob

s3 = boto3.client('s3')
res = boto3.resource('s3')

def writeCsv(df, outputPath):
    if outputPath.startswith("s3n://"):
        bucketName = outputPath[6:outputPath.find("/",6)]
        filePath = outputPath[outputPath.find("/",6) + 1:]
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        res.Bucket(bucketName).put_object(Body=csv_buffer.getvalue(), Key=filePath)
    else:
        if not os.path.exists(os.path.dirname(outputPath)):
            os.makedirs(os.path.dirname(outputPath))
        df.to_csv(outputPath,index=False)

def main():

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    print('Following months of data being downloaded: ')
    date_range = [datetime.datetime.strftime(x, '%Y-%m') for x in list(pd.date_range(start=start_date, end=end_date, freq='m'))]
    print(date_range)

    out_path = os.getcwd()+'/test.tif'
    out_path_lowres = os.getcwd()+'/test_lowres.tif'
    #out_path = '/home/ec2-user/irrigationai/test.tif'
    working_directory = os.path.dirname(out_path)
    #working_directory = '/home/ec2-user/irrigationai'
    print(out_path)
    print(out_path_lowres)

    for m in date_range:
        modisDates = url.getDates("MOD13Q1",m)
        print('Following cycles will be downloaded for the month: ')
        print(modisDates)

        for d in modisDates:
            print('HDF files for the following cycle will be downloaded: ')
            print(d)
            tiles = url.getUrls("MOD13Q1", d)
            tiles.sort(key=lambda x: x[1])
            print(tiles)
            print(len(tiles))

            startTime = datetime.datetime.now()
            #ndvi_files = []
            files = []
            file_count = 0

            for tile in tiles:
                file_count += 1
                print(file_count, tile)
                try:
                    try:
                        hdf_file = url.pull(tile[0], working_directory)
                    except:
                        try:
                            print('Attempt #2!')
                            hdf_file = url.pull(tile[0], working_directory)
                        except:
                            print('hdf cannot be downloaded!')

                    ext = os.path.splitext(hdf_file)[1]
                    #ndvi_files.append(extract.ndviToRaster(hdf_file,hdf_file.replace(ext,".ndvi.tif")))
                    files.append(extract.chosenToRaster(hdf_file,hdf_file.replace(ext,".chosen.tif")))
                except:
                    print('hdf cannot be downloaded/rastered!')

            #init.mosaic(ndvi_files,out_path)
            init.mosaic(files,out_path,out_path_lowres)
            endTime = datetime.datetime.now()
            print(f"Done. Elapsed time {endTime-startTime}")

            res.Bucket('irrigationai-data').upload_file(out_path,'MODIS/MOD13Q1-multiband-2008-2009/'+d+'_4bands_full.tif')
            res.Bucket('irrigationai-data').upload_file(out_path_lowres,'MODIS/MOD13Q1-multiband-2008-2009/'+d+'_4bands_lowres.tif')
            print('Saved into S3!')

        # octvi.globalVi("MOD13Q1","2008-05-08","./example_standard.tif")
        # print('Success so far! Saving to S3..')
        # res.Bucket('irrigationai-data').upload_file('example_standard.tif','MODIS/working/example_standard.tif')
        # print('Saved into S3!')

if __name__=="__main__":
    main()
