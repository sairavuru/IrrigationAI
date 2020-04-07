import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

N_BANDS = 4
N_CLASSES = 1 # Irrigated/Non Irrigated
CLASS_WEIGHTS = [0.1,0.9]
N_EPOCHS = 100
UPCONV = True
PATCH_SZ = 320 # should divide by 16

def predict(x, model, patch_sz=160, n_classes=1):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    print(img_height, img_width)
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    #print(npatches_vertical,npatches_horizontal)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]

    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]
    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
    #print(x0,x1,y0,y1)
    patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    #print(prediction.shape, patches_predict.shape)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
    #print(x0,x1,y0,y1,i,j,k, npatches_horizontal, npatches_vertical)
    prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]

def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150], # Land
        1: [223, 194, 125],
        #, # rainfed
        2: [27, 120, 55] # Irrigated
    }

    z_order = {
        1: 0,
        2: 1
        #,
        #3: 0
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for ch in range(3):
        pict[ch,:,:][mask[0,:,:] > threshold] = colors[1][ch]
        #pict(1,:,:)[mask[1,:,:]> threshold] =colors[0][0]
        #for i in range(1, 3):
        # cl = z_order[i]
        # for ch in range(2):
        # pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
        return pict

if __name__ == '__main__':
    model = get_model()
    model.load_weights(weights_path)
    test_id = '09'
    img = normalize(tiff.imread('./Data/{}.tif'.format(test_id))) # make channels last
    mymat = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
    result_stg = mymat >0.5
    final_tif = result_stg.astype(np.float16)
    tiff.imsave('result.tif', final_tif)
    map = picture_from_mask(mymat, 0.5)
    #mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1]) # make channels first
    #map = picture_from_mask(mask, 0.5)
    #tiff.imsave('result.tif', (255*mask).astype('uint8'))
    #tiff.imsave('result.tif', mymat)
    tiff.imsave('map.tif', map)
    print("Prediction Complete")