import os.path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import os, sys, glob, random, math
import boto3

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import keras as k
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
#tf.enable_eager_execution()

s3 = boto3.client('s3')
res = boto3.resource('s3')

def get_rand_patch(img, mask, sz=160):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    #print(len(img.shape), img.shape[0], img.shape[1], img.shape[0:1], mask.shape[0:1])
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

    # Apply some random transformations
    random_transformation = np.random.randint(1,8)
    if random_transformation == 1:  # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        pass

    return patch_img, patch_mask

def get_patches_val():
    trainIds = [str(i).zfill(2) for i in range(11, 12)]  # all availiable ids: from "01" to "24"
    x_dict = dict()
    y_dict = dict()
    sz = 320
    n_patches = 100000000
    #print('Reading images')
    for img_id in trainIds:
        #print(tiff.imread('./Data/{}.tif'.format(img_id)).shape)
        print(img_id + ' read')
        img_m = normalize(tiff.imread('./Data/{}.tif'.format(img_id)))
        #print(img_m.shape)
        mask_stg1 = tiff.imread('./Label/{}.tif'.format(img_id))[:,:,np.newaxis] # / 255
        mask_stg2 = mask_stg1 >= 40.
        mask = mask_stg2.astype(int)
        #mask = mask_stg1
        #print(mask.shape)
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        x_dict[img_id] = img_m[:, :, :]
        y_dict[img_id] = mask[:, :, :]
        #print(img_id + ' read')
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        #print("debug")
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    #print('Generated {} patches'.format(total_patches))
        yield np.array(img_patch), np.array(mask_patch)
        #yield np.array(x), np.array(y)

def get_patches():
    trainIds = [str(i).zfill(2) for i in range(1, 11)]  # all availiable ids: from "01" to "24"
    x_dict = dict()
    y_dict = dict()
    sz = 320
    n_patches = 1000000000
    #print('Reading images')
    for img_id in trainIds:
        print(img_id + ' read')
        #print(tiff.imread('./Data/{}.tif'.format(img_id)).shape)
        img_m = normalize(tiff.imread('./Data/{}.tif'.format(img_id)))
        #print(img_m.shape)
        mask_stg1 = tiff.imread('./Label/{}.tif'.format(img_id))[:,:,np.newaxis] # / 255
        mask_stg2 = mask_stg1 >= 40.
        mask = mask_stg2.astype(int)
        #mask = mask_stg1
        #print(mask.shape)
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        x_dict[img_id] = img_m[:, :, :]
        y_dict[img_id] = mask[:, :, :]
        #print(img_id + ' read')
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        #print("debug")
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    #print('Generated {} patches'.format(total_patches))
        yield np.array(img_patch), np.array(mask_patch)
        #yield np.array(x), np.array(y)

def acc_m(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    acc = K.mean(tf.cast(tf.equal(y_true,y_pred), tf.float32))
    return acc


def recall_m(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    true_positives = K.sum(y_true*y_pred)
    positives = K.sum(y_true)
    recall = true_positives / (positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    true_positives = K.sum(y_true*y_pred)
    predicted_positives = K.sum(y_pred)
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


w_pos = 0.94
w_neg = 0.06


def unet_model(n_classes=1, im_sz=160, n_channels=3, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.1, 0.9]):
    droprate = 0.2
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    # inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_1 = Dropout(droprate)(pool4_1)

    n_filters *= growth_factor
    pool4_1 = BatchNormalization()(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    pool4_2 = Dropout(droprate)(pool4_2)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)

    n_filters //= growth_factor
    if upconv:
        up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
    else:
        up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
    up6_1 = BatchNormalization()(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
    conv6_1 = Dropout(droprate)(conv6_1)

    n_filters //= growth_factor
    if upconv:
        up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
    else:
        up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
    conv6_2 = Dropout(droprate)(conv6_2)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(droprate)(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(droprate)(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)
    # conv10 = Conv2D(n_classes, (1, 1), activation='relu')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    def weighted_binary_crossentropy(y_true, y_pred):
        # threshold = 40.
        # labels = tf.cast(y_true > threshold, tf.float32)
        # class_loglosses = K.sum(K.binary_crossentropy(y_true, y_pred)*(y_true*w_pos+(1.0-y_true)*w_neg))
        # weight_sum = K.sum(y_true*w_pos+(1.0-y_true)*w_neg)
        # return class_loglosses/weight_sum
        return tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=10.0)

    # K.sum(class_loglosses * K.constant(class_weights))

    def margin_loss(y_true, y_pred):
        threshold = 40.0
        neg_margin = 10.0
        pos_margin = 30.0
        labels = tf.cast(y_true > threshold, tf.float32)
        neg_loss = (1.0 - labels) * tf.square(tf.maximum(0.0, y_pred - threshold + neg_margin)) * w_neg
        pos_loss = labels * (tf.square(tf.maximum(0.0, threshold - y_pred + pos_margin))) * w_pos
        loss = tf.reduce_sum(neg_loss + pos_loss)
        weight_sum = K.sum(labels * w_pos + (1.0 - labels) * w_neg)
        return loss / weight_sum

    # K.sum(class_loglosses * K.constant(class_weights))

    # model.compile(optimizer=Adam(lr=0.0005), loss=weighted_binary_crossentropy, metrics=['accuracy', recall_m, precision_m])
    # model.compile(optimizer=Adam(lr=0.0005), loss=margin_loss)
    model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy", metrics=[acc_m, recall_m, precision_m, f1_m])
    # print(K.eval(model.optimizer.lr))
    # model.summary()
    return model




weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x



N_BANDS = 4
N_CLASSES = 1  # Irrigated/Non Irrigated
CLASS_WEIGHTS = [0.1,0.9]
N_EPOCHS = 100
UPCONV = True
PATCH_SZ = 320  # should divide by 16

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)

def predict(x, model, patch_sz=160, n_classes=1):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    #print(img_height, img_width)
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
        print ("test")
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
    patches_predict = model.predict(patches_array, batch_size=1)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    #print(prediction.shape, patches_predict.shape)
    #print(patches_predict.shape)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        #print(x0,x1,y0,y1,i,j,k)
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],  # Land
        1: [223, 194, 125],
        #,  # rainfed
       2: [27, 120, 55]    # Irrigated
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
    #    cl = z_order[i]
    #    for ch in range(2):
    #        pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict


if __name__ == '__main__':
    model = get_model()
    model.load_weights(weights_path)

    files = glob.glob("*_downsampled.tif")
    print(files)

    for f in files:
        #in_file = f.split('/')[1]
        in_file = f
        out_path = in_file.split('.')[0] + '_result.tif'
        print(in_file, out_path)


        img = normalize(tiff.imread('./{}'.format(in_file)))  # make channels last

        mymat = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2, 0, 1])

        result_stg = mymat > 0.4
        final_tif = result_stg.astype(np.float16)
        tiff.imsave(out_path, final_tif)

        res.Bucket('irrigationai-data').upload_file(out_path, 'MODIS/MOD13Q1-multiband-2008-2009/predicted/' + out_path)

    print("Prediction Complete")