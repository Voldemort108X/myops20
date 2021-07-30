import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPool2D, concatenate, UpSampling2D
# from config import *
from tensorflow.keras import backend as K

smooth = 1.
def dice_coef(y_true, y_pred):
    print('this is y_true shape',y_true.shape)
    print('this is y_pred shape',y_pred.shape)
    y_true_f = K.flatten(y_true)
    
    print(y_true_f)
    y_pred_f = K.flatten(y_pred)
    print(y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    print(intersection)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def UNet_model(img_rows, img_cols, learning_rate):
    inputs = Input((img_rows, img_cols, 1))
    print("input shape bf",inputs.shape)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    print("conv1 shape bf", conv1.shape)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    print("conv1 shape",conv1.shape)
    print("pool1 shape", pool1.shape)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    print("conv2 shape", conv2.shape)
    print("pool2 shape", pool2.shape)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    print("conv3 shape", conv3.shape)
    print("pool3 shape", pool3.shape)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv4)
    print("conv4 shape", conv4.shape)
    print("pool4 shape", pool4.shape)
    
    up5 = UpSampling2D()(conv4)
    up5 = concatenate([up5,conv3])
    print('up6 shape',up5.shape)

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    print('conv6 shape',conv5.shape)

    up6 = UpSampling2D()(conv5)
    print('up7 shape', up6.shape)
    up6 = concatenate([up6,conv2])

    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling2D()(conv6)
    up7 = concatenate([up7,conv1])

    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv7)
    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(conv7) #fully convolutional layer
    
    print("conv10.shape", conv8.shape)
    model = Model(inputs=[inputs], outputs=[conv8])

    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    
    return model

if __name__ == "__main__":
    model = UNet_model(256,256,1e-5)
    print(model.summary())