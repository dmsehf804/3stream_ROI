"""
Train our temporal-stream CNN on optical flow frames.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from spatial_train_model import get_model, freeze_all_but_top, freeze_all_but_mid_and_top
from spatial_train_data import DataSet, get_generators
import time
import os.path
from os import makedirs
from I3D_model import Inception_Inflated3d
from keras.optimizers import SGD, Adam
import os

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D

from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from I3D_model import conv3d_bn
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

os.environ["CUDA_VISIBLE_DEVICES"]="0"

NUM_FRAMES = 25
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2
NUM_CLASSES = 101
WEIGHTS = 'rgb_imagenet_and_kinetics'

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    
    model.compile(
        optimizer=SGD(lr=0.001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.summary()
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def train(num_of_snip=5, saved_weights=None,
        class_limit=None, image_shape=(224, 224),
        load_to_memory=False, batch_size=32, nb_epoch=1000, name_str=None):

    # Get local time.
    time_str = time.strftime("%y%m%d%H%M", time.localtime())

    # if name_str == None:
    #     name_str = WEIGHTS
    print(name_str)
    # Callbacks: Save the model.
    directory1 = os.path.join('logs', 'checkpoints', name_str)
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    checkpointer = ModelCheckpoint(
        filepath=os.path.join(directory1, '{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Callbacks: TensorBoard
    directory2 = os.path.join('logs', 'TB', name_str)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    tb = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early stoper
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save results.
    directory3 = os.path.join('logs', 'csv', name_str)
    if not os.path.exists(directory3):
        os.makedirs(directory3)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + \
        str(timestamp) + '.log'))
    reduce_lr = ReduceLROnPlateau(patience=10, min_lr=0.00001)
    print("class_limit = ", class_limit)

    if image_shape is None:
        data = DataSet(
                class_limit=class_limit
                )
    else:
        data = DataSet(
                image_shape=image_shape,
                class_limit=class_limit
                )
    
    # Get generators.
    generators = get_generators(data=data, image_shape=image_shape, batch_size=batch_size)

    # Get the model.
    #model = get_model(data=data)

 

    model = Inception_Inflated3d(include_top=False,
                weights=WEIGHTS,
                input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
                classes=NUM_CLASSES)
    model = Dropout(0.8)(model)

    model = conv3d_bn(model, 101, 1, 1, 1, padding='same', 
            use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1')

    num_frames_remaining = int(model.shape[1])
    model = Reshape((num_frames_remaining,101))(model)

    # logits (raw scores for each class)
    model = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                output_shape=lambda s: (s[0], s[2]))(model)

    # if not endpoint_logit:
    #     model = Activation('softmax', name='prediction')(model)
    # if saved_weights is None:
    #     print("Loading network from ImageNet weights.")
    #     print("Get and train the top layers...")
    #     model = freeze_all_but_top(model)
    #     model = train_model(model, 10, generators)
    # else:
    #     print("Loading saved model: %s." % saved_weights)
    #     model.load_weights(saved_weights)

    # print("Get and train the mid layers...")
    # model = freeze_all_but_mid_and_top(model)
    model = train_model(model, 10, generators, [tb, early_stopper, csv_logger, checkpointer, reduce_lr])
    

def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    saved_weights = None
    class_limit = None  # int, can be 1-101 or None
    num_of_snip = 1 # number of chunks used for each video
    image_shape=(224, 224)
    load_to_memory = False  # pre-load the sequencea in,o memory
    batch_size = 1
    nb_epoch = 500
    name_str = WEIGHTS
    "=============================================================================="

    train(num_of_snip=num_of_snip, saved_weights=saved_weights,
            class_limit=class_limit, image_shape=image_shape,
            load_to_memory=load_to_memory, batch_size=batch_size,
            nb_epoch=nb_epoch, name_str=name_str)

if __name__ == '__main__':
    main()
