import os
import open3d
import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from keras.models import Model, Sequential
from sklearn.preprocessing import normalize

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)
    
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = ops.eye(num_features)

    def __call__(self, x):
        x = ops.reshape(x, (-1, self.num_features, self.num_features))
        xxt = ops.tensordot(x, x, axes=(2, 2))
        xxt = ops.reshape(xxt, (-1, self.num_features, self.num_features))
        return ops.sum(self.l2reg * ops.square(xxt - self.eye))

def tnet(inputs, num_features):
    # Initialise bias as the identity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(512, 3))
x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(8, activation="softmax")(x)

base_model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
base_model.load_weights("./pointnet.weights.h5")
base_model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])

path = "../test data/pcd/"

xrot = np.pi / 9
yrot = -np.pi / 6

for file in (os.listdir(path)):

    pcd = open3d.io.read_point_cloud(path + file)
    
    pcd.transform([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,-1,0],
                   [0,0,0,1]])

    R = pcd.get_rotation_matrix_from_xyz((xrot, yrot, 0.1))
    
    pcd.rotate(R, center=(0, 0, 0))
    
    pcd.transform([[-1, 0, 0, 0],
                   [ 0, 1, 0, 0],
                   [-1,-1, 1, 0],
                   [ 0, 0, 0, 1]])
                   
    center = pcd.get_center() 
    pcd.scale(0.4, center=center)
    center = pcd.get_center() 
    pcd.translate(-center)
    
    sample = np.asarray(pcd.points)

    sample2 = []
    sample2.clear()

    for i in range (0, sample.shape[0]): 
        if sample[i][2] > -2 and sample[i][2] < 1 and sample[i][1] > -0.1:
            sample2.append(sample[i])   
            
    if sample2 is not None and len(sample2) >= 512:
        sample2 = np.asarray(sample2)
        pcd2 = open3d.geometry.PointCloud()
        pcd2.points = open3d.utility.Vector3dVector(sample2)
        center = pcd2.get_center()               
        pcd2.translate(-center)    
        sample2 = np.asarray(pcd2.points)
        sample2 = normalize(sample2, norm="l1")
        sample2 = sample2[np.random.choice(np.arange(len(sample2)), 512)]   
        sample2 = np.reshape(sample2, (1, 512, 3))
        prediction = base_model.predict(sample2)
        print("predicted: " + str(np.argmax(prediction)) + ", actual: " + file[0])