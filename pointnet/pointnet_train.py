import gc
import os
import csv
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
from keras import ops
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from keras import layers
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.models import Model, Sequential
from sklearn.utils import shuffle

class AccHistory(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        global plot_step_acc
        global plot_step_loss
        global epoch_total_step_counter
        global accumulator_step_acc
        global accumulator_step_loss
        global new_path
        try:  
            plot_step_acc.append(logs.get("accuracy"))
            plot_step_loss.append(logs.get("loss"))
            epoch_total_step_counter = epoch_total_step_counter + 1
            with open(new_path + "step_history.csv",'a', newline='') as f:
                writer=csv.writer(f)
                writer.writerow([logs.get("accuracy"),logs.get("loss")])
                f.close()
        except Exception as e:
            print(e)


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

##############################################
#################### LOAD ####################
##############################################

# train_data = pickle.load(open("./test pcd/pcd_test.pickle", "rb"))
# train_classes = pickle.load(open("./test pcd/pcd_classes.pickle", "rb"))

train_data = pickle.load(open("./pcd pickle/train_data.pickle", "rb"))
train_classes = pickle.load(open("./pcd pickle/train_classes.pickle", "rb"))

train_data = np.asarray(train_data)
train_classes = np.asarray(train_classes)

print(train_data.shape)

##############################################
################ PARAMETERS ##################
##############################################

X, y = shuffle(train_data, train_classes)

for i in range (0, 10):
    X, y = shuffle(X, y, random_state=np.random.randint(low=0, high=10000))

epoch_size = 3
number_of_splits = 5
kfold = StratifiedKFold(n_splits=number_of_splits, shuffle=False)

overall_accuracy = 0
overall_loss = 0
overall_val_accuracy = 0
overall_val_loss = 0

epoch_total_step_counter = 0

fold_no = 1

avg_accuracy = []
avg_loss = []
avg_val_accuracy = []
avg_val_loss = []
avg_step_accuracy = []
avg_step_loss = []

plot_step_acc = []
plot_step_loss = []

accumulator_step_acc = []
accumulator_step_loss = []
accumulator_epoch_acc = []
accumulator_epoch_loss = []
accumulator_epoch_val_acc = []
accumulator_epoch_val_loss = []

y = np.argmax(y, axis=1)

##############################################
################## TRAIN #####################
##############################################

try:
    for a, b in kfold.split(X, y):
        print('------------------------------------------------------------------------')
        print("Training for fold: " + str(fold_no))
        
        ##############################################
        ################### MODEL ####################
        ##############################################

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
        opt = keras.optimizers.Adam(learning_rate=0.0001)

        base_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
     
        ##############################################
        ################ PARA / VAR ##################
        ##############################################
        
        epoch_total_step_counter = 0
                              
        plot_step_acc.clear()
        plot_step_loss.clear()
        
        ##############################################
        ################# TRAINING ###################
        ##############################################
        
        new_path = "./models/model-" + str(fold_no) + "/"
        
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        
        with open(new_path + "step_history.csv",'w', newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['accuracy', 'loss'])
            f.close()
            
        y2 = to_categorical(y, num_classes = 8)
        
        history = base_model.fit(X[a], y2[a], validation_data=(X[b], y2[b]), callbacks=[AccHistory()], epochs=epoch_size, verbose=1)
        hist_df = pd.DataFrame(history.history)
        
        base_model.save_weights(new_path + "model.weights.h5")
        # base_model.save(new_path + "model.model")
         
        with open(new_path + "history.csv", mode='w') as csv_file: 
            hist_df.to_csv(csv_file)
            csv_file.close()
        
        ##############################################
        ######### OVERALL ACC. CALCULATION ###########
        ##############################################

        overall_accuracy = overall_accuracy + history.history['accuracy'][epoch_size-1]
        overall_loss = overall_loss + history.history['loss'][epoch_size-1]
        overall_val_accuracy = overall_val_accuracy + history.history['val_accuracy'][epoch_size-1]
        overall_val_loss = overall_val_loss + history.history['val_loss'][epoch_size-1]
        
        accumulator_epoch_acc.append(history.history['accuracy'])
        accumulator_epoch_loss.append(history.history['loss'])
        accumulator_epoch_val_acc.append(history.history['val_accuracy'])
        accumulator_epoch_val_loss.append(history.history['val_loss'])
        
        accumulator_step_acc.append(np.array(plot_step_acc))
        accumulator_step_loss.append(np.array(plot_step_loss))
        
        ##############################################
        ########## FIG. OF MODEL ACCURACY ############
        ##############################################
        
        ep1 = np.arange(1, epoch_total_step_counter+1, dtype = int)
        ep2 = np.arange(1, epoch_size+1, dtype = int)
        
        f1, ax1 = plt.subplots(figsize=(8,6), dpi=300)
        ax1.set_title("model accuracy & loss") 
        ax1.set_xlabel("steps")     
        ax1.set_ylabel("loss")
        ax1.plot(ep1, plot_step_loss, color="coral", label="loss")
        ax1.grid()
        ax2 = ax1.twinx()
        # ax2.set_ylabel("accuracy", fontweight="bold", color="steelblue")
        ax2.set_ylabel("accuracy")
        ax2.plot(ep1, plot_step_acc, color="steelblue", label="accuracy")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")
        f1.tight_layout()
        f1.savefig(new_path + "step-accuracy.pdf")
        f1.clf()
                         
        f2, ax3 = plt.subplots(figsize=(8, 6), dpi=300)  
        ax3.set_title("model accuracy & loss")        
        ax3.set_xlabel("epochs")
        ax3.set_ylabel("loss / val. loss")
        ax3.plot(ep2, history.history['loss'], color="red", label="training loss")
        ax3.plot(ep2, history.history['val_loss'], color="coral", label="validation loss")
        ax3.grid()
        ax4 = ax3.twinx()
        ax4.set_ylabel("accuracy / val. accuracy")      
        ax4.plot(ep2, history.history['accuracy'], color="steelblue", label="training accuracy")
        ax4.plot(ep2, history.history['val_accuracy'], color="seagreen", label="validation accuracy")
        lines3, labels3 = ax3.get_legend_handles_labels()
        lines4, labels4 = ax4.get_legend_handles_labels()
        ax4.legend(lines3 + lines4, labels3 + labels4, frameon=False, loc="center right")
        f2.tight_layout()
        f2.savefig(new_path + "epoch-accuracy.pdf")
        f2.clf()
        
        ##############################################
        ########## AVG. ACC. CALCULATION #############
        ##############################################

        if fold_no == 1:
            for i in range(0, epoch_size):
                avg_accuracy.append(history.history['accuracy'][i])
                avg_val_accuracy.append(history.history['val_accuracy'][i])
                avg_loss.append(history.history['loss'][i])
                avg_val_loss.append(history.history['val_loss'][i])
        else:
            for i in range(0, epoch_size):
                avg_accuracy[i] = avg_accuracy[i] + history.history['accuracy'][i]
                avg_val_accuracy[i] = avg_val_accuracy[i] + history.history['val_accuracy'][i]
                avg_loss[i] = avg_loss[i] + history.history['loss'][i]
                avg_val_loss[i] = avg_val_loss[i] + history.history['val_loss'][i]
                               
        if fold_no == 1:
            for i in range(0, epoch_total_step_counter):
                avg_step_accuracy.append(plot_step_acc[i])
                avg_step_loss.append(plot_step_loss[i])
        else:
            for i in range(0, epoch_total_step_counter):
                avg_step_accuracy[i] = avg_step_accuracy[i] + plot_step_acc[i]
                avg_step_loss[i] = avg_step_loss[i] + plot_step_loss[i]
 
        fold_no = fold_no + 1
        
        del base_model
        K.clear_session()
        gc.collect()
   
        ##############################################
        ################ END OF FOR ##################
        ##############################################
    
    with open("./models/overall_accuracy.txt", "w") as overall_accuracy_txt:
            overall_accuracy_txt.write("overall_accuracy:     " + str(overall_accuracy / float(number_of_splits))     + "\n" + "overall_loss:     " + str(overall_loss / float(number_of_splits)) + "\n")
            overall_accuracy_txt.write("overall_val_accuracy: " + str(overall_val_accuracy / float(number_of_splits)) + "\n" + "overall_val_loss: " + str(overall_val_loss / float(number_of_splits)))
    overall_accuracy_txt.close()

    ##############################################
    ########## AVG. ACC. CALCULATION #############
    ##############################################

    for i in range(0, epoch_size):
        avg_accuracy[i] = avg_accuracy[i] / float(number_of_splits)
        avg_val_accuracy[i] = avg_val_accuracy[i] / float(number_of_splits)
        avg_loss[i] = avg_loss[i] / float(number_of_splits)
        avg_val_loss[i] = avg_val_loss[i] / float(number_of_splits) 
            
    for i in range(0, epoch_total_step_counter):
        avg_step_accuracy[i] = avg_step_accuracy[i] / float(number_of_splits)
        avg_step_loss[i] = avg_step_loss[i] / float(number_of_splits)  

    ##############################################
    ########## FIG. OF TOTAL ACCURACY ############
    ##############################################

    f4, ax5 = plt.subplots(figsize=(8,6 ), dpi=300)
    ax5.set_title("overall model accuracy & loss")
    ax5.set_xlabel("steps")
    ax5.set_ylabel("loss")
    for loss in accumulator_step_loss:
        ax5.plot(ep1, loss, color="coral", linestyle='--', linewidth=1)
    ax5.grid()
    ax6 = ax5.twinx()
    ax6.set_ylabel("accuracy")
    for acc in accumulator_step_acc:
        ax6.plot(ep1, acc, color="steelblue", linestyle='--', linewidth=1)
    patch1 = Line2D([0], [0], color="steelblue", label='accuracy')
    patch2 = Line2D([0], [0], color="coral", label='loss')
    ax6.legend(handles=[patch1, patch2], frameon=False, loc="center right")  
    f4.tight_layout()
    f4.savefig("./models/overall-step-accuracy.pdf")
    f4.clf()
               
    f5, ax7 = plt.subplots(figsize=(8, 6), dpi=300)   
    ax7.set_title("overall model accuracy & loss")     
    ax7.set_xlabel("epochs")
    ax7.set_ylabel("loss / val. loss")
    for loss in accumulator_epoch_loss:
        ax7.plot(ep2, loss, color="red", label="training loss", linestyle='--', linewidth=1)
    for loss in accumulator_epoch_val_loss:
        ax7.plot(ep2, loss, color="coral", label="validation loss", linestyle='--', linewidth=1)
    ax7.grid()
    ax8 = ax7.twinx()
    ax8.set_ylabel("accuracy / val. accuracy")
    for acc in accumulator_epoch_acc:
        ax8.plot(ep2, acc, color="steelblue", label="training accuracy", linestyle='--', linewidth=1)
    for acc in accumulator_epoch_val_acc:
        ax8.plot(ep2, acc, color="seagreen", label="validation accuracy", linestyle='--', linewidth=1)
    patch1 = Line2D([0], [0], color="steelblue", label='training accuracy')
    patch2 = Line2D([0], [0], color="seagreen", label='validation accuracy')
    patch3 = Line2D([0], [0], color="red", label='training loss')
    patch4 = Line2D([0], [0], color="coral", label='validation loss')
    ax8.legend(handles=[patch1, patch2, patch3, patch4], frameon=False, loc="center right")
    f5.tight_layout()
    f5.savefig("./models/overall-epoch-accuracy.pdf")
    f5.clf()
    
    ##############################################
    ########## FIG. OF AVG. ACCURACY #############
    ##############################################

    f6, ax9 = plt.subplots(figsize=(8,6), dpi=300)
    ax9.set_title("average model accuracy & loss") 
    ax9.set_xlabel("steps")    
    ax9.set_ylabel("loss")
    ax9.plot(ep1, avg_step_loss, color="coral", label="loss")
    ax9.grid()
    ax10 = ax9.twinx()
    ax10.set_ylabel("accuracy")
    ax10.plot(ep1, avg_step_accuracy, color="steelblue", label="accuracy")
    lines9, labels9 = ax9.get_legend_handles_labels()
    lines10, labels10 = ax10.get_legend_handles_labels()
    ax10.legend(lines9 + lines10, labels9 + labels10, frameon=False, loc="center right")
    f6.tight_layout() 
    f6.savefig("./models/average-step-accuracy.pdf")
    f6.clf()
    
    f7, ax11 = plt.subplots(figsize=(8, 6), dpi=300)
    ax11.set_title("average model accuracy & loss")       
    ax11.set_xlabel("epochs")
    ax11.set_ylabel("loss / val. loss")
    ax11.plot(ep2, avg_loss, color="red", label="training loss")
    ax11.plot(ep2, avg_val_loss, color="coral", label="validation loss")
    ax11.grid()
    ax12 = ax11.twinx() 
    ax12.set_ylabel("accuracy / val. accuracy")
    ax12.plot(ep2, avg_accuracy, color="steelblue", label="training accuracy")
    ax12.plot(ep2, avg_val_accuracy, color="seagreen", label="validation accuracy")
    lines11, labels11 = ax11.get_legend_handles_labels()
    lines12, labels12 = ax12.get_legend_handles_labels()
    ax12.legend(lines11 + lines12, labels11 + labels12, frameon=False, loc="center right")
    f7.tight_layout()
    f7.set_size_inches(8, 6)
    f7.savefig("./models/average-epoch-accuracy.pdf", dpi=300)
    f7.clf()
    
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno, e)