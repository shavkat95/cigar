# coding=utf-8
# Copyright 512 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CIFAR100-N dataset."""

# todos:
# - class_weights
# - generator
# - augmentation



from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras import activations

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def shapes(*x):
    for idx, item in enumerate(x):
        print(f"arg_{idx}: {item.shape}") if hasattr(item, "shape") else print(f"arg_{idx}: {len(item)}")



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return (a[p], b[p])

print('\n img_train.shape: ')
print(x_train.shape)

print('\n img_train[0].shape: ')
print(x_train[0].shape)

(x_train, y_train) = unison_shuffled_copies(x_train, y_train)

x_train, x_test = x_train/255, x_test/255
y_train, y_test = tf.keras.utils.to_categorical(y_train, 100), tf.keras.utils.to_categorical(y_test, 100)

shapes(x_train, x_test, y_train, y_test, y_train, y_test)

img_inputs = tf.keras.Input(shape=(32, 32, 3))



def build_model(dp_rate = 0.5):

    def build_cnn(im_inputs, run = 1):
        if run == 2:
            mod = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, 1),
                tf.keras.layers.ELU(alpha=0.9),
                tf.keras.layers.Conv2D(32, 3, 2),
                tf.keras.layers.ELU(alpha=0.9),
                tf.keras.layers.Reshape((12, 12, 32, 1)),
                
            ])(im_inputs)
        elif run == 1:
            mod = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, 3, 1),
                tf.keras.layers.Conv2D(16, 3)
            ])(im_inputs)
        return mod
    
    old_x_1 = build_cnn(img_inputs)
    old_x_2 = build_cnn(img_inputs)
    old_x_3 = build_cnn(img_inputs)
    old_x_4 = build_cnn(img_inputs)
    x_5 = build_cnn(img_inputs)
    x_6 = build_cnn(img_inputs)
    x_7 = build_cnn(img_inputs)
    x_8 = build_cnn(img_inputs)


    x_1 = build_cnn(keras.layers.Add()([old_x_1, x_5]), 2)
    x_2 = build_cnn(keras.layers.Add()([old_x_2, x_6]), 2)
    x_3 = build_cnn(keras.layers.Add()([old_x_3, x_7]), 2)
    x_4 = build_cnn(keras.layers.Add()([old_x_4, x_8]), 2)
    x_5 = build_cnn(keras.layers.Add()([x_5, old_x_1]), 2)
    x_6 = build_cnn(keras.layers.Add()([x_6, old_x_2]), 2)
    x_7 = build_cnn(keras.layers.Add()([x_7, old_x_3]), 2)
    x_8 = build_cnn(keras.layers.Add()([x_8, old_x_4]), 2)
    
    
    x_1_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_1, x_2]))
    max_pool_1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_1_f)
    avg_pool_1 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(x_1_f)
    x_1_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Concatenate()([max_pool_1, avg_pool_1]))

    x_1_f = tf.keras.layers.Dense(110)(x_1_f)


    x_2_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_2, x_3]))
    max_pool_2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_2_f)
    avg_pool_2 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(x_2_f)
    x_2_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Multiply()([max_pool_2, avg_pool_1]))

    x_2_f = tf.keras.layers.Dense(110)(keras.layers.Concatenate()([x_2_f, x_1_f]))
    
    x_3_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_3, x_4]))
    max_pool_3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_3_f)
    avg_pool_3 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(x_3_f)
    x_3_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Add()([max_pool_3, avg_pool_2]))

    x_3_f = tf.keras.layers.Dense(110)(keras.layers.Concatenate()([x_1_f, x_3_f]))
    
    x_4_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_4, x_5]))
    max_pool_4 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_4_f)
    avg_pool_4 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(x_4_f)
    x_4_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Add()([max_pool_4, avg_pool_3]))

    x_4_f = tf.keras.layers.Dense(110)(keras.layers.Concatenate()([x_4_f, x_2_f]))
    
    x_5_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_5, x_6]))
    max_pool_5 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_5_f)
    avg_pool_5 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(x_5_f)
    x_5_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Add()([max_pool_5, avg_pool_4]))

    x_5_f = tf.keras.layers.Dense(110)(keras.layers.Concatenate()([x_5_f, x_3_f]))
    
    x_6_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_6, x_7]))
    max_pool_6 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_6_f)
    avg_pool_6 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(x_6_f)
    x_6_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Add()([max_pool_6, avg_pool_5]))

    x_6_f = tf.keras.layers.Dense(110)(x_6_f)
    
    x_7_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_7, x_8]))
    max_pool_7 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_7_f)
    avg_pool_7 = tf.keras.layers.AveragePooling3D(pool_size=(2, 2, 1))(x_7_f)
    x_7_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Multiply()([max_pool_7, avg_pool_6]))

    x_7_f = tf.keras.layers.Dense(110)(keras.layers.Concatenate()([x_7_f, x_5_f]))
    
    x_8_f = tf.keras.Sequential([
        tf.keras.layers.Conv3D(128, 4, (1,1,1)),
        tf.keras.layers.ELU(alpha=0.8),
        tf.keras.layers.Conv3D(128, 4, (2,1,1))])(keras.layers.Add()([x_8, x_1]))
    max_pool_8 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(x_8_f)
    x_8_f = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dp_rate),
        tf.keras.layers.Dense(128),
        keras.layers.ELU(alpha=0.8),
    ])(keras.layers.Concatenate()([max_pool_8, avg_pool_7]))

    x_8_f = tf.keras.layers.Dense(110)(keras.layers.Concatenate()([x_8_f, x_6_f]))

    x_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
    ])(keras.layers.Concatenate()([x_1_f, x_7_f]))

    x_1 = tf.keras.layers.Concatenate()([x_1,x_6_f])
    x_1 = keras.layers.BatchNormalization()(x_1)
    
    x_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
    ])(tf.keras.layers.Concatenate()([x_2_f,x_8_f]))

    x_2 = keras.layers.BatchNormalization()(x_2)

    
    x_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
    ])(x_3_f)

    x_3 = tf.keras.layers.Concatenate()([x_3,x_1_f])
    x_3 = keras.layers.BatchNormalization()(x_3)
    
    
    x_4 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
    ])(x_4_f)

    x_4 = tf.keras.layers.Concatenate()([x_4,x_2_f])
    x_4 = keras.layers.BatchNormalization()(x_4)
    
    
    x_5 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
    ])(x_5_f)

    x_5 = tf.keras.layers.Concatenate()([x_5,x_3_f])
    x_5 = keras.layers.BatchNormalization()(x_5)

    x_6 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
    ])(x_6_f)

    x_6 = tf.keras.layers.Concatenate()([x_6,x_4_f])
    x_6 = keras.layers.BatchNormalization()(x_6)
    
    
    x_7 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7)
    ])(x_7_f)


    x_7 = tf.keras.layers.Concatenate()([x_7,x_5_f])
    x_7 = keras.layers.BatchNormalization()(x_7)

    
    
    x_8 = tf.keras.Sequential([
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7),
        tf.keras.layers.Dense(100),
        keras.layers.ELU(alpha=0.7)
    ])(x_8_f)



    x_8 = tf.keras.layers.Concatenate()([x_8,x_6_f])

    x_8 = keras.layers.BatchNormalization()(x_8)



    x_1_1 =tf.keras.layers.Dense(128)(keras.layers.Concatenate()([x_1, x_2]))

    x_1_2 = tf.keras.layers.Dense(128)(keras.layers.Concatenate()([x_3, x_4]))

    x_2_1 =tf.keras.layers.Dense(128, activation='sigmoid')(keras.layers.Multiply()([x_1_1, x_1_2]))

    x_2_2 =tf.keras.layers.Dense(128, activation='sigmoid')(keras.layers.Multiply()([x_5, x_6]))

    x_3_1 =tf.keras.layers.Dense(256)(keras.layers.Concatenate()([x_2_1, x_7]))

    x_3_1 = keras.layers.Concatenate()([x_3_1, x_1_1])

    x_3_2 =tf.keras.layers.Dense(256)(keras.layers.Concatenate()([x_2_2, x_8]))

    x_3_2 = keras.layers.Concatenate()([x_3_2, x_1_2])


    x_5 = tf.keras.layers.Dense(512)(keras.layers.Concatenate()([x_3_1, x_3_2]))

    x_5 = keras.layers.BatchNormalization()(x_5)

    top_dropoutrate = dp_rate
    
    def mk_wires(x_x, run = -1):
        
        old_w_1 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        w_2 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        w_3 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        w_4 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        w_5 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        w_6 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        w_7 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        w_8 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
        
        old_w_1 = tf.keras.layers.Dense(64, use_bias=False)(old_w_1)
        w_2 = tf.keras.layers.Dense(64, use_bias=False)(w_2)
        w_3 = tf.keras.layers.Dense(64, use_bias=False)(w_3)
        w_4 = tf.keras.layers.Dense(64, use_bias=False)(w_4)
        w_5 = tf.keras.layers.Dense(64, use_bias=False)(w_5)
        w_6 = tf.keras.layers.Dense(64, use_bias=False)(w_6)
        w_7 = tf.keras.layers.Dense(64, use_bias=False)(w_7)
        w_8 = tf.keras.layers.Dense(64, use_bias=False)(w_8)
        
        w_1 = tf.keras.layers.Multiply()([old_w_1, w_2])
        old_w_2 = tf.keras.layers.Multiply()([w_2, w_3])
        w_3 = tf.keras.layers.Multiply()([w_3, w_4])
        w_4 = tf.keras.layers.Multiply()([w_4, w_5])
        w_5 = tf.keras.layers.Multiply()([w_5, w_6])
        w_6 = tf.keras.layers.Multiply()([w_6, w_7])
        w_7 = tf.keras.layers.Multiply()([w_7, w_8])
        w_8 = tf.keras.layers.Multiply()([w_8, old_w_1])
        
        w_1 = tf.keras.layers.Dense(64, use_bias=False)(w_1)
        old_w_2 = tf.keras.layers.Dense(64, use_bias=False)(old_w_2)
        w_3 = tf.keras.layers.Dense(64, use_bias=False)(w_3)
        w_4 = tf.keras.layers.Dense(64, use_bias=False)(w_4)
        w_5 = tf.keras.layers.Dense(64, use_bias=False)(w_5)
        w_6 = tf.keras.layers.Dense(64, use_bias=False)(w_6)
        w_7 = tf.keras.layers.Dense(64, use_bias=False)(w_7)
        w_8 = tf.keras.layers.Dense(64, use_bias=False)(w_8)

        
        w_1 = tf.keras.layers.Multiply()([w_1, w_3])
        w_2 = tf.keras.layers.Multiply()([old_w_2, w_4])
        w_3 = tf.keras.layers.Multiply()([w_3, w_5])
        w_4 = tf.keras.layers.Multiply()([w_4, w_6])
        w_5 = tf.keras.layers.Multiply()([w_5, w_7])
        w_6 = tf.keras.layers.Multiply()([w_6, w_8])
        w_7 = tf.keras.layers.Multiply()([w_7, w_1])
        w_8 = tf.keras.layers.Multiply()([w_8, old_w_2])

        # w_1 = keras.layers.ELU(alpha=0.9)(w_1)
        w_2 = keras.layers.ELU(alpha=0.7)(w_2)
        w_3 = keras.layers.ELU(alpha=0.5)(w_3)
        w_4 = keras.layers.ELU(alpha=0.4)(w_4)
        w_5 = keras.layers.ELU(alpha=0.3)(w_5)
        w_6 = keras.layers.ELU(alpha=0.2)(w_6)
        w_7 = keras.layers.ELU(alpha=0.1)(w_7)
        w_8 = keras.layers.ELU(alpha=0.5)(w_8)
        
        def mk_more_wires(w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8):
            output_ls = [w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8]
            i = 0
            for layer in output_ls:
                output_ls[i] = tf.keras.layers.Concatenate()([w_1[:, (8*i):(8*(i+1))], w_2[:, (8*i):(8*(i+1))], w_3[:, (8*i):(8*(i+1))], w_4[:, (8*i):(8*(i+1))], w_5[:, (8*i):(8*(i+1))], w_6[:, (8*i):(8*(i+1))], w_7[:, (8*i):(8*(i+1))], w_8[:, (8*i):(8*(i+1))]])
                i+=1
            return output_ls
        
        [w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8] = mk_more_wires(w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8)
                
        w_1 = tf.keras.layers.Dense(64, )(w_1)
        w_2 = tf.keras.layers.Dense(64, )(w_2)
        w_3 = tf.keras.layers.Dense(64, )(w_3)
        w_4 = tf.keras.layers.Dense(64, )(w_4)
        w_5 = tf.keras.layers.Dense(64, )(w_5)
        w_6 = tf.keras.layers.Dense(64, )(w_6)
        w_7 = tf.keras.layers.Dense(64, )(w_7)
        w_8 = tf.keras.layers.Dense(64, )(w_8)
        
        res = tf.keras.layers.Concatenate()([w_4, w_5, w_6, w_7, w_8, w_1, w_2, w_3])
        
        if run == 1:
            w_1 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            w_2 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            w_3 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            w_4 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            w_5 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            w_6 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            w_7 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            w_8 = tf.keras.layers.Dropout(top_dropoutrate)(x_x)
            
            w_1 = tf.keras.layers.Dense(64, use_bias=False)(w_1)
            w_2 = tf.keras.layers.Dense(64, use_bias=False)(w_2)
            w_3 = tf.keras.layers.Dense(64, use_bias=False)(w_3)
            w_4 = tf.keras.layers.Dense(64, use_bias=False)(w_4)
            w_5 = tf.keras.layers.Dense(64, use_bias=False)(w_5)
            w_6 = tf.keras.layers.Dense(64, use_bias=False)(w_6)
            w_7 = tf.keras.layers.Dense(64, use_bias=False)(w_7)
            w_8 = tf.keras.layers.Dense(64, use_bias=False)(w_8)
        
            x_x = tf.keras.layers.Concatenate()([w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8])
        
        res = tf.keras.layers.Add()([res, x_x])
        
        res = tf.keras.layers.LayerNormalization()(res)
        
        return res
        
        

    x_5 = mk_wires(x_5, run = 1)

    x_5 = mk_wires(x_5)

    x_5 = mk_wires(x_5)

    x_5 = mk_wires(x_5)

    x_5 = mk_wires(x_5)

    x_5 = mk_wires(x_5)

    x_5 = mk_wires(x_5)

    x_5 = mk_wires(x_5)

    x_5 = keras.layers.ELU(alpha=0.3)(x_5)

    x_5 = tf.keras.layers.Dense(256, use_bias=False)(x_5)

    x_5 = keras.layers.ELU(alpha=0.2)(x_5)

    outputs = tf.keras.layers.Dense(100, activation='softmax')(x_5)

    model = keras.Model(inputs=img_inputs, outputs=outputs, name="cifar_model")
    
    opt = keras.optimizers.RMSprop(learning_rate=0.001)

    model.compile(optimizer=opt, 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model



# old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

# model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print('model.summary(): ')

model.summary()

print('\n \n \n ')

class CustomCallback(keras.callbacks.Callback):


    def on_epoch_end(self, epoch, logs=None):
        print('\n')
        # keys = list(logs.keys())
        # print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        print('\n')
        global x_train
        global y_train
        (x_train, y_train) = unison_shuffled_copies(x_train, y_train)


my_callbacks = [
    # keras.callbacks.EarlyStopping(patience=2),
    keras.callbacks.ModelCheckpoint(filepath='cifar_model.{val_loss:.2f}.keras'),
    # keras.callbacks.TensorBoard(log_dir='./logs'),
    CustomCallback()
]

model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')  # The file needs to end with the .keras extension

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.2)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.2)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')


old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.2)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')



old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')



old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')


old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')


old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')


old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')


old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')


old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')


old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')




old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')

old_model = tf.keras.models.load_model('cifar_model.keras')

model = build_model(dp_rate=0.5)

model.set_weights(old_model.get_weights())

opt = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, callbacks = my_callbacks, batch_size=16, validation_data=(x_test, y_test), epochs=1)

model.save('cifar_model.keras')










# ds = tfds.load('imagenet2012', split='train', shuffle_files=True)







print('\n run successful')