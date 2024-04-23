# coding=utf-8
# Copyright 2024 The TensorFlow Datasets Authors.
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



from __future__ import annotations
import tensorflow as tf
from tensorflow import keras

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

print('\n img_train.shape: ')
print(x_train.shape)

print('\n img_train[0].shape: ')
print(x_train[0].shape)

x_train, x_test = x_train/255, x_test/255
y_train, y_test = tf.keras.utils.to_categorical(y_train, 100), tf.keras.utils.to_categorical(y_test, 100)

shapes(x_train, x_test, y_train, y_test, y_train, y_test)

img_inputs = tf.keras.Input(shape=(32, 32, 3))



def build_model():

    def build_cnn(im_inputs):
        mod = tf.keras.Sequential([
            tf.keras.layers.Conv2D(8, 2, 1),
            tf.keras.layers.Conv2D(8, 2, 1),
        ])(im_inputs)
        return mod
    
    x_1 = build_cnn(img_inputs)
    x_2 = build_cnn(img_inputs)
    x_3 = build_cnn(img_inputs)
    x_4 = build_cnn(img_inputs)
    x_5 = build_cnn(img_inputs)
    x_6 = build_cnn(img_inputs)
    x_7 = build_cnn(img_inputs)
    x_8 = build_cnn(img_inputs)
    
    
    x_1_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_1, x_2]))
    
    x_2_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_2, x_3]))
    
    x_3_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_3, x_4]))
    
    x_4_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_4, x_5]))
    
    x_5_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_5, x_6]))
    
    x_6_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_6, x_7]))
    
    x_7_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_7, x_8]))
    
    x_8_f = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Conv2D(8, 2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024),
    ])(keras.layers.Add()([x_8, x_1]))

    x_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_1_f)

    x_1 = tf.keras.layers.Add([x_1,x_7_f])

    
    
    x_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_2_f)

    x_2 = tf.keras.layers.Add([x_2,x_8_f])

    
    x_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_3_f)

    x_3 = tf.keras.layers.Add([x_3,x_1_f])
    
    
    x_4 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_4_f)

    x_4 = tf.keras.layers.Add([x_4,x_2_f])
    
    
    x_5 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_5_f)

    x_5 = tf.keras.layers.Add([x_5,x_3_f])

    x_6 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_6_f)

    x_6 = tf.keras.layers.Add([x_6,x_4_f])
    
    
    x_7 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_7_f)

    x_7 = tf.keras.layers.Add([x_7,x_5_f])

    
    
    x_8 = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1024, activation='softmax')
    ])(x_8_f)

    x_8 = tf.keras.layers.Add([x_8,x_6_f])



    x_1_1 =tf.keras.layers.Dense(1024)(keras.layers.Add()([x_1, x_2]))

    x_1_2 = tf.keras.layers.Dense(1024)(keras.layers.Add()([x_3, x_4]))

    x_2_1 =tf.keras.layers.Dense(1024, activation='relu')(keras.layers.Add()([x_1_1, x_5]))

    x_2_2 =tf.keras.layers.Dense(1024, activation='relu')(keras.layers.Add()([x_1_2, x_6]))

    x_3_1 =tf.keras.layers.Dense(1024, activation='softmax')(keras.layers.Concatenate()([x_2_1, x_7]))

    x_3_1 = keras.layers.Add()([x_3_1, x_1_1])

    x_3_2 =tf.keras.layers.Dense(1024, activation='softmax')(keras.layers.Concatenate()([x_2_2, x_8]))

    x_3_2 = keras.layers.Add()([x_3_2, x_1_2])


    x_4 = tf.keras.layers.Dense(1024)(keras.layers.Concatenate()([x_3_1, x_3_2]))

    x_4 = keras.layers.LayerNormalization()(x_4)

    

    x_5 = tf.keras.layers.Dense(1024)(x_4)



    x_5 = tf.keras.layers.Dropout(0.25)(x_5)

    x_5 = tf.keras.layers.Dense(1024)(x_5)

    x_5 = keras.layers.LeakyReLU(negative_slope=0.05)(x_5)

    outputs = tf.keras.layers.Dense(100, activation='softmax')(x_5)

    model = keras.Model(inputs=img_inputs, outputs=outputs, name="cifar_model")
    
    opt = keras.optimizers.Adam(learning_rate=0.0000001)

    model.compile(optimizer=opt, 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    model.save('cifar_model.keras')
    
    return model



model = build_model()

model = tf.keras.models.load_model('cifar_model.keras')

opt = keras.optimizers.Adam(learning_rate=0.00001)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


print('model.summary(): ')

model.summary()

print('\n \n \n ')

model.fit(x_train, y_train, batch_size=16, validation_data=(x_test, y_test), epochs=20)

model.save('cifar_model.keras')  # The file needs to end with the .keras extension



















# ds = tfds.load('imagenet2012', split='train', shuffle_files=True)







print('\n run successful')