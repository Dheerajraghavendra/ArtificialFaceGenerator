#!/usr/bin/python

from __future__ import absolute_import, division, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from IPython import display
#from PIL import Image
from pathlib import Path
import os
import time

AUTOTUNE = tf.data.experimental.AUTOTUNE    #Have to read more

p = Path('train/processed_celeba_small/celeba')

images = list(p.glob('*.jpg'))
n = len(images)

filenames_ds = tf.data.Dataset.list_files(str(p/'*.jpg'))

BATCH_SIZE = 256
HEIGHT = 32
WIDTH = 32
STEPS_PER_EPOCH = np.ceil(n/BATCH_SIZE)

def image_from_pathstring(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img*2-1  #normalize to [-1,1]   (initially (0,1))
    return tf.image.resize(img, [WIDTH, HEIGHT])

imgs = filenames_ds.map(image_from_pathstring, num_parallel_calls=AUTOTUNE)


#for f in imgs.take(1):
#    plt.imshow(f.numpy())
#plt.show()

def show_batch(image_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        sub = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        #plt.axis('off')


'''
def prepare_for_train(ds, cache=True, shuffle_buf_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size = shuffle_buf_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = prepare_for_train(imgs)
train_iter = iter(train_ds)
image_batch = next(train_iter)
'''

#show_batch(image_batch.numpy())

IP_SIZE = 100

#t = imgs.shuffle(BATCH_SIZE)

def create_discriminator():
    model = tf.keras.Sequential()
    
    model.add(layers.ZeroPadding2D(1, input_shape=[32, 32, 3]))
    model.add(layers.Conv2D(64, (4,4), strides=(2,2), padding = 'valid'))
    model.add(layers.LeakyReLU())
    print model.output_shape

    model.add(layers.ZeroPadding2D(1))
    model.add(layers.Conv2D(128, (4,4), strides=(2,2), padding='valid'))  #inputshape = [16, 16, 64]
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print model.output_shape

    model.add(layers.ZeroPadding2D(1))
    model.add(layers.Conv2D(256, (4,4), strides=(2,2), padding='valid'))  #inputshape = [8, 8, 128]
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    print model.output_shape

    model.add(layers.Flatten()) #inputshape = [4, 4, 256]
    model.add(layers.Dense(1))

    return model

def create_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(2*2*512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((2, 2, 512)))
    
    print model.output_shape

    model.add(layers.Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', use_bias=False))
    print model.output_shape
    #assert model.output_shape == (None, 2, 2, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', use_bias=False))  #inputshape = [4, 4, 256]
    print model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', use_bias=False))  #inputshape = [8, 8, 128]
    print model.output_shape
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', use_bias=False, activation='tanh'))  #inputshape = [16, 16, 64]
    print model.output_shape
    assert model.output_shape == (None, 32, 32, 3)

    return model


generator = create_generator()
noise = tf.random.normal([1, IP_SIZE])
discriminator = create_discriminator()
#generated_image = generator(noise, training=False)
#print generated_image, type(generated_image)
#plt.imshow(generated_image[0, :, :, 2]);
#plt.show()

bin_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def loss_discriminator(real_output, fake_output):
    real_loss = bin_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = bin_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss+fake_loss

def loss_generator(fake_output):
    return bin_cross_entropy(tf.ones_like(fake_output), fake_output)

opt_generator = tf.keras.optimizers.Adam(1e-4)
opt_discriminator = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer = opt_generator, discriminator_optimizer = opt_discriminator, generator=generator, discriminator=discriminator)

EPOCHS=40
NUM_SAMPLES=16
seed = tf.random.normal([NUM_SAMPLES, IP_SIZE])

@tf.function
def train_step(image_batch):
    noise = tf.random.normal([BATCH_SIZE, IP_SIZE])
    with tf.GradientTape() as tape_gen, tf.GradientTape() as tape_disc:
        generated_images = generator(noise, training=True)
        
        real_y = discriminator(image_batch, training=True)
        fake_y = discriminator(generated_images, training=True)

        loss_disc = loss_discriminator(real_y, fake_y)
        loss_gen = loss_generator(fake_y)
        
        gradients_gen  = tape_gen.gradient(loss_gen, generator.trainable_variables)
        gradients_disc = tape_disc.gradient(loss_disc, discriminator.trainable_variables)

    opt_generator.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    opt_discriminator.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

def start_train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch+1, seed)

        if(epoch)%15==0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        print "Time for epoch ", epoch+1, " is ",time.time()-start, " sec"

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

def generate_and_save_images(model, epoch, ip_sample):
    op_sample = model(ip_sample, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(op_sample.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow((op_sample[i, :, :, :]+1)/2)
        #plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

dataset = imgs.shuffle(50000).batch(BATCH_SIZE)
start_train(dataset, EPOCHS)
