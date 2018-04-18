from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import argparse
import os
import sys
import time
import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.datasets import mnist

from official.mnist import dataset as mnist_dataset
from official.mnist import mnist
from official.utils.arg_parsers import parsers

import tensorflow.contrib.eager as tfe
#from tensorflow.examples.tutorials.mnist import input_data

class DQNAgent:
    def __init__(self, data_format):
        self.num_classes = 10
        self.model = self._build_model(data_format)
        self.learning_rate = .1

    def _loss(self, logits, labels):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))

    def _compute_accuracy(self,logits, labels):
        predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
        labels = tf.cast(labels, tf.int64)
        batch_size = int(logits.shape[0])
        return tf.reduce_sum(
            tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size

    def _build_model(self, data_format):
        # CNN
        if data_format == 'channels_first':
            input_shape = [1, 28, 28]
        else:
            assert data_format == 'channels_last'
            input_shape = [28, 28, 1]

        l = tf.keras.layers
        max_pool = l.MaxPooling2D(
            (2, 2), (2, 2), padding='same', data_format=data_format)
        # The model consists of a sequential chain of layers, so tf.keras.Sequential
        # (a subclass of tf.keras.Model) makes for a compact description.
        return tf.keras.Sequential([
                l.Reshape((-1,28,28), input_shape=(784,)),
                l.Conv2D(
                    32,
                    5,
                    padding='same',
                    data_format=data_format,
                    activation=tf.nn.relu),
                max_pool,
                l.Conv2D(
                    64,
                    5,
                    padding='same',
                    data_format=data_format,
                    activation=tf.nn.relu),
                max_pool,
                l.Flatten(),
                l.Dense(1024, activation=tf.nn.relu),
                l.Dropout(0.4),
                l.Dense(10)
            ])

    def _train(self, model, optimizer, dataset, step_counter, log_interval=None):
        """Trains model on `dataset` using `optimizer`."""
        start = time.time()
        for (batch, (images, labels)) in enumerate(tfe.Iterator(dataset)):
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                10, global_step=step_counter):
                with tfe.GradientTape() as tape:
                    #print("image shape: ",images.shape)
                    logits = model(images, training=True)
                    #print("logits shape: ",logits.shape)

                    loss_value = self._loss(logits, labels)
                    tf.contrib.summary.scalar('loss', loss_value)
                    tf.contrib.summary.scalar('accuracy', self._compute_accuracy(logits, labels))
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)
                if log_interval and batch % log_interval == 0:
                    rate = log_interval / (time.time() - start)
                    print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
                    start = time.time()

    def _test(self,model, dataset):
        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tfe.metrics.Mean('loss')
        accuracy = tfe.metrics.Accuracy('accuracy')

        for (images, labels) in tfe.Iterator(dataset):
            logits = model(images, training=False)
            avg_loss(self._loss(logits, labels))
            accuracy(
                tf.argmax(logits, axis=1, output_type=tf.int64),
                tf.cast(labels, tf.int64))
        print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
                (avg_loss.result(), 100 * accuracy.result()))
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', avg_loss.result())
            tf.contrib.summary.scalar('accuracy', accuracy.result())        
""" def load_mnist():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

 """
class MNISTEagerArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(MNISTEagerArgParser, self).__init__(parents=[
            parsers.EagerParser(),
            parsers.ImageModelParser()])

        self.add_argument(
            '--log_interval', '-li',
            type=int,
            default=10,
            metavar='N',
            help='[default: %(default)s] batches between logging training status')
        self.add_argument(
            '--output_dir', '-od',
            type=str,
            default=None,
            metavar='<OD>',
            help='[default: %(default)s] Directory to write TensorBoard summaries')
        self.add_argument(
            '--lr', '-lr',
            type=float,
            default=0.01,
            metavar='<LR>',
            help='[default: %(default)s] learning rate')
        self.add_argument(
            '--momentum', '-m',
            type=float,
            default=0.5,
            metavar='<M>',
            help='[default: %(default)s] SGD momentum')
        self.add_argument(
            '--no_gpu', '-nogpu',
            action='store_true',
            default=False,
            help='disables GPU usage even if a GPU is available')

        self.set_defaults(
            data_dir='/tmp/tensorflow/mnist/input_data',
            model_dir='/tmp/tensorflow/mnist/checkpoints/',
            batch_size=100,
            train_epochs=10)

if __name__ == "__main__":
    tfe.enable_eager_execution()

    #tf.logging.set_verbosity(tf.logging.INFO)
    parser = MNISTEagerArgParser()
    flags = parser.parse_args()
    print(flags)


    # Automatically determine device and data_format
    (device, data_format) = ('/gpu:0', 'channels_first')
    if flags.no_gpu or tfe.num_gpus() <= 0:
        (device, data_format) = ('/cpu:0', 'channels_last')
    # If data_format is defined in FLAGS, overwrite automatically set value.
    if flags.data_format is not None:
        data_format = flags.data_format
    print('Using device %s, and data format %s.' % (device, data_format))

    # load data
    #(x_train, y_train), (x_test, y_test) = load_mnist()

    train_ds = mnist_dataset.train(flags.data_dir).shuffle(60000).batch(flags.batch_size)
    test_ds = mnist_dataset.test(flags.data_dir).batch(flags.batch_size)
    
    # Construct model
    dqnagent = DQNAgent(data_format)
    model = dqnagent.model
    model.summary()
    optimizer = tf.train.MomentumOptimizer(flags.lr, flags.momentum)

    # Create file writers for writing TensorBoard summaries.
    if flags.output_dir:
        # Create directories to which summaries will be written
        # tensorboard --logdir=<output_dir>
        # can then be used to see the recorded summaries.
        train_dir = os.path.join(flags.output_dir, 'train')
        test_dir = os.path.join(flags.output_dir, 'eval')
        tf.gfile.MakeDirs(flags.output_dir)
    else:
        train_dir = None
        test_dir = None

    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)
    test_summary_writer = tf.contrib.summary.create_file_writer(
        test_dir, flush_millis=10000, name='test')

# Create and restore checkpoint (if one exists on the path)
    checkpoint_prefix = os.path.join(flags.model_dir, 'ckpt')
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tfe.Checkpoint(
        model=model, optimizer=optimizer, step_counter=step_counter)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(tf.train.latest_checkpoint(flags.model_dir))

    # Train and evaluate for a set number of epochs.
    with tf.device(device):
        for _ in range(flags.train_epochs):
            start = time.time()
            with summary_writer.as_default():
                #Run train
                dqnagent._train(model, optimizer, train_ds, step_counter, flags.log_interval)
            end = time.time()
            print('\nTrain time for epoch #%d (%d total steps): %f' %
                (checkpoint.save_counter.numpy() + 1,
                    step_counter.numpy(),
                    end - start))
            with test_summary_writer.as_default():
                dqnagent._test(model, test_ds)
            checkpoint.save(checkpoint_prefix)


