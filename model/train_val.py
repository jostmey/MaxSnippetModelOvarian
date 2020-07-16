#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-11-28
# License: See LICENSE
# Purpose: Evaluate repertoire model on holdout cross-validation
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import h5py
import numpy as np
import tensorflow as tf
import os
from Models import *

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--holdouts', help='Holdout samples', type=str, nargs='+', required=True)
parser.add_argument('--seed', help='Seed for initializing model parameters', type=int, required=True)
args = parser.parse_args()

##########################################################################################
# Data
##########################################################################################

# Settings
#
holdouts = args.holdouts
cohorts = ['2018-04-17']
diagnoses = ['Normal', 'Malignant']
diagnoses_positive = ['Malignant']

# Load data
#
with h5py.File('../dataset/database.h5', 'r') as db:
  ds = db['diagnoses'][...]
  ms = db['motifs'][...]
  ss = db['samples'][...]
  cs = db['cohorts'][...]

# Select cohort
#
is_cohorts = []
for cohort in cohorts:
  is_cohorts.append(
    np.where(cs['name'].astype(str) == cohort)[0][0]
  )
is_sample = []
for i_cohort in is_cohorts:
  i0_sample, i1_sample = cs[i_cohort]['range_samples']
  is_sample += list(range(i0_sample, i1_sample))
ss_cohorts = ss[is_sample]

# Select diagnoses
#
is_diagnoses = []
for diagnosis in diagnoses:
  is_diagnoses.append(
    np.where(ds['name'].astype(str) == diagnosis)[0][0]
  )
is_sample = []
for i_diagnosis in is_diagnoses:
  is_sample += np.where(ss_cohorts['index_diagnosis'] == i_diagnosis)[0].tolist()
ss_diagnoses = ss_cohorts[is_sample]

# Split samples
#
is_train = list(range(ss_diagnoses.size))
is_val = []
for holdout in holdouts:
  i_holdout = np.where(ss_diagnoses['name'].astype(str) == holdout)[0][0]
  is_train.remove(i_holdout)
  is_val.append(i_holdout)
ss_train = ss_diagnoses[is_train]
ss_val = ss_diagnoses[is_val]

# Assemble motifs
#
ms_train = []
for s_train in ss_train:
  i0_motif, i1_motif = s_train['range_motifs']
  ms_train.append(
    ms[i0_motif:i1_motif]
  )
ms_val = []
for s_val  in ss_val:
  i0_motif, i1_motif = s_val['range_motifs']
  ms_val.append(
    ms[i0_motif:i1_motif]
  )
ms_train_val = ms_train+ms_val

# Assemble labels
#
ys_train = np.zeros(ss_train.size, dtype=np.float64)
ys_val = np.zeros(ss_val.size, dtype=np.float64)
for diagnosis_positive in diagnoses_positive:
  i_positive = np.where(ds['name'].astype(str) == diagnosis_positive)[0][0]
  ys_train[ss_train['index_diagnosis'] == i_positive] = np.float64(1.0)
  ys_val[ss_val['index_diagnosis'] == i_positive] = np.float64(1.0)
ys_train_val = np.concatenate([ys_train, ys_val], axis=0)

# Amino acid factors
#
fs_aminoacid = np.genfromtxt('../lib/atchley_factors.csv', delimiter=',')[:,1:].astype(np.float32)  # Assumes order of residues matches database
fs_norm = (fs_aminoacid-np.mean(fs_aminoacid, axis=0))/np.std(fs_aminoacid, axis=0)

##########################################################################################
# Operators
##########################################################################################

# Settings
#
motif_size = ms_train[0]['indices_aminoacid'][0].size
learning_rate = 0.01
seed = args.seed
num_models = 16384

# Sample inputs
#
indices_aminoacid = tf.placeholder(tf.int8, [None, motif_size])
frequencies = tf.placeholder(tf.float64, [None])
frequencies_baseline = tf.placeholder(tf.float64, [None])
label = tf.placeholder(tf.float64)

factors_aminoacid = tf.convert_to_tensor(fs_norm, dtype=tf.float64)

# Instantiate models
#
models = Models(factors_aminoacid, num_models)

# Prepare the features and normalize their values
#
with tf.variable_scope('features'):

  frequencies_expand = tf.expand_dims(frequencies, axis=1)
  features = models.features(indices_aminoacid, frequencies, frequencies_baseline, dtype=tf.float64)

  features_sample = tf.reduce_sum(tf.multiply(frequencies_expand, features), axis=0, keepdims=True)
  squares_sample = tf.reduce_sum(tf.multiply(frequencies_expand, tf.square(features)), axis=0, keepdims=True)
  number_sample = tf.cast(1.0, tf.float64)

  features_total = tf.get_variable(
    'features_total', shape=features_sample.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=features_sample.dtype, trainable=False
  )
  squares_total = tf.get_variable(
    'squares_total', shape=squares_sample.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=squares_sample.dtype, trainable=False
  )
  number_total = tf.get_variable(
    'number_total', shape=number_sample.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=number_sample.dtype, trainable=False
  )

  accumulate_features = features_total.assign_add(features_sample)
  accumulate_squares = squares_total.assign_add(squares_sample)
  accumulate_number = number_total.assign_add(number_sample)
  accumulate_features = tf.group(*[accumulate_features, accumulate_squares, accumulate_number])

  means_total = features_total/number_total
  variances_total = squares_total/number_total-tf.square(means_total)

  means = tf.get_variable(
    'means', shape=means_total.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=means_total.dtype, trainable=False
  )
  variances = tf.get_variable(
    'variances', shape=variances_total.get_shape(),
    initializer=tf.constant_initializer(1.0),
    dtype=variances_total.dtype, trainable=False
  )

  store_means = means.assign(means_total)
  store_variances = variances.assign(variances_total)
  store_features = tf.group(*[store_means, store_variances])

  features_norm = (features-means)/tf.sqrt(variances)

# Run the model and normalize its output
#
with tf.variable_scope('models'):

  logits = models.logits(features_norm, seed=seed, dtype=tf.float64)
  logits_sample = tf.reduce_max(logits, axis=0)

# Evaluate the performance of the models
#
with tf.variable_scope('metrics'):

  labels = tf.tile(tf.reshape(label, [1]), [num_models])
  probabilities_sample = tf.sigmoid(logits_sample)

  costs_sample = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_sample, labels=labels)
  accuracies_sample = tf.cast(tf.equal(tf.round(labels), tf.round(probabilities_sample)), tf.float64)
  number_sample = tf.constant(1.0, dtype=tf.float64)

  costs_total = tf.get_variable(
    'costs_total', shape=costs_sample.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=costs_sample.dtype, trainable=False
  )
  accuracies_total = tf.get_variable(
    'accuracies_total', shape=accuracies_sample.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=accuracies_sample.dtype, trainable=False
  )
  number_total = tf.get_variable(
    'number_total', shape=number_sample.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=number_sample.dtype, trainable=False
  )

  accumulate_costs = costs_total.assign_add(costs_sample)
  accumulate_accuracies = accuracies_total.assign_add(accuracies_sample)
  accumulate_number = number_total.assign_add(tf.constant(1.0, dtype=tf.float64))
  accumulate_metrics = tf.group(*[accumulate_costs, accumulate_accuracies, accumulate_number])

  reset_costs = costs_total.assign(tf.zeros_like(costs_total))
  reset_accuracies = accuracies_total.assign(tf.zeros_like(accuracies_total))
  reset_number = number_total.assign(tf.zeros_like(number_total))
  reset_metrics = tf.group(*[reset_costs, reset_accuracies, reset_number])

  costs = costs_total/number_total
  accuracies = accuracies_total/number_total
  index_bestmodel = tf.argmin(costs, axis=0)

# Fit the models to the data
#
with tf.variable_scope('optimizer'):

  optimizer = tf.train.AdamOptimizer(learning_rate)
  grads_params_sample = optimizer.compute_gradients(tf.reduce_sum(costs_sample), var_list=tf.trainable_variables())
  number_sample = tf.constant(1.0, dtype=tf.float64)
  step = tf.constant(1, dtype=tf.int64)

  grads_total = [
    tf.Variable(tf.zeros_like(param.initialized_value()), dtype=param.initialized_value().dtype, trainable=False) \
    for grad, param in grads_params_sample
  ]
  number_total = tf.get_variable(
    'number_total', shape=number_sample.get_shape(),
    initializer=tf.constant_initializer(0.0),
    dtype=number_sample.dtype, trainable=False
  )
  step_total = tf.get_variable(
    'step_total', shape=step.get_shape(),
    initializer=tf.constant_initializer(0),
    dtype=step.dtype, trainable=False
  )

  accumulate_gradients = tf.group(*[
    grads_total[index].assign_add(grad) for index, (grad, param) in enumerate(grads_params_sample)
  ])
  accumulate_number = number_total.assign_add(tf.constant(1.0, dtype=tf.float64))
  accumulate_optimizer = tf.group(*[accumulate_gradients, accumulate_number])
  
  reset_gradients = tf.group(*[
    grad.assign(tf.zeros_like(grad)) for grad in grads_total
  ])
  reset_number = number_total.assign(tf.zeros_like(number_total))
  reset_optimizer = tf.group(*[reset_gradients, reset_number])

  apply_gradients = optimizer.apply_gradients([
    (grads_total[index]/number_total, param) for index, (grad, param) in enumerate(grads_params_sample)
  ])
  apply_step = step_total.assign_add(step)
  apply_optimizer = tf.group(*[apply_gradients, apply_step])

# Store model
#
saver = tf.train.Saver(max_to_keep=100)

# Create operator to initialize session
#
initializer = tf.global_variables_initializer()

##########################################################################################
# Session
##########################################################################################

# Settings
#
num_train = ys_train.size
num_val = ys_val.size
num_train_val = ys_train_val.size
num_iterations = 2500
cutoff = 65536

# Suppress log messages
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Open session
#
with tf.Session() as session:

  # Initialize variables
  #
  session.run(initializer)

  # Check for save models
  #
  if not os.path.isfile('bin/models_holdouts='+','.join(holdouts)+'-seed='+str(seed)+'/last.ckpt.index'):

    # Normalize features
    #
    for i in range(num_train_val):
      session.run(
        accumulate_features,
        feed_dict={
          indices_aminoacid: ms_train_val[i]['indices_aminoacid'][:cutoff,:],
          frequencies: ms_train_val[i]['frequency'][:cutoff],
          frequencies_baseline: ms_train_val[i]['frequency_baseline'][:cutoff]
        }
      )
    session.run(store_features)

  # Restore saved models
  #
  else:
    saver.restore(session, 'bin/models_holdouts='+','.join(holdouts)+'-seed='+str(seed)+'/last.ckpt')

  # Training iterations
  #
  for iteration in range(num_iterations):

    # Train models
    #
    session.run((reset_metrics, reset_optimizer))
    for i in range(num_train):
      session.run(
        (accumulate_metrics, accumulate_optimizer),
        feed_dict={
          indices_aminoacid: ms_train[i]['indices_aminoacid'][:cutoff,:],
          frequencies: ms_train[i]['frequency'][:cutoff],
          frequencies_baseline: ms_train[i]['frequency_baseline'][:cutoff],
          label: ys_train[i]
        }
      )
    i_total, cs_train, as_train, i_bestfit_train = session.run((step_total, costs, accuracies, index_bestmodel))

    # Validate models
    #
    session.run(reset_metrics)
    ps_val = np.zeros((num_val, num_models), dtype=np.float64)
    for i in range(num_val):
      ps_val[i,:], _ = session.run(
        (probabilities_sample, accumulate_metrics),
        feed_dict={
          indices_aminoacid: ms_val[i]['indices_aminoacid'][:cutoff,:],
          frequencies: ms_val[i]['frequency'][:cutoff],
          frequencies_baseline: ms_val[i]['frequency_baseline'][:cutoff],
          label: ys_val[i]
        }
      )
    cs_val, as_val = session.run((costs, accuracies))

    # Update models
    #
    session.run(apply_optimizer)

    # Print report
    #
    print(
      i_total,
      np.mean(cs_train)/np.log(2.0, dtype=np.float64),
      100.0*np.mean(as_train),
      np.mean(cs_val)/np.log(2.0, dtype=np.float64),
      100.0*np.mean(as_val),
      i_bestfit_train,
      cs_train[i_bestfit_train]/np.log(2.0, dtype=np.float64),
      100.0*as_train[i_bestfit_train],
      cs_val[i_bestfit_train]/np.log(2.0, dtype=np.float64),
      100.0*as_val[i_bestfit_train],
      ','.join([ str(p) for p in ps_val[:,i_bestfit_train] ]),
      sep='\t', flush=True
    )

    # Periodically save the models
    #
    if i_total%500 == 0 and i_total > 0:
      saver.save(session, 'bin/models_holdouts='+','.join(holdouts)+'-seed='+str(seed)+'/iteration='+str(i_total)+'.ckpt')

  # Save model
  #
  saver.save(session, 'bin/models_holdouts='+','.join(holdouts)+'-seed='+str(seed)+'/last.ckpt')


