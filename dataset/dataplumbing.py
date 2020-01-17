#########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-07-26
# Environment: Python3
# Purpose: Utilities for creating a datbase of immune receptor sequences
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import csv
import numpy as np
import os.path
import h5py
from itertools import combinations

##########################################################################################
# Load Samples
##########################################################################################

def load_sequences(path_tsv):
  sequences = dict()
  with open(path_tsv, 'r') as stream:
    reader = csv.DictReader(stream, delimiter='\t')
    for row in reader:
      sequence = row['aminoAcid']
      quantity = np.float64(row['frequencyCount (%)'])
      status = row['sequenceStatus']
      if 'In' in status and 'X' not in sequence: 
        if sequence not in sequences:
          sequences[sequence] = quantity
        else:
          sequences[sequence] += quantity
  return sequences

def trim_sequences(sequences, trim_front, trim_rear, min_length):
  sequences_trim = dict()
  for sequence, quantity in sequences.items():
    trim_front_adjusted = trim_front
    trim_rear_adjusted = trim_rear
    while len(sequence)-trim_front_adjusted-trim_rear_adjusted < min_length:
      trim_front_adjusted -= 1
      trim_rear_adjusted -= 1
    if trim_front_adjusted < 0:
      trim_front_adjusted = 0
    if trim_rear_adjusted < 0:
      trim_rear_adjusted = 0
    if trim_rear_adjusted > 0:
      sequence_trim = sequence[trim_front_adjusted:-trim_rear_adjusted]
    else:
      sequence_trim = sequence[trim_front_adjusted:]
    if len(sequence_trim) >= min_length:
      if sequence_trim not in sequences_trim:
        sequences_trim[sequence_trim] = quantity
      else:
        sequences_trim[sequence_trim] += quantity
  return sequences_trim

def list_templates(motif_window, motif_size):
  templates = dict()
  index_template = 0
  for motif_window_ in range(motif_size, motif_window+1):
    templates[motif_window_] = dict()
    window = list(range(motif_window_))
    for template_tuple in combinations(window[1:-1], motif_size-2):
      template = list(template_tuple)
      template.insert(0, window[0])
      template.append(window[-1])
      templates[motif_window_][index_template] = template
      index_template += 1
  return templates

def sequences_to_motifs(sequences, templates):
  motif_size = len(list(list(templates.values())[0].values())[0])
  motif_window = max(templates.keys())
  motifs = dict()
  for sequence, quantity in sequences.items():
    num_types = 0
    for motif_window_ in range(motif_size, motif_window+1):
      stop = len(sequence)-motif_window_+1
      if stop >= 0:
        for i in range(stop):
          sequence_window = sequence[i:i+motif_window_]
          for index_template, template in templates[motif_window_].items():
            motif = ''
            for i in template:
              motif = motif+sequence_window[i]
            motif = motif+':'+str(index_template)
            if motif not in motifs:
              motifs[motif] = quantity
            else:
              motifs[motif] += quantity
  total = 0.0
  for quantity in motifs.values():
    total += quantity
  for motif, quantity in motifs.items():
    motifs[motif] = quantity/total
  return motifs

##########################################################################################
# Create Database
##########################################################################################

def insert_diagnosis(path_db, diagnosis):
  dtype_diagnosis = [
    ('name', 'S512')
  ]
  flag = 'r+' if os.path.isfile(path_db) else 'w'
  with h5py.File(path_db, flag) as db:
    if 'diagnoses' not in db:
      ds_db = db.create_dataset('diagnoses', (0,), dtype_diagnosis, maxshape=(None,))
    else:
      ds_db = db['diagnoses']
    if diagnosis not in ds_db[...].astype(str):
      ds_db.resize(ds_db.size+1, axis=0)
      ds_db[-1] = diagnosis
    index_diagnosis = list(ds_db[...].astype(str)).index(diagnosis)
  return index_diagnosis

def insert_motifs(path_db, motifs, templates):
  motif_size = len(list(list(templates.values())[0].values())[0])
  dtype_aminoacid = [
    ('residue', 'S1')
  ]
  dtype_template = [
    ('indices_position', str(motif_size)+'i1')
  ]
  dtype_motif = [
    ('indices_aminoacid', str(motif_size)+'i1'),
    ('index_template', 'i4'),
    ('frequency', 'f8'),
  ]
  flag = 'r+' if os.path.isfile(path_db) else 'w'
  with h5py.File(path_db, flag) as db:
    if 'aminoacids' not in db:
      as_db = db.create_dataset('aminoacids', (20,), dtype_aminoacid)
      as_db[...] = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'], dtype='S1')
    else:
      as_db = db['aminoacids']
    aminoacids_list = as_db[...].astype(str)
    aminoacids_dict = { aa: index for index, aa in enumerate(aminoacids_list) }
    if 'templates' not in db:
      templates_ = dict()
      for templates_size in templates.values():
        for index_template, template in templates_size.items():
          templates_[index_template] = template
      ts = np.zeros(len(templates_), dtype=dtype_template)
      for index_template, template in templates_.items():
        ts[index_template]['indices_position'] = template
      ts_db = db.create_dataset('templates', (len(templates_),), dtype_template)
      ts_db[...] = ts
    if 'motifs' not in db:
      ms_db = db.create_dataset('motifs', (0,), dtype_motif, maxshape=(None,))
    else:
      ms_db = db['motifs']
    ms = np.zeros(len(motifs), dtype=dtype_motif)
    for i, motif in enumerate(sorted(motifs, key=motifs.get, reverse=True)):
      aminoacids, index_template = motif.split(':')
      for j, aminoacid in enumerate(aminoacids):
        ms[i]['indices_aminoacid'][j] = aminoacids_dict[aminoacid]
      ms[i]['index_template'] = str(index_template)
      ms[i]['frequency'] = motifs[motif]
    ms_db.resize(ms_db.size+ms.size, axis=0)
    ms_db[-ms.size:] = ms
    range_motifs = [ms_db.size-ms.size, ms_db.size]
  return range_motifs

def insert_sample(path_db, name, index_diagnosis, range_motifs):
  dtype_sample = [
    ('name', 'S512'),
    ('index_diagnosis', 'i1'),
    ('range_motifs', '2i8')
  ]
  flag = 'r+' if os.path.isfile(path_db) else 'w'
  with h5py.File(path_db, flag) as db:
    if 'samples' not in db:
      ss_db = db.create_dataset('samples', (0,), dtype_sample, maxshape=(None,))
    else:
      ss_db = db['samples']
    s = np.zeros(None, dtype=dtype_sample)
    s['name'] = name
    s['index_diagnosis'] = index_diagnosis
    s['range_motifs'] = range_motifs
    index_sample = ss_db.size
    ss_db.resize(ss_db.size+1, axis=0)
    ss_db[-1] = s
    return index_sample

def insert_cohort(path_db, name, range_samples):
  dtype_cohort = [
    ('name', 'S512'),
    ('range_samples', '2i8')
  ]
  flag = 'r+' if os.path.isfile(path_db) else 'w'
  with h5py.File(path_db, flag) as db:
    if 'cohorts' not in db:
      cs_db = db.create_dataset('cohorts', (0,), dtype_cohort, maxshape=(None,))
    else:
      cs_db = db['cohorts']
    c = np.zeros(None, dtype=dtype_cohort)
    c['name'] = name
    c['range_samples'] = range_samples
    cs_db.resize(cs_db.size+1, axis=0)
    cs_db[-1] = c


