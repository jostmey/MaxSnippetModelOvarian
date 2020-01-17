#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2018-02-01
# Purpose: Load repertoire samples and prepare the data for the model
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import dataplumbing as dp

##########################################################################################
# Settings
##########################################################################################

# Cohort settings
#
cohort = '2018-04-17'
base_dir = 'downloads/v2/'

# Motif settings
#
motif_window = 4
motif_size = 3
trim_front = 3
trim_rear = 3

# Database settings
#
path_db = 'database.h5'

##########################################################################################
# Build Database
##########################################################################################

# Load and prepare auxilary data
#
templates = dp.list_templates(motif_window, motif_size)

# Holds the range of samples in cohort
#
range_samples = [-1, -1]

# Loop over every sample
#
for diagnosis in ['Normal', 'Benign', 'Malignant']:
  for patient in range(1, 11):

    # Sample name and path
    #
    name = 'O-'+str(patient)+diagnosis[0]
    path = base_dir+name+'.tsv'

    # Load and process sequences
    #
    sequences = dp.load_sequences(path)
    sequences = dp.trim_sequences(sequences, trim_front, trim_rear, motif_size)
    motifs = dp.sequences_to_motifs(sequences, templates)

    # Save the sample and its data
    #
    index_diagnosis = dp.insert_diagnosis(path_db, diagnosis)
    range_motifs = dp.insert_motifs(path_db, motifs, templates)
    index_sample = dp.insert_sample(path_db, name, index_diagnosis, range_motifs)

    # Update the range of samples in cohort
    #
    if range_samples[0] == -1:
      range_samples[0] = index_sample
    range_samples[1] = index_sample+1

# Save cohort
#
dp.insert_cohort(path_db, cohort, range_samples)

