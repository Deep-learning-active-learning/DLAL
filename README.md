# DLAL

This is the ReadMe documentation for DLAL, a proposed Deep Learning Based Active Learning using two networks, VQNN and VPLNN, to rank, query, and pseudolabel vectors in order to minimize the amount of labeled data needed to train models. Please read through the relevant project paper before perusing through this codebase.
Built on sklearn: https://scikit-learn.org/stable/

## DLAL
This is the location where the models are trained and utilized within the AL framework.
Includes code that uses VQNN and VPLNN to excute DLAL.

## utils
Contains various utilities used throughout the codebase in a single, easy to use place.
All functions used multiple time are found here, unless the relevant code is considered complex enough to
warrant its own file (EX: inf_density, egl).

## inf_density
Calculates density metrics for the given problem space. 
Built on Modal: https://github.com/modAL-python/modAL

## mval
Python implementation of the official Matlab codebase for "MVAL: Maximizing Variance for Active Learning." Official codebase can be found here: https://github.com/YazhouTUD/MVAL

## egl
Expected Gradient Length implementation.

## genDataset
Script that generates training sets + test/validation sets for both VQNN and VPLNN.

## calc_baseline
Code used to run and evaluate baseline AL algorithms.

## make_plots
Simple script that searches for relevant pickled dictionaries and plots all computed AL algorithms' performance over time for the requested dataset.

## HyperOpt
Used to perform hyperparameter tuning for algorithms. Uses TPE via the hyperopt package.
