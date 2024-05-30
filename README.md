# Code for "Digital outcome measures from smartwatch data relate to non-motor features of Parkinson’s disease"
 published in [NPJ Parkinson's Disease](https://rdcu.be/dJjyV
) and on [medrxiv](https://doi.org/10.1101/2023.09.12.23295406)

There is also a [blog post](https://communities.springernature.com/posts/the-role-of-smartwatch-data-in-monitoring-non-motor-symptoms-of-parkinson-s-disease)
 
 
## Summary
Using data from PPMI we investigate the utility of digital sensor data for the monitoring of non-motor symptoms in Parkinson's disease. We compute digital weekly averages and relate those to clinical scores from visits. We further investigate whether such digital averages can predict the clinical scores on an individual level with machine learning models. Finally, we evaluate the rate of change of both digital and clinical measures and check for associations.
 
 
## Structure of this repository
This repository is split into 
 - 1_DataPrepocessing: which deals with loading of data from PPMI via an adapted version of pympi, cleaning of data, and extraction of progression estimates
 - 2_Analysis: all conducted relevant analysis (digital weekly average correlation with clinical scores, predicion of clinical scores, rate of change association)
 - 3_Figures: code to reproduce all relevant figures
 - scripts: helper functions
 - environment: anaconda .yml file to recreate python environment
