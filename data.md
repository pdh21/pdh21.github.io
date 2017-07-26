---
layout: page
title: Data Analysis
permalink: /data/
---
The field of Astronomy is unique among the sciences in that astronomers cannot conduct experiments, they can only observe. The observations are made by telescopes which are incredibly expensive to both build and run, and the amount of data coming from these facilities is huge.
 
My training in Astronomy has made me a specialist in selecting appropriate statistical techniques to extract as much scientific information from observations as possible, be it in Astronomy or elsewhere.

### [DISCUS: The Data Intensive Science Centre at the University of Sussex](http://www.sussex.ac.uk/discus/)
As Research Fellow for DISCUS, I provide statistical and data analysis consultancy for both academia and industry. For consultancy enquiries,
 please [contact me](../contact/index.html) or [DISCUS](http://www.sussex.ac.uk/discus/contact).


Here are a few examples of inter-disciplinary projects I have worked on:

- - - 


### [Variation in random capillary blood glucose and HbA1c as predictors of cystic fibrosis related diabetes (CFRD)](http://www.cysticfibrosisjournal.com/article/S1569-1993(15)30348-9/pdf)
The Leeds CF unit implemented electronic care records (ECR) in 2007
coded for all aspect of CF disease. Subjectively, patterns of disease progression have
become evident, including increased variation in blood glucose measures prior to
diagnosis of CFRD. The aim of this study was to determine objectively whether

* random capillary blood glucose (RCBG) variation and HbA1c differs between
those who develop CFRD in the 3 years preceding diagnosis and controls,
* univariate and/or multivariate analysis using longitudinal datasets can predict
onset of CFRD

We demonstrated that epidemiological datasets extracted from ECR reveal early potential
for prediction of CFRD from multivariate datasets. Multivariate Gaussian mixtures
modelling of HbA1c and RCBG show potential as an indicator of CFRD that can
be monitored over time, which might be realised with larger sample sizes.

- - -

### [Gaussian process classification of Alzheimer's disease and mild cognitive impairment from resting-state fMRI](https://doi.org/10.1016/j.neuroimage.2015.02.037)

![](../Figures/GP_neuro.jpg){:style="float: right;margin-right: 7px;margin-top: 7px;"}


Multivariate pattern analysis and statistical machine learning techniques are attracting increasing interest from the 
neuroimaging community. Researchers and clinicians are also increasingly interested in the study of 
functional-connectivity patterns of brains at rest and how these relations might change in conditions like Alzheimer's 
disease or clinical depression.
 
In this study we investigate the efficacy of a specific multivariate statistical machine
 learning technique to perform patient stratification from functional-connectivity patterns of brains at rest. Whilst 
 the majority of previous approaches to this problem have employed support vector machines (SVMs) we investigate the 
 performance of Bayesian Gaussian process logistic regression (GP-LR) models with linear and non-linear 
 covariance functions. GP-LR models can be interpreted as a Bayesian probabilistic analogue to kernel SVM classifiers.
  However, GP-LR methods confer a number of benefits over kernel SVMs. Whilst SVMs only return a binary class 
  label prediction, GP-LR, being a probabilistic model, provides a principled estimate of the probability of 
  class membership. Class probability estimates are a measure of the confidence the model has in its predictions, 
  such a confidence score may be extremely useful in the clinical setting. Additionally, if miss-classification costs 
  are not symmetric, thresholds can be set to achieve either strong specificity or sensitivity scores. Since GP-LR models are Bayesian, computationally expensive cross-validation hyper-parameter grid-search methods can 
   be avoided.
    
We apply these methods to a sample of 77 subjects; 27 with a diagnosis of probable AD, 50 with a diagnosis of 
a-MCI and a control sample of 39. All subjects underwent a MRI examination at 3 T to obtain a 7 minute and 20 second 
resting state scan. Our results support the hypothesis that GP-LR models can be effective at performing patient 
stratification: the implemented model achieves 75% accuracy disambiguating healthy subjects from subjects with amnesic 
mild cognitive impairment and 97% accuracy disambiguating amnesic mild cognitive impairment subjects from those with 
Alzheimer's disease, accuracies are estimated using a held-out test set. Both results are significant at the 1% level.