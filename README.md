# Early Classification of Space Objects based on Astrometric Time Series Data

This repository contains code for classifying Resident Space Objects (RSOs) based on their Vector Covariance Message (VCM) time series data using machine learning (ML) techniques. 

## Project Overview

Accurate and timely classification of space objects is crucial for Space Situational Awareness (SSA). This project explores the use of supervised ML algorithms for early time series classification of RSOs, addressing challenges like data imbalance and irregular sampling.

### Key Features

- Employs advanced time series classification algorithms such as InceptionTime and TSiT for regular and irregular time series, respectively.
- Introduces time regularization techniques and investigates the impact of synthetic data on classification accuracy.
- Explores various input combinations and dataset variations to optimize performance. 
- Evaluates the trade-off between classification accuracy and timeliness for early RSO identification.

### Dataset

This project utilizes VCM data, which comprise 22,303 RSOs over a period of six months (9/1/2022-2/28/2023). VCM data consist of RSO ephemerides from a high-precision special perturbations orbit propagator and estimator using tracking observations, VCMs are issued by the US Space Force (USSF) Space Command (USSPACECOM) and were provided through an Orbital Data Request (ODR) the authors submitted to the 18th Space Defense Squadron (18th SDS). The VCM dataset are divided into three classes, Payload (P), Rocket Body (R), and Debris (D), for a total of 21,966 RSOs. The remaining RSOs instead are categorized as Unknown and have been excluded from the study. It must be noted that the VCMs are reported at non-uniform time intervals, and might not be available for certain RSOs.

### Algorithms Tested

- **Traditional ML:** Random Forest, Logistic Regression, K-Nearest Neighbor, Support Vector Machine, Naive Bayes
- **Deep Learning:** Multi-Layer Perceptron
- **Time Series Specific:** InceptionTime, TSiT

## Code Example (coming soon)

The provided Python code "main_TSiT" demonstrates:
* Loading an irregular time series pre-trained model (TSiT).
* Performing inference on a test dataset. 

<!--
The pre-trained model contained in the folder "pre_trained_model_Nst_10" is based on:
* batch size of 64 for the training data and 128 for the testing data
* NN's depth = 3
* convolutional dropout = 0
* number of steps/measurements $n_{ST} =$ 10 
* number of variables/features $n_{V} =$ 14 (epoch, position vector, velocity vector, ballistic coefficient, solar radiation pressure coefficient, semi-major axis, eccentricity, inclination, argument of perigee, right ascension of the ascending node)
* augmentation factor $n_{AUG} =$ (0, 5, 0) for (P, R, D)
* maximum time interval cut-off $\Delta t_{cut} =$ of (2, 2, Inf) for (P, R, D)
* Cross-Entropy Loss Flat as loss function

Resulting in F1 scores (0-1) of 0.80 for Payload, 0.80 for Rocket Body, and 0.93 for Debris classes.

The pre-trained model contained in the folder "pre_trained_model_Nst_15" differs from the previous one for:
* convolutional dropout = 0.01
* number of steps/measurements $n_{ST} =$ 15 

Resulting in F1 scores (0-1) of 0.84 for Payload, 0.82 for Rocket Body, and 0.94 for Debris classes.
-->

### Dependencies

- tsai ([https://github.com/timeseriesAI/tsai](https://github.com/timeseriesAI/tsai))
- os (part of the Python standard library, [https://docs.python.org/3/library/os.html](https://docs.python.org/3/library/os.html))
- pickle (part of the Python standard library, [https://docs.python.org/3/library/pickle.html](https://docs.python.org/3/library/pickle.html))

## Citing

If you find this project research useful, please cite our work:

```
@inproceedings{MLforVCM_AMOS24,
author = {Lavezzi, Giovanni and Mun Siew, Peng and Wu, Di and Folcik, Zachary and Rodriguez-Fernandez, Victor and Price, Jeffrey and Linares, Richard},
year = {2024},
month = {09},
pages = {},
booktitle = {25th Advanced Maui Optical and Space Surveillance Technologies},
publisher = {Maui Economic Development Board},
address = {Maui, HI},
title = {Early Classification of Space Objects based on Astrometric Time Series Data}
}
```

## Acknowledgments

The authors thank Deshaun Hutchinson and Amanda Harman at the US Space Force 18 SDS for their efforts in collecting and providing the VCMs used in this study. 

Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

