# Multi-View Time Series Feature Fusion and Case-Based Reasoning for Explainable Automated Robotic Surgical Skill Assessment

The implementation of Multi-View Time Series Feature Fusion and Case-Based Reasoning for Explainable Automated Robotic Surgical Skill Assessment

## Data

We use two public datasets (six surgical tasks): JIGSAWS (suturing, knot tying, needle passing) and ROSMA (pea on a peg, post, and sleeve, wire chaser) for our experiments. Please download these two datasets using the links provided in these papers.

For JIGSAWS, we apply self-proclaimed annotation.
For ROSMA, we provide high-quality annotation by two senior surgeons; the annotation will be released upon the acceptance of the paper.

We organize the pre-processed dataset into pickle format and then save them into the 'datasets/jigsaws-kinematic' folder and 'datasets/rosma-kinematic' folder. 

## Requirments

All Python packages needed are listed in the requirments.txt file and can be installed as follows:

```
conda config --append channels conda-forge
conda create --name <your env name> --file requirements.txt
```

## Expriments

To run each experiment, you can run the following commands.

```
python CBR_module/jigsaws-su-expCBR-Feature-Fusion.py
python CBR_module/jigsaws-kt-expCBR-Feature-Fusion.py
python CBR_module/jigsaws-np-expCBR-Feature-Fusion.py

python CBR_module/rosma-pp-expCBR-Feature-Fusion.py
python CBR_module/rosma-ps-expCBR-Feature-Fusion.py
python CBR_module/rosma-wc-expCBR-Feature-Fusion.py
```

The results of all experiments are saved in the 'results' folder.

Example output "CBR_su_feature_fusion_scaled_new_four_views_LOSO5_results.csv" is in the 'results/jigsaws' folder.

## Overview of the framework


![Framework](images/framework.pdf)