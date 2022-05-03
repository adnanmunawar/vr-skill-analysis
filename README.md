# Surgical Skill Analysis with Virtual Reality
### [Course Wiki Page](https://ciis.lcsr.jhu.edu/doku.php?id=courses:456:2022:projects:456-2022-08:project-08) | [Volumetric Drilling GitHub](https://github.com/adnanmunawar/volumetric_drilling) | [AMBF](https://github.com/WPI-AIM/ambf)
This repo provides code access and documentation of our project to develop an objective, technical assessment for Otolaryngology-Head and Neck Surgery (OHNS).

## Overview

Current evidence has shown that higher-volume surgeons with superior technical skills yield better patient outcomes. Current methods for the evaluation of surgical skill​ include:

- Objective Structured Assessments of Technical Skills (OSATS)​
- Society for Improving Medical Professional Learning platform

However, some issues with these evaluations include bias from the evaluator, as well as poor intra-evaluator reliability.

Mastoidectomy Procedures​ are highly precise surgeries that require the drilling of temporal bone. Temporal Bone Simulators​ include stereoscopic vision and haptic feedback​, but automated metrics have limited results​. An objective, technical assessment is needed for Otolaryngology-Head and Neck Surgery.

## 1. Team Information
**Team Members:** Aditya Khandeshi, Liza Naydanova, Alexandra Szewc
**Mentors:** Max Li, Adnan Munawar, Dr. Danielle Trakimas, Nimesh Nagururu, Dr. Creighton, Dr. Unberath, Dr. Taylor

## 2. Repository Contents

| Item | Path |
|----------------------------------|--------------------------------------------------------|
| Simulator Setup Documentation    | `~/AMBF Simulator and Phantom Omni Setup.pdf`          |
| Feature Extraction Dev Scripts   | `~/feature_validation/feature_engineering_development` |
| Feature Extraction Data          | `~/feature_validation/data/`                           |
| Feature Validation Scripts       | `~/feature_validation/`                                |
| Feature Validation Data          | `~/feature_validation/[FEATURE NAME]/`                 |
| Feature Extraction Script        | `~/extract_features.py`                                |
| Feature Documentation            | `~/Documentation of Extracted Features.pdf`            |
| Model Implementations            | `~/models/`                                            |

Please note that feature extraction code documentation is included in the form of function comments for each extracted feature in the `extract_features.py` file.