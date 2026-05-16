# KGNFR: Injecting Structured Knowledge Graphs to Enhance Multi-Label Non-Functional Requirements Classification

This repository contains the code for the paper **"KGNFR: Injecting Structured Knowledge Graphs to Enhance Multi-Label Non-Functional Requirement Classification"** – a knowledge graph construction and fusion framework for multi-label non-functional requirements (NFR) classification.

## Overview

The project introduces KGNFR, a novel framework that enhances non-functional requirements classification by leveraging structured knowledge graphs. Our approach jointly constructs and fuses domain-specific knowledge, enabling improved representation and aggregation of multi-label NFR features. This repository includes all the necessary scripts, models, and utilities to reproduce the experiments presented in the paper.

## Requirements

The code has been developed and tested using the following packages:

- **torch** ~= 2.2.1+cu121
- **tqdm** ~= 4.66.2
- **six** ~= 1.12.0
- **boto3** ~= 1.9.227
- **requests** ~= 2.31.0
- **botocore** ~= 1.12.227
- **sentencepiece** ~= 0.2.0
- **scikit-learn** ~= 1.3.2
- **torchsummary** ~= 1.5.1

## Installation

It is recommended to set up a virtual environment before installing the dependencies. You can install the required packages using pip. For example:

```bash
pip install torch==2.2.1+cu121 tqdm==4.66.2 six==1.12.0 boto3==1.9.227 requests==2.31.0 botocore==1.12.227 sentencepiece==0.2.0 scikit-learn==1.3.2 torchsummary==1.5.1
