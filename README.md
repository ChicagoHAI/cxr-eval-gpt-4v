# Codebase for GPT-4V Evaluation in Radiology Report Generation Task

This repository details the experiments conducted as discussed in [our paper](https://www.arxiv.org/abs/2407.12176) on the evaluation of radiology report generation using GPT-4V(ison) model series (GPT-4-vision-preview (11/15/2023) and GPT-4o).

## Experiments Overview

- **Experiment 1:** Systematic Evaluation of Direct Report Generation
- **Experiment 2:** Medical Image Reasoning Capability
- **Experiment 3:** Report Synthesis Given Medical Conditions

We utilize the [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) dataset [1], the [IU X-RAY](https://openi.nlm.nih.gov/faq) dataset [2] and the [CheXpert Plus](https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1) dataset [3], evaluated using Microsoft's Azure OpenAI service. Access to these APIs and datasets must be secured prior to conducting the experiments.

## Setup Instructions

### Environment Preparation

To set up the project environment, use the following command:

```
conda env create --name=<env_name> -f cxr-eval.yml
```

### Model Preparation

Our codebase integrates CheXbert [4] for report labeling and RadGraph [5] for evaluation. Download the required model checkpoints from the [repository here](https://github.com/rajpurkarlab/CXR-Report-Metric) and place them in the `./models/` directory.

### Data Preparation

Extract the 'indication', 'findings', and 'impression' sections from the original report and ensure the exclusion of studies with empty impression or indication sections. Prepare the following CSV files in your input directory:

- `ind_list_sample.csv` with columns 'study_id', 'path_image'
- `gt_indc_sample.csv` with columns 'study_id', 'report'
- `gt_findings_sample.csv` with columns 'study_id', 'report'
- `gt_imp_sample.csv` with columns 'study_id', 'report'

## Running Experiments

### Experiment 1: Direct Report Generation

| Prompt                     | Prompt Type | Description                                              |
|----------------------------|-------------|----------------------------------------------------------|
| Prompt 1.1 Basic generation| 1_1         | Direct report generation based on chest X-ray images     |
| Prompt 1.2 +Indication     | 1_2         | Enhancement by including the indication section          |
| Prompt 1.3 +Instruction    | 1_3         | Enhancement by providing medical condition instructions |
| Prompt 1.4 Chain-of-Thought| 1_4         | Medical condition labeling followed by report synthesis  |
| Prompt 1.5 Few-shot        | 1_5         | Few-shot learning with a few examples                    |

Execute the following commands for each dataset:

```
python gen_repo.py \
  --api <Azure api key> \
  --endpoint <Azure endpoint> \
  --type <prompt type> \
  --p <prompt directory> \
  --m <image directory> \
  --i <input directory> \
  --o <output directory>
```

Following data files will be stored in your output directory:
- `err_list_{prompt_type}.csv` with columns 'study_id', 'error'
- `gen_findings_sample_{prompt_type}.csv` with columns 'study_id', 'report'
- `gen_imp_sample_{prompt_type}.csv` with columns 'study_id', 'report'

### Experiment 2&3: Task Decomposition
| Prompt                             | Prompt Type | Description                                                          |
|------------------------------------|-------------|----------------------------------------------------------------------|
| Prompt 2.1: 2-Class Reasoning      | 2_1         | 2-class Medical condition labeling directly from chest X-ray images |
| Prompt 3.1: Report Synthesis       | 3_1         | Report generation using provided positive and negative conditions   |

Prepare an additional CSV `gt_labels_sample.csv` for Experiment 3 with columns: 'study_id', 14 CXR conditions and put it in your input directory. Follow the [CheXbert](https://github.com/stanfordmlgroup/CheXbert) instructions for labeling. Noted thta Experiment 1_4 will also need to run the following command.

```
python gen_label.py \
  --api <Azure api key> \
  --endpoint <Azure endpoint> \
  --type <prompt type> \
  --p <prompt directory> \
  --m <image directory> \
  --i <input directory> \
  --o <output directory>
```

Following data files will be stored in your output directory:
- `err_list_{prompt_type}.csv` with columns 'study_id', 'error'
- `gen_labels_sample_{prompt_type}.csv` with columns 'study_id', 14 CXR conditions (only for Experiment 2)
- `gen_imp_sample_{prompt_type}.csv` with columns 'study_id', 'report' (only for Experiment 3)
- `gen_findings_sample_{prompt_type}.csv` with columns 'study_id', 'report' (only for Experiment 3)

## Evaluation

General evaluation can be conducted using:

```
python evaluate.py --gt_path <path> --gen_path <path> --out_path <output CSV>
```

Refer to `label_processing.ipynb` for F1 calculation and `hypothesis_test.ipynb` for hypothesis testing in Experiment 2.

## References
[1] Johnson, Alistair, Pollard, Tom, Mark, Roger, Berkowitz, Seth, and Steven Horng. "MIMIC-CXR Database" (version 2.0.0). PhysioNet (2019). https://doi.org/10.13026/C2JT1Q.

[2] Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Ro-driguez, Sameer Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 23(2):304–310, 2016.

[3] Pierre Chambon, Jean-Benoit Delbrouck, Thomas Sounack, Shih-Cheng Huang, Zhihong Chen, Maya Varma, Steven QH Truong, Chu The Chuong, and Curtis P. Langlotz. "Chexpert plus: Augmenting a large chest x-ray dataset with text radiology reports, patient demographics and additional image formats." arXiv preprint arXiv:2405.19538 (2024).

[4] Smit, Akshay, Saahil Jain, Pranav Rajpurkar, Anuj Pareek, Andrew Y. Ng, and Matthew P. Lungren. "CheXbert: combining automatic labelers and expert annotations for accurate radiology report labeling using BERT." arXiv preprint arXiv:2004.09167 (2020).

[5] Jain, Saahil, Ashwin Agrawal, Adriel Saporta, Steven QH Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon et al. "Radgraph: Extracting clinical entities and relations from radiology reports." arXiv preprint arXiv:2106.14463 (2021).
