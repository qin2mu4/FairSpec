
# Implementation of SIGIR26 paper "FairSpec: Expert Specialization for Fair LLM-based Recommendation"

## Abstract
The integration of Large Language Models (LLMs) into recommender systems has introduced a powerful paradigm for personalized recommedation. However, due to biases stemming from sensitive user attributes (e.g., gender, age), LLMs may generate discriminatory and unfair recommendations, even for users with similar preferences. Existing methods to improve LLM-based recommendation fairness, like prompt engineering or data-processing, are often unstable and fail to address unfairness rooted in pre-training data and model parameters. While fine-tuning approaches show promise, improving fairness typically comes at the cost of degraded recommendation performance, as the model’s parameters conflates user preferences with sensitive attributes. To address this, we propose **FairSpec**, a novel lightweight fine-tuning framework designed to enhance fairness while preserving recommendation quality. FairSpec has a Parameter-Efficient Mixture-of-Experts architecture (PE-MoE) with a fairness-aware router. The experts are specialized into utility experts and fairness experts, and the latter are adversarially trained to suppress sensitive information. Furthermore, FairSpec introduces Fairness-Aware Direct Preference Optimization (FA-DPO), which employs a composite reward function to jointly optimize recommendation performance and fairness. Results on three public benchmarks with four LLM backbones demonstrate that FairSpec achieves the best overall performance-fairness balance, attaining the top average rank across all metrics. Furthermore, visualizations of routing weights illustrate the strategic behavior and effectiveness of the proposed fairness-aware routing mechanism.

## Method
<img width="1071" height="615" alt="image" src="https://github.com/user-attachments/assets/4b7cc651-0f49-40cd-8438-b5cac4223aa1" />

## How to Run
1. Prepare data: Extract data.zip into ./data
2. Create conda environment:
    ```bash
    conda env create -f environment.yml
    ```
3. Run bash: 
    ```bash
    bash run_book.sh
    bash run_ml1m.sh
    bash run_post.sh
    ```

## Acknowledgments
Our code is based on the following repositories: https://github.com/Ablustrund/LoRAMoE

## Cite
```
@inproceedings{zheng2026FairSpec,
  title={FairSpec: Expert Specialization for Fair LLM-based Recommendation},
  author={Yuchen Zheng, Xuan Pan, Jing Wang, Chuanchang Zhang, Xi Lin, Chunyao Song, Xiangrui Cai, and Xiaojie Yuan.},
  booktitle={Proceedings of the 49th international ACM SIGIR conference on research and development in information retrieval},
  pages={},
  year={2026}
}
```
