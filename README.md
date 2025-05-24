# StellarCodeBio: Architecting Next-Generation AI for Scientific Discovery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
**[阅读中文版 (Read in Chinese)](README_zh.md)** 
<!-- Future consideration: A dedicated logo/banner for StellarCodeBio -->

Welcome to the official repository of the **StellarCodeBio** initiative!

StellarCodeBio is a deeply personal and ambitious long-term exploration into the future of Artificial Intelligence. It moves beyond current data-intensive paradigms, aspiring to architect "Research Models" – AI systems capable of **efficient learning from limited data, robust causal reasoning, seamless integration of symbolic knowledge, and drawing profound inspiration from biological neural computation**, particularly Spiking Neural Networks (SNNs). Our ultimate vision is to forge AI that can not only assist but potentially drive fundamental scientific discovery and contribute to solving complex global challenges.

This repository marks the public commencement of the StellarCodeBio journey, beginning with a foundational project in computational drug discovery.

**Initiator & Lead Researcher:** ShengYao Liang
**Affiliation:** Independent Researcher (StellarCodeBio Initiative)
**Email:** pikeshuaiwe@gmail.com
**ORCID:** [https://orcid.org/0009-0002-3713-8700](https://orcid.org/0009-0002-3713-8700)

---

## The StellarCodeBio Vision: Towards Cognitive AI Engines – Learning the Essence

The recent successes of Large Language Models have been transformative. Yet, the path to truly cognitive AI – systems that understand the world with depth and adapt with an efficiency rivaling human intuition – requires a paradigm shift. StellarCodeBio is born from a conviction that such a shift is possible through the synergistic fusion of several frontier AI domains.

Our guiding principles and core research pillars include:

*   🧠 **Biologically Inspired Architectures (The "Engine"):** At the heart of StellarCodeBio lies the exploration of Spiking Neural Networks (SNNs). We believe their event-driven, temporally dynamic nature holds immense potential for energy-efficient and powerful computation, more closely mirroring the brain's mechanisms. Our initial work on a Spiking Language Model (SLM) serves as a testament to this direction.
*   💡 **Data-Efficient Learning (The "Fuel Efficiency"):** We aim to break free from the "big data" bottleneck. Inspired by human learning, we leverage **Meta-Learning (Learning to Learn)** and few-shot learning techniques to empower our models to generalize broadly and adapt rapidly to new tasks, rules, or environments with minimal data.
*   🧩 **Neuro-Symbolic AI (The "Logic & Knowledge Core"):** True intelligence requires more than pattern recognition. We are committed to integrating symbolic reasoning, knowledge graphs, and logical rules directly into our neural architectures. This fusion aims to imbue our AI with explainability, robustness, and the ability to manipulate abstract concepts – moving closer to understanding "essence" rather than just surface statistics.
*   🔗 **Causal Inference (The "Understanding Why"):** To build AI that can truly interact with and shape the world, it must understand cause and effect. StellarCodeBio will explore methods to instill causal reasoning capabilities, enabling models to make robust predictions under intervention and generate more meaningful explanations.
*   🚀 **The "Research Model" (The Ultimate Goal):** The convergence of these pillars is envisioned to culminate in a "Research Model" – a highly autonomous AI system capable of posing hypotheses, designing experiments (virtual or otherwise), interpreting results, and collaborating with human researchers to accelerate the pace of scientific discovery across various domains, from drug development to fundamental physics. This is a long-term, "moonshot" aspiration.

*(A Personal Note on Motivation: This ambitious journey is also fueled by a profound personal drive to explore the very nature of intelligence and to push the boundaries of what's considered possible. It's an endeavor born from deep introspection, a fascination with complex systems (be they biological, societal, or computational), and an unwavering belief in the power of focused, unconventional thinking to make a tangible impact. While the path is undoubtedly arduous, the pursuit itself is a source of immense intellectual vitality.)*

This pIC50 prediction project, detailed below, represents an initial, practical step in applying machine learning principles while laying the groundwork for the more advanced SNN-based and cognitive architectures envisioned under StellarCodeBio.

---

## Project I: An Efficient Machine Learning-Based Prediction Model for JAK2 Inhibitor pIC50

This first public project under the StellarCodeBio initiative provides a validated machine learning model for predicting the pIC50 values of potential Janus Kinase 2 (JAK2) inhibitors, a crucial task in early-stage drug discovery.

**Original Manuscript Context:**
The research for this pIC50 model was first submitted to the *Journal of Chemical Information and Modeling* (Manuscript ID: ci-2025-00977b). Following a transfer recommendation recognizing its scientific merit, it was then submitted to *ACS Omega* (Manuscript ID: ao-2025-043259). 
A preprint detailing this work, titled "An Efficient Machine Learning-Based Prediction Model for JAK2 Inhibitor pIC50," is available on ChemRxiv:
**DOI: [10.26434/chemrxiv-2025-3v3gw-v3](https://doi.org/10.26434/chemrxiv-2025-3v3gw-v3)**
While ACS Omega ultimately determined the work's primary focus was more aligned with machine learning applications than their specific chemistry readership (leading to a decision not to send for external peer-review), they acknowledged the quality of the work. This open-source release is motivated by the desire to share these findings and the developed tool with the broader scientific community, particularly those who may find it useful and may not have access to similar proprietary tools. We believe in the power of open science to accelerate research for all.

### Abstract (for pIC50 model)
**Background:** Janus Kinase 2 (JAK2) is a key kinase in cellular signal transduction. Its abnormal activation is closely related to various myeloproliferative neoplasms and inflammatory diseases. Developing selective JAK2 inhibitors is an important direction in drug discovery. Accurate prediction of compound inhibitory activity ($pIC_{50}$) against JAK2 is crucial for accelerating the discovery and optimization of lead compounds.
**Objective:** This study aims to utilize public resources from the ChEMBL database, combined with machine learning methods, to build a computational model capable of efficiently and accurately predicting the $pIC_{50}$ values of JAK2 inhibitors.
**Methods:** We collected compounds targeting human JAK2 (ChEMBL ID: CHEMBL2971) and their $IC_{50}$ (nM) activity data from the ChEMBL database. After data cleaning and standardization (converting $IC_{50}$ to $pIC_{50}$, averaging duplicates), a dataset of 5546 compounds was obtained. RDKit (v2022.9.5) was used to calculate Morgan fingerprints, MACCS Keys, and 13 physicochemical/topological descriptors. Feature selection based on LightGBM importance resulted in 345 features for the final XGBoost model (v3.0.1). Hyperparameters were tuned using 5-fold GridSearchCV, and early stopping was employed.
**Results (Re-run on May 24, 2025):** The final XGBoost model demonstrated good predictive performance on an independent test set, achieving:
    *   Coefficient of Determination ($R^2$): **0.6828**
    *   Root Mean Square Error (RMSE): **0.6334**
    *   Mean Absolute Error (MAE): **0.4761**
    (Training set $R^2$ was 0.9367, indicating good fit while controlling overfitting.)
**Conclusion:** This study successfully constructed an XGBoost-based prediction model for JAK2 inhibitor $pIC_{50}$. Utilizing readily accessible molecular descriptors, the model demonstrated good prediction accuracy and robustness on an external test set. This model is provided as an efficient virtual screening tool to aid the early discovery and optimization process of JAK2 inhibitors.

### Features (of this pIC50 model project)
*   Transparent and reproducible pipeline for JAK2 pIC50 prediction.
*   Utilizes publicly available ChEMBL data.
*   Employs standard cheminformatics tools (RDKit) and machine learning libraries (XGBoost, Scikit-learn, LightGBM).
*   Includes Python scripts for:
    1.  `data_processing.py`: Downloading, cleaning, and featurizing ChEMBL data from ChEMBL (Target: CHEMBL2971). Generates `final_data.csv`.
    2.  `optimize_model.py`: Calculates feature importance using LightGBM from `final_data.csv`. Generates `feature_importance.csv`.
    3.  `modeltraining.py`: Trains, tunes (GridSearchCV), and evaluates the final XGBoost model using `final_data.csv` and `feature_importance.csv`. Saves the trained model artifacts and a prediction plot.
*   Provides pre-generated `final_data.csv` (5546 compounds, SMILES, pIC50, and 2200+ initial features before selection) and `feature_importance.csv` (ranked features) for convenience.
*   Detailed instructions for reproducing the results from scratch or using pre-generated files.

### Directory Structure

StellarCodeBio/
├── modeltraining.py # Main script for XGBoost model training and evaluation
├── data_processing.py # Script for data download and preprocessing
├── optimize_model.py # Script for feature importance calculation
├── final_data.csv # Preprocessed data with all initial features (~5546 compounds)
├── feature_importance.csv # Feature importance scores from LightGBM
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
├── README.md # This file

*(The `.joblib` model file will be generated by `modeltraining.py` in the main directory).*

### Requirements
*   Python 3.9+ (Developed and tested with Python 3.11)
*   Key dependencies (see `requirements.txt` for full list and specific versions used in development):
    ```
    chembl_webresource_client
    pandas
    numpy
    rdkit-pypi 
    tqdm
    scikit-learn
    lightgbm
    xgboost==3.0.1 
    joblib
    matplotlib
    seaborn
    scipy
    ```

### Installation
1.  Clone this repository:
    ```bash
    git clone https://github.com/ShengyaoLiang/StellarCodeBio.git
    cd StellarCodeBio
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv venv 
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note on RDKit:* If `pip install rdkit-pypi` fails, please refer to the official RDKit installation guide for alternative methods (e.g., using Conda).

### Usage - Step-by-Step Reproduction
To fully reproduce the results reported in this project, run the scripts in the following order:

1.  **Install Dependencies:**
    Ensure all packages from `requirements.txt` are installed in your environment.
    ```bash
    pip install -r requirements.txt
    ```

2.  **(Option A - Reproduce from scratch) Data Processing and Feature Calculation:**
    This script downloads the latest data for human JAK2 (CHEMBL ID: CHEMBL2971) from the ChEMBL database, performs cleaning (retaining exact IC50 values, removing duplicates/missing data), converts IC50 to pIC50, and calculates molecular features (Morgan fingerprints, MACCS Keys, and 13 physicochemical/topological descriptors).
    ```bash
    python data_processing.py
    ```
    **Output:** `final_data.csv` (containing SMILES, pIC50, and over 2200 initial feature columns for approximately 5546 compounds).

3.  **(Option A - Reproduce from scratch) Feature Importance Calculation:**
    This script reads `final_data.csv`, trains a LightGBM model using all calculated features, and assesses the importance of each feature.
    ```bash
    python optimize_model.py
    ```
    **Input:** `final_data.csv`
    **Output:** `feature_importance.csv` (two columns: 'feature' and 'importance', sorted by importance).

4.  **Model Training and Evaluation:**
    This script reads `final_data.csv` and `feature_importance.csv`, selects the top 350 most important features (of which ~345 are typically found present in `final_data.csv`), splits the data, standardizes features, performs GridSearchCV for XGBoost hyperparameter tuning (this step can be time-consuming), trains the final model with early stopping, evaluates its performance, and saves the model artifacts and a prediction plot.
    ```bash
    python modeltraining.py 
    ```
    **Inputs:** `final_data.csv`, `feature_importance.csv`
    **Outputs:**
    *   `jak2_pci50_model_tuned_top350_xgb3_v5.joblib` (Dictionary containing the trained XGBoost model, the StandardScaler object, and the list of 345 feature names used).
    *   Performance metrics (R², RMSE, MAE) printed to the console.

    **(Option B - Using Pre-generated Files):** If you have downloaded the `final_data.csv` and `feature_importance.csv` provided in this repository, you can skip steps 2 and 3 and directly run `python modeltraining.py`.

### Expected Results (from re-run on May 24, 2025)
*   **GridSearchCV Best R² (on validation folds):** ~0.6516
*   **Final Model Test Set Performance:**
    *   R-squared (R²): ~0.6828
    *   RMSE: ~0.6334
    *   MAE: ~0.4761
*   **Number of Trees in Final Model (determined by Early Stopping):** ~342

*(Please note: Slight variations in results can occur due to differences in computational environments or the stochastic nature of some algorithms, even with fixed random seeds. The hyperparameters identified by GridSearchCV may also vary slightly if multiple combinations yield similar cross-validation scores.)*

### Model Details (for pIC50 predictor)
*   **Target:** Janus Kinase 2 (JAK2) - ChEMBL ID: CHEMBL2971
*   **Activity Data:** pIC50 values.
*   **Features:** 345 selected features comprising Morgan fingerprints, MACCS Keys, and RDKit physicochemical/topological descriptors.
*   **Algorithm:** XGBoost (Version 3.0.1 used in re-runs)
*   **Hyperparameters (Best found in re-run on May 24, 2025):** `{'colsample_bytree': 0.75, 'gamma': 0.25, 'learning_rate': 0.03, 'max_depth': 13, 'reg_alpha': 0.25, 'reg_lambda': 0.55, 'subsample': 0.8}` (The script will perform GridSearchCV to find optimal parameters for each specific run).

---

## Future Directions for StellarCodeBio
This pIC50 prediction model is just the beginning. Future explorations under the StellarCodeBio initiative may include (but are not limited to):
*   **Developing Advanced Spiking Neural Network (SNN) Architectures:** For tasks requiring temporal data processing and energy efficiency, starting with foundational Spiking Language Models (SLMs).
*   **Neuro-Symbolic Integration with SNNs:** Combining the strengths of SNNs with symbolic knowledge and reasoning.
*   **Meta-Learning for Data-Efficient SNNs:** Enabling SNNs to rapidly adapt to new tasks or data with minimal examples.
*   **Causal Inference in Biological Systems:** Building models that aim to understand underlying causal mechanisms.
*   **Towards a "Research Model":** Incrementally building towards more autonomous AI systems capable of assisting or even driving scientific inquiry.

We believe in open and collaborative research. Stay tuned for updates!

## Contributing to StellarCodeBio
We are excited about the potential of StellarCodeBio and welcome contributions, collaborations, discussions, and feedback. Please feel free to:
*   Open an issue on the [GitHub Issues page](https://github.com/ShengyaoLiang/StellarCodeBio/issues) for bug reports, feature requests, or questions.
*   Fork the repository and submit pull requests with your improvements.
*   Reach out to ShengYao Liang (pikeshuaiwe@gmail.com) for collaboration inquiries.

## License
The contents of this repository, under the StellarCodeBio initiative, are licensed under the **MIT License**. See the `LICENSE` file for details.

## Acknowledgements
*   The ChEMBL database for providing essential bioactivity data.
*   The RDKit community for their invaluable cheminformatics toolkit.
*   The developers and communities behind Scikit-learn, XGBoost, LightGBM, Pandas, NumPy, Matplotlib, Seaborn, and Joblib.
*   Any individuals who provided specific help or inspiration.

---