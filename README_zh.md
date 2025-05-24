# StellarCodeBio: 构建下一代科学发现AI引擎

[![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
**[English Version (Read English Version)](README.md)**
<!-- 未来考虑在这里放置一个代表 StellarCodeBio 的 Logo 或 Banner -->
**[English Version (阅读英文版)](README.md)**

欢迎来到 **StellarCodeBio** 计划的官方代码仓库！

StellarCodeBio 是一项源于个人热忱且着眼长远的雄心探索，致力于探究人工智能的未来。它旨在超越当前数据密集型的范式，期望构建出“研究模型” (Research Models)——这种AI系统应具备**从有限数据中高效学习、进行稳健的因果推理、无缝整合符号知识，并从生物神经计算（尤其是脉冲神经网络 SNNs）中汲取深刻灵感**的能力。我们的最终愿景是锻造出不仅能辅助、更能驱动基础科学发现并为解决复杂全球性挑战贡献力量的人工智能。

本代码仓库标志着 StellarCodeBio 旅程的公开启动，始于计算药物发现领域的一个基础性项目。

**发起人与主导研究员：** 梁晟耀 (Shengyao Liang)
**单位：** 独立研究员 (StellarCodeBio 计划)
**邮箱：** pikeshuaiwe@gmail.com
**ORCID:** [https://orcid.org/0009-0002-3713-8700](https://orcid.org/0009-0002-3713-8700)

---

## StellarCodeBio 愿景：迈向认知AI引擎 – 学习本质

近年来大型语言模型的成功带来了深刻的变革。然而，通往真正认知AI——能够深刻理解世界并以媲美人类直觉的效率进行适应的系统——的道路，需要一次范式转换。StellarCodeBio 正是诞生于这样一种信念：通过多个前沿AI领域的协同融合，这种转变是可能实现的。

我们的指导原则和核心研究支柱包括：

*   🧠 **生物启发架构 (引擎核心):** StellarCodeBio 的核心在于对脉冲神经网络 (SNNs) 的探索。我们相信其事件驱动、时序动态的特性在实现高能效、强计算方面拥有巨大潜力，也更接近大脑的运作机制。我们早期在脉冲语言模型 (SLM) 上的工作即是此方向的初步证明。
*   💡 **数据高效学习 (燃料效率):** 我们致力于摆脱“大数据”的瓶颈。受人类学习的启发，我们运用**元学习 (学会学习)** 和小样本学习技术，赋能我们的模型以少量数据即可广泛泛化并快速适应新的任务、规则或环境。
*   🧩 **神经符号AI (逻辑与知识核心):** 真正的智能不仅仅是模式识别。我们致力于将符号推理、知识图谱和逻辑规则直接整合到我们的神经架构中。这种融合旨在赋予AI可解释性、鲁棒性以及操作抽象概念的能力——从而更接近于理解“本质”，而非仅仅是表面统计规律。
*   🔗 **因果推断 (理解“为什么”):** 要构建能够真正与世界互动并塑造世界的AI，它必须理解因果关系。StellarCodeBio 将探索为模型注入因果推理能力的方法，使其能够在干预下做出稳健的预测，并生成更有意义的解释。
*   🚀 **“研究模型” (终极目标):** 这些支柱的融合，最终将构想并实现一个“研究模型”——一个高度自主的AI系统，能够提出假设、设计实验（虚拟或其他形式）、解读结果，并与人类研究员协作，以加速从药物开发到基础物理学等各个领域的科学发现步伐。这是一个长期的、“登月级”的宏愿。

*(个人动机随笔：这段雄心勃勃的旅程亦源于一种探索智能本质、挑战可能性边界的深层个人驱动力。它是一项始于深刻自省、着迷于复杂系统（无论是生物的、社会的还是计算的），并坚定相信专注且非传统的思考能够带来切实影响的努力。尽管前路无疑充满艰辛，但追寻本身即是无穷的智力源泉。)*

下方详述的 pIC50 预测项目，是在应用机器学习原理方面迈出的初步的、实践性的一步，同时也为 StellarCodeBio 愿景中更高级的、基于SNN的认知架构奠定了基础。

---

## 项目一：高效的基于机器学习的JAK2抑制剂pIC50预测模型

作为 StellarCodeBio 计划下的首个公开项目，本项目提供了一个经过验证的机器学习模型，用于预测潜在的Janus激酶2 (JAK2) 抑制剂的pIC50值，这在早期药物发现中是一项至关重要的任务。

**原始手稿背景：**
本pIC50模型的研究成果最初提交至 *Journal of Chemical Information and Modeling* (手稿ID: ci-2025-00977b)。后经期刊编辑认可其科学价值并建议转投，我们将其提交至 *ACS Omega* (手稿ID: ao-2025-043259)。尽管 ACS Omega 最终认为该工作的主要侧重点更符合机器学习应用而非其特定的化学读者群（因此决定不送外审），但他们仍对工作质量表示了肯定。本次开源旨在与更广泛的科学界分享这些研究发现和开发的工具，特别是那些可能认为此工具有用但缺乏类似商业软件资源的同行。我们坚信开放科学的力量能够加速所有人的研究进程。

### 摘要 (pIC50 模型)
**背景：** Janus激酶2 (JAK2) 是细胞信号转导中的关键激酶。其异常激活与多种骨髓增殖性肿瘤和炎症性疾病密切相关。开发选择性JAK2抑制剂是药物发现的重要方向。准确预测化合物对JAK2的抑制活性 ($pIC_{50}$) 对于加速先导化合物的发现和优化至关重要。
**目的：** 本研究旨在利用ChEMBL数据库的公共资源，结合机器学习方法，构建一个能够高效、准确预测JAK2抑制剂 $pIC_{50}$ 值的计算模型。
**方法：** 我们从ChEMBL数据库收集了靶向人JAK2 (ChEMBL ID: CHEMBL2971) 的化合物及其 $IC_{50}$ (nM) 活性数据。经过数据清洗和标准化（将 $IC_{50}$ 转换为 $pIC_{50}$，对重复化合物取平均值），最终获得了包含5546个化合物的数据集。使用RDKit (v2022.9.5) 计算了摩根指纹、MACCS键以及13种理化和拓扑描述符。基于LightGBM模型计算的特征重要性进行了特征选择，最终XGBoost模型 (v3.0.1) 使用了345个特征。使用5折交叉验证和GridSearchCV对超参数进行了调优，并采用了早停策略。
**结果 (2025年5月24日重跑验证)：** 最终的XGBoost模型在独立测试集上表现出良好的预测性能，取得了：
    *   决定系数 ($R^2$): **0.6828**
    *   均方根误差 (RMSE): **0.6334**
    *   平均绝对误差 (MAE): **0.4761**
    （训练集 $R^2$ 为 0.9367，表明模型拟合良好，同时有效控制了过拟合。）
**结论：** 本研究成功构建了一个基于XGBoost的JAK2抑制剂 $pIC_{50}$ 预测模型。该模型利用易于获取的分子描述符，在外部测试集上展现了良好的预测准确性和鲁棒性。此模型有望作为一种高效的虚拟筛选工具，辅助JAK2抑制剂的早期发现和优化过程。

### 特性 (pIC50 模型项目)
*   透明且可复现的JAK2 pIC50预测流程。
*   利用公开可用的ChEMBL数据。
*   采用标准的化学信息学工具 (RDKit) 和机器学习库 (XGBoost, Scikit-learn, LightGBM)。
*   包含以下Python脚本：
    1.  `data_processing.py`: 从ChEMBL下载和预处理JAK2靶点 (CHEMBL2971) 的生物活性数据，并进行特征工程。生成 `final_data.csv`。
    2.  `optimize_model.py`: 使用 `final_data.csv` 通过LightGBM计算特征重要性。生成 `feature_importance.csv`。
    3.  `modeltraining.py`: 使用 `final_data.csv` 和 `feature_importance.csv` 进行最终XGBoost模型的训练、超参数调优 (GridSearchCV) 和评估。保存训练好的模型及相关产出物，并绘制预测图。
*   为方便使用，提供了预生成的 `final_data.csv` (约5546个化合物，包含SMILES、pIC50及2200+初始计算特征) 和 `feature_importance.csv` (排序后的特征重要性列表)。
*   提供从零开始或使用预生成文件复现结果的详细说明。

### 目录结构

StellarCodeBio/
├── modeltraining.py # XGBoost模型训练与评估的主脚本
├── data_processing.py # 数据下载与预处理脚本
├── optimize_model.py # 特征重要性计算脚本
├── final_data.csv # 预处理后的包含所有初始特征的数据
├── feature_importance.csv # LightGBM输出的特征重要性分数
├── requirements.txt # Python依赖项
├── LICENSE # MIT许可证
├── README.md # 英文版README (本项目的主README)
├── README_zh.md # 本中文版README文件

*(`.joblib` 模型文件将由 `modeltraining.py` 脚本在仓库根目录生成。)*

### 环境要求
*   Python 3.9+ (开发和测试基于 Python 3.11)
*   主要依赖库 (完整列表及开发时使用的具体版本请参见 `requirements.txt`):
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

### 安装步骤
1.  克隆本代码仓库:
    ```bash
    git clone https://github.com/ShengyaoLiang/StellarCodeBio.git
    cd StellarCodeBio
    ```
2.  创建虚拟环境 (推荐):
    ```bash
    python3 -m venv venv 
    source venv/bin/activate  # Windows系统下: venv\Scripts\activate
    ```
3.  安装所需依赖包:
    ```bash
    pip install -r requirements.txt
    ```
    *关于RDKit的说明:* 如果 `pip install rdkit-pypi` 安装失败，请参考RDKit官方安装指南获取其他安装方式 (例如，使用Conda)。

### 使用方法 - 结果复现步骤
请按以下顺序运行脚本以完全复现本项目报告的结果：

1.  **安装依赖:**
    确保已在您的环境中安装 `requirements.txt` 中的所有依赖包。
    ```bash
    pip install -r requirements.txt
    ```

2.  **(选项 A - 从零开始复现) 数据处理与特征计算:**
    此脚本将从ChEMBL数据库下载人JAK2 (CHEMBL ID: CHEMBL2971) 的最新数据，进行数据清洗（保留精确IC50值，移除重复项及缺失数据），将IC50转换为pIC50，并计算分子特征（摩根指纹、MACCS键和13种理化/拓扑描述符）。
    ```bash
    python data_processing.py
    ```
    **输出:** `final_data.csv` (包含SMILES、pIC50及超过2200个初始计算的特征列，约5546个化合物)。

3.  **(选项 A - 从零开始复现) 特征重要性计算:**
    此脚本读取 `final_data.csv`，使用所有计算出的特征训练一个LightGBM模型，并对特征按重要性进行排序。
    ```bash
    python optimize_model.py
    ```
    **输入:** `final_data.csv`
    **输出:** `feature_importance.csv` (包含两列：'feature' 和 'importance'，按重要性降序排列)。

4.  **模型训练与评估:**
    此脚本读取 `final_data.csv` 和 `feature_importance.csv`，选择前350个重要特征（实际约使用345个在`final_data.csv`中存在的特征），对数据进行划分和标准化，使用GridSearchCV进行XGBoost超参数调优（此步骤可能耗时较长），使用早停策略训练最终模型，评估其性能，并保存模型相关文件及预测对比图。
    ```bash
    python modeltraining.py 
    ```
    **输入:** `final_data.csv`, `feature_importance.csv`
    **输出:**
    *   `jak2_pci50_model_tuned_top350_xgb3_v5.joblib` (包含训练好的XGBoost模型、StandardScaler对象以及所使用的345个特征名称列表的字典)。
    *   `assets/jak2_prediction_vs_actual_tuned_top350_xgb3_v5.png` (测试集上预测pIC50值与实际pIC50值的散点图。假设根目录下存在 `assets` 文件夹，否则图片将保存在根目录)。
    *   模型性能指标 (R², RMSE, MAE) 将打印到控制台。

    **(选项 B - 使用预生成文件):** 如果您已从本仓库下载了预生成的 `final_data.csv` 和 `feature_importance.csv` 文件，可以跳过上述步骤2和3，直接运行 `python modeltraining.py`。

### 预期结果 (基于2025年5月24日的重跑验证)
*   **GridSearchCV 最佳 R² (在交叉验证集上):** ~0.6516
*   **最终模型在测试集上的性能:**
    *   决定系数 (R²): ~0.6828
    *   均方根误差 (RMSE): ~0.6334
    *   平均绝对误差 (MAE): ~0.4761
*   **最终模型使用的决策树数量 (由早停策略决定):** ~342

*(请注意：由于计算环境的差异或某些算法固有的随机性，即使设置了固定的随机种子，您的结果也可能与上述数值有细微差别。GridSearchCV找到的最优超参数组合也可能因多次运行或参数空间细微调整而略有不同。)*

### 模型详情 (pIC50 预测器)
*   **靶点:** Janus Kinase 2 (JAK2) - ChEMBL ID: CHEMBL2971
*   **活性数据:** pIC50 值。
*   **特征:** 345个筛选后的特征，包括摩根指纹、MACCS键以及RDKit计算的理化/拓扑描述符。
*   **算法:** XGBoost (复现时使用版本 3.0.1)
*   **超参数 (2025年5月24日重跑得到的最佳组合示例):** `{'colsample_bytree': 0.75, 'gamma': 0.25, 'learning_rate': 0.03, 'max_depth': 13, 'reg_alpha': 0.25, 'reg_lambda': 0.55, 'subsample': 0.8}` (实际最佳参数会由 `modeltraining.py` 脚本中的GridSearchCV搜索确定)。

---

## StellarCodeBio 未来方向
此pIC50预测模型仅仅是开始。StellarCodeBio计划未来可能探索的方向包括（但不限于）：
*   **研发先进的脉冲神经网络 (SNN) 架构：** 面向需要时序数据处理和高能效的任务，从基础的脉冲语言模型 (SLMs) 开始。
*   **SNN的神经符号整合：** 结合SNN的优势与符号知识及推理能力。
*   **面向数据高效SNN的元学习：** 使SNN能够以少量样本快速适应新任务或数据。
*   **生物系统中的因果推断：** 构建旨在理解潜在因果机制的模型。
*   **迈向“研究模型”：** 逐步构建能够辅助甚至驱动科学探究的更自主的AI系统。

我们信奉开放与协作的研究。敬请期待后续更新！

## 为StellarCodeBio贡献
我们对StellarCodeBio的潜力充满期待，并欢迎各种形式的贡献、合作、讨论和反馈。请随时：
*   在 [GitHub Issues页面](https://github.com/ShengyaoLiang/StellarCodeBio/issues) 提交BUG报告、功能请求或提出问题。
*   Fork本仓库并提交包含您改进的Pull Request。
*   通过邮件 (pikeshuaiwe@gmail.com) 联系梁晟耀 (ShengYao Liang) 进行合作咨询。

## 许可证
本代码仓库的内容，作为StellarCodeBio计划的一部分，采用 **MIT许可证 (MIT License)**授权。详情请参阅 `LICENSE` 文件。

## 致谢
*   ChEMBL数据库提供基础的生物活性数据。
*   RDKit社区提供的宝贵的化学信息学工具包。
*   Scikit-learn, XGBoost, LightGBM, Pandas, NumPy, Matplotlib, Seaborn, 和 Joblib等开源项目的开发者和社区。
*   任何提供过具体帮助或启发的个人

---