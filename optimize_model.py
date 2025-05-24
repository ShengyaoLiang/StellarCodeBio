import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, GraphDescriptors
import matplotlib.pyplot as plt
import logging
import warnings

def main():
    # 读取数据
    try:
        data = pd.read_csv('final_data.csv') # 确认使用最新的数据文件
    except FileNotFoundError:
        print("错误：找不到 final_data.csv。请先运行 data_processing.py 生成该文件。")
        return
    except Exception as e:
        print(f"读取 final_data.csv 时出错: {e}")
        return

    # -- 直接从 data 中选择所有特征列 --
    # 获取 Morgan 指纹列名
    morgan_cols = [col for col in data.columns if col.startswith('morgan_')]
    # 获取 MACCS Keys 列名
    maccs_cols = [col for col in data.columns if col.startswith('maccs_')]
    # 定义基础描述符列名
    base_desc_cols = ['MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    # 定义 RDKit 描述符列名 (确保与 data_processing.py 中生成的一致)
    # 这里列出一些常用的，请根据 data_processing.py 的实际输出调整
    rdkit_desc_cols = [
        'NumAromaticRings', 'NumRotatableBonds', 'RingCount', 'NumHeteroatoms', 'HeavyAtomCount',
        'NumAmideBonds', 'BalabanJ', 'BertzCT', 'FractionCSP3', 'MolMR',
        'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
        'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedRings'
        # 添加其他在 data_processing.py 中计算的 RDKit 描述符...
    ]

    # 合并所有可能的特征列名
    all_possible_feature_cols = morgan_cols + maccs_cols + base_desc_cols + rdkit_desc_cols

    # 筛选出实际存在于 data 中的特征列（排除 target 和 SMILES）
    target_col = 'pCI50'
    smiles_col = 'canonical_smiles' # 假设 SMILES 列名为 canonical_smiles
    existing_feature_cols = [col for col in data.columns if col in all_possible_feature_cols and col != target_col and col != smiles_col]

    # 检查是否有特征列被找到
    if not existing_feature_cols:
        print("错误：在 final_data.csv 中找不到任何预期的特征列。")
        print(f"找到的列: {list(data.columns)}")
        print(f"预期的特征前缀/名称: morgan_, maccs_, {base_desc_cols}, {rdkit_desc_cols}")
        return

    print(f"找到 {len(existing_feature_cols)} 个特征列用于训练。")

    # 准备数据
    X = data[existing_feature_cols].values
    y = data[target_col].values

    # 检查是否有 NaN 值
    if np.isnan(X).any():
        print("警告：特征数据 X 中包含 NaN 值。正在尝试用列均值填充...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    if np.isnan(y).any():
        print("警告：目标数据 y 中包含 NaN 值。将移除包含 NaN 的行...")
        nan_mask = ~np.isnan(y)
        X = X[nan_mask]
        y = y[nan_mask]
        if len(y) == 0:
            print("错误：移除 NaN 后目标数据为空。")
            return

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 创建并训练 LightGBM 模型 ---
    print("\nTraining LightGBM model...")
    # 可以设置一些基础参数，或者后续进行超参数优化
    lgbm_model = lgb.LGBMRegressor(
        random_state=42,
        n_estimators=500, # 增加树的数量
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1, # 无限制
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1 # 使用所有可用核心
    )

    # 训练模型 (可以加入 early stopping)
    lgbm_model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 eval_metric='rmse', # 或者 'r2'
                 callbacks=[lgb.early_stopping(100, verbose=True)]) # 提前停止
                 # verbose=False # 训练时不打印过多信息

    # --- 评估模型 ---
    y_pred_train = lgbm_model.predict(X_train)
    y_pred_test = lgbm_model.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    print(f"\n训练集 R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}")
    print(f"测试集 R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}")

    # --- 计算并保存 LightGBM 特征重要性 ---
    print("\nCalculating and saving feature importances...")
    importances = lgbm_model.feature_importances_

    if len(existing_feature_cols) != len(importances):
        print(f"错误：特征名称数量 ({len(existing_feature_cols)}) 与重要性得分数量 ({len(importances)}) 不匹配！")
        print("将跳过特征重要性文件的保存。")
    else:
        feature_importance = pd.DataFrame({
            'feature': existing_feature_cols,
            'importance': importances
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("特征重要性已保存到 feature_importance.csv")
        print("\n最重要的特征 (基于 LightGBM):")
        print(feature_importance.head(10))

if __name__ == "__main__":
    # 配置日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # 忽略特定警告
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')
    main()