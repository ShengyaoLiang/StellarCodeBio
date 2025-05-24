import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from tqdm.auto import tqdm  # 用于显示进度条

# ChEMBL Target ID for human JAK2
TARGET_CHEMBL_ID = 'CHEMBL2971'
OUTPUT_FILENAME = 'final_data.csv'

def fetch_chembl_data(target_id):
    """
    Fetches bioactivity data for a given ChEMBL target ID.
    """
    print(f"Fetching data for target: {target_id}")
    activity = new_client.activity
    # Search for IC50 activities, measured in nM, for the specific target
    res = activity.filter(
        target_chembl_id=target_id,
        standard_type="IC50",
        standard_units="nM"
    ).only(
        'activity_id', 'assay_chembl_id', 'canonical_smiles',
        'molecule_chembl_id', 'standard_relation', 'standard_value',
        'standard_units', 'standard_type', 'target_chembl_id'
    )
    df = pd.DataFrame(res)
    print(f"Fetched {len(df)} records.")
    return df

def clean_data(df):
    """
    Cleans the raw ChEMBL data.
    """
    print("Cleaning data...")
    # Ensure standard_value is numeric and drop rows where it's not
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df.dropna(subset=['standard_value', 'canonical_smiles'])

    # Keep only exact measurements ('=' relationship)
    df = df[df['standard_relation'] == '=']

    # Remove entries with non-positive IC50 values
    df = df[df['standard_value'] > 0]

    print(f"{len(df)} records remaining after initial cleaning.")
    return df

def calculate_pci50(df):
    """
    Converts IC50 (nM) to pCI50.
    pCI50 = -log10(IC50 in M)
    """
    print("Calculating pCI50...")
    df['IC50_M'] = df['standard_value'] * 1e-9  # Convert nM to M
    # Add a small constant to avoid log10(0) if any IC50_M is extremely small or zero after potential future processing steps
    df['pCI50'] = -np.log10(df['IC50_M'] + 1e-12) # Added small constant for safety
    df = df.drop(columns=['standard_value', 'standard_relation', 'standard_units', 'standard_type', 'IC50_M'])
    print("pCI50 calculation finished.")
    return df

def handle_duplicates(df):
    """
    Handles duplicate entries for the same molecule.
    Aggregates pCI50 values by taking the mean.
    """
    print("Handling duplicates...")
    # Group by SMILES and calculate the mean pCI50
    df_agg = df.groupby('canonical_smiles')['pCI50'].mean().reset_index()
    # Keep other relevant info (like molecule_chembl_id) by merging back
    # We might lose some assay/activity specific info here, focusing on molecule-potency relationship
    df_unique_mols = df.drop_duplicates(subset=['canonical_smiles'])[['canonical_smiles', 'molecule_chembl_id']]
    df_final = pd.merge(df_agg, df_unique_mols, on='canonical_smiles', how='left')
    print(f"{len(df_final)} unique molecules remaining after aggregation.")
    return df_final

def generate_features(df):
    """
    Generates molecular features (Morgan fingerprints, MACCS Keys, and descriptors) from SMILES.
    """
    print("Generating molecular features...")
    molecules = [Chem.MolFromSmiles(smi) for smi in tqdm(df['canonical_smiles'], desc="Parsing SMILES")]
    # Filter out invalid SMILES
    valid_idx = [i for i, mol in enumerate(molecules) if mol is not None]
    df = df.iloc[valid_idx].reset_index(drop=True)
    molecules = [molecules[i] for i in valid_idx]
    print(f"{len(df)} valid molecules for feature generation.")

    # Morgan Fingerprints (ECFP4 equivalent)
    fp_morgan = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in tqdm(molecules, desc="Generating Morgan Fingerprints")]
    fp_morgan_df = pd.DataFrame([list(arr) for arr in fp_morgan], columns=[f'morgan_{i}' for i in range(2048)])

    # MACCS Keys Fingerprints
    fp_maccs = [AllChem.GetMACCSKeysFingerprint(m) for m in tqdm(molecules, desc="Generating MACCS Keys")]
    fp_maccs_df = pd.DataFrame([list(arr) for arr in fp_maccs], columns=[f'maccs_{i}' for i in range(167)]) # MACCS keys have 167 bits (index 0 unused)

    # RDKit Descriptors
    basic_descriptors = []
    basic_desc_names = [
        'MolWt', 'MolLogP', 'NumHDonors', 'NumHAcceptors', 'TPSA'
    ]
    extended_descriptors = []
    extended_desc_names = [
        'FractionCSP3', 'NumAromaticRings', 'NumAliphaticRings',
        'NumRotatableBonds', 'ExactMolWt', 'qed', 'BalabanJ', 'BertzCT'
    ]
    all_desc_names = basic_desc_names + extended_desc_names

    # Calculate all descriptors in one loop
    for mol in tqdm(molecules, desc="Calculating Descriptors"):
        desc_values = []
        try:
            # Calculate basic descriptors
            desc_values.extend([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol)
            ])
            # Calculate extended descriptors
            desc_values.extend([
                Descriptors.FractionCSP3(mol),
                Descriptors.NumAromaticRings(mol),
                Descriptors.NumAliphaticRings(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.ExactMolWt(mol),
                Chem.QED.qed(mol), # QED is in Chem.QED module
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol)
            ])
            extended_descriptors.append(desc_values) # Append the combined list
        except Exception as e:
            # Handle potential errors during descriptor calculation
            print(f"Warning: Could not calculate all descriptors for a molecule. Error: {e}. Filling with NaN.")
            extended_descriptors.append([np.nan] * len(all_desc_names))

    desc_df = pd.DataFrame(extended_descriptors, columns=all_desc_names)
    # Handle potential NaNs from descriptor calculation errors
    desc_df = desc_df.fillna(desc_df.median()) # Fill with median

    # Combine features (Molecule Info + Morgan FP + MACCS FP + Descriptors)
    final_df = pd.concat([
        df[['molecule_chembl_id', 'canonical_smiles', 'pCI50']],
        fp_morgan_df,
        fp_maccs_df,
        desc_df
    ], axis=1)
    print("Feature generation finished.")
    return final_df


def main():
    """
    Main function to run the data processing pipeline.
    """
    raw_data = fetch_chembl_data(TARGET_CHEMBL_ID)
    if raw_data.empty:
        print("No data fetched. Exiting.")
        return

    cleaned_data = clean_data(raw_data)
    if cleaned_data.empty:
        print("No data after cleaning. Exiting.")
        return

    pci50_data = calculate_pci50(cleaned_data)

    unique_data = handle_duplicates(pci50_data)
    if unique_data.empty:
        print("No data after handling duplicates. Exiting.")
        return

    featured_data = generate_features(unique_data)
    if featured_data.empty:
        print("No data after feature generation. Exiting.")
        return

    # Save the final processed data
    featured_data.to_csv(OUTPUT_FILENAME, index=False)
    print(f"Processed data saved to {OUTPUT_FILENAME}")
    print(f"Final dataset shape: {featured_data.shape}")

if __name__ == '__main__':
    # Before running: pip install chembl_webresource_client pandas numpy rdkit tqdm
    # Make sure you have the RDKit cheminformatics toolkit installed.
    main() 
