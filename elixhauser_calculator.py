#This script is for icd-10 code only
#the comorbidity measures are based on the Elixhauser Comorbidity Software Refined for ICD-10-CM v2025.
# there are 38 comorbidity measures instead of 31 in previous versions
# there are 2 weight-schemes, mortality and readmission
#(https://hcup-us.ahrq.gov/toolssoftware/comorbidityicd10/comorbidity_icd10.jsp#indices)

import pandas as pd
import numpy as np
import re
from typing import Union, Dict, List, Tuple
import pickle

#load the icd-cormobdity mapping dictionary
with open('icd10_dict.pkl', 'rb') as f:
    icd10_dict = pickle.load(f)
      
#load the mortality weights mapping dictionary
with open('mw.pkl', 'rb') as f:
    mw = pickle.load(f)
#load the mortality weights mapping dictionary
with open('rw.pkl', 'rb') as f:
    rw = pickle.load(f)    
    
#load the mortality weights mapping dictionary
with open('elixhauser.pkl', 'rb') as f:
    elixhauser = pickle.load(f)    

def calculate_elixhauser_score(df: pd.DataFrame, 
                             id_col: str = 'id', 
                             icd_col: str = 'icd_code',
                             score_type: str = 'unweighted',
                             weight_scheme: str = 'elixhauser'
                             ) -> pd.DataFrame:
    """
    Efficiently calculate Elixhauser comorbidity scores for patients.
    Optimized for large datasets (10,000+ rows).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing patient IDs and ICD codes
    id_col : str
        Column name containing patient IDs
    icd_col : str
        Column name containing ICD codes
    score_type : str
        'unweighted' for binary scoring, 'weighted' for weighted scoring
    weight_scheme : str
        Weighting scheme for weighted scoring:
        - 'mw': weights for calculating mortality index 
        - 'rw': weight for calculating hospital readmission index  
        - 'elixhauser': Original Elixhauser weights (default)
       
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with patient IDs, individual comorbidity flags, and total score
    """
    
    # Validate inputs
    if score_type not in ['unweighted', 'weighted']:
        raise ValueError("score_type must be 'unweighted' or 'weighted'")
    
    if weight_scheme not in ['mw', 'rw', 'elixhauser']:
        raise ValueError("weight_scheme must be one of: 'mw', 'rw', 'elixhauser'")
    
 
    # Step 1: Normalize ICD codes efficiently
    data = df.copy()
    data['normalized_icd'] = data[icd_col].astype(str).str.replace('.', '', regex=False).str.upper()
    
    # Step 2: Get comorbidity definitions
    comorbidity_patterns = icd10_dict
    
    # Step 3: Create lookup dictionary for fast matching
#     print("Building ICD code lookup table...")
    icd_to_comorbidities = _build_icd_lookup(comorbidity_patterns)
    
    # Step 4: Vectorized comorbidity detection
#     print("Detecting comorbidities...")
    data['comorbidities'] = data['normalized_icd'].map(
        lambda x: _find_matching_comorbidities(x, icd_to_comorbidities)
    ).fillna('')
    
    # Step 5: Aggregate by patient using efficient groupby
#     print("Aggregating by patient...")
    patient_comorbidities = (
        data.groupby(id_col)['comorbidities']
        .apply(lambda x: '|'.join(x))
        .reset_index()
    )
    
    # Step 6: Create binary matrix efficiently
    all_comorbidities = list(comorbidity_patterns.keys())
    results = pd.DataFrame({id_col: patient_comorbidities[id_col]})
    
    for comorbidity in all_comorbidities:
        results[comorbidity] = patient_comorbidities['comorbidities'].str.contains(
            f'\\b{comorbidity}\\b', regex=True, na=False
        ).astype(int)
    
    # Step 7: Handle exclusions efficiently
    results = _handle_exclusions_vectorized(results)
    
    # Step 8: Calculate scores
    if score_type == 'weighted':
        weights = _get_elixhauser_weights(weight_scheme)
        score_cols = [col for col in results.columns if col != id_col]
        weight_values = np.array([weights.get(col, 0) for col in score_cols])
        results['elixhauser_score'] = np.dot(results[score_cols].values, weight_values)
    else:
        score_cols = [col for col in results.columns if col != id_col]
        results['elixhauser_score'] = results[score_cols].sum(axis=1)
    
#     print(f"Completed processing. Found {results['elixhauser_score'].sum()} total comorbidities.")
    return results

    
def _build_icd_lookup(comorbidity_patterns: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Build efficient lookup table: ICD prefix -> list of comorbidities."""
    lookup = {}
    
    for comorbidity, patterns in comorbidity_patterns.items():
        for pattern in patterns:
            # For each pattern, create entries for all possible prefixes
            for i in range(3, len(pattern) + 1):  # Start from length 3 for efficiency
                prefix = pattern[:i]
                if prefix not in lookup:
                    lookup[prefix] = []
                if comorbidity not in lookup[prefix]:
                    lookup[prefix].append(comorbidity)
    
    return lookup

def _find_matching_comorbidities(icd_code: str, lookup: Dict[str, List[str]]) -> str:
    """Find all comorbidities that match an ICD code using prefix lookup."""
    if not icd_code or pd.isna(icd_code):
        return ''
    
    matching_comorbidities = set()
    
    # Check all possible prefixes of the ICD code
    for i in range(3, len(icd_code) + 1):
        prefix = icd_code[:i]
        if prefix in lookup:
            matching_comorbidities.update(lookup[prefix])
    
    return '|'.join(matching_comorbidities) if matching_comorbidities else ''

def _get_elixhauser_weights(weight_scheme: str = 'elixhauser') -> Dict[str, float]:
    """Get weights for Elixhauser comorbidities based on different schemes."""
    
    weight_schemes = {
        'mw':mw,
        'rw':rw,       
        'elixhauser': elixhauser,
         }
    
    if weight_scheme not in weight_schemes:
        raise ValueError(f"Unknown weight scheme: {weight_scheme}")
    
    return weight_schemes[weight_scheme]

def _handle_exclusions_vectorized(results: pd.DataFrame) -> pd.DataFrame:
    """Handle mutually exclusive conditions using vectorized operations."""
    
    # If complicated hypertension is present, remove uncomplicated
    mask = results['HTN_CX'] == 1
    results.loc[mask, 'HTN_UNCX'] = 0
    
    # If complicated diabetes is present, remove uncomplicated
    mask = results['DIAB_CX'] == 1
    results.loc[mask, 'DIAB_UNCX'] = 0
    
    # If metastatic cancer is present, remove solid tumor
    mask = results['CANCER_METS'] == 1
    results.loc[mask, 'CANCER_SOLID'] = 0
    
    return results

def compare_weight_schemes(df: pd.DataFrame, 
                         id_col: str = 'id', 
                         icd_col: str = 'icd_code'
                         ) -> pd.DataFrame:
    """
    Compare different weighting schemes for the same dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing patient IDs and ICD codes
    id_col : str
        Column name containing patient IDs
    icd_col : str
        Column name containing ICD codes
    icd_version : str
        'icd9' or 'icd10' for ICD version
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with patient IDs and scores from all weighting schemes
    """
    
    schemes = ['unweighted','mw', 'rw', 'elixhauser']
    results = None
    
    for scheme in schemes:
        if scheme == 'unweighted':
            score_result = calculate_elixhauser_score(
                df, id_col=id_col, icd_col=icd_col, 
                score_type='unweighted', 
            )
            score_col = f'{scheme}_score'
        else:
            score_result = calculate_elixhauser_score(
                df, id_col=id_col, icd_col=icd_col, 
                score_type='weighted', weight_scheme=scheme,
            )
            score_col = f'{scheme}_score'
        
        if results is None:
            results = score_result[[id_col, 'elixhauser_score']].copy()
            results.rename(columns={'elixhauser_score': score_col}, inplace=True)
        else:
            results[score_col] = score_result['elixhauser_score']
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Create larger sample dataset for performance testing
    np.random.seed(42)
    
    # Generate sample data
    n_patients = 2000
    n_codes_per_patient = np.random.poisson(5, n_patients)  # Average 5 codes per patient
    
    icd10_samples = ['I50.9', 'E11.9', 'I10', 'C78.1', 'C25.9', 'F32.9', 'I48.0', 'N18.6', 'E66.9', 'J44.1', 'I25.9', 'E78.5', 'K21.9', 'M79.3', 'Z87.891']
    
    sample_data = []
    for patient_id in range(1, n_patients + 1):
        n_codes = n_codes_per_patient[patient_id - 1]
        codes = np.random.choice(icd10_samples, size=n_codes, replace=True)
        for code in codes:
            sample_data.append({'patient_id': patient_id, 'diagnosis_code': code})
    
    large_sample = pd.DataFrame(sample_data)
    print(f"Created sample dataset with {len(large_sample)} rows and {large_sample['patient_id'].nunique()} patients")
    
       
    # Show sample results
    results = calculate_elixhauser_score(
        large_sample.head(100),  # Show first 100 rows for demo
        id_col='patient_id',
        icd_col='diagnosis_code',
        score_type='unweighted'
        
    )
    print(results.head(10))
    
    # Test different weighting schemes
    print("\nTesting different weighting schemes...")
    comparison = compare_weight_schemes(
        large_sample.head(100),
        id_col='patient_id',
        icd_col='diagnosis_code'
    )
    
    print(f"\nScore Comparison (first 10 patients):")
    print(comparison.head(10))
    
    print(f"\nScore Statistics by Weighting Scheme:")
    for col in comparison.columns:
        if col != 'patient_id':
            scores = comparison[col]
            print(f"{col}: Mean={scores.mean():.2f}, Std={scores.std():.2f}, Range=[{scores.min():.1f}, {scores.max():.1f}]")
