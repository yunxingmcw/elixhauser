import pandas as pd
import numpy as np
import re
from typing import Union, Dict, List, Tuple

def calculate_elixhauser_score(df: pd.DataFrame, 
                             id_col: str = 'id', 
                             icd_col: str = 'icd_code',
                             score_type: str = 'unweighted',
                             icd_version: str = 'icd10') -> pd.DataFrame:
    """
    Calculate Elixhauser comorbidity scores for patients.
    
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
    icd_version : str
        'icd9' or 'icd10' for ICD version
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with patient IDs, individual comorbidity flags, and total score
    """
    
    # Validate inputs
    if score_type not in ['unweighted', 'weighted']:
        raise ValueError("score_type must be 'unweighted' or 'weighted'")
    
    if icd_version not in ['icd9', 'icd10']:
        raise ValueError("icd_version must be 'icd9' or 'icd10'")
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Normalize ICD codes (remove dots, convert to uppercase)
    data['normalized_icd'] = data[icd_col].astype(str).str.replace('.', '').str.upper()
    
    # Get comorbidity definitions and weights
    comorbidity_defs = _get_elixhauser_definitions(icd_version)
    weights = _get_elixhauser_weights(score_type)
    
    # Initialize results DataFrame
    unique_ids = data[id_col].unique()
    results = pd.DataFrame({id_col: unique_ids})
    
    # Calculate each comorbidity
    for comorbidity, icd_patterns in comorbidity_defs.items():
        results[comorbidity] = 0
        
        for patient_id in unique_ids:
            patient_codes = data[data[id_col] == patient_id]['normalized_icd'].tolist()
            
            # Check if patient has this comorbidity
            has_comorbidity = any(
                _matches_pattern(code, patterns) 
                for code in patient_codes 
                for patterns in icd_patterns
            )
            
            results.loc[results[id_col] == patient_id, comorbidity] = int(has_comorbidity)
    
    # Handle mutually exclusive conditions
    results = _handle_exclusions(results)
    
    # Calculate total score
    comorbidity_cols = [col for col in results.columns if col != id_col]
    
    if score_type == 'unweighted':
        results['elixhauser_score'] = results[comorbidity_cols].sum(axis=1)
    else:  # weighted
        weighted_scores = []
        for _, row in results.iterrows():
            score = sum(row[col] * weights.get(col, 0) for col in comorbidity_cols)
            weighted_scores.append(score)
        results['elixhauser_score'] = weighted_scores
    
    return results

def _get_elixhauser_definitions(icd_version: str) -> Dict[str, List[List[str]]]:
    """Get ICD code patterns for Elixhauser comorbidities."""
    
    if icd_version == 'icd10':
        return {
            'congestive_heart_failure': [['I099', 'I110', 'I130', 'I132', 'I255', 'I420', 'I425', 'I426', 'I427', 'I428', 'I429', 'I43', 'I50', 'P290']],
            'cardiac_arrhythmias': [['I441', 'I442', 'I443', 'I456', 'I459', 'I47', 'I48', 'I49', 'R000', 'R001', 'R008', 'T821', 'Z450', 'Z950']],
            'valvular_disease': [['A520', 'I05', 'I06', 'I07', 'I08', 'I091', 'I098', 'I34', 'I35', 'I36', 'I37', 'I38', 'I39', 'Q230', 'Q231', 'Q232', 'Q233', 'Z952', 'Z953', 'Z954']],
            'pulmonary_circulation': [['I26', 'I27', 'I280', 'I288', 'I289']],
            'peripheral_vascular': [['I70', 'I71', 'I731', 'I738', 'I739', 'I771', 'I790', 'I792', 'K551', 'K558', 'K559', 'Z958', 'Z959']],
            'hypertension_uncomplicated': [['I10']],
            'hypertension_complicated': [['I11', 'I12', 'I13', 'I15']],
            'paralysis': [['G041', 'G114', 'G801', 'G802', 'G81', 'G82', 'G830', 'G831', 'G832', 'G833', 'G834', 'G839']],
            'other_neurological': [['G10', 'G11', 'G12', 'G13', 'G20', 'G21', 'G22', 'G254', 'G255', 'G312', 'G318', 'G319', 'G32', 'G35', 'G36', 'G37', 'G40', 'G41', 'G931', 'R470', 'R56']],
            'chronic_pulmonary': [['I278', 'I279', 'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47', 'J60', 'J61', 'J62', 'J63', 'J64', 'J65', 'J66', 'J67', 'J684', 'J701', 'J703']],
            'diabetes_uncomplicated': [['E100', 'E101', 'E106', 'E108', 'E109', 'E110', 'E111', 'E116', 'E118', 'E119', 'E120', 'E121', 'E126', 'E128', 'E129', 'E130', 'E131', 'E136', 'E138', 'E139', 'E140', 'E141', 'E146', 'E148', 'E149']],
            'diabetes_complicated': [['E102', 'E103', 'E104', 'E105', 'E107', 'E112', 'E113', 'E114', 'E115', 'E117', 'E122', 'E123', 'E124', 'E125', 'E127', 'E132', 'E133', 'E134', 'E135', 'E137', 'E142', 'E143', 'E144', 'E145', 'E147']],
            'hypothyroidism': [['E00', 'E01', 'E02', 'E03', 'E890']],
            'renal_failure': [['I120', 'I131', 'N032', 'N033', 'N034', 'N035', 'N036', 'N037', 'N052', 'N053', 'N054', 'N055', 'N056', 'N057', 'N18', 'N19', 'N250', 'Z490', 'Z491', 'Z492', 'Z940', 'Z992']],
            'liver_disease': [['B18', 'I85', 'I864', 'I982', 'K70', 'K711', 'K713', 'K714', 'K715', 'K717', 'K72', 'K73', 'K74', 'K760', 'K762', 'K763', 'K764', 'K768', 'K769', 'Z944']],
            'peptic_ulcer': [['K257', 'K259', 'K267', 'K269', 'K277', 'K279', 'K287', 'K289']],
            'aids': [['B20', 'B21', 'B22', 'B24']],
            'lymphoma': [['C81', 'C82', 'C83', 'C84', 'C85', 'C88', 'C96', 'C900', 'C902']],
            'metastatic_cancer': [['C77', 'C78', 'C79', 'C80']],
            'solid_tumor': [['C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C30', 'C31', 'C32', 'C33', 'C34', 'C37', 'C38', 'C39', 'C40', 'C41', 'C43', 'C45', 'C46', 'C47', 'C48', 'C49', 'C50', 'C51', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C60', 'C61', 'C62', 'C63', 'C64', 'C65', 'C66', 'C67', 'C68', 'C69', 'C70', 'C71', 'C72', 'C73', 'C74', 'C75', 'C76', 'C97']],
            'rheumatoid_arthritis': [['L940', 'L941', 'L943', 'M05', 'M06', 'M315', 'M32', 'M33', 'M34', 'M35', 'M45', 'M461', 'M468', 'M469']],
            'coagulopathy': [['D560', 'D561', 'D562', 'D568', 'D569', 'D65', 'D66', 'D67', 'D68', 'D691', 'D693', 'D694', 'D695', 'D696']],
            'obesity': [['E66']],
            'weight_loss': [['E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'R634', 'R64']],
            'fluid_electrolyte': [['E222', 'E86', 'E87']],
            'blood_loss_anemia': [['D500']],
            'deficiency_anemia': [['D508', 'D509', 'D51', 'D52', 'D53']],
            'alcohol_abuse': [['F10', 'E52', 'G621', 'I426', 'K292', 'K700', 'K703', 'K709', 'T51', 'Z502', 'Z714', 'Z721']],
            'drug_abuse': [['F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F18', 'F19', 'Z715', 'Z722']],
            'psychoses': [['F20', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29', 'F302', 'F312', 'F315']],
            'depression': [['F204', 'F313', 'F314', 'F315', 'F32', 'F33', 'F341', 'F412']]
        }
    
    else:  # ICD-9
        return {
            'congestive_heart_failure': [['39891', '40201', '40211', '40291', '40401', '40403', '40411', '40413', '40491', '40493', '4254', '4255', '4257', '4258', '4259', '428']],
            'cardiac_arrhythmias': [['4260', '42613', '4267', '4269', '42610', '42612', '4270', '4271', '4272', '4273', '4274', '4276', '4278', '4279', '7850', '99601', '99604', 'V450', 'V533']],
            'valvular_disease': [['0932', '394', '395', '396', '397', '424', '7463', '7464', '7465', '7466', 'V422', 'V433']],
            'pulmonary_circulation': [['4150', '4151', '416', '4170', '4178', '4179']],
            'peripheral_vascular': [['0930', '4373', '440', '441', '4431', '4432', '4438', '4439', '4471', '5571', '5579', 'V434']],
            'hypertension_uncomplicated': [['401']],
            'hypertension_complicated': [['402', '403', '404', '405']],
            'paralysis': [['3341', '342', '343', '3440', '3441', '3442', '3443', '3444', '3445', '3446', '3449']],
            'other_neurological': [['331', '332', '333', '3341', '335', '340', '341', '345', '3481', '3483', '7803', '7843']],
            'chronic_pulmonary': [['4168', '4169', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505', '5064', '5081', '5088']],
            'diabetes_uncomplicated': [['2500', '2501', '2502', '2503']],
            'diabetes_complicated': [['2504', '2505', '2506', '2507', '2508', '2509']],
            'hypothyroidism': [['2409', '243', '244', '2461', '2468']],
            'renal_failure': [['40301', '40311', '40391', '40402', '40403', '40412', '40413', '40492', '40493', '585', '586', '5880', 'V420', 'V451']],
            'liver_disease': [['07022', '07023', '07032', '07033', '07044', '07054', '4560', '4561', '4562', '570', '571', '5722', '5728', '5733', '5734', '5738', '5739', 'V427']],
            'peptic_ulcer': [['5317', '5319', '5327', '5329', '5337', '5339', '5347', '5349']],
            'aids': [['042', '043', '044']],
            'lymphoma': [['200', '201', '202', '2030', '2386']],
            'metastatic_cancer': [['196', '197', '198', '199']],
            'solid_tumor': [['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '170', '171', '172', '174', '175', '176', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195']],
            'rheumatoid_arthritis': [['446', '7010', '7100', '7101', '7102', '7103', '7104', '7140', '7141', '7142', '725']],
            'coagulopathy': [['286', '2871', '2873', '2874', '2875']],
            'obesity': [['278']],
            'weight_loss': [['260', '261', '262', '263', '7832', '7994']],
            'fluid_electrolyte': [['2536', '276']],
            'blood_loss_anemia': [['2800']],
            'deficiency_anemia': [['2801', '2808', '2809', '281']],
            'alcohol_abuse': [['265', '2652', '2911', '2912', '2913', '2915', '2918', '2919', '3030', '3039', '3050', '3575', '4255', '5353', '5710', '5711', '5712', '5713', 'V113']],
            'drug_abuse': [['292', '304', '3052', '3053', '3054', '3055', '3056', '3057', '3058', '3059', 'V6542']],
            'psychoses': [['295', '297', '298']],
            'depression': [['2962', '2963', '2965', '3004', '309', '311']]
        }

def _get_elixhauser_weights(score_type: str) -> Dict[str, float]:
    """Get weights for Elixhauser comorbidities (van Walraven et al. weights)."""
    
    if score_type == 'unweighted':
        return {}  # No weights needed for unweighted scoring
    
    # van Walraven et al. weights (commonly used)
    return {
        'congestive_heart_failure': 7,
        'cardiac_arrhythmias': 5,
        'valvular_disease': -1,
        'pulmonary_circulation': 4,
        'peripheral_vascular': 2,
        'hypertension_uncomplicated': 0,
        'hypertension_complicated': 0,
        'paralysis': 7,
        'other_neurological': 6,
        'chronic_pulmonary': 3,
        'diabetes_uncomplicated': 0,
        'diabetes_complicated': 0,
        'hypothyroidism': 0,
        'renal_failure': 5,
        'liver_disease': 11,
        'peptic_ulcer': 0,
        'aids': 0,
        'lymphoma': 9,
        'metastatic_cancer': 12,
        'solid_tumor': 4,
        'rheumatoid_arthritis': 0,
        'coagulopathy': 3,
        'obesity': -4,
        'weight_loss': 6,
        'fluid_electrolyte': 5,
        'blood_loss_anemia': -2,
        'deficiency_anemia': -2,
        'alcohol_abuse': 0,
        'drug_abuse': -7,
        'psychoses': 0,
        'depression': -3
    }

def _matches_pattern(icd_code: str, patterns: List[str]) -> bool:
    """Check if ICD code matches any of the given patterns."""
    for pattern in patterns:
        if icd_code.startswith(pattern):
            return True
    return False

def _handle_exclusions(results: pd.DataFrame) -> pd.DataFrame:
    """Handle mutually exclusive conditions according to Elixhauser methodology."""
    
    # If complicated hypertension is present, remove uncomplicated
    mask = results['hypertension_complicated'] == 1
    results.loc[mask, 'hypertension_uncomplicated'] = 0
    
    # If complicated diabetes is present, remove uncomplicated
    mask = results['diabetes_complicated'] == 1
    results.loc[mask, 'diabetes_uncomplicated'] = 0
    
    # If metastatic cancer is present, remove solid tumor
    mask = results['metastatic_cancer'] == 1
    results.loc[mask, 'solid_tumor'] = 0
    
    return results

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'patient_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'diagnosis_code': ['I50.9', 'E11.9', 'I10', 'C78.1', 'C25.9', 'F32.9', 'I48.0', 'N18.6', 'E66.9']
    })
    
    print("Sample Data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    # Calculate unweighted scores
    unweighted_scores = calculate_elixhauser_score(
        sample_data, 
        id_col='patient_id', 
        icd_col='diagnosis_code',
        score_type='unweighted',
        icd_version='icd10'
    )
    
    print("Unweighted Elixhauser Scores:")
    print(unweighted_scores[['patient_id', 'elixhauser_score']])
    print("\n" + "="*50 + "\n")
    
    # Calculate weighted scores
    weighted_scores = calculate_elixhauser_score(
        sample_data, 
        id_col='patient_id', 
        icd_col='diagnosis_code',
        score_type='weighted',
        icd_version='icd10'
    )
    
    print("Weighted Elixhauser Scores:")
    print(weighted_scores[['patient_id', 'elixhauser_score']])
    print("\n" + "="*50 + "\n")
    
    # Show detailed breakdown for first patient
    print("Detailed breakdown for Patient 1:")
    patient_1_details = weighted_scores[weighted_scores['patient_id'] == 1]
    comorbidity_cols = [col for col in patient_1_details.columns if col not in ['patient_id', 'elixhauser_score']]
    active_comorbidities = [col for col in comorbidity_cols if patient_1_details[col].iloc[0] == 1]
    print(f"Active comorbidities: {active_comorbidities}")
