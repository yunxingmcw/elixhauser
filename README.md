# elixhauser
This is a python implementation of the SAS Elixhauser Comorbidity Software Refined for ICD-10-CM v2025.
See details at (https://hcup-us.ahrq.gov/toolssoftware/comorbidityicd10/comorbidity_icd10.jsp#indices)

1. The icd10-comorbididty mapping is stored in the icd10_dict and the weights are mw for mortality and rw for readmission. 

2. The example input file is a synthetic file containing IDs and ICD codes. 

3. To use the code, the ICD-code file needs to be converted to long-format, containing the 2 columns: an ID column and an ICD-code column, each ID may have multiple ICD codes

