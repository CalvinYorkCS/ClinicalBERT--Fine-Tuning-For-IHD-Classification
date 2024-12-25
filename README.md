# <p align="center">ClinicalBERT Fine-tuning for Ischemic Heart Disease Detection</p>

### <p align="center">Group Members</p>
<p align="center">Calvin York, Chen Zhao, Jianing Wen, Suchithra Moolinti</p>

## Abstract
Clinical notes contain information about patients beyond structured data such as lab values or medications. However, clinical notes have been underused relative to structured data because notes are high dimensional and sparse. We aim to develop and evaluate a continuous representation of clinical notes. The goal for this project is to fine tune Bio_ClinicalBERT, which is a widely used pre-trained model on clinical data, for Ischemic Heart Disease (IHD) Detection. The initial preprocessing of this model includes standardizing text, splitting data input into manageable chunks, and creating labels for IHD. Binary labels are created using ICD-9 further utilised for training of models. The model is fine-tuned using a classification head, with hyperparameter optimization ensuring optimal learning rates and batch sizes. Evaluation metrics such as accuracy, precision, recall, F1-score demonstrate the model’s high performance, achieving accurate and reliable predictions of IHD presence. 
Introduction

The healthcare industry has experienced huge advancements with respect to patient health care data management because most of the medical institutes started utilizing digital data records for this data, Electronic Health Records (EHR’s). These records are widely used to study and analyze the patient histories and predict the care. Patient data stored in an Electronic Health Record(EHR) gives a clear overview of patient status, which can save a lot of time when they come back for the consultation. Machine learning may be useful for analysis because data is added to an EHR on a daily basis. To find trends and enhance predictions, machine learning algorithms use structured elements in EHR data, like as lab findings or electrocardiography measures. Clinical notes and other unstructured, high-dimensional, and sparse data are challenging to include into clinical machine learning models. Our objective is to develop a clinical note modeling framework that can generate medical forecasts and reveal clinical insights. There is substantial clinical utility in clinical notes. 

Throughout their hospital stay and admission history, a patient may be linked to hundreds of notes. With the time constraints , the clinicians working in the intensive care unit will have to go through a large volume of clinical notes  and it’s quite tough in making accurate clinical predictions. Making accurate clinical predictions may require reading a large volume of clinical notes. In practice, systems that generate precise predictions based on clinical notes may be helpful because this can reduce a doctor's burden. But there are challenges in data integration and utilizing the data for understanding and improving the care have not been realized. Additionally, to use this wide range of data for predictive insights, early detection or treatment optimisation haven’t fully been realized. 

For this project, we have utilized clinical notes with fine-tuned BERT models to analyze readmission factors and evaluate their performance in predicting Ischemic Heart Disease using metrics such as the confusion matrix and ROC curve. We additionally benchmarked ClinicalBERT against a vanilla BERT model, highlighting the superior performance of domain-specific embeddings. 

## Background

As discussed above, although the clinical notes are very useful for evaluation and they play a critical role for healthcare providers with the variable terminology and syntax, they do have challenges. ClinicalBERT is a domain-specific variant of BERT (Bidirectional Encoder Representations from Transformers) designed for clinical text processing. This is initially developed based on the pretrained BERT model by fine tuning on a large scale EHR data which includes clinical notes from the MIMIC-III dataset. This fine-tuning enables it to better understand the clinical language, including medical terminology, abbreviations, and contextual relationships unique to healthcare settings. ClinicalBERT is used for addressing challenges such as extracting patient details, predicting diagnoses, and improving classification tasks by leveraging the information embedded in unstructured clinical notes.  

In this study we have majorly highlighted Ischemic Heart Disease (IHD), also known as Coronary Artery Disease (CAD), which is a leading cause of mortality worldwide. It results from the narrowing of coronary arteries due to plaque buildup, reducing blood flow to the heart and potentially leading to angina, myocardial infarction, or heart failure. Accurate classification of IHD is critical for timely intervention and improved patient outcomes. Historically, IHD classification has relied on structured data from EHRs, such as laboratory results, imaging studies, and procedural records. 

In the past, research has largely focused on structured data, utilising machine learning and statistical techniques to analyse numerical and categorical variables. One of the researchers has demonstrated the efficacy of structured EHR data in predicting IHD outcomes. However, there is a lack of research into how clinical notes could complement or enhance these methods. Free-text data may encode subtle indicators of IHD that are absent in structured fields, offering opportunities to refine classification accuracy and deepen understanding of patient presentations. Our main objective is to demonstrate the potential use of clinical notes in IHD classification. By utilising clinical notes we would like to make the best use of the ClinicalBERT for predicting the IHD efficiently to demonstrate the use of clinical notes.

## Methodology

We conducted the experiment with the Hyperparameter Tuning, then designed a training pipeline  to evaluate the performance of BERT and ClinicalBERT. The Methodology is described as below:

### 4.1 Hyperparameter Tuning

We have used Optuna framework for the hyperparameter optimisation with the main focus on adjusting the learning rate and batch size using a validation set of measure performance. The effectiveness of different hyperparameter configurations was evaluated based on validation loss over a single epoch of training, ensuring a balanced trade-off between computational efficiency and model performance.

### 4.2 Training Pipeline

The training pipeline is mainly designed to process data, fine-tune ClinicalBERT and evaluate its performance. 

Major stages of the pipeline are as follows:
Data Collection: Selected ICD9 and notes tables from MIMIC-III using BigQuery
Data Preprocessing: Creation of IHD labels, processing text, etc.
Data Split: Split data into train, val, and test datasets with 2000, 1000, and 1000 patients respectively
Initialize Model: Initialized ClinicalBERT with a Linear Classification Head
Hyperparameter tuning: Determined best hyperparameters using Optuna
Fine Tuning:  Fine tuned the model on the training data for note-by-note prediction
Voting: Used a voting equation to create predictions on a patient-by-patient basis
Evaluation: Evaluated the model and compared it to a similar BERT-base and fine tuned the model.

### 4.3 Voting Equation

Once the model is fine tuned to for predictions on a note-by-note basis, predictions can be made patient-by-patient through use of a voting equation.
pmaxn +pmeann n/c1 + n/c 
The equation utilizes the max probability across patient notes in order to capture notes that highly indicate presence of IHD. It also incorporates the mean across notes to reduce risk of individual notes having too large of an impact. n (number of subsequences), and c (hyperparameter chosen), are used for scaling in order to account for how much weight the number of notes and the length of a subsequence should have.

### 4.4 Performance Evaluation

Our evaluation used several metrics to assess performance. We additionally compared the 
performance of ClinicalBERT with BERT-base to highlight the advantage of domain-specific pre-training for clinical text. 

Metrics included:  
Validation Loss: A key indicator of generalization performance during fine-tuning.  
Classification Accuracy: Measuring the correct prediction of IHD labels.  
Precision, Recall, and F1 Score: Capturing the balance between false positives and false negatives for a nuanced evaluation.  

## Dataset 

We use the Multiparameter Intelligent Monitoring in Intensive Care III (MIMIC-III) dataset hosted on PhysioNet for our model development and experiment. It consists of 46,520 patients in the intensive care unit of the Beth Israel Deaconess Medical Center (BIDMC) between 2001 and 2012, and it has 2,083,180 total clinical note events across the patients. The data in MIMIC-III has been de-identified, and it is freely available to researchers worldwide, it encompasses a diverse and very large population of ICU patients and it contains highly granular data, including vital signs, laboratory results, and medications. 




### 5.1 Dataset Description 

The clinical data managed in the dataset is as follows:
Progress notes by care providers
Nurse verified time-stamped physiological measurements (for example, hourly documentation of heart rate, arterial blood pressure, or respiratory rate)
Patient demographics and in-hospital mortality.
Laboratory test results (for example, hematology, chemistry, and microbiology results).
Discharge summaries and reports of electrocardiogram and imaging studies.
Billing-related information such as International Classification of Disease, 9th Edition (ICD-9) codes, Diagnosis Related Group (DRG) codes, and Current Procedural Terminology (CPT) codes.

MIMIC-III is a relational database consisting of 26 tables. We are specifically focusing on the following four key tables from the dataset:
Admissions: Contains comprehensive records of patient admissions, including admission times, discharge times, and demographic information such as age, ethnicity, and insurance type.This table helps us analyze admission trends, patient demographics, and the length of hospital stays, which are critical factors in assessing patient risk profiles.
Patients: Provides core patient information, including unique patient identifiers, date of birth, gender, and mortality status. By correlating patient demographics with admission data, we aim to study the influence of factors like age and gender on mortality rates and disease outcomes.
Diagnoses_icd: Contains detailed diagnostic information with ICD-9 (International Classification of Diseases, 9th Revision) codes assigned to each patient during their hospital stay. We can identify the most common diagnoses, tracking comorbidities, and this data will help explore patterns in disease prevalence across different patient demographics. We will be using it to validate predictive models that correlate diagnosis with patient outcomes and mortality.
Noteevents: Comprises unstructured clinical notes, including nursing notes, physician observations, discharge summaries and reports. Additionally, it includes metadata like the note category and timestamp. By analyzing clinical notes, we aim to derive contextual information that complements structured data, enhancing the prediction accuracy of our models.


### 5.2 Data Preprocessing

The clinical notes dataset is loaded into a DataFrame (df_notes) and sorted by SUBJECT_ID, HADM_ID, and CHARTDATE to ensure chronological order. Another dataset, containing ICD-9 diagnosis codes, is loaded into df_icd9, where missing codes are replaced with "Unknown."
Text Preprocessing:
Text is converted into lowercase, and newlines were replaced with spaces. Additionally, text preprocessing cleans the clinical notes by removing de-identified information, numerical prefixes, and medical abbreviations while standardizing terms like "Dr." to "doctor" and "M.D." to "md." This ensures that the data is well-labeled, cleaned, and structured, making it ready for further analysis or input into models like ClinicalBERT. Patient notes were then concatenated into 318-word chunks. To identify patients with Coronary Artery Disease (CAD), ICD-9 codes starting with specific prefixes are used. 
Labels Creation: 
The creation of labels for Coronary Artery Disease (CAD) involves identifying patients with ICD-9 codes that begin with 410, 411, 412, 413, or 414, which are associated with Ischemic Heart Disease.Patients with these codes are identified using their SUBJECT_ID, and a new column, HAS_CAD, is added to df_notes to indicate whether a patient has CAD.
Training Table:
Final training table is constructed, containing essential columns for the machine learning task. This table includes the preprocessed clinical text (cleaned and standardized using the preprocessing function), a label indicating the presence of CAD, and the corresponding patient ID. This organized structure ensures that the data is ready for training models, allowing for accurate prediction and analysis of CAD presence based on clinical notes.

### 5.3 Training Data And Test Data
Train, Val, and Test datasets were created by first selecting 4000 total patients, then splitting them into 2000 for training, 1000 for validation, and 1000 for testing. The datasets were then created by selecting all clinical notes corresponding to each of the 4000 patients.

## Results

We developed ClinicalBERT, a model of clinical notes whose representations can be used for clinical tasks. Before evaluating its performance as a model, we study its performance.
Here are the few results: 

#### Note-by-Note Evaluation (Before Voting)

ClinicalBERT achieved high recall (98.38%), its capability to identify potential IHD cases with minimal false negatives. However, the overall accuracy (47.55%) and precision (34.81%) indicated challenges in correctly classifying non-IHD cases due to noise and variability in individual notes. The resulting F1 score (51.42%) highlighted the trade-off between precision and recall.

Accuracy: 0.4755276564774381 <br>
Precision: 0.34808068901268135 <br>
Recall: 0.9838250308990274 <br>
F1 Score: 0.5142263292419178 <br>

#### Patient-by-Patient Evaluation [after Voting]

By aggregating predictions at the patient level [Table2], ClinicalBERT exhibited significantly improved performance, with accuracy rising to 87.57% and precision reaching 84.30%. The recall (69.12%) and F1 score (75.96%) suggest that patient-level aggregation mitigates variability in note-level predictions and better captures overall disease patterns.

Accuracy: 0.8756530825496343 <br>
Precision: 0.8430493273542601 <br>
Recall: 0.6911764705882353 <br>
F1 Score: 0.7595959595959596 <br>

### Below figures shows Precision, Recall and ROC curve before voting, with_BERT and with_ClincalBERT.
 
#### ClinicalBERT, Beforevoting:

Fig1a. Before Voting ROC Curve 

![ROC Curve Before Voting](https://github.com/CalvinYorkCS/ClinicalBERT--Fine-Tuning-For-IHD-Classification/blob/main/images/before_voting_roc_curve.png)

Fig1b. Before Voting Precision-Recall Curve

![Precision-Recall Curve Before Voting](https://github.com/CalvinYorkCS/ClinicalBERT--Fine-Tuning-For-IHD-Classification/blob/main/images/before_voting_precision_recall_curve.png)


#### BERT-base, After Voting: 
Fig2a. ROC Curve - BERT

![ROC Curve - BERT](https://github.com/CalvinYorkCS/ClinicalBERT--Fine-Tuning-For-IHD-Classification/blob/main/images/bert_base_roc_curve.png)

Fig2b. Precision Recall Curve - BERT

![Precision Recall Curve- BERT](https://github.com/CalvinYorkCS/ClinicalBERT--Fine-Tuning-For-IHD-Classification/blob/main/images/bert_base_precision_recall_curve.png)

#### ClinicalBERT, After Voting:
Fig3a. ROC Curve - ClinicalBERT

![ROC Curve- ClinicalBERT](https://github.com/CalvinYorkCS/ClinicalBERT--Fine-Tuning-For-IHD-Classification/blob/main/images/clinicalbert_roc_curve.png)


Fig3b. Precision Recall Curve - ClinicalBERT

![Precision Recall Curve- ClinicalBERT](https://github.com/CalvinYorkCS/ClinicalBERT--Fine-Tuning-For-IHD-Classification/blob/main/images/clinicalbert_precision_recall_curve.png)


## Discussion

Our project aimed to evaluate the effectiveness of clinicalBERT, a model pre-trained on clinical texts, in classifying Ischemic Heart Disease (IHD). The main findings indicate that clinicalBERT was able to learn valuable representations for IHD classification. This is in contrast to BERT-base, which showed limitations in generalizing to this specific clinical context.

### Detailed Description of Results
The results showed that clinicalBERT effectively captured the nuances of clinical language related to IHD, facilitating more accurate predictions compared to BERT-base. This was particularly evident in its ability to differentiate between complex cases that typically challenge general-purpose language models like BERT-base.

### Limitations of the Study 
Despite the promising outcomes, our study has limitations. The reliance on a single dataset and the potential biases inherent in clinical text sources could affect the generalizability of the results. Additionally, the model's performance could be limited by the quality of data preprocessing and the initial scope of training data, our initial training on 2000 patients, which took approximately 5.5 hours with an A100 on Google Colab, suggests that performance could be significantly enhanced by training on many more patients with superior resources.

### Future Directions for the Study
To address these limitations and enhance model robustness, future work will focus on implementing more advanced preprocessing techniques to prevent data leakage and refining the model's ability to process data post-IHD diagnosis. Expanding the dataset and utilizing better computational resources are also planned to further improve the model's accuracy and efficiency. Moreover, we intend to include logistic regression and Self Attention visualization in our comparative analysis and integrate both clinical notes and structured data for a more holistic approach.

Data Availability: The data used in this study is available for public access. https://physionet.org/content/mimiciii/1.4/


## Conclusion
This study demonstrates utilising ClinicalBERT to predict Ischemic Heart Disease (IHD) in patients using unstructured clinical notes. By fine-tuning ClinicalBERT on domain-specific text data, we were able to extract valuable insights. The performance comparison with BERT-base highlights the advantage of pretraining on clinical datasets, which enables ClinicalBERT to better understand the complexities of medical language. Our results suggest that incorporating clinical notes into predictive models offers a possibility for improving disease classification, complementing traditional methods that rely solely on structured data and manual efforts. Although the goal was not to surpass all existing methods, this study is to prove the potential of clinical notes in enhancing diagnostic precision.

## References
a. https://www.aimspress.com/article/doi/10.3934/mbe.2024058?viewType=HTML <br>
b. https://aclanthology.org/2024.findings-acl.916/ <br>
c. Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet. https://doi.org/10.13026/C2XW26.  <br>
d. https://github.com/nwams/ClinicalBERT-Deep-Learning--Predicting-Hospital-Readmission-Using-Transformer  <br>
e. https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT?text=Paris+is+the+%5BMASK%5D+of+France. <br>
f. https://arxiv.org/abs/1904.03323 <br>
