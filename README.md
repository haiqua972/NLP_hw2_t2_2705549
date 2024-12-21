# Task 2: Governing vs. Opposition Party Classification
# 1. Project Overview
This project focuses on Task 2 from the shared task on "Ideology and Power Identification in Parliamentary Debates" which involves classifying whether a speaker’s party is currently governing (label 0) or in opposition (label 1). Using a set of parliamentary speeches extracted from the ParlaMint corpus, this project explores both fine-tuning a multilingual BERT model and employing a zero-shot inference strategy using Llama-3.1-8B.
# 2. Dataset
The dataset includes parliamentary speeches from multiple countries, each labeled according to whether the speaker belongs to a governing party or an opposition party. For this task, we focused exclusively on data from a single, non-English speaking country to challenge the model's ability to handle language-specific nuances.
# 3. Model Training and Evaluation
## 3.1.	Fine-Tuning Approach:
*	**Model:** We used BertForSequenceClassification with a multilingual BERT base, fine-tuned for binary classification.
*	**Training Details:**
Batch Size: 16, Epochs: 3, Optimizer: AdamW with a learning rate of 2e-5, Loss Function: Cross-Entropy
*	The model was trained on both the translated English text and the original language text of the speeches to compare performance across different linguistic inputs.
## 3.2.	Zero-Shot Inference:
*	**Model:** Llama-3.1-8B
*	Employed without any fine-tuning, this model was used to infer the classification based on its pre-trained knowledge, leveraging its multilingual capabilities.
# 4. Data Preprocessing
Data was tokenized using BERT tokenizer and split into training and testing datasets with a 90/10 split, maintaining a stratified distribution of labels to handle class imbalances.
# 5. Experimental Results
The models were evaluated based on accuracy, precision, recall, and F1 score:
## 5.1. Fine-tuned models:
Fine-tuned models showed varied performance, with the English text model generally performing better than the model trained on original language texts.

![image](https://github.com/user-attachments/assets/dd61b6a8-49c6-48b5-85d3-1b718081209c)

## 5.2. Zero-shot inference:
Zero-shot inference provided a baseline to understand how well the causal model could generalize without specific training, with insightful outcomes in cross-lingual contexts.

![image](https://github.com/user-attachments/assets/4b24a3a8-0bcd-4db2-afc7-021f1c3c4be0)

# 6. Comparative Analysis
Comparative results highlighted the strengths and weaknesses of both approaches:
* Fine-tuned models excelled in accuracy and adaptability to specific linguistic features.
*	Zero-shot models demonstrated robust generalization but were limited in precision and recall, indicating potential improvements in context understanding.
# 7. Discussion and Improvements
The imbalance in the dataset posed challenges, particularly in achieving high precision and recall. Future work could explore more sophisticated balancing techniques, deeper linguistic feature engineering, or the use of ensemble methods to enhance model robustness. Further experimentation with different language-specific BERT models might also yield improvements.
# 8. Conclusion
This project underscored the potential and challenges of using advanced NLP techniques for political discourse analysis. While the fine-tuned models performed well on trained linguistic contexts, the zero-shot approach offered valuable insights into model generalization across languages.
# 9. Usage
## 9.1. Prerequisites
Install the required libraries:

!pip install transformers datasets torch sklearn
## 9.2. Steps

1.	**Upload Dataset:** Place your dataset in Google Drive and set the file path.
  
2.	**Run the Notebook:** Execute the script step-by-step in a Python/Colab notebook.
## 9.3. File Structure
*	**dataset/:** Contains the parliamentary speeches dataset (not included in this repo for privacy).
*	**scripts/:** Python scripts for fine-tuning, evaluation, and inference.
*	**results/:** Generated metrics and evaluation logs.
# 10. Contact
Haiqua Meerub: haiquameerub972@gmail.com


