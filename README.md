# AutoJudge — Predicting Programming Problem Difficulty

AutoJudge is a machine learning system that predicts the difficulty of programming problems using only their textual descriptions. It outputs both a difficulty class (**Easy**, **Medium**, **Hard**) and a difficulty score on a **1–10 scale**. The system is trained using the provided dataset (`problems_data.jsonl`) and includes a simple web interface built using **Flask** for demonstration purposes.

## Dataset

Each problem in the provided dataset contains:
- Problem description  
- Input description  
- Output description  
- Difficulty class label  
- Difficulty score  

No external data sources are used.

## Preprocessing and Feature Extraction

For preprocessing, missing values are handled and all text fields are combined into a single input. Feature extraction includes:
- TF-IDF vectors  
- Text length  
- Mathematical symbol count  
- Keyword frequencies (e.g., dp, graph, recursion)

## Models and Evaluation

Various models were evaluated using an **80/20 train–test split**.

### Classification
- Logistic Regression  
- Support Vector Machine (LinearSVC)  
- Random Forest  

**Final classifier:** Linear SVM

### Regression
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

**Final regressor:** Gradient Boosting Regressor

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```
Run the web application:
```bash
python app.py
```
Open in browser:
```cpp
http://127.0.0.1:5000
```