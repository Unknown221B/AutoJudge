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

For preprocessing, missing values are handled and all text fields are combined internally into a single input. Feature extraction includes:
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

**Final classifier:** Random Forest  
**Accuracy:** 54.31%

**Confusion Matrix:**
```text
[[ 32  66  38]
 [ 11 374  40]
 [ 19 202  41]]
```

From the matrix we can conclude that **Medium** problems are most often classified correctly, although there is a noticeable overlap between Medium and Hard classes. Thus there is an inherent ambiguity in estimating difficulty levels using text-only features.

### Regression
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

**Final regressor:** Random Forest Regressor  
**Mean Absolute Error (MAE):** 1.69  
**Root Mean Squared Error (RMSE):** 2.05

> **Note:**  The difficulty class and difficulty score are predicted independently using separate models, and no explicit numeric thresholds were defined between difficulty classes.

## How to Run

Clone the repository:
```bash
git clone <https://github.com/Unknown221B/AutoJudge.git>
cd AutoJudge
```

Install dependencies:
```bash
pip install -r requirements.txt
```
> **Note:**  If the above command does not work, use:
> 
> ```bash
> python -m pip install -r requirements.txt
> ```

Run the web application:
```bash
python app.py
```
Open in browser:
```cpp
http://127.0.0.1:5000
```
## Demo
The video file is included in the repository.  
[Repository Link](./AutoJudge%20Demo%20Video%20Final.mp4)  
[Google Drive Link](https://drive.google.com/file/d/1xGW8AZBnOVZjP3yMnH7o1lIWCwc5vAHK/view?usp=sharing)  

## Project Report
A detailed report on the entire project is given here:  
[Project Report link](./AutoJudge%20Report.pdf)   
