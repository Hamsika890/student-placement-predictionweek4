# Week 4 – Predictive Modeling and Evaluation

Project: Student Placement Prediction and Skill-Gap Analysis Using Python  
Prepared by: Hamsika Chinthakindi

## Files
- `src/train_model.py` – reproducible model training and evaluation script
- `notebooks/week4_modeling.ipynb` – step-by-step notebook
- `reports/Week4_Predictive_Modeling_Evaluation_Hamsika.docx` – Week 4 report
- `reports/week4_model_metrics.csv` – model comparison metrics
- `reports/week4_confusion_matrices.csv` – confusion matrix counts

## Required existing dataset
Keep the Week 2 raw dataset at:
`data/raw/student_placement_raw_data.csv`

## Run
From the repository root:
```bash
pip install -r requirements.txt
python src/train_model.py
```
Or open `notebooks/week4_modeling.ipynb` and run all cells from the repository root or from the `notebooks/` folder.

## Notes
The report's precision, recall and F1-score are for the `Not Placed` class. `salary_lpa` is excluded to avoid post-outcome leakage. `student_id` is an identifier and excluded. The test metrics are dataset-specific estimates, not guarantees.
