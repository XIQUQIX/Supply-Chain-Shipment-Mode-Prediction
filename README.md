# Supply Chain Shipment Mode Prediction

## Abstract
This project develops a machine learning model to predict the shipment mode (Air, Truck, Air Charter, Ocean) for supply chain deliveries using a dataset with features like country, product details, and delivery metrics. Implemented in Python using TensorFlow, the model achieves an accuracy of 87.93% on the test set. The project includes data preprocessing, feature engineering, neural network training, and performance evaluation with a confusion matrix and classification report.

## Project Overview
This Jupyter Notebook (`trial.ipynb`) implements a machine learning pipeline to predict the shipment mode in a supply chain dataset (`Delivery_Dataset.csv`). The dataset contains 33 features, including country, product group, delivery dates, and costs. The goal is to classify deliveries into four shipment modes: Air, Truck, Air Charter, or Ocean. The project uses a neural network built with TensorFlow/Keras, achieving a test accuracy of 87.93%.

## Dataset
- **Source**: `input/Delivery_Dataset.csv` (not included; assumed to be a supply chain dataset, e.g., from Kaggle).
- **Features**: 33 columns, including `Country`, `Product Group`, `Line Item Quantity`, `Line Item Value`, `Shipment Mode` (target), and others.
- **Target**: `Shipment Mode` (4 classes: Air, Truck, Air Charter, Ocean).
- **Size**: ~10,000 rows (after preprocessing: 6,974 training, 2,990 testing).

## Requirements
To run the notebook, install the following Python libraries:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
```
- Python version: 3.10 or higher.
- Ensure the dataset file (`Delivery_Dataset.csv`) is placed in the `input/` directory.

### Project Structure
- `input/`: Directory containing the dataset (`Delivery_Dataset.csv`).
- `trial.ipynb`: Main Jupyter Notebook with the end-to-end pipeline.
- Outputs: Confusion matrix plot and classification report printed in the notebook.

### Workflow
1. **Data Loading**: Load the CSV dataset using Pandas.
2. **Preprocessing**:
   - Drop irrelevant columns (e.g., `ID`, high-cardinality columns like `PQ #`).
   - Handle missing values: Fill `Dosage` with mode, `Line Item Insurance` with mean, drop rows with missing `Shipment Mode`.
   - Feature engineering: Extract year/month/day from date columns, apply binary and one-hot encoding for categorical features.
   - Split data: 70% training, 30% testing.
   - Scale features using `StandardScaler`.
3. **Modeling**:
   - Build a neural network with two hidden layers (128 neurons each, ReLU activation) and a softmax output layer (4 classes).
   - Compile with Adam optimizer and sparse categorical cross-entropy loss.
   - Train with early stopping (patience=3) for up to 100 epochs.
4. **Evaluation**:
   - Test accuracy: 87.93%.
   - Generate confusion matrix and classification report to analyze per-class performance (e.g., Air: 0.90 F1-score; Ocean: 0.73 F1-score).
5. **Visualization**: Plot a confusion matrix using Seaborn.

## Results
- **Test Accuracy**: 87.93%.
- **Classification Report**:
  - Air: Precision=0.90, Recall=0.90, F1=0.90 (1,796 samples).
  - Truck: Precision=0.87, Recall=0.86, F1=0.86 (880 samples).
  - Air Charter: Precision=0.84, Recall=0.83, F1=0.84 (201 samples).
  - Ocean: Precision=0.75, Recall=0.71, F1=0.73 (113 samples).
- **Observation**: The model performs well on Air and Truck but struggles with Ocean due to class imbalance (fewer samples).


## Potential Improvements
- **Class Imbalance**: Use SMOTE or class weights to improve performance on minority classes (e.g., Ocean).
- **Feature Engineering**: Include external data (e.g., weather, geopolitical risks) or derive new features.
- **Model Alternatives**: Try Random Forest or XGBoost for comparison.
- **Hyperparameter Tuning**: Optimize neural network architecture or training parameters.

## Notes
- The dataset may have inconsistencies (e.g., date formats), which are handled by the preprocessing function.
- The high feature dimensionality (771 after one-hot encoding) may benefit from dimensionality reduction (e.g., PCA).
- For reproducibility, the random seed is set to 1 in the train-test split.

## License
This project is for educational purposes and uses open-source libraries. Ensure the dataset's license allows use for your purposes.
