import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================================================================
# Configuration Constants
# ==============================================================================
# Path constants are crucial for accessing files in the defined project structure:
# Root/
#   ├── dataset/
#   │   └── Iris.csv
#   ├── src/
#   │   └── main.py (current file)
#   └── output/ (will be created)

# Determine the absolute path of the directory containing main.py ('src')
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# Path to the data file: step up from 'src' then go into 'dataset'
DATA_FILE_PATH = os.path.join(os.path.dirname(ROOT_DIR), "dataset", "Iris.csv")

# Define the root output directory path
OUTPUT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "output")
# Define subdirectories for different types of output
TREE_DIR = os.path.join(OUTPUT_DIR, "decision_trees")
PLOT_DIR = os.path.join(OUTPUT_DIR, "box_plots")
# NEW DIRECTORY: for CSV summary files
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary_metrics") 

N_SPLITS = 6  # Required number of folds for cross-validation (Assignment 1)
SEED = 42     # Random seed for reproducibility of data split and tree training

# ==============================================================================
# 1. Data Loading and Preparation
# ==============================================================================
def load_and_prepare_data():
    """
    Loads the Iris dataset, handles potential FileNotFoundError, and separates
    data into features (X), labels (y), and metadata (names).
    """
    print(f"Attempting to load data from: {DATA_FILE_PATH}")
    try:
        df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}. Please check the path.")
        return None, None, None, None

    # Prepare features (X): drop the target ('Species') and unnecessary identifier ('Id')
    X = df.drop(columns=["Species", "Id"])
    # Prepare labels (y): the target variable
    y = df["Species"]
    
    class_names = y.unique()
    feature_names = X.columns.tolist()
    
    print(f"Data loaded successfully. Total samples: {len(df)}")
    print(f"Features: {feature_names}, Classes: {class_names}")
    return X, y, class_names, feature_names

# ==============================================================================
# 2. Cross-Validation, Training, and Evaluation
# ==============================================================================
def run_cross_validation(X, y, feature_names):
    """
    Executes the 6-fold stratified cross-validation process.
    Trains the Decision Tree, saves the visualization, calculates metrics, 
    and records all test set predictions.
    """
    # Initialize Stratified K-Fold for stratified sampling (Assignment 1 requirement)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    fold_results = []
    all_predictions = [] # Stores predictions for summary output (Assignment 3 requirement)
    
    # Create necessary output directories
    os.makedirs(TREE_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True) 
    
    print(f"\nStarting {N_SPLITS}-Fold Stratified Cross-Validation...")

    # Iterate through each of the 6 folds
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Processing Fold {fold}/{N_SPLITS} ---")
        
        # Split data for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 2. Construct a decision tree using the training data subset
        dt_classifier = DecisionTreeClassifier(random_state=SEED)
        dt_classifier.fit(X_train, y_train)
        
        # Save Decision Tree Visualization (Assignment 2 requirement)
        plt.figure(figsize=(18, 12))
        plot_tree(dt_classifier, 
                  feature_names=feature_names, 
                  class_names=dt_classifier.classes_, 
                  filled=True, rounded=True, fontsize=10)
        tree_filename = os.path.join(TREE_DIR, f"DecisionTree_Fold{fold}.png")
        plt.title(f"Decision Tree - Fold {fold}", fontsize=14)
        plt.savefig(tree_filename)
        plt.close()
        print(f"Decision tree visualization saved to {tree_filename}")
        
        # 3. Predict the class labels of the instances in the test set
        y_pred = dt_classifier.predict(X_test)
        
        # Record the predicted labels (Assignment 3 requirement)
        for idx, actual, predicted in zip(test_index, y_test.tolist(), y_pred):
            all_predictions.append({
                'Sample_Index_in_Original_Dataset': idx,
                'Actual_Label': actual,
                'Predicted_Label': predicted,
                'Fold': fold
            })

        # Compute Overall Accuracy (Assignment 3 metric)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Compute class-wise metrics (Precision, Recall, Specificity) (Assignment 3 metrics)
        cm = confusion_matrix(y_test, y_pred, labels=dt_classifier.classes_)
        fold_metrics = {'Fold': fold, 'Accuracy': accuracy}
        
        # Calculate class-wise metrics
        for i, cls in enumerate(dt_classifier.classes_):
            # Calculate components from the 3x3 confusion matrix
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP 
            FP = cm[:, i].sum() - TP 
            TN = cm.sum() - (TP + FN + FP)
            
            # Precision: TP / (TP + FP)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            
            # Recall (Sensitivity): TP / (TP + FN)
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            
            # Specificity: TN / (TN + FP)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

            fold_metrics[f'{cls}_Precision'] = precision
            fold_metrics[f'{cls}_Recall'] = recall
            fold_metrics[f'{cls}_Specificity'] = specificity
        
        fold_results.append(fold_metrics)
        
    results_df = pd.DataFrame(fold_results)
    
    # Finalize and save the predicted labels summary (CSV saved to 'summary_metrics' folder)
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.sort_values(by=['Sample_Index_in_Original_Dataset'], inplace=True)
    predictions_df.to_csv(os.path.join(SUMMARY_DIR, "predicted_labels_summary.csv"), index=False)
    print("\nPredicted labels summary saved to the 'output/summary_metrics' folder.")
    
    return results_df

# ==============================================================================
# 3. Quantitative Analysis and Plotting
# ==============================================================================
def perform_analysis_and_plot(results_df, class_names):
    """
    Performs the final quantitative assessment by calculating mean/variance 
    and generating the required box plots (Assignment 4b).
    """
    
    # Calculate the mean and variance of each metric (Assignment 4b)
    mean_metrics = results_df.drop('Fold', axis=1).mean().rename('Mean')
    variance_metrics = results_df.drop('Fold', axis=1).var().rename('Variance')
    analysis_df = pd.concat([mean_metrics, variance_metrics], axis=1)
    
    # Save metrics summary to the 'summary_metrics' folder
    analysis_df.to_csv(os.path.join(SUMMARY_DIR, "metrics_summary.csv"))

    print("\n" + "="*50)
    print("             Quantitative Assessment")
    print("="*50)
    print("\n--- Mean and Variance of Performance Metrics (Assignment 4b) ---")
    print(analysis_df.to_string())

    # Draw box plots for class-wise metrics (Assignment 4b)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Prepare data for box plotting (long format required for Seaborn 'hue')
    class_metrics_data = []
    for _, row in results_df.iterrows():
        for name in class_names:
            class_metrics_data.append({'Class': name, 'Metric': 'Precision', 'Value': row[f'{name}_Precision']})
            class_metrics_data.append({'Class': name, 'Metric': 'Recall', 'Value': row[f'{name}_Recall']})
            class_metrics_data.append({'Class': name, 'Metric': 'Specificity', 'Value': row[f'{name}_Specificity']})

    metrics_plot_df = pd.DataFrame(class_metrics_data)

    plt.figure(figsize=(12, 6))
    # Box plots summarizing metrics for each class separately (required visualization)
    sns.boxplot(x='Metric', y='Value', hue='Class', data=metrics_plot_df)
    plt.title('Box Plots of Class-wise Performance Metrics Across 6 Folds')
    plt.ylabel('Metric Value')
    plt.xlabel('Metric Type')
    plt.legend(title='Iris Class')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plot_filename = os.path.join(PLOT_DIR, 'BoxPlots_Metrics.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"\nBox plot visualization saved to {plot_filename}")

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
def main():
    """Main entry point to orchestrate the assignment tasks."""
    
    print("--- Lab Assignment 1: Decision Tree Cross-Validation ---")
    
    # Step 1: Load and prepare data
    X, y, class_names, feature_names = load_and_prepare_data()
    
    if X is None:
        return

    # Step 2 & 3: Run cross-validation, train models, compute metrics, and record predictions
    results_df = run_cross_validation(X, y, feature_names)

    # Step 4: Perform quantitative analysis and plotting
    perform_analysis_and_plot(results_df, class_names)
    
    # Provide guidance for the required qualitative analysis
    print("\n" + "="*50)
    print("           Qualitative Assessment Guide")
    print("="*50)
    print("Action Required: Qualitatively assess how different the decision trees are (Assignment 4a).")
    print(f"Review the six image files in the '{TREE_DIR}' directory and comment on:")
    print("1. Root feature selection in each fold.")
    print("2. Tree depth and complexity.")
    print("3. Variations in critical split points (thresholds).")
    
    print("\n*** Submission Checklist ***")
    print("Ensure the following files are submitted:")
    print("1. Implementation Document (summary and analysis).")
    print("2. This code file (src/main.py).")
    print("3. Decision Tree images (in output/decision_trees/).")
    print("4. Box Plots image (in output/box_plots/).")
    print("5. Metrics summary (in output/summary_metrics/metrics_summary.csv).")
    print("6. Predicted labels summary (in output/summary_metrics/predicted_labels_summary.csv).")

if __name__ == "__main__":
    # Standard practice to ensure 'main()' runs when the script is executed directly
    main()
