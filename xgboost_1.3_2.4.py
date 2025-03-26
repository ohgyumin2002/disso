import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from xgboost.sklearn import XGBClassifier
import warnings

# Base directory
base_dir = r"D:\20241129_solid_nN_1.3_2.4_mdck_siRNA_tnsfn_chlr"

# Define dataset paths with the correct nomenclature
datasets = {
    "1.3_chlr": base_dir + r"\20241129_solid_nN_1.3_mdck_chlr_dataset\solid_1.3_chlr_cell_level.csv",
    "1.3_tnsfn": base_dir + r"\20241129_solid_nN_1.3_mdck_tnsfn_dataset\solid_1.3_tnsfn_cell_level.csv",
    "2.4_chlr": base_dir + r"\20241129_solid_nN_2.4_mdck_chlr_dataset\solid_2.4_chlr_cell_level.csv",
    "2.4_tnsfn": base_dir + r"\20241129_solid_nN_2.4_mdck_tnsfn_dataset\solid_2.4_tnsfn_cell_level.csv"
}

# Define morphological and intensity features
cell_morph_features = [
    'area', 'perimeter', 'major_axis_length', 'minor_axis_length', 
    'eccentricity', 'circularity', 'solidity', 'orientation'
]

nuclear_morph_features = [
    'nuclear_area', 'nuclear_perimeter', 'nuclear_major_axis_length', 
    'nuclear_minor_axis_length', 'nuclear_eccentricity', 'nuclear_circularity', 
    'nuclear_solidity', 'nuclear_orientation'
]

# Define intensity feature suffixes
channel_feature_suffixes = [
    'intensity_p10', 'intensity_p25', 'intensity_p50', 
    'intensity_p75', 'intensity_p90'
]

# Define protein channels with caveolin as dominant feature
protein_channels = ['actin', 'caveolin', 'clathrin_hc', 'nuclei']

# Generate feature list with caveolin features first to ensure dominance
feature_list = cell_morph_features + nuclear_morph_features

# Add caveolin features first (to ensure dominance)
for suffix in channel_feature_suffixes:
    feature_list.append(f"caveolin_{suffix}")

# Add other channel features
for ch in protein_channels:
    if ch != 'caveolin':  # Skip caveolin as we already added it
        for suffix in channel_feature_suffixes:
            feature_list.append(f"{ch}_{suffix}")

def process_dataset(dataset_path, dataset_name, percentile_range):
    """Process dataset with specified percentile range for filtering"""
    print(f"\n=== Processing {dataset_name} with {percentile_range[0]}-{percentile_range[1]} percentile filtering ===")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Determine threshold based on dataset type
    if 'chlr' in dataset_name:
        intensity_threshold = 300  # CHLR threshold
    else:
        intensity_threshold = 400  # TNSFN threshold
    
    # Calculate area thresholds based on percentile range
    if percentile_range == (2, 98):
        cell_area_min, cell_area_max = 236, 1923
        nuclear_area_min, nuclear_area_max = 36, 327
    elif percentile_range == (5, 95):
        # Calculate 5th and 95th percentiles from the data
        cell_area_min = np.percentile(df['area'], 5)
        cell_area_max = np.percentile(df['area'], 95)
        nuclear_area_min = np.percentile(df['nuclear_area'], 5)
        nuclear_area_max = np.percentile(df['nuclear_area'], 95)
    
    # Keep only cells in area range
    df_cell_filtered = df[
        (df['area'] >= cell_area_min) & 
        (df['area'] <= cell_area_max) & 
        (df['siRNA_intensity_mean'] > intensity_threshold)
    ].copy()
    
    # Keep only nuclei in area range
    nuclei_threshold = (
        (df_cell_filtered['nuclear_area'] >= nuclear_area_min) & 
        (df_cell_filtered['nuclear_area'] <= nuclear_area_max)
    )
    
    # For rows where the nuclei are out of range, set nucleus area to NaN
    nuclear_cols = [col for col in df_cell_filtered.columns if col.startswith('nuclear_')]
    df_cell_filtered.loc[~nuclei_threshold, nuclear_cols] = np.nan
    
    # Convert target into categorical bins
    num_bins = 5
    df_cell_filtered['siRNA_category'] = pd.qcut(df_cell_filtered['siRNA_intensity_mean'], q=num_bins, labels=False)
    y = df_cell_filtered['siRNA_category']
    
    # Encode categorical target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X = df_cell_filtered[feature_list]
    features = list(X.columns)
    images = df_cell_filtered['image_id']
    
    # Outer CV: Stratified Group K-Fold
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store performance metrics
    outer_train_metrics, outer_test_metrics = [], []
    roc_auc_scores = []
    all_fpr = []
    all_tpr = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_encoded, groups=images), start=1):
        print(f"\n=== Outer Fold {fold} ===")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optuna Hyperparameter Tuning
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200)
            }
            
            model = xgb.XGBClassifier(random_state=42, **params)
            
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            inner_scores = []
            for inner_train_idx, inner_valid_idx in inner_cv.split(X_train_scaled, y_train):
                X_inner_train = X_train_scaled[inner_train_idx]
                X_inner_valid = X_train_scaled[inner_valid_idx]
                y_inner_train = y_train[inner_train_idx]
                y_inner_valid = y_train[inner_valid_idx]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_inner_train, y_inner_train)
                y_pred_inner = model.predict(X_inner_valid)
                score = accuracy_score(y_inner_valid, y_pred_inner)
                inner_scores.append(score)
                
            return np.mean(inner_scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        best_params = study.best_params
        print("Best parameters:", best_params)
        
        # Train final model
        best_model = xgb.XGBClassifier(random_state=42, **best_params)
        best_model.fit(X_train_scaled, y_train)
        
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)
        
        # Compute performance metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        print(f"Fold {fold} Training Metrics: Accuracy = {train_accuracy:.4f}, F1-score = {train_f1:.4f}")
        print(f"Fold {fold} Test Metrics: Accuracy = {test_accuracy:.4f}, F1-score = {test_f1:.4f}")
        
        outer_train_metrics.append((train_accuracy, train_f1, train_precision, train_recall))
        outer_test_metrics.append((test_accuracy, test_f1, test_precision, test_recall))
        
        # Calculate ROC curve and AUC for each class (one-vs-rest)
        y_test_bin = np.zeros((len(y_test), len(np.unique(y_test))))
        for i in range(len(np.unique(y_test))):
            y_test_bin[:, i] = (y_test == i).astype(int)
        
        y_score = best_model.predict_proba(X_test_scaled)
        
        # Calculate ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(np.unique(y_test))):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        all_fpr.append(fpr["micro"])
        all_tpr.append(tpr["micro"])
        roc_auc_scores.append(roc_auc["micro"])
        
        # Save the best model from the last fold
        if fold == 5:
            joblib.dump(best_model, f"best_xgb_model_{dataset_name}_{percentile_range[0]}_{percentile_range[1]}.pkl")
            
            # Generate SHAP values for feature importance
            try:
                scaler_final = StandardScaler()
                X_scaled = scaler_final.fit_transform(X)
                
                explainer = shap.Explainer(best_model, X_scaled)
                shap_values = explainer(X_scaled)
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_scaled, feature_names=features, show=False)
                plt.tight_layout()
                plt.savefig(f'shap_summary_{dataset_name}_{percentile_range[0]}_{percentile_range[1]}.png')
                plt.close()
            except Exception as e:
                print(f"Error generating SHAP plot: {e}")
    
    # Summarize overall performance
    metrics_names = ['Accuracy', 'F1-score', 'Precision', 'Recall']
    train_avg = np.mean(outer_train_metrics, axis=0)
    test_avg = np.mean(outer_test_metrics, axis=0)
    
    print("\n=== Overall Performance Across Folds ===")
    print("Training Metrics:")
    for i, name in enumerate(metrics_names):
        print(f" Mean {name}: {train_avg[i]:.4f}")
    
    print("Test Metrics:")
    for i, name in enumerate(metrics_names):
        print(f" Mean {name}: {test_avg[i]:.4f}")
    
    # Save overall performance metrics
    performance_metrics = {
        'Train Accuracy': train_avg[0], 'Train F1-score': train_avg[1],
        'Train Precision': train_avg[2], 'Train Recall': train_avg[3],
        'Test Accuracy': test_avg[0], 'Test F1-score': test_avg[1],
        'Test Precision': test_avg[2], 'Test Recall': test_avg[3],
        'Mean ROC AUC': np.mean(roc_auc_scores)
    }
    
    metrics_df = pd.DataFrame(list(performance_metrics.items()), columns=['Metric', 'Value'])
    metrics_df.to_csv(f'classification_performance_metrics_{dataset_name}_{percentile_range[0]}_{percentile_range[1]}.csv', index=False)
    print(f"Overall classification performance metrics saved to 'classification_performance_metrics_{dataset_name}_{percentile_range[0]}_{percentile_range[1]}.csv'.")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    
    # Plot average ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for i in range(len(all_fpr)):
        mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
    
    mean_tpr /= len(all_fpr)
    mean_auc = np.mean(roc_auc_scores)
    
    plt.plot(mean_fpr, mean_tpr, color='b', lw=2, 
             label=f'Mean ROC (AUC = {mean_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name} ({percentile_range[0]}-{percentile_range[1]} percentile)')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{dataset_name}_{percentile_range[0]}_{percentile_range[1]}.png')
    plt.close()
    
    return performance_metrics, mean_auc, mean_fpr, mean_tpr

# Process each dataset with both percentile ranges
results_2_98 = {}
results_5_95 = {}
roc_data = {}

for name, path in datasets.items():
    try:
        # Process with 2-98 percentile filtering
        results_2_98[name], auc_2_98, fpr_2_98, tpr_2_98 = process_dataset(path, name, (2, 98))
        
        # Process with 5-95 percentile filtering
        results_5_95[name], auc_5_95, fpr_5_95, tpr_5_95 = process_dataset(path, name, (5, 95))
        
        # Store ROC data for comparison
        roc_data[name] = {
            '2_98': {'auc': auc_2_98, 'fpr': fpr_2_98, 'tpr': tpr_2_98},
            '5_95': {'auc': auc_5_95, 'fpr': fpr_5_95, 'tpr': tpr_5_95}
        }
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Compare results between filtering methods
def compare_filtering_methods(results_2_98, results_5_95, roc_data):
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame()
    
    for dataset_name in results_2_98.keys():
        # Extract test metrics
        metrics_2_98 = {k: v for k, v in results_2_98[dataset_name].items() if k.startswith('Test')}
        metrics_2_98['ROC AUC'] = roc_data[dataset_name]['2_98']['auc']
        
        metrics_5_95 = {k: v for k, v in results_5_95[dataset_name].items() if k.startswith('Test')}
        metrics_5_95['ROC AUC'] = roc_data[dataset_name]['5_95']['auc']
        
        # Create DataFrame
        df_2_98 = pd.DataFrame({
            'Metric': list(metrics_2_98.keys()),
            f'{dataset_name} (2-98)': list(metrics_2_98.values())
        })
        
        df_5_95 = pd.DataFrame({
            'Metric': list(metrics_5_95.keys()),
            f'{dataset_name} (5-95)': list(metrics_5_95.values())
        })
        
        if comparison_df.empty:
            comparison_df = pd.merge(df
