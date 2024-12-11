import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def train_xgboost_models(xgboost_train_data_df, psu_grps, grp_leads):
    models_dict = {psu_occ: '' for psu_occ in grp_leads}
    xgboost_train_data_df = xgboost_train_data_df.drop('force_id', axis=1)
    for glkeep in grp_leads:
        drop_cols = [grp for grp in psu_grps if glkeep in grp][0]
        drop_cols = [col for col in drop_cols if col in xgboost_train_data_df.columns]
        # 1. Prepare the data
        # Assuming `train_xgboost_df` has features and a target column named 'outcome'
        xgb_df_curr = xgboost_train_data_df[~xgboost_train_data_df[glkeep].isna()]
        X = xgb_df_curr.drop(columns=drop_cols)
        y = xgb_df_curr[glkeep]
        y = y.replace({99: 0})

        # 3. Initialize and Train the XGBoost Model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,  # Number of trees
            learning_rate=0.1,  # Step size shrinkage
            max_depth=3,  # Maximum depth of a tree
            random_state=42,
            use_label_encoder=False,  # Disable label encoding (deprecated in XGBoost)
            eval_metric="logloss"  # Evaluation metric
        )

        xgb_model.fit(X, y)

        models_dict[glkeep] = xgb_model

    return models_dict


def validate_xgboost_models(xgboost_train_data_df, psu_grps, grp_leads):
    tr_results = {psu_occ: {'acc': '', 'auc': '', 'class_report': ''} for psu_occ in grp_leads}
    xgboost_train_data_df = xgboost_train_data_df.drop('force_id', axis=1)
    for glkeep in grp_leads:
        drop_cols = [grp for grp in psu_grps if glkeep in grp][0]
        drop_cols = [col for col in drop_cols if col in xgboost_train_data_df.columns]
        # 1. Prepare the data
        # Assuming `train_xgboost_df` has features and a target column named 'outcome'
        xgb_df_curr = xgboost_train_data_df[~xgboost_train_data_df[glkeep].isna()]
        X = xgb_df_curr.drop(columns=drop_cols)
        y = xgb_df_curr[glkeep]
        y = y.replace({99: 0})

        # 2. Train-Test Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 3. Initialize and Train the XGBoost Model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,  # Number of trees
            learning_rate=0.1,  # Step size shrinkage
            max_depth=3,  # Maximum depth of a tree
            random_state=42,
            use_label_encoder=False,  # Disable label encoding (deprecated in XGBoost)
            eval_metric="logloss"  # Evaluation metric
        )

        xgb_model.fit(X_train, y_train)

        # 4. Make Predictions
        y_pred = xgb_model.predict(X_val)
        y_pred_proba = xgb_model.predict_proba(X_val)[:, 1]  # Probability of the positive class

        # 5. Evaluate the Model
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        class_report = classification_report(y_val, y_pred)
        tr_results[glkeep]['acc'] = accuracy
        tr_results[glkeep]['auc'] = auc
        tr_results[glkeep]['class_report'] = class_report

    return tr_results

def format_xgboost_results_to_prompt(accuracy, auc, class_report, tst_pt_pred):
    """
    Format XGBoost model results into a report for inclusion in a prompt.

    Parameters:
        accuracy (float): Model accuracy on the validation set.
        auc (float): Area under the ROC curve for the validation set.
        class_report (dict): Classification report as a dictionary.
        tst_pt_pred (str): Prediction for the test patient (e.g., 'Yes' or 'No').

    Returns:
        str: A formatted report as a string.
    """

    # Construct the prompt
    # report = (
    #     f"Additionally, an XGBoost model estimates the predicted probability of the occluded outcome being 'yes' to be: {tst_pt_pred}.\n"
    #     f"This prediction is based on a model trained and validated on the set of clinical features you were given. "
    #     f"The dataset includes a similar cohort of patients, but not the patient in question.\n"
    #     f"Below are the results obtained on the similar cohort (excluding the test patient)\n"
    #     f"Accuracy on Validation Set: {accuracy:.2f}\n"
    #     f"AUC (ROC Curve): {auc:.2f}\n\n"
    #     f"Classification Report:\n{class_report}\n\n")

    report = (
        f"Additionally, an XGBoost model estimates the predicted probability of the occluded outcome being 'yes' to be: {tst_pt_pred}.\n"
    f"This prediction is based on a model trained and validated on the set of clinical features you were given. "
    f"The dataset includes a similar cohort of patients, but not the patient in question.\n\n"
    f"The XGBoost model performs very well so take it's prediction with a lot of weight. "
    f"Specifically, it achieved the following performance metrics on the validation set (excluding the test patient):\n"
    f" - Accuracy: {accuracy:.2f}\n"
    f" - Area Under the Receiver Operating Characteristic (AUC-ROC) Curve: {auc:.2f}\n\n"
    f"In this model, a predicted value of 0 indicates the model predicts the occluded outcome did not occur for a given patient, "
    f"while a predicted value of 1 indicates the model predicts it did occur.\n\n"
    f"Classification Report:\n{class_report}\n\n"
    )




    return report


def fetch_xgboost_report(xgboost_test_data_df, models_dict, models_results_dict, psu_grps, grp_leads):
    output_dict = {'force_id': [], 'psu_occlusion': [], 'xgboost_report': []}
    for force_id in xgboost_test_data_df.force_id:
        xgb_fid_test_curr = xgboost_test_data_df[xgboost_test_data_df['force_id'] == force_id]
        xgb_fid_test_curr = xgb_fid_test_curr.drop('force_id', axis=1)
        for glkeep in grp_leads:
            model_curr = models_dict[glkeep]
            model_results_curr = models_results_dict[glkeep]
            tr_acc = model_results_curr['acc']
            tr_auc = model_results_curr['auc']
            tr_class_report = model_results_curr['class_report']

            drop_cols = [grp for grp in psu_grps if glkeep in grp][0]
            drop_cols = [col for col in drop_cols if col in xgb_fid_test_curr.columns]
            # 1. Prepare the data
            xgb_test_df_curr = xgb_fid_test_curr.copy()
            xgb_test_df_curr[glkeep] = xgb_test_df_curr[glkeep].fillna(0)
            X_test = xgb_test_df_curr.drop(columns=drop_cols)
            y_pred_proba = model_curr.predict_proba(X_test)[:, 1]  # Probability of the positive class

            report_curr = format_xgboost_results_to_prompt(tr_acc, tr_auc, tr_class_report, y_pred_proba)

            output_dict['force_id'].append(force_id)
            output_dict['psu_occlusion'].append(glkeep)
            output_dict['xgboost_report'].append(report_curr)

    return output_dict





