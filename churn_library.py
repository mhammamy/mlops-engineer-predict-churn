
'''
Churn Prediction Library

This module provides functions for predicting customer churn, including:
1. Importing data
2. Exploratory data analysis (EDA)
3. Encoding categorical variables
4. Feature engineering
5. Generating classification reports
6. Plotting feature importance
7. Training and evaluating models

Functions:
---------
- import_data(pth): Reads a CSV file and returns a DataFrame.
- perform_eda(df): Performs EDA and saves figures.
- encoder_helper(df, category_lst, response): Encodes categorical features.
- perform_feature_engineering(df, response): Splits data into train and test sets.
- classification_report_image(y_train, y_test, y_train_preds, y_test_preds):
Saves classification report images.
- feature_importance_plot(model, X_data, output_pth): Plots and saves feature importance.
- train_models(X_train, X_test, y_train, y_test): Trains, evaluates, and saves models.
'''


# import libraries

import os
import joblib  # Use joblib for model persistence
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set_theme()

plot_roc_curve = RocCurveDisplay.from_estimator


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

FILE_PATH = r"./data/bank_data.csv"

MODEL_DIR = './models/'

EDA_OUTPUT_DIR = 'images/eda'

RESULTS_OUTPUT_DIR = 'images/results'

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    print("Loading data...")

    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def _plot_and_save_histogram(df, column, output_dir):
    '''Plot and save histograms'''
    plt.figure(figsize=(20, 10))
    df[column].hist()
    plt.savefig(os.path.join(output_dir, f'{column.lower()}_histogram.png'))
    plt.close()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    print("Performing EDA...")

    # Create the directory if it doesn't exist
    output_dir = EDA_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for column in (
        'Churn',
        'Customer_Age',
        'Marital_Status',
            'Total_Trans_Ct'):
        _plot_and_save_histogram(df, column, output_dir)

    # Plot and save the correlation heatmap
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(20, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()


def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    print("--- Encoding dataframe...")

    df = df.copy()

    for category in category_lst:
        encoded_column = []
        gender_groups = df.groupby(category)[response].mean()

        for val in df[category]:
            encoded_column.append(gender_groups.loc[val])

        df[f'{category}_{response}'] = encoded_column

    return df


def perform_feature_engineering(df, category_lst, response='Churn'):
    '''
    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name
        [optional argument that could be used for naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

    print("Performing feature engineering...")

    df = encoder_helper(df, category_lst, response=response)

    y = df[response]
    X = df[KEEP_COLS]  # pylint: disable=invalid-name
    X_train, X_test, y_train, y_test = train_test_split(  # pylint: disable=invalid-name
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def _save_classification_report(report, report_title):
    # Create the directory if it doesn't exist
    output_dir = RESULTS_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 2.2))
    plt.text(0.01, 1, report, {'fontsize': 12}, fontfamily='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir, f'{
                report_title.lower().replace(
                    ' ', '_')}.png'))
    plt.close()


def classification_report_image(  # pylint: disable=too-many-arguments
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf
):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    print("--- Generating classification reports...")

    for y, y_preds, report_name in (
        (y_test, y_test_preds_rf, 'Random Forest Classification Report'),
        (y_train, y_train_preds_rf, 'Random Forest Training Classification Report'),
        (y_test, y_test_preds_lr, 'Logistic Regression Classification Report'),
        (y_train, y_train_preds_lr, 'Logistic Regression Training Classification Report'),
    ):
        _save_classification_report(
            classification_report(y, y_preds), report_name
        )


def feature_importance_plot(model, X_data, output_dir=RESULTS_OUTPUT_DIR):  # pylint: disable=invalid-name
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_dir: dir to store the figure

    output:
             None
    '''

    print("--- Generating feature importance plot...")

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()


def train_models(  # pylint: disable=too-many-locals
    X_train,  # pylint: disable=invalid-name
    X_test,  # pylint: disable=invalid-name
    y_train,
    y_test,
    use_saved_models=False
):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search

    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

    print("Training/Loading models...")

    model_dir = MODEL_DIR
    rfc_model_path = os.path.join(model_dir, 'rfc_model.pkl')
    lrc_model_path = os.path.join(model_dir, 'logistic_model.pkl')

    # Check if the Random Forest model exists
    if use_saved_models and os.path.exists(rfc_model_path):
        print("--- Loading existing Random Forest model...")
        rfc = joblib.load(rfc_model_path)
    else:
        print("Training new Random Forest model...")
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['log2', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        rfc = RandomForestClassifier(random_state=42)
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        rfc = cv_rfc.best_estimator_
        joblib.dump(rfc, rfc_model_path)

    # Check if the Logistic Regression model exists
    if use_saved_models and os.path.exists(lrc_model_path):
        print("--- Loading existing Logistic Regression model...")
        lrc = joblib.load(lrc_model_path)
    else:
        print("--- Training new Logistic Regression model...")
        lrc = LogisticRegression(solver='liblinear', max_iter=5000)
        lrc.fit(X_train, y_train)
        joblib.dump(lrc, lrc_model_path)

    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    X = pd.concat(  # pylint: disable=invalid-name
        [X_train, X_test],  # pylint: disable=invalid-name
        ignore_index=True
    )
    feature_importance_plot(rfc, X, output_dir=RESULTS_OUTPUT_DIR)


if __name__ == "__main__":
    custumers_df = import_data(FILE_PATH)
    perform_eda(custumers_df)
    (custumers_X_train,
     custumers_X_test,
     custumers_y_train,
     custumers_y_test) = perform_feature_engineering(custumers_df,
                                                     CAT_COLUMNS,
                                                     response='Churn')
    train_models(
        custumers_X_train,
        custumers_X_test,
        custumers_y_train,
        custumers_y_test,
        use_saved_models=True)
