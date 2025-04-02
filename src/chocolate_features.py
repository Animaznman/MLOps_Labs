import pandas as pd
import numpy as np 
import pickle
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split


# Given a data, oriented around the flavors_of_cacao.csv,
# process the data into something usable.
def process_data(data, target):
    df = pd.read_csv(data)
    df.columns = df.columns.str.strip().str.replace('"', '', regex=False).str.replace(' ', '_', regex=False).str.lower()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['cocoa_percent'] = df['cocoa_percent'].apply(lambda x: x.split('%')[0])

    
    # Going to try out Robert's stuff here
    # encoder = OneHotEncoder(sparse_output = False)
    # object_df = data.select_dtypes(include = ["object"])
    # encoded = encoder.fit_transform(object_df)
    # encoded_df = pd.DataFrame(encoded, columns = encoder.get_feature_names_out(object_df.columns))
    # merge_encoded_df = pd.concat([encoded_df, data.select_dtypes(exclude['object'])], axis = 1)
    y = df['rating']
    X = df.drop(columns = ['rating'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    # Create pipeline for imputing and scaling numeric variables
    # one-hot encoding categorical variables, and select features based on chi-squared value
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, make_column_selector(dtype_include=["int", "float"])),
        ("cat", categorical_transformer, make_column_selector(dtype_exclude=["int", "float"]))
    ])


    clf = Pipeline(
        steps=[("preprocessor", preprocessor)]
    )

    # Create new train and test data using the pipeline
    clf.fit(X_train, y_train)
    train_new = clf.transform(X_train)
    test_new = clf.transform(X_test)

    # Transform to dataframe and save as a csv
    train_new = pd.DataFrame.sparse.from_spmatrix(train_new)
    test_new = pd.DataFrame.sparse.from_spmatrix(test_new)
    train_new['y'] = y_train
    test_new['y'] = y_test
    return train_new, test_new, clf

def save_data(train_new, test_new, train_name, test_name, clf, clf_name):
    train_new.to_csv(train_name)
    test_new.to_csv(test_name)
    
    # Save pipeline
    with open(clf_name,'wb') as f:
        pickle.dump(clf,f)

if __name__=="__main__":
    
    
    params = yaml.safe_load(open("params.yaml"))["features"]
    data = params["data"]
    target = params["target"]
    process_data(data, target)
    train_new, test_new, clf = process_data(data, target)
    save_data(train_new, test_new, 'data/chocolate_processed_train.csv', 'data/chocolate_processed_test.csv', clf, 'data/chocolate_pipeline.pkl')