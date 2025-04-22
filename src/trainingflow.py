from metaflow import FlowSpec, step

class RegressorTrainFlow(FlowSpec):

    @step
    def start(self):
        import pandas as pd
        import numpy as np
        from sklearn.impute import KNNImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split

        # Load and preprocess dataset
        df = pd.read_csv('./data/flavors_of_cacao.csv')
        df.columns = df.columns.str.strip().str.replace('"', '', regex=False).str.replace(' ', '_', regex=False).str.lower()
        df['cocoa_percent'] = df['cocoa_percent'].str.rstrip('%').astype(float)  # Ensure numeric type for cocoa_percent

        y = df['rating']
        X = df.drop(columns=['rating'])

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        # Create a kNN imputer for numeric variables
        numeric_transformer = Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),  # Impute using kNN
            ("scaler", StandardScaler())  # Scale numeric features
        ])

        # One-hot encoding for categorical variables
        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Combine transformations in a ColumnTransformer
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=["int", "float"])),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude=["int", "float"]))
        ])

        # Build a pipeline with preprocessing
        pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

        # Fit the pipeline to training data
        pipeline.fit(X_train)

        # Transform training and testing sets
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        # Convert transformed data to DataFrames for easier manipulation
        X_train_transformed = pd.DataFrame.sparse.from_spmatrix(X_train_transformed)
        X_test_transformed = pd.DataFrame.sparse.from_spmatrix(X_test_transformed)

        # Add back the target variable to the transformed DataFrame
        X_train_transformed['y'] = y_train.reset_index(drop=True)
        X_test_transformed['y'] = y_test.reset_index(drop=True)

        self.train_data = X_train_transformed.drop(columns=['y'])
        self.test_data = X_test_transformed.drop(columns=['y'])
        self.train_values = y_train
        self.test_values = y_test
        self.next(self.train_knn, self.train_svm)

    @step
    def train_knn(self):
        from sklearn.neighbors import KNeighborsRegressor
        import numpy as np

        self.train_data = self.train_data.to_numpy(dtype=float)  # Correct conversion for DataFrame
        self.model = KNeighborsRegressor()
        self.model.fit(self.train_data, self.train_values)
        self.next(self.choose_model)

    @step
    def train_svm(self):
        from sklearn import svm
        import numpy as np

        # Convert DataFrame to NumPy array
        self.train_data = self.train_data.to_numpy(dtype=float)  # Correct conversion for DataFrame
        
        self.model = svm.SVR()
        self.model.fit(self.train_data, self.train_values)
        self.next(self.choose_model)





    @step
    def choose_model(self, inputs):
        from sklearn.metrics import mean_squared_error
        import mlflow
        from mlflow.sklearn import log_model
        import numpy as np

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        experiment_name = "metaflow-experiment"

        # Ensure MLflow experiment exists
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        # Function to evaluate models
        def score(inp):
            inp.test_data = inp.test_data.to_numpy(dtype=float)  # Ensure proper format
            inp.test_values = inp.test_values.to_numpy(dtype=float).ravel()  # Convert to 1D array
            
            predictions = inp.model.predict(inp.test_data)
            mse = mean_squared_error(inp.test_values, predictions)
            return inp.model, mse

        # Select the best model
        self.results = sorted(map(score, inputs), key=lambda x: x[1])
        self.model = self.results[0][0]  # Pick model with lowest MSE

        # Log best model to MLflow
        with mlflow.start_run():
            try:
                log_model(self.model, artifact_path="metaflow_train", registered_model_name="metaflow-wine-regressor")
            except Exception as e:
                print(f"MLflow Logging Error: {e}")

        self.next(self.end)



    # def choose_model(self, inputs):
    #     import mlflow
    #     mlflow.set_tracking_uri('http://127.0.0.1:5000')
    #     mlflow.set_experiment('metaflow-experiment')

    #     def score(inp):
    #         return inp.model, inp.model.score(inp.test_data, inp.test_labels)

    #     self.results = sorted(map(score, inputs), key=lambda x: -x[1])
    #     self.model = self.results[0][0]
    #     with mlflow.start_run():
    #         mlflow.sklearn.log_model(self.model, artifact_path = 'metaflow_train', registered_model_name="metaflow-wine-model")
    #         mlflow.end_run()
    #     self.next(self.end)

    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %f' % res for res in self.results))
        print('Model:', self.model)


if __name__=='__main__':
    RegressorTrainFlow()