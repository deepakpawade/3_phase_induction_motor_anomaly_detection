from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import numpy as np


class Scaler:
    def __init__(self):
        self.scaler = None
        self.gmm = None

    def fit_labelled_data(self, X, y):
        scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scores = []
        pipelines = []

        for scaler in scalers:
            pipeline = make_pipeline(scaler, SVC())
            pipeline.fit(X_train, y_train)
            pipelines.append(pipeline)

            y_pred = pipeline.predict(X_val)
            scores.append(accuracy_score(y_val, y_pred))

        best_pipeline_index = np.argmax(scores)
        self.scaler = pipelines[best_pipeline_index].steps[0][1]

    def fit_unlabelled_data(self, dataframe):
        scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
        data = dataframe
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
        scaled_train_data = []

        for scaler in scalers:
            scaler.fit(train_data)
            scaled_train_data.append(scaler.transform(train_data))

        gmms = []
        for scaled_data in scaled_train_data:
            gmm = GaussianMixture(n_components=1, covariance_type='full')
            gmm.fit(scaled_data)
            gmms.append(gmm)

        scores = []
        for gmm, scaler in zip(gmms, scalers):
            scaled_val_data = scaler.transform(val_data)
            score = gmm.score(scaled_val_data)
            scores.append(score)

        best_scaler_index = np.argmax(scores)
        self.scaler = scalers[best_scaler_index]

        scaled_data = self.scaler.transform(data)
        gmm = gmms[best_scaler_index]
        self.gmm = gmm

    def transform(self, data):
        return self.scaler.transform(data)

    def score(self, data):
        scaled_data = self.scaler.transform(data)
        return self.gmm.score(scaled_data)












































# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.mixture import GaussianMixture
# import numpy as np


# class Scaler:
#     def get_optimal_scalar_labelled(X,y):
#         scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#         scores = []
#         pipe = make_pipeline(MinMaxScaler(), SVC())

#         pipe.fit(X_train, y_train)
#         y_pred = pipe.predict(X_val)
#         scores.append( accuracy_score(y_val, y_pred))

#         pipe2 = make_pipeline(StandardScaler(), SVC())
#         pipe2.fit(X_train, y_train)
#         y_pred2 = pipe2.predict(X_val)
#         scores.append( accuracy_score(y_val, y_pred2))


#         pipe3 = make_pipeline(RobustScaler(), SVC())
#         pipe3.fit(X_train, y_train)
#         y_pred3 = pipe3.predict(X_val)
#         scores.append( accuracy_score(y_val, y_pred3))

#         best_scaler_index = np.argmax(scores)
#         best_scaler = scalers[best_scaler_index]
#         scaled_data = best_scaler.transform(X)
#         return scaled_data

#     def get_optimal_scalar_unlabelled(dataframe):
#         scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]

#         # Load the unlabelled data
#         data = dataframe

#         # Split the data into training and validation sets
#         train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

#         # Fit and transform the data using each scaler
        
#         scaled_train_data = []
#         for scaler in scalers:
#             scaler.fit(train_data)
#             scaled_train_data.append(scaler.transform(train_data))

#         # Train a Gaussian mixture model for each scaler
#         gmms = []
#         for scaled_data in scaled_train_data:
#             gmm = GaussianMixture(n_components=1, covariance_type='full')
#             gmm.fit(scaled_data)
#             gmms.append(gmm)

#         # Evaluate the performance of each model using the validation set
#         scores = []
#         for gmm, scaler in zip(gmms, scalers):
#             scaled_val_data = scaler.transform(val_data)
#             score = gmm.score(scaled_val_data)
#             scores.append(score)

#         # Select the scaler with the best performance
#         best_scaler_index = np.argmax(scores)
#         best_scaler = scalers[best_scaler_index]
#         scaled_data = best_scaler.transform(data)
#         return scaled_data