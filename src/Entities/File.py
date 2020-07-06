from pathlib import Path
import pandas as pd
from Services.FileSystem import Folder_Management, File_Management
import os
from Services.Configuration.Config import Config
from Services.Processing import PreProcessing
from time import sleep
import numpy as np
import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from enum import Enum
import seaborn as sns
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

sns.set()


class PredictiveColumn(Enum):
    RUNTIME = 'runtime'
    MEMORY = 'memory.max_usage_in_bytes'


class File:
    def __init__(self, full_name: str, tool_folder: Path, raw_df=None):
        """
        the constructor for the class
        :param full_name:
        :param tool_folder:
        :param merged_file:
        """
        # Provides information whether the entity is a merged too file or a "real" file
        if raw_df is not None:
            self.merged_file = True
        else:
            self.merged_file = False

        if not self.merged_file:
            # Full path to the source file
            self.path = Path(Config.DATA_RAW_DIRECTORY, full_name)
        # Name of file with extension
        self.full_name = full_name

        if not self.merged_file:
            # Name of file without extension
            self.name = os.path.splitext(full_name)[0]
        else:
            self.name = full_name

        self.verified = True

        # Check if its a merged file or not
        if self.merged_file:
            self.raw_df = raw_df
        else:
            # Load data set depending on memory saving modes
            if not Config.MEMORY_SAVING_MODE:
                self.raw_df = File_Management.read_file(self.full_name)
                if self.raw_df is None:
                    self.verified = False
                    return
            else:
                self.raw_df = pd.DataFrame()

        # Pre process the raw data set
        self.preprocessed_df = PreProcessing.pre_process_data_set(self.raw_df)

        # Setup required data sets
        self.runtime_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
        self.memory_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

        self.predicted_runtime_values = pd.DataFrame(columns=['y', 'y_hat'])
        self.predicted_memory_values = pd.DataFrame(columns=['y', 'y_hat'])

        self.runtime_evaluation_percentage_mean = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.runtime_evaluation_percentage_var = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.memory_evaluation_percentage_mean = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.memory_evaluation_percentage_var = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.runtime_feature_importance = pd.DataFrame()
        self.memory_feature_importance = pd.DataFrame()

        self.verify_file()

        # Return, because the file is not eligible to be evaluated.
        if not self.verified:
            return

        # The folder where all reports and plots are getting stored, only created if file is valid
        self.folder = Folder_Management.create_file_folder(tool_folder, self.name)

        if self.folder is not None:
            self.verified = True
        else:
            self.verified = False


    def load_data(self):
        """
        Loads the data set. Only used if memory saving mode is active
        :return:
        """
        if not self.merged_file:
            self.raw_df = File_Management.read_file(self.full_name)
            self.preprocessed_df = PreProcessing.pre_process_data_set(self.raw_df)
            return
        else:
            return

    def get_raw_df_statistics(self):
        """
        Returns column, row and feature count of the raw data set
        :return:
        """
        columns: int = len(self.raw_df.columns)
        rows: int = len(self.raw_df.index)
        features: int = columns - 1
        return columns, rows, features

    def get_pre_processed_df_statistics(self):
        """
        Returns column, row and feature count of the raw data set
        :return:
        """
        columns: int = len(self.preprocessed_df.columns)
        rows: int = len(self.preprocessed_df.index)
        features: int = columns - 1
        return columns, rows, features

    def read_raw_data(self):
        """
        Read the file into memory
        :return:
        """
        # Merged files do not have a real path, so they can not be read from a csv file
        if self.merged_file:
            return

        try:
            self.raw_df = pd.read_csv(self.full_name)
        except OSError as ex:
            logging.error(ex)
            self.raw_df = None

    def verify_file(self):
        """
        Check if the file passes all requirements to be able to be evaluated
        :return:
        """
        columns, rows, features = self.get_raw_df_statistics()
        if rows < Config.MINIMUM_ROW_COUNT:
            logging.warning(f"{self.name} has not sufficient rows ({rows}).")
            logging.warning("The file will not be evaluated.")
            sleep(1)
            self.verified = False

        if columns < Config.MINIMUM_COLUMN_COUNT:
            logging.warning(f"{self.name} has not sufficient columns ({columns}).")
            logging.warning("The file will not be evaluated.")
            sleep(1)
            self.verified = False

        # check for infinity values
        for column in self.preprocessed_df:
            if self.preprocessed_df[column].any() > np.iinfo('i').max:
                logging.warning(f"Detected infinity values in preprocessed data set!")
                logging.warning(f"File will not be evaluated.")
                self.verified = False

    def predict_runtime(self):
        """
        Predicts the runtime for a complete data set.
        :return:
        """
        df = self.preprocessed_df.copy()

        if Config.RUNTIME_LABEL not in df:
            logging.warning(f"Label {Config.RUNTIME_LABEL} could not be found!")
            return

        model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, random_state=1)

        y = df[Config.RUNTIME_LABEL]
        del df[Config.RUNTIME_LABEL]
        X = df

        source_row_count = len(X)

        X_indices = (X != 0).any(axis=1)
        X = X.loc[X_indices]
        y = y.loc[X_indices]

        if source_row_count != len(X) and Config.VERBOSE:
            logging.info(f"Removed {source_row_count - len(X)} row(s). Source had {source_row_count}.")

        X = PreProcessing.normalize_X(X)
        X = PreProcessing.variance_selection(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        model.fit(X_train, y_train)

        # Feature importance
        self.calculate_feature_importance(model, df, True)

        y_test_hat = model.predict(X_test)
        y_train_hat = model.predict(X_train)
        train_score = r2_score(y_train, y_train_hat)
        test_score = r2_score(y_test, y_test_hat)

        over_fitting = False
        if train_score > test_score * 2:
            over_fitting = True

        self.runtime_evaluation = self.runtime_evaluation.append(
            {'File Name': self.name, "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": over_fitting,
             "Initial Row Count": len(self.raw_df.index),
             "Initial Feature Count": len(self.raw_df.columns) - 1, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

        self.predicted_runtime_values = pd.concat([pd.Series(y_test).reset_index()[Config.RUNTIME_LABEL], pd.Series(y_test_hat)],
                                                  axis=1)
        self.predicted_runtime_values.rename(columns={"runtime": "y", 0: "y_hat"}, inplace=True)

    def predict_memory(self):
        """
        Predicts the memory usage for a complete data set.
        :return:
        """
        df = self.preprocessed_df.copy()

        if Config.MEMORY_LABEL not in df:
            logging.warning(f"Label {Config.MEMORY_LABEL} could not be found!")
            return

        model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, random_state=1)

        y = df[Config.MEMORY_LABEL]
        del df[Config.MEMORY_LABEL]
        X = df

        source_row_count = len(X)

        X_indexes = (X != 0).any(axis=1)

        X = X.loc[X_indexes]
        y = y.loc[X_indexes]

        if source_row_count != len(X) and Config.VERBOSE:
            logging.info(f"Removed {source_row_count - len(X)} rows. Source had {source_row_count}.")

        X = PreProcessing.normalize_X(X)
        X = PreProcessing.variance_selection(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

        model.fit(X_train, y_train)

        # Feature importance
        self.calculate_feature_importance(model, df, False)

        y_test_hat = model.predict(X_test)
        y_train_hat = model.predict(X_train)
        test_score = r2_score(y_test, y_test_hat)
        train_score = r2_score(y_train, y_train_hat)

        over_fitting = False
        if train_score > test_score * 2:
            over_fitting = True

        self.memory_evaluation = self.memory_evaluation.append(
            {'File Name': self.name, "Test Score": test_score,
             "Train Score": train_score, "Potential Over Fitting": over_fitting,
             "Initial Row Count": len(self.raw_df.index),
             "Initial Feature Count": len(self.raw_df.columns) - 1, "Processed Row Count": len(X),
             "Processed Feature Count": X.shape[1]}, ignore_index=True)

        self.predicted_memory_values = pd.concat(
            [pd.Series(y_test).reset_index()[Config.MEMORY_LABEL], pd.Series(y_test_hat)],
            axis=1)
        self.predicted_memory_values.rename(columns={"runtime": "y", 0: "y_hat"}, inplace=True)

    def free_memory(self):
        """
        Release not required memory for memory saving mode.
        :return:
        """
        if not Config.MEMORY_SAVING_MODE:
            return

        self.raw_df = None
        self.preprocessed_df = None

    def predict_row_removal(self, column: str):
        """
        Predict the value for the column specified, while removing data from the original df
        :param column: the column that should be predicted.
        :return:
        """
        df = self.preprocessed_df.copy()

        if column not in df:
            return

        averages_per_repetition = pd.Series()

        final_evaluation = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94',
                     '95', '96', '97', '98', '99', 'Rows', 'Features'])

        y = df[column]
        del df[column]
        X = df

        X = PreProcessing.normalize_X(X)
        X = PreProcessing.variance_selection(X)

        column_count, row_count, feature_count = self.get_pre_processed_df_statistics()

        for i in range(0, Config.REPETITIONS, 1):
            if Config.VERBOSE:
                logging.info(f"Started repetition # {i + 1}")
            k_folds = self.k_folds(X, y)
            averages_per_repetition = averages_per_repetition.append(k_folds).mean()
            # Remove the mean of the index column
            del averages_per_repetition[0]

            averages_per_repetition = averages_per_repetition.fillna(0)
            final_evaluation = final_evaluation.append(averages_per_repetition, ignore_index=True)

        if column == PredictiveColumn.MEMORY.value:
            self.memory_evaluation_percentage_mean = pd.Series(final_evaluation.mean())
            self.memory_evaluation_percentage_mean['Rows'] = row_count
            self.memory_evaluation_percentage_mean['Features'] = feature_count
            self.memory_evaluation_percentage_var = pd.Series(final_evaluation.var())
            self.memory_evaluation_percentage_var['Rows'] = row_count
            self.memory_evaluation_percentage_var['Features'] = feature_count
        elif column == PredictiveColumn.RUNTIME.value:
            self.runtime_evaluation_percentage_mean = pd.Series(final_evaluation.mean())
            self.runtime_evaluation_percentage_mean['Rows'] = row_count
            self.runtime_evaluation_percentage_mean['Features'] = feature_count
            self.runtime_evaluation_percentage_var = pd.Series(final_evaluation.var())
            self.runtime_evaluation_percentage_var['Rows'] = row_count
            self.runtime_evaluation_percentage_var['Features'] = feature_count
        else:
            logging.warning(f"Could not detect predictive column: {column}")

    def k_folds(self, X, y):
        """
        Trains the model to predict the total time
        :param X:
        :param y:
        :return:
        """
        try:
            model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, random_state=1)

            kf = KFold(n_splits=Config.K_FOLDS)
            k_fold_scores = pd.DataFrame(
                columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94',
                         '95', '96', '97', '98', '99'])
            counter = 0
            for train_index, test_index in kf.split(X):
                r2scores = []
                # Iterate from 0 to 101. Ends @ 100, reduce by 1
                for i in range(0, 101, 1):
                    if i <= 90 and i % 10 == 0:
                        r2scores.append(self.calculate_fold(model, i, X, y, train_index, test_index))
                    if 99 >= i > 90:
                        r2scores.append(self.calculate_fold(model, i, X, y, train_index, test_index))

                k_fold_scores = k_fold_scores.append(
                    {'0': r2scores[0], '10': r2scores[1], '20': r2scores[2], '30': r2scores[3], '40': r2scores[4],
                     '50': r2scores[5], '60': r2scores[6], '70': r2scores[7], '80': r2scores[8],
                     '90': r2scores[9], '91': r2scores[10], '92': r2scores[11], '93': r2scores[12],
                     '94': r2scores[13], '95': r2scores[14], '96': r2scores[15], '97': r2scores[16],
                     '98': r2scores[17], '99': r2scores[18]}, ignore_index=True)

                counter += 1

            return k_fold_scores

        except BaseException as ex:
            print(ex)
            k_fold_scores = pd.DataFrame(0, index=np.arange(Config.K_FOLDS),
                                         columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92',
                                                  '93', '94', '95', '96', '97', '98', '99'])
            return k_fold_scores

    def calculate_fold(self, model, i, X, y, train_index, test_index):
        """
        Calculates the r2 scores
        :param model:
        :param i:
        :param X:
        :param y:
        :param train_index:
        :param test_index:
        :return:
        """
        # Create a deep copy, to keep the original data set untouched
        train_index_copy = train_index

        # Calculate amount of rows to be removed
        rows = int(len(train_index_copy) * i / 100)

        # Remove rows by random index
        train_index_copy = PreProcessing.remove_random_rows(train_index_copy, rows)
        X_train, X_test = X[train_index_copy], X[test_index]
        y_train, y_test = y[train_index_copy], y[test_index]

        model.fit(X_train, y_train)
        y_test_hat = model.predict(X_test)
        return r2_score(y_test, y_test_hat)

    def generate_reports(self):
        """
        Generate file specific reports
        :return:
        """
        # Predicted values
        if not self.runtime_evaluation.empty:
            self.runtime_evaluation.to_csv(os.path.join(self.folder, "runtime_evaluation_report.csv"), index=False)

        if not self.memory_evaluation.empty:
            self.memory_evaluation.to_csv(os.path.join(self.folder, "memory_evaluation_report.csv"), index=False)

        if not self.predicted_memory_values.empty:
            self.predicted_memory_values.to_csv(os.path.join(self.folder, "predicted_memory_report.csv"), index=False)

        if not self.predicted_runtime_values.empty:
            self.predicted_runtime_values.to_csv(os.path.join(self.folder, "predicted_runtime_report.csv"), index=False)

        if not self.runtime_evaluation_percentage_mean.empty:
            self.runtime_evaluation_percentage_mean.to_csv(os.path.join(self.folder, "runtime_row_removal_mean.csv"),
                                                           index=False)

        if not self.runtime_evaluation_percentage_var.empty:
            self.runtime_evaluation_percentage_var.to_csv(os.path.join(self.folder, "runtime_row_removal_var.csv"),
                                                          index=False)

        if not self.memory_evaluation_percentage_mean.empty:
            self.memory_evaluation_percentage_mean.to_csv(os.path.join(self.folder, "memory_row_removal_mean.csv"),
                                                          index=False)

        if not self.memory_evaluation_percentage_var.empty:
            self.memory_evaluation_percentage_var.to_csv(os.path.join(self.folder, "memory_row_removal_var.csv"),
                                                         index=False)

    def generate_plots(self):
        """
        Helper to call all plotting functions
        :return:
        """
        self.plot_predicted_values()
        self.plot_percentage_removal()
        self.plot_feature_importance(True)
        self.plot_feature_importance(False)
        self.plot_feature_to_label_correlation(True)

    def plot_predicted_values(self):
        """
        Plots the predicted values for the unmodified dataset
        :return:
        """

        ax = None

        if not self.predicted_memory_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="memory", data=self.predicted_memory_values)
            ax.set(xscale="log", yscale="log")

        if not self.predicted_runtime_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="runtime", data=self.predicted_runtime_values)
            ax.set(xscale="log", yscale="log")

        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(self.folder, "predicated_log_values.png"))
        fig.clf()

        ax = None

        if not self.predicted_memory_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="memory", data=self.predicted_memory_values)

        if not self.predicted_runtime_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="runtime", data=self.predicted_runtime_values)

        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(self.folder, "predicated_values.png"))
        fig.clf()

    def plot_percentage_removal(self):

        if not Config.PERCENTAGE_REMOVAL:
            return

        ax = None

        if not self.runtime_evaluation_percentage_mean.empty:
            mean = self.runtime_evaluation_percentage_mean
            del mean['Rows']
            del mean['Features']

            ax = sns.lineplot(data=mean, label="mean", palette="tab10", linewidth=2.5)

        if not self.runtime_evaluation_percentage_var.empty:
            var = self.runtime_evaluation_percentage_var
            del var['Rows']
            del var['Features']

            ax = sns.lineplot(data=var, label="var", palette="tab10", linewidth=2.5)

        # Remove not required rows from dfs

        ax.set(xlabel='% rows removed', ylabel='R^2 Score')

        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(self.folder, "percentage_removal_prediction.png"))
        fig.clf()

    def calculate_feature_importance(self, model, df, runtime: bool):
        """
        Calculates the feature importance for the given model
        """
        feats = {}  # a dict to hold feature_name: feature_importance
        for feature, importance in zip(df.columns, model.feature_importances_):
            feats[feature] = importance  # add the name/value pair

        importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
        importance.sort_values(by='Gini-importance', inplace=True, ascending=False)

        importance_indices = importance[importance['Gini-importance'].gt(0.01)].index
        if runtime:
            self.runtime_feature_importance = importance.T[importance_indices]
        else:
            self.memory_feature_importance = importance.T[importance_indices]

    def plot_feature_importance(self, runtime: bool):
        """
        Plots the feature importance for each evaluation
        """
        if runtime:
            if self.runtime_feature_importance.empty:
                return
            ax = sns.barplot(data=self.runtime_feature_importance)
        else:
            if self.memory_feature_importance.empty:
                return
            ax = sns.barplot(data=self.memory_feature_importance)

        ax.set(xlabel='Feature', ylabel='Gini Index')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.legend()
        fig = ax.get_figure()
        # Save the fig
        if runtime:
            fig.savefig(os.path.join(self.folder, "runtime_feature_importance.png"), bbox_inches='tight')
        else:
            fig.savefig(os.path.join(self.folder, "memory_feature_importance.png"), bbox_inches='tight')
        fig.clf()

    def plot_feature_to_label_correlation(self, runtime: bool):
        important_features = []

        if runtime:
            for col in self.runtime_feature_importance.columns:
                important_features.append(col)

            important_features.append(Config.RUNTIME_LABEL)

            data = self.preprocessed_df[important_features]
            corr = data.corr()

            # Generate a mask for the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=np.bool))

            # Set up the matplotlib figure
            f, ax = plt.subplots(figsize=(11, 9))

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})

            fig = ax.get_figure()
            if runtime:
                fig.savefig(os.path.join(self.folder, "runtime_correlation_matrix.png"), bbox_inches='tight')
            else:
                fig.savefig(os.path.join(self.folder, "memory_correlation_matrix.png"), bbox_inches='tight')
            fig.clf()

