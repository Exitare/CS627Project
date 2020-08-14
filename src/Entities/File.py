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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sns.set()


class File:
    def __init__(self, full_name: str, tool_folder: Path, raw_df=None):
        """
        the constructor for the class
        :param full_name:
        :param tool_folder:
        :param raw_df:
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
        if not Config.MEMORY_SAVING_MODE:
            self.preprocessed_df = PreProcessing.pre_process_data_set(self.raw_df)
        else:
            self.preprocessed_df = pd.DataFrame()

        # Setup required data sets
        self.runtime_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
        self.memory_evaluation = pd.DataFrame(
            columns=['File Name', 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

        self.predicted_runtime_values = pd.DataFrame(columns=['y', 'y_hat'])
        self.predicted_memory_values = pd.DataFrame(columns=['y', 'y_hat'])

        self.runtime_evaluation_percentage_folds = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99', 'Rows', 'Features'])

        self.runtime_evaluation_percentage_rows = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99'])

        self.memory_evaluation_percentage_folds = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99'])

        self.memory_evaluation_percentage_rows = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80',
                     '90', '91', '92', '93', '94', '95', '96', '97', '98',
                     '99'])

        self.runtime_feature_importance = pd.DataFrame()
        self.memory_feature_importance = pd.DataFrame()

        self.pca_analysis_df = pd.DataFrame()
        self.pca_model = None

        if not Config.MEMORY_SAVING_MODE:
            self.verify()

        # Return, because the file is not eligible to be evaluated.
        if not self.verified:
            return

        # The folder where all reports and plots are getting stored, only created if file is valid
        self.folder = Folder_Management.create_file_folder(tool_folder, self.name)

        if self.folder is not None:
            self.verified = True
        else:
            self.verified = False

        # Determines if a file is already evaluated or not
        self.evaluated = False

    # Loading
    def load_raw_data(self):
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

    def verify(self):
        """
        Check if the file passes all requirements to be able to be evaluated
        :return:
        """
        columns, rows, features = self.get_raw_df_statistics()
        if rows < Config.MINIMUM_ROW_COUNT:
            if Config.VERBOSE:
                logging.warning(f"{self.name} has insufficient rows ({rows}).")
                logging.warning("The file will not be evaluated.")
                sleep(1)
            self.verified = False

        if columns < Config.MINIMUM_COLUMN_COUNT:
            if Config.VERBOSE:
                logging.warning(f"{self.name} has insufficient columns ({columns}).")
                logging.warning("The file will not be evaluated.")
                sleep(1)
            self.verified = False

        # check for infinity values
        for column in self.preprocessed_df:
            if self.preprocessed_df[column].any() > np.iinfo('i').max:
                if Config.VERBOSE:
                    logging.warning(f"Detected infinity values in preprocessed data set!")
                    logging.warning(f"File will not be evaluated.")
                self.verified = False

    # Prediction
    def predict(self, label: str):
        """
        Predicts the runtime for a complete data set.
        :return:
        """
        df = self.preprocessed_df.copy()

        if label not in df:
            return

        model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, max_depth=Config.FOREST_MAX_DEPTH,
                                      random_state=1)

        y = df[label]
        del df[label]
        X = df

        source_row_count = len(X)

        X_indices = (X != 0).any(axis=1)
        X = X.loc[X_indices]
        y = y.loc[X_indices]

        if source_row_count != len(X) and Config.VERBOSE:
            logging.info(f"Removed {source_row_count - len(X)} row(s). Source had {source_row_count}.")

        X = PreProcessing.variance_selection(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)

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

        if label == Config.RUNTIME_LABEL:
            self.runtime_evaluation = self.runtime_evaluation.append(
                {'File Name': self.name, "Test Score": test_score,
                 "Train Score": train_score, "Potential Over Fitting": over_fitting,
                 "Initial Row Count": len(self.raw_df.index),
                 "Initial Feature Count": len(self.raw_df.columns) - 1, "Processed Row Count": len(X),
                 "Processed Feature Count": X.shape[1]}, ignore_index=True)

            self.predicted_runtime_values = pd.concat(
                [pd.Series(y_test).reset_index()[Config.RUNTIME_LABEL], pd.Series(y_test_hat)],
                axis=1)
            self.predicted_runtime_values.rename(columns={"runtime": "y", 0: "y_hat"}, inplace=True)
        else:
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

    def predict_row_removal(self, label: str):
        """
        Predict the value for the column specified, while removing data from the original df
        :param label: the label that should be predicted.
        :return:
        """
        df = self.preprocessed_df.copy()

        if label not in df:
            return

        averages_per_repetition = pd.Series()

        k_folds_evaluation = pd.DataFrame(
            columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94',
                     '95', '96', '97', '98', '99'])

        k_folds_row_count = []

        y = df[label]
        del df[label]
        X = df

        X = PreProcessing.normalize_X(X)
        X = PreProcessing.variance_selection(X)

        for i in range(0, Config.REPETITIONS, 1):
            if Config.VERBOSE:
                logging.info(f"Started repetition # {i + 1}")
            k_folds, k_folds_row_count = self.__calculate_k_folds(X, y)
            averages_per_repetition = averages_per_repetition.append(k_folds).mean()

            # Remove the mean of the index column
            del averages_per_repetition[0]

            averages_per_repetition = averages_per_repetition.fillna(0)
            k_folds_evaluation = k_folds_evaluation.append(averages_per_repetition, ignore_index=True)

        if label == Config.MEMORY_LABEL:
            self.memory_evaluation_percentage_folds = k_folds_evaluation
            self.memory_evaluation_percentage_folds = self.memory_evaluation_percentage_folds.append(
                pd.Series(k_folds_evaluation.mean()), ignore_index=True)
            self.memory_evaluation_percentage_folds = self.memory_evaluation_percentage_folds.append(
                pd.Series(k_folds_evaluation.var()), ignore_index=True)
            self.memory_evaluation_percentage_rows = k_folds_row_count.iloc[0]
        elif label == Config.RUNTIME_LABEL:
            self.runtime_evaluation_percentage_folds = k_folds_evaluation
            self.runtime_evaluation_percentage_folds = self.runtime_evaluation_percentage_folds.append(
                pd.Series(k_folds_evaluation.mean()), ignore_index=True)
            self.runtime_evaluation_percentage_folds = self.runtime_evaluation_percentage_folds.append(
                pd.Series(k_folds_evaluation.var()), ignore_index=True)
            self.runtime_evaluation_percentage_rows = k_folds_row_count.iloc[0]
        else:
            logging.warning(f"Could not detect predictive column: {label}")

    def pca_analysis(self, label: str):
        """
        Generates a pca analysis
        """
        df = self.preprocessed_df.copy()
        del df[label]
        X = df

        X = PreProcessing.variance_selection(X)

        self.pca_model = PCA()
        self.pca_model.fit_transform(X)
        self.pca_analysis_df = X

        # loadings = pd.DataFrame(self.pca_model.components_.T, index=df.columns)
        # loadings.to_csv(Path.joinpath(self.folder, "pca_eigenvecotors.csv"), index=True)

    def __calculate_k_folds(self, X, y):
        """
        Trains the model to predict the total time
        :param X:
        :param y:
        :return:
        """
        try:

            kf = KFold(n_splits=Config.K_FOLDS)
            k_fold_scores = pd.DataFrame(
                columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94',
                         '95', '96', '97', '98', '99'])
            k_fold_rows = pd.DataFrame(
                columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92', '93', '94',
                         '95', '96', '97', '98', '99'])

            for train_index, test_index in kf.split(X):
                r2scores = []
                # Iterate from 0 to 101. Ends @ 100, reduce by 1
                for i in range(0, 101, 1):
                    if i <= 90 and i % 10 == 0:
                        r2scores.append(self.__calculate_single_fold(i, X, y, train_index, test_index))
                    if 99 >= i > 90:
                        r2scores.append(self.__calculate_single_fold(i, X, y, train_index, test_index))

                k_fold_scores = k_fold_scores.append(
                    {'0': r2scores[0][0], '10': r2scores[1][0], '20': r2scores[2][0], '30': r2scores[3][0],
                     '40': r2scores[4][0], '50': r2scores[5][0], '60': r2scores[6][0], '70': r2scores[7][0],
                     '80': r2scores[8][0], '90': r2scores[9][0], '91': r2scores[10][0], '92': r2scores[11][0],
                     '93': r2scores[12][0], '94': r2scores[13][0], '95': r2scores[14][0], '96': r2scores[15][0],
                     '97': r2scores[16][0], '98': r2scores[17][0], '99': r2scores[18][0]}, ignore_index=True)

                k_fold_rows = k_fold_rows.append(
                    {'0': r2scores[0][1], '10': r2scores[1][1], '20': r2scores[2][1], '30': r2scores[3][1],
                     '40': r2scores[4][1], '50': r2scores[5][1], '60': r2scores[6][1], '70': r2scores[7][1],
                     '80': r2scores[8][1], '90': r2scores[9][1], '91': r2scores[10][1], '92': r2scores[11][1],
                     '93': r2scores[12][1], '94': r2scores[13][1], '95': r2scores[14][1], '96': r2scores[15][1],
                     '97': r2scores[16][1], '98': r2scores[17][1], '99': r2scores[18][1]}, ignore_index=True)

            return k_fold_scores, k_fold_rows

        except BaseException as ex:
            print(ex)
            k_fold_scores = pd.DataFrame(0, index=np.arange(Config.K_FOLDS),
                                         columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92',
                                                  '93', '94', '95', '96', '97', '98', '99'])
            k_fold_rows = pd.DataFrame(0, index=np.arange(Config.K_FOLDS),
                                       columns=['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '91', '92',
                                                '93', '94',
                                                '95', '96', '97', '98', '99'])
            return k_fold_scores, k_fold_rows

    def __calculate_single_fold(self, i, X, y, train_index, test_index):
        """
        Calculates the r2 scores
        :param i:
        :param X:
        :param y:
        :param train_index:
        :param test_index:
        :return:
        """
        model = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS, max_depth=Config.FOREST_MAX_DEPTH,
                                      random_state=1)

        # Calculate amount of rows to be removed
        rows = int(len(train_index) * i / 100)
        # Remove rows by random index
        train_index = PreProcessing.remove_random_rows(train_index, rows)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_test_hat = model.predict(X_test)
        return r2_score(y_test, y_test_hat), len(train_index)

    # Reports
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

        if not self.runtime_evaluation_percentage_folds.empty:
            self.runtime_evaluation_percentage_folds.to_csv(
                os.path.join(self.folder, "runtime_row_removal_predictions.csv"), index=False)

        if not self.runtime_evaluation_percentage_rows.empty:
            self.runtime_evaluation_percentage_rows.T.to_csv(os.path.join(self.folder, "row_removal_rows.csv"),
                                                             index=False)

        if not self.memory_evaluation_percentage_folds.empty:
            self.memory_evaluation_percentage_folds.to_csv(
                os.path.join(self.folder, "memory_row_removal_predictions.csv"), index=False)

        if not self.memory_evaluation_percentage_rows.empty:
            self.memory_evaluation_percentage_rows.T.to_csv(os.path.join(self.folder, "row_removal_rows.csv"),
                                                            index=False)

    # Plots
    def generate_plots(self):
        """
        Helper to call all plotting functions
        :return:
        """
        self.plot_predicted_values(True)
        self.plot_predicted_values(False)
        self.plot_percentage_removal()
        self.plot_feature_importance(True)
        self.plot_feature_importance(False)
        self.plot_feature_to_label_correlation(True)
        self.plot_feature_to_label_correlation(False)
        self.__plot_pca_analysis()

    def plot_predicted_values(self, log_scale: bool):
        """
        Plots the predicted values for the unmodified data set
        :return:
        """

        ax = None

        if not self.predicted_memory_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="memory", data=self.predicted_memory_values)
            if log_scale:
                ax.set(xscale="log", yscale="log")

        if not self.predicted_runtime_values.empty:
            ax = sns.scatterplot(x='y', y='y_hat', label="runtime", data=self.predicted_runtime_values)
            if log_scale:
                ax.set(xscale="log", yscale="log")

        if ax is None:
            logging.warning("Could not plot predicted values, because axis where None")
            return

        ax.legend()
        fig = ax.get_figure()
        if log_scale:
            fig.savefig(os.path.join(self.folder, "predicated_log_values.png"))
        else:
            fig.savefig(os.path.join(self.folder, "predicated_values.png"))
        fig.clf()
        plt.close('all')

    def plot_percentage_removal(self):

        if not Config.PERCENTAGE_REMOVAL:
            return

        ax = None

        if not self.runtime_evaluation_percentage_folds.empty:
            ax = sns.lineplot(data=self.runtime_evaluation_percentage_folds, palette="tab10", linewidth=2.5,
                              dashes=False)

        if not self.memory_evaluation_percentage_folds.empty:
            ax = sns.lineplot(data=self.memory_evaluation_percentage_folds, palette="tab10", linewidth=2.5,
                              dashes=False)

        if ax is None:
            return

        ax.set(xlabel='Folds (Forelast is mean, last is var)', ylabel='R^2 Score')
        ax.legend()
        fig = ax.get_figure()
        fig.savefig(os.path.join(self.folder, "percentage_removal_prediction.png"))
        fig.clf()
        plt.close('all')

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
        plt.close('all')

    def plot_feature_to_label_correlation(self, runtime: bool):
        """
        Plots the correlation between the feature and labels
        """
        important_features = []

        if runtime:
            if self.runtime_feature_importance.empty or Config.RUNTIME_LABEL not in self.raw_df:
                return

            for col in self.runtime_feature_importance.columns:
                important_features.append(col)

            important_features.append(Config.RUNTIME_LABEL)

            data = self.preprocessed_df[important_features]
            corr = data.corr()

            mask = np.triu(np.ones_like(corr, dtype=np.bool))
            f, ax = plt.subplots(figsize=(11, 9))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})

            fig = ax.get_figure()
            fig.savefig(os.path.join(self.folder, "runtime_correlation_matrix.png"), bbox_inches='tight')
            fig.clf()

        else:
            if self.memory_feature_importance.empty or Config.MEMORY_LABEL not in self.raw_df:
                return

            for col in self.memory_feature_importance.columns:
                important_features.append(col)

            important_features.append(Config.MEMORY_LABEL)

            data = self.preprocessed_df[important_features]
            corr = data.corr()

            mask = np.triu(np.ones_like(corr, dtype=np.bool))
            f, ax = plt.subplots(figsize=(11, 9))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})

            fig = ax.get_figure()
            fig.savefig(os.path.join(self.folder, "memory_correlation_matrix.png"), bbox_inches='tight')
            fig.clf()
            plt.close('all')

    def __plot_pca_analysis(self):
        features = range(self.pca_model.n_components_)
        plt.bar(features, self.pca_model.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder, "pca_features.png"), bbox_inches='tight')
        plt.clf()
        plt.close('all')

    # Cleanup
    def free_memory(self):
        """
        Release not required memory for memory saving mode.
        :return:
        """
        if not Config.MEMORY_SAVING_MODE:
            return

        self.raw_df = None
        self.preprocessed_df = None
