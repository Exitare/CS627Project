from pathlib import Path
import pandas as pd
from Services.FileSystem import Folder_Management, File_Management
import os
from Services.Configuration.Config import Config
from Services.Processing import PreProcessing
from time import sleep
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Services.Predictions import Predictions
from Services.Helper import Data_Frame_Helper
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans

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

        # List of labels present in the data set. The list is ideally a copy of the list specific in the config.ini
        # If a label is missing the data set it will not be present in here, and therefore not evaluated
        self.detected_labels = []
        self.verified = True

        # Check if its a merged file or not
        if self.merged_file:
            self.raw_df = raw_df
        else:
            # Load data set depending on memory saving modes
            if not Config.MEMORY_SAVING_MODE:
                self.raw_df = File_Management.read_file(self.full_name)
                if self.raw_df is None:
                    if Config.DEBUG:
                        logging.info("Raw df is empty!")
                    self.verified = False
                    return
            else:
                self.raw_df = pd.DataFrame()

        # Pre process the raw data set
        if not Config.MEMORY_SAVING_MODE:
            self.preprocessed_df = PreProcessing.pre_process_data_set(self.raw_df)
            self.detect_labels()
        else:
            self.preprocessed_df = pd.DataFrame()

        # Contains the test, train score, name of the file, and additional data for each file
        self.evaluation_results = pd.DataFrame(
            columns=['File Name', "Label", 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])
        # Contains the test, train score, name of the file, and additional data for each file and each split
        self.split_evaluation_results = pd.DataFrame()
        # Contains y and y_hat values
        self.predicted_results = pd.DataFrame(columns=["Label", 'y', 'y_hat'])
        # Contains the features importances for each file
        self.feature_importances = pd.DataFrame()
        # Contains the pca object with supporting functions for all labels
        self.pca = dict()
        # Contains the pca scores calculated
        self.pca_scores = dict()
        # Contains all pca components as df for each label
        self.pca_components_data_frames = dict()
        # Contains all simple dfs. Needs to be a dict, because simple dfs needs to be created for a target value
        # So key is the label, and value is a list of multiple simple dataframes
        self.simple_dfs = dict()
        self.simple_dfs_evaluation = pd.DataFrame(
            columns=['File Name', "Label", 'Train Score', 'Test Score', 'Potential Over Fitting', 'Initial Row Count',
                     'Initial Feature Count', 'Processed Row Count', 'Processed Feature Count'])

        # Prepare the internal data structure
        # self.prepare_internal_data_structure()

        if not Config.MEMORY_SAVING_MODE:
            self.verify()

        # Return, because the file is not eligible to be evaluated.
        if not self.verified:
            if Config.DEBUG:
                logging.info(f"{self.name} is not verified!")
                sleep(1)
            return

        # The folder where all reports and plots are getting stored, only created if file is valid
        self.folder = Folder_Management.create_file_folder(tool_folder, self.name)
        if self.folder is not None:
            self.verified = True
        else:
            self.verified = False

        self.simple_df_folder = Folder_Management.create_folder(Path.joinpath(self.folder, "Simple"))
        self.split_folder = Folder_Management.create_folder(Path.joinpath(self.folder, "Splits"))
        # Determines if a file is already evaluated or not
        self.evaluated = False

    def load_memory_sensitive_data(self):
        """
        Loads all the data and prepares data sets, which was skipped due to memory saving mode
        """
        self.__load_preprocess_raw_data()
        self.detect_labels()

    # Loading and preprocessing
    def __load_preprocess_raw_data(self):
        """
        Loads the data set and preprocesses it
        Only used if memory saving mode is active
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

    def detect_labels(self):
        """
        Detects if the labels specified in the config.ini are present for the preproccesed data set
        """
        for label in Config.LABELS:
            if label in self.preprocessed_df:
                if Config.DEBUG:
                    logging.info(f"Found label {label} in file {self.name}")
                self.detected_labels.append(label)

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
            if Config.DEBUG:
                logging.warning(f"{self.name} has insufficient rows ({rows}).")
                logging.warning("The file will not be evaluated.")
                sleep(1)
            self.verified = False

        if columns < Config.MINIMUM_COLUMN_COUNT:
            if Config.DEBUG:
                logging.warning(f"{self.name} has insufficient columns ({columns}).")
                logging.warning("The file will not be evaluated.")
                logging.warning(self.raw_df)
                input()
                sleep(1)
            self.verified = False

        # check for infinity values
        for column in self.preprocessed_df:
            if self.preprocessed_df[column].any() > np.iinfo('i').max:
                if Config.DEBUG:
                    logging.warning(f"Detected infinity values in preprocessed data set!")
                    logging.warning(f"File will not be evaluated.")
                self.verified = False

        # Check if columns will pass variance selection
        for label in self.detected_labels:
            if label in self.preprocessed_df:
                check_df = self.preprocessed_df.copy()
                del check_df[label]
                check_df = PreProcessing.variance_selection(check_df)

                if 'numpy' not in str(type(check_df)):
                    self.verified = False

    # Prediction
    def predict(self, label: str):
        """
        Predicts the runtime for a complete data set.
        :return:
        """
        try:

            model, train_score, test_score, over_fitting, X, y_test, y_test_hat \
                = Predictions.predict(label, self.preprocessed_df.copy())

            # Calculate feature importances
            temp_df = self.preprocessed_df.copy()
            del temp_df[label]
            # self.__calculate_feature_importance(label, model, temp_df)

            self.evaluation_results = self.evaluation_results.append(
                {
                    'File Name': self.name,
                    "Label": label,
                    "Test Score": test_score,
                    "Train Score": train_score,
                    "Potential Over Fitting": over_fitting,
                    "Initial Row Count": len(self.raw_df.index),
                    "Initial Feature Count": len(self.raw_df.columns) - 1,
                    "Processed Row Count": len(X),
                    "Processed Feature Count": X.shape[1]
                }, ignore_index=True)

            temp = pd.concat(
                [pd.Series(y_test).reset_index()[label], pd.Series(y_test_hat)],
                axis=1)
            temp.rename(columns={label: 'y', 0: "y_hat"}, inplace=True)

            self.predicted_results["y"] = temp["y"]
            self.predicted_results["y_hat"] = temp["y_hat"]
            self.predicted_results["Label"].fillna(label, inplace=True)

        except BaseException as ex:
            logging.exception(ex)

    def predict_splits(self, label: str):
        """
        Split the data into parts, and predicts results using only one part after another.
        """
        try:
            df = self.preprocessed_df.copy()

            # how many rows should one part contain
            total_rows = int(len(df))
            rows_per_chunk: int = int(len(df) * 0.1)

            chunks = Data_Frame_Helper.split_df(df, rows_per_chunk)

            part = 0
            for chunk in chunks:
                model, train_score, test_score, over_fitting, X, y_test, y_test_hat \
                    = Predictions.predict(label, chunk.copy())

                if model is None:
                    logging.warning("Could not create predictions because of insufficient data!")
                    continue

                # Calculate feature importances
                temp_df = chunk.copy()
                del temp_df[label]
                # self.__calculate_feature_importance(label, model, temp_df)

                self.split_evaluation_results = self.split_evaluation_results.append(
                    {
                        'File Name': self.name,
                        "Label": label,
                        "Test Score": test_score,
                        "Train Score": train_score,
                        "Potential Over Fitting": over_fitting,
                        "Initial Row Count": len(chunk),
                        "Initial Feature Count": len(chunk.columns),
                        "Processed Row Count": len(X),
                        "Processed Feature Count": X.shape[1],
                        "Total rows": total_rows,
                        "Part": part
                    }, ignore_index=True)
                part += 1

        except BaseException as ex:
            logging.exception(ex)

    def pca_analysis(self, label: str):
        """
        Generates a pca analysis
        """

        try:
            df = self.preprocessed_df.copy()

            y = df[label]
            del df[label]
            X = df

            pca = PCA()
            pca_data = pca.fit(X)
            component_id = next(x for x, val in enumerate(pca.explained_variance_ratio_.cumsum()) if val > 0.8)
            pca = PCA(n_components=component_id + 1)
            pca_scores = pca.fit_transform(X)

            self.pca[label] = pca
            self.pca_scores[label] = pca_scores

        except BaseException as ex:
            logging.exception(ex)

    def k_means(self, label: str):
        """

        """
        inertias = []

        # Creating 10 K-Mean models while varying the number of clusters (k)

        data = self.pca[label]

        for k in range(1, 6):
            model = KMeans(n_clusters=k)

            # Fit model to samples
            model.fit(data.n_components_.iloc[:, :3])

            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        plt.plot(range(1, 10), inertias, '-p', color='gold')
        plt.xlabel('number of clusters, k')
        plt.ylabel('inertia')

        plt.show()

    def get_best_performing_split(self, label: str):
        """
        Returns the best performing split
        """
        data = Data_Frame_Helper.get_label_data(self.split_evaluation_results, label)
        if data is None:
            return None

        row = data.loc[data['Test Score'].idxmax()]
        return int(row['Part'])

    def create_simple_data_set(self):
        """
        Creates simple data sets based on the feature importances
        """

        # All simple dataframes will be stored in here
        for label in self.detected_labels:
            try:
                X = self.preprocessed_df.copy()

                if label in X:
                    del X[label]

                y = self.preprocessed_df[label]

                if len(X.columns) < 10:
                    if Config.DEBUG:
                        logging.debug("Simple dataset evaluation will not be done because not enough columns.")
                    return

                estimator = RandomForestRegressor(n_estimators=Config.FOREST_ESTIMATORS,
                                                  max_depth=Config.FOREST_MAX_DEPTH,
                                                  random_state=1)

                # Create simple data sets
                for i in range(20, 100, 20):
                    features_to_select: int = int(len(X.columns) * (i / 100))

                    selector = RFE(estimator, n_features_to_select=features_to_select, step=1)
                    selector = selector.fit(X, y)
                    df = X[X.columns[selector.get_support(indices=True)]].copy()
                    df[label] = y

                    # Add new simple dataset to list
                    if label not in self.simple_dfs:
                        self.simple_dfs[label] = list()
                        self.simple_dfs[label].append(df)
                    else:
                        self.simple_dfs[label].append(df)

            except BaseException as ex:
                logging.exception(ex)

    def evaluate_simple_data_set(self):
        """
        Evaluates the simple data sets
        """

        for label, data_sets in self.simple_dfs.items():
            for data_set in data_sets:
                data = data_set.copy()
                model, train_score, test_score, over_fitting, X, y_test, y_test_hat = Predictions.predict(label,
                                                                                                          data)

                # Store the simple df evaluation in a dataframe and in a list
                self.simple_dfs_evaluation = self.simple_dfs_evaluation.append(
                    {
                        "File Name": self.name,
                        "Label": label,
                        "Test Score": test_score,
                        "Train Score": train_score,
                        "Potential Over Fitting": over_fitting,
                        "Initial Row Count": len(self.raw_df.index),
                        "Initial Feature Count": len(self.raw_df.columns) - 1,
                        "Processed Row Count": len(X),
                        "Processed Feature Count": X.shape[1]
                    },
                    ignore_index=True)

    # Reports
    def generate_reports(self):
        """
        Generate file specific reports
        :return:
        """

        # Report for evaluation results based on the whole data set
        for label in self.evaluation_results["Label"].unique():
            data = Data_Frame_Helper.get_label_data(self.evaluation_results, label)
            data.to_csv(Path.joinpath(self.folder, f"{label}_evaluation_report.csv"), index=False)

        # Report for y and y_hat
        for label in self.predicted_results["Label"].unique():
            data = Data_Frame_Helper.get_label_data(self.predicted_results, label)
            data.to_csv(Path.joinpath(self.folder, f"{label}_predicted_values_report.csv"), index=False)

        # Report for the split evaluation
        for label in self.split_evaluation_results["Label"].unique():
            data = Data_Frame_Helper.get_label_data(self.split_evaluation_results, label)
            data.to_csv(Path.joinpath(self.split_folder, f"{label}_split_evaluation_report.csv"), index=False)

        # Report for combined datasets (whole, splits)
        combined_evaluation_data_set = self.__create_combined_evaluation_data_set()
        for label in combined_evaluation_data_set["Label"].unique():
            data = Data_Frame_Helper.get_label_data(combined_evaluation_data_set, label)
            data.to_csv(Path.joinpath(self.folder, f"{label}_combined_evaluation_report.csv"), index=False)

        # Report the dataframes for simple df
        for label, data_sets in self.simple_dfs.items():

            if self.simple_df_folder is None:
                continue

            counter = 0
            for data_set in data_sets:
                if data_set.empty:
                    if Config.DEBUG:
                        logging.debug("Dataset is empty! Skipping")
                    continue

                data_set.to_csv(Path.joinpath(self.simple_df_folder, f"{label}_simple_df_{counter}.csv"))
                counter += 1

        # Report the evaluations for simple df
        for label in self.simple_dfs_evaluation["Label"].unique():

            if self.simple_df_folder is None:
                continue

            data = Data_Frame_Helper.get_label_data(self.simple_dfs_evaluation, label)
            data.to_csv(Path.joinpath(self.simple_df_folder, f"{label}_simple_df_evaluation.csv"))

    def __create_combined_evaluation_data_set(self) -> dict:
        """
        Creates a data set containing the full evaluation and the splits
        """
        compare_evaluations = pd.DataFrame()

        for label in self.split_evaluation_results["Label"].unique():
            data = Data_Frame_Helper.get_label_data(self.split_evaluation_results, label)
            data['split'] = True
            compare_evaluations = compare_evaluations.append(data)

        for label in self.evaluation_results["Label"].unique():
            data = Data_Frame_Helper.get_label_data(self.evaluation_results, label)
            data['split'] = False
            compare_evaluations = compare_evaluations.append(data)

        for label in self.simple_dfs_evaluation["Label"].unique():
            data = Data_Frame_Helper.get_label_data(self.simple_dfs_evaluation, label)
            data['split'] = False
            compare_evaluations = compare_evaluations.append(data)

        return compare_evaluations

    # Plots
    def generate_plots(self):
        """
        Helper to call all plotting functions
        :return:
        """
        self.__plot_predicted_values(True)
        self.__plot_predicted_values(False)
        # self.__plot_feature_importance()
        # self.__plot_feature_to_label_correlation()
        self.__plot_pca_analysis()
        self.__plot_pca_analysis_scatter()
        self.__plot_simple_df_test_scores()

    def __plot_simple_df_test_scores(self):
        """
        Plots the test scores for all simple df
        """

        try:
            for label in self.simple_dfs_evaluation["Label"].unique():
                data = self.simple_dfs_evaluation[self.simple_dfs_evaluation["Label"] == label].copy()
                ax = sns.barplot(x="Processed Feature Count", y="Test Score", data=data)
                ax.set(xlabel='Features', ylabel='Test Score')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
                fig = ax.get_figure()
                fig.savefig(os.path.join(self.simple_df_folder, f"{label}_test_scores.png"), bbox_inches='tight')
                fig.clf()
                plt.close('all')

        except BaseException as ex:
            logging.exception(ex)

    def __plot_predicted_values(self, log_scale: bool):
        """
        Plots the predicted values for the unmodified data set
        :return:
        """
        try:
            for label in self.predicted_results["Label"].unique():
                # Get data associated to the label
                data = self.predicted_results[self.predicted_results["Label"] == label]
                ax = sns.scatterplot(x='y', y='y_hat', label=label, data=data)

                if log_scale:
                    ax.set(xscale="log", yscale="log")

                if ax is None:
                    logging.warning("Could not plot predicted values, because axis where None")
                    return

                ax.legend()
                fig = ax.get_figure()
                if log_scale:
                    fig.savefig(os.path.join(self.folder, f"{label}_predicated_log_values.png"))
                else:
                    fig.savefig(os.path.join(self.folder, f"{label}_predicated_values.png"))
                fig.clf()
                plt.close('all')
        except BaseException as ex:
            logging.exception(ex)

    def __plot_feature_importance(self):
        """
        Plots the feature importance for each evaluation
        """

        for label in self.feature_importances["Label"].unique():

            data = Data_Frame_Helper.get_label_data(self.feature_importances, label)
            indices = data[data['Gini-importance'].gt(0.01)].index
            feature_importance = data.T[indices]

            if feature_importance.empty:
                continue

            ax = sns.barplot(data=feature_importance)
            ax.set(xlabel='Feature', ylabel='Gini Index')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
            ax.legend()
            fig = ax.get_figure()
            fig.savefig(os.path.join(self.folder, f"{label}_feature_importance.png"), bbox_inches='tight')
            fig.clf()
            plt.close('all')

    def __plot_feature_to_label_correlation(self):
        """
        Plots the correlation between the feature and labels
        """

        for label in self.feature_importances["Label"].unique():

            data = Data_Frame_Helper.get_label_data(self.feature_importances, label)

            indices = data[data['Gini-importance'].gt(0.01)].index
            feature_importance = data.T[indices]

            important_features = []
            for col in feature_importance.columns:
                important_features.append(col)

            important_features.append(label)

            data = self.preprocessed_df[important_features]
            corr = data.corr()

            mask = np.triu(np.ones_like(corr, dtype=np.bool))
            f, ax = plt.subplots(figsize=(11, 9))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})

            fig = ax.get_figure()
            fig.savefig(os.path.join(self.folder, f"{label}_correlation_matrix.png"), bbox_inches='tight')
            fig.clf()

    def __plot_pca_analysis(self):
        """
        Plots all features and their weight
        """
        try:
            for label, pca in self.pca.items():
                if pca is None:
                    continue

                features = range(pca.n_components_)
                plt.bar(features, pca.explained_variance_ratio_, color='black')
                plt.xlabel('PCA features')
                plt.ylabel('variance %')
                plt.xticks(features)
                plt.xticks(rotation=90, fontsize=8)
                plt.tight_layout()
                plt.savefig(Path.joinpath(self.folder, f"{label}_pca_features.png"), bbox_inches='tight')
                plt.clf()
                plt.close('all')

        except BaseException as ex:
            if Config.DEBUG:
                logging.exception(ex)
                input()
            return

    def __plot_pca_analysis_scatter(self):
        """
        Plots the clustering of the first most important pca components
        """

        try:
            for label, data in self.pca_components_data_frames.items():
                if data.empty:
                    continue

                temp_data = data.copy()
                temp_data[label] = np.log(temp_data[label])

                ax = sns.scatterplot(x=data[0], y=data[1],
                                     hue=label,
                                     data=temp_data)
                ax.set(xlabel='Component 1', ylabel='Component 2')
                ax.legend()
                fig = ax.get_figure()
                fig.savefig(Path(self.folder, f"{label}_pca_cluster.png"), bbox_inches='tight')
                fig.clf()
                plt.close('all')
        except BaseException as ex:
            if Config.DEBUG:
                logging.exception(ex)
                input()

    def __calculate_feature_importance(self, label: str, model, df):
        """
        Calculates the feature importance for the given model
        """

        # TODO: Fix mixing of all importances
        feats = {}  # a dict to hold feature_name: feature_importance
        for feature, importance in zip(df.columns, model.feature_importances_):
            feats[feature] = importance  # add the name/value pair

        importance = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
        importance.sort_values(by='Gini-importance', inplace=True, ascending=False)
        importance = importance.reset_index()
        importance["Label"] = label
        importance.rename(columns={'index': 'Column'}, inplace=True)
        self.feature_importances = self.feature_importances.append(importance)

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

    @staticmethod
    def __lower_threshold(threshold: float) -> float:
        if threshold > 0.1:
            threshold -= 0.1
        elif 0.1 >= threshold > 0:
            threshold -= 0.01

        return round(threshold, 2)
