

def process_merged_tool_version():
    """
    Processes all detected tool versions
    :return:
    """
    pass


def process_single_files():
    pass


def process_single_file_data_removal():
    """TBD"""
    pass


def process_data_sets():
    """
    Processes all data sets, read from the given folder
    :return:
    """

    if len(Runtime_Datasets.RAW_FILE_PATHS) == 0:
        print("No files found to evaluate. Stopping")
        FolderManagement.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
        sys.exit()

    # Get similar files in directory

    # merged file prediction
    for file, path in Runtime_Datasets.RAW_SIMILAR_FILES.items():
        input()
        data_frames = []

        for filename in path:
            General_File_Service.read_file(filename)
            data_frames.append(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)

        merged_df = pd.concat(data_frames)
        print(f"Evaluating {file}")
        Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET = merged_df
        General_File_Service.create_tool_folder(file)
        PreProcessing.pre_process_data_set(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)
        Misc.set_general_file_data(file)

        if Runtime_File_Data.EVALUATED_FILE_ROW_COUNT < Config.Config.MINIMUM_ROW_COUNT:
            if Runtime_Datasets.COMMAND_LINE_ARGS.verbose:
                print(f"Data set {file} has insufficient row count and will be ignored. ")

            Runtime_Datasets.EXCLUDED_FILES = Runtime_Datasets.EXCLUDED_FILES.append(
                {'File': file, 'Rows': Runtime_File_Data.EVALUATED_FILE_ROW_COUNT}, ignore_index=True)
            FolderManagement.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
            continue

        Single_Predictions.compare_real_to_predicted_data()

        if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
            if Runtime_File_Data.EVALUATED_FILE_NO_USEFUL_INFORMATION:
                print(
                    f"File {Runtime_File_Data.EVALUATED_FILE_NAME} does not contain useful information. Skipping...")
                continue
            print("Removing data by percentage")
            Predict_Data_Removal.removal_helper()

        generate_file_report_files()

    # Single file prediction
    for filename in Runtime_Datasets.RAW_FILE_PATHS:
        try:
            print(f"Evaluating {filename}")
            # Generate tool folder
            General_File_Service.create_tool_folder(filename)
            # Load the data set
            General_File_Service.read_file(filename)
            # Pre process the df
            PreProcessing.pre_process_data_set(Runtime_File_Data.EVALUATED_FILE_RAW_DATA_SET)
            Misc.set_general_file_data(filename)

            # Test for infinite values
            for column in Runtime_File_Data.EVALUATED_FILE_PREPROCESSED_DATA_SET:
                if Runtime_File_Data.EVALUATED_FILE_PREPROCESSED_DATA_SET[column].any() > np.iinfo('i').max:
                    continue

            # Check if dataset row count is equal or greater compared to the treshold set in the config
            if Runtime_File_Data.EVALUATED_FILE_ROW_COUNT < Config.Config.MINIMUM_ROW_COUNT:
                if Runtime_Datasets.COMMAND_LINE_ARGS.verbose:
                    print(f"Data set {filename} has insufficient row count and will be ignored. ")

                FolderManagement.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
                Runtime_Datasets.EXCLUDED_FILES = Runtime_Datasets.EXCLUDED_FILES.append(
                    {'File': filename, 'Rows': Runtime_File_Data.EVALUATED_FILE_ROW_COUNT}, ignore_index=True)
                continue

            # Working on full data set
            print("Evaluation full data set")
            Single_Predictions.compare_real_to_predicted_data()

            # Remove data by percentage
            if Runtime_Datasets.COMMAND_LINE_ARGS.remove:
                if Runtime_File_Data.EVALUATED_FILE_NO_USEFUL_INFORMATION:
                    if Runtime_Datasets.COMMAND_LINE_ARGS.verbose:
                        print(f"File {Runtime_File_Data.EVALUATED_FILE_NAME}"
                              f"does not contain useful information. Skipping...")
                    continue

                print("Removing data by percentage")
                Predict_Data_Removal.removal_helper()

            generate_file_report_files()

        except Exception as ex:
            print("error occurred in process_data_sets()")
            print(ex)
            FolderManagement.remove_folder(Runtime_Folders.CURRENT_WORKING_DIRECTORY)
            sys.exit()
