def remove_bad_columns(dataframe):
    """

    :param dataframe:
    :return:
    """

    print("Removing columns...")

    columns = ['ref_file_filetype',
               'parameters.analysis_type.algorithmic_options.algorithmic_options_selector',
               'parameters.analysis_type.io_options.io_options_selector',
               'parameters.analysis_type.scoring_options.scoring_options_selector',
               'parameters.fastq_input.fastq_input_selector', 'parameters.rg.rg_selector',
               'parameters.reference_source.index_a', 'parameters.analysis_type.analysis_type_selector',
               'job_runner_name', 'handler', 'destination_id', 'parameters.reference_source.ref_file']

    for column in columns:
        del dataframe[column]
    # use pd to remove
    return dataframe


def convert_factorial_to_numerical(dataframe):
    """
    Converts categorical datacolumns to numerical
    :param dataframe:
    :return:
    """

    print("Decoding categorical data columns...")

    cleanup_nums = {"fastq_input2_filetype": {"none": 0, "uncompressed": 1, "compressed": 2},
                    "fastq_input1_filetype": {"compressed": 0, "uncompressed": 1}}

    dataframe.replace(cleanup_nums, inplace=True)
    return dataframe
