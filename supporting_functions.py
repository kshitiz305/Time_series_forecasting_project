import math

import pandas as pd
import numpy as np
from statsmodels.graphics.tukeyplot import results
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import scipy.stats as stats
from pmdarima.arima import auto_arima
import time
import statistics
import matplotlib.pyplot as plt

#supporting functions --------------------------------------------------------------

def column_analyzer(data, developer_mode, ghost_mode, enhanced_diagnostic_mode):
    def dtype_detection(column_data):
        detected_data_types = []
        for item in column_data:
            if isinstance(item, (int, float)):
                detected_data_types.append(type(item))
        unique_dtypes = set(detected_data_types)
        return unique_dtypes

    print()
    types_detected = []
    length_logged = []
    column_names = data.columns.to_list()
    print("---")
    print("Column data with compatible data types detected:")
    print()
    for i in column_names:
        length_logged.append(len(data[i]))
        column_dtypes = dtype_detection(data[i])
        types_detected.append(column_dtypes)

        if not column_dtypes:
            continue

        print(f"Analysis results - Column name: {i};")
        print(f"Types detected: {column_dtypes}")
        print("-")

    print("---")
    print(f"Columns in file: {column_names}")
    if "date" or "DATE" in column_names and ghost_mode == 0:
        print("WARNING!!!!")
        print("Date column detected.")
    print()

def error_metric(historic_data, forecasted_data, purpose, metric):
    dev_mode = 0 #place developer mode here
    improved_diagnostics = 0 #place enhanced diagnostics mode here
    observed = np.array(historic_data)
    forecast = np.array(forecasted_data)

    if len(observed) != len(forecast):
        raise ValueError("Lengths of observed and forecast do not match")

    #MAE(Mean absolute error)
    mae_calculations = np.mean(np.abs(observed - forecast))
    #Root mean square error(RMSE)
    rmse_calculations = np.sqrt( np.mean((observed - forecast)**2 ))
    #mean absolue percentage error
    mape_calculations = np.mean( np.abs((observed - forecast) / (observed + 1e-10) )) * 100
    descriptor = ["Mean Absolte error(MAE)", "Root mean square error(RMSE)", "mean absolue percentage error(MAPE)"]
    results = [mae_calculations, rmse_calculations, f"{mape_calculations}%"]
    if dev_mode == 1:
        for element in range(len(descriptor)):
            print(f"{descriptor[element]} : {results[element]}")
    if improved_diagnostics == 1:
        error_metric_cat = "error metric"
        #apply csv generator ro generate a error metric report
        dataframe_generator(descriptor, results, error_metric_cat, purpose, developer_mode=0, metric=metric, transmutation="Athena")

def data_extermination(array_names, array_cat):
    data_used = array_names
    data_cat = array_cat

    assertion_trigger = 1
    if assertion_trigger == 1:
        assert(type(data_used) == list or type(data_used) == np.ndarray)
        assert(type(data_cat) == list or type(data_cat) == np.ndarray)

    for data_source in data_used:
        data_source.clear()
        del (data_source)

    for array in range(len(data_used)):
        try:
            print(data_used[array][0])
        except:
            print(f"{data_cat[array]} data deletion confirmed")

def diagostics_check(data,data_catagory, developer_mode, tsf_dev_mode_override, ghost_mode, purpose, metric):
    delay = 0
    if delay == 1:
        time.sleep(2)
    cat_copy = data_catagory[:]
    if developer_mode == 1 and tsf_dev_mode_override != 1 and ghost_mode != 1:
        print(f"{cat_copy} diagnostics analysis is initializing")
    if delay == 1:
        time.sleep(5)

    negatives = 0
    positves = 0
    zeros = 0
    for element in data:
        if element > 0:
            positves += 1
        elif element < 0:
            negatives += 1
        elif element == 0:
            zeros += 1

    unique_values = set(data)

    if ghost_mode != 1:
        print("Diagnostics complete")
    mutation_protocol = "Athena"
    diagnostics_cat = f"{data_catagory} enhanced diagnostics"
    descriptor = [f"Data type for {data_catagory.lower()}: ", f"Data length for {data_catagory.lower()}: ", "Fragment: ",
                  f"Number of unique values for {data_catagory.lower()}: ",
                  f"Percentage of values that are unique for {data_catagory.lower()} data: ", "Negative values: ", "Positive values: ", "Zero values: "]
    analysis_elements = [f"{type(data)}", f"{len(data)}",data[:10], f"{len(unique_values)}",
                         f"{round((len(unique_values) / len(data)) * 100,3)}%", negatives, positves,zeros]

    if developer_mode == 1 and ghost_mode != 1:
        print("-------------------------------")
        for output_element in range(len(descriptor)):
            print(f"Descriptor: {descriptor[output_element]} | Result: {analysis_elements[output_element]}")
        print("-------------------------------")

    #CSV generation of diagnostic data
    dataframe_generator(descriptor, analysis_elements, diagnostics_cat, purpose, developer_mode, metric,mutation_protocol)
    if delay == 1:
        time.sleep(3)

def dataframe_generator(description_column, results_column, catagory, purpose, developer_mode, metric, transmutation):
    def data_cataloging(mutation_col_1, mutation_col_2, data_logging_01, data_logging_02):
        assert (type(mutation_col_1) == str and type(mutation_col_2) == str)
        dataframe = pd.DataFrame({
            mutation_col_1: data_logging_01,
            mutation_col_2: data_logging_02
        })
        return dataframe

    cleared_mutations = ["Athena", "Hephestus", "Chronos", "Tzench", "Tracker", "Isolator"]
    if transmutation not in cleared_mutations:
        return "Mutation no cleared. Dataframe generation halted"

    #Prepare for transmutation integration
    #add one parameter for identifying transmutation status
    assert(len(description_column) == len(results_column))
    assert (len(purpose) != 0)
    generation_confirmation = 0
    saving_confirmation = 0
    dataframe_desription = description_column
    dataframe_specifics = results_column
    dataframe_purpose = purpose
    dataframe = pd.DataFrame()
    mutation_protocol_command = transmutation

    #column transmutation for modularity protocols
    #statistical analysis names("Athena")
    standard_c1 = "Description"
    standard_c2 = f"Analysis results({metric})"
    #forecast column isolation("Hephestus")
    hephestus_c1 = "Data point indexes" #modify to dates when date column is integrated
    hephestus_c2 = f"Forecast data points({metric})"
    #Processing time logging
    chrono_c1 = "Function name"
    chrono_c2 = "Processing time"
    #Transformation value tzeench tracking
    tzeench_c1 = f"Pre-change value({metric})"
    tzeench_c2 = f"Post-change value({metric})"
    #Column names for "tracker mutation"
    tracker_c1 = "Pre-modification"
    tracker_c2 = "Post-modification"
    #Column names for isolator
    isolator_C1 = "Date"
    isolator_C2 = f"{purpose} in {metric}"

    generation_signal = 1
    if generation_signal != 1:
        gen_sig_reset = 1
        generation_signal = gen_sig_reset

    if generation_signal == 1:
        if mutation_protocol_command == "Athena":
            dataframe = data_cataloging(standard_c1, standard_c2, dataframe_desription, dataframe_specifics)
            generation_confirmation += 1
        elif mutation_protocol_command == "Hephestus":
            dataframe = data_cataloging(hephestus_c1, hephestus_c2, dataframe_desription, dataframe_specifics)
            generation_confirmation += 1
        elif mutation_protocol_command == "Chronos":
            dataframe = data_cataloging(chrono_c1, chrono_c2, dataframe_desription, dataframe_specifics)
            generation_confirmation += 1
        elif mutation_protocol_command == "Tzench":
            dataframe = data_cataloging(tzeench_c1, tzeench_c2,dataframe_desription, dataframe_specifics)
            generation_confirmation += 1
        elif mutation_protocol_command == "Tracker":
            dataframe = data_cataloging(tracker_c1, tracker_c2, dataframe_desription, dataframe_specifics)
            generation_confirmation += 1
        elif mutation_protocol_command == "Isolator":
            dataframe = data_cataloging(isolator_C1, isolator_C2, dataframe_desription, dataframe_specifics)
            generation_confirmation += 1

    csv_file_name = ""

    if mutation_protocol_command == "Hephestus":
        if developer_mode == 1:
            csv_file_name = f"{catagory.capitalize()} {dataframe_purpose.lower()} CSV data isolation.csv"
        else:
            csv_file_name = f"{dataframe_purpose.capitalize()} CSV forecasted data isolation.csv"
    elif mutation_protocol_command == "Tzench":
        if developer_mode == 1:
            csv_file_name = (f"{catagory.capitalize()} {dataframe_purpose.lower()} CSV transformation tracking.csv")
        else:
            csv_file_name = (f"{dataframe_purpose.capitalize()} CSV transformation tracking.csv")
    else:
        if developer_mode == 1:
            csv_file_name = f"{catagory.capitalize()} {dataframe_purpose.lower()} CSV data analysis results.csv"
        else:
            csv_file_name = f"{dataframe_purpose.capitalize()} CSV data analysis results.csv"

    if developer_mode == 1:
        print(f"CSV file name: {csv_file_name}")

    #saving it as a csv
    dataframe.to_csv(csv_file_name, index=True)

    if developer_mode == 1:
        if generation_confirmation == 1 and saving_confirmation == 1:
            print(f"File name: {csv_file_name}")

def data_analysis_suite(data, purpose, catagory, developer_mode, metric, ghost_mode):
    dev_mode_override = 1

    def IQR_deviation(data_input):
        return_mode = 1
        q1 = np.median(data_input[:10])
        q3 = np.median(data_input[10:])
        iqr = q3 - q1
        quartile_deviation = iqr / 2
        if return_mode == 1:
            return quartile_deviation
        else:
            print(quartile_deviation)
    def advanced_augmentations(data, mutation_protocol): #refine to completion
        assert(type(data) == list or type(data) == np.ndarray)
        assert(type(mutation_protocol) == str)
        mutations = ["Percentile ranks", "IQM", "Semi-quartile range"]

        if mutation_protocol not in mutations:
            return "Mutation incompatible"
        elif mutation_protocol == "Percentile ranks":
            assert(len(data) != 0)

            cnt_below_25 = 0
            cnt_between_25_75 = 0
            cnt_above_75 = 0

            percenile_25 = np.percentile(data, 25)
            percentile_75 = np.percentile(data, 75)
            for value in data:
                if value < percenile_25:
                    cnt_below_25 += 1
                elif value >= percenile_25 and value <= percentile_75:
                    cnt_between_25_75 += 1
                else:
                    cnt_above_75 += 1

            pr_c1 = ["0 - 25", "25 - 75", "75 - 100"]
            pr_c2 = [cnt_below_25, cnt_between_25_75, cnt_above_75]
            pr_catagory = "Percentile ranks"
            dataframe_generator(pr_c1, pr_c2, pr_catagory, purpose, developer_mode, metric, transmutation="Tracker")


        elif mutation_protocol == "IQM":
            assert(len(data) != 0)
            percenile_25 = np.percentile(data, 25)
            percentile_75 = np.percentile(data, 75)
            iqm_data_points = data[(data >= percenile_25) & (data <= percentile_75)]
            iqm_results = np.mean(iqm_data_points)
            return iqm_results

        elif mutation_protocol == "Semi-quartile range":
            assert(len(data) != 0)
            iqr = np.subtract(*np.percentile(data, [75, 25]))
            semi_iqr = iqr/2
            return semi_iqr
    def data_analysis_arm_QUANTILE_REFACTORING(quantile_data):
        metric_updated_quantile_data = []
        for index in quantile_data:
            refactor = f"{round(index, 3)} {metric}"
            metric_updated_quantile_data.append(refactor)
        quantile_data = metric_updated_quantile_data
        return list(quantile_data)

    print()
    if ghost_mode != 1:
        print("----")
        print("Processing initilized")
    print(f"Data catagory: {catagory}")

    if developer_mode == 1 and dev_mode_override == 0 and ghost_mode != 1: #add ghost mode here
        print(f"Data type: {type(data)}")
        print(f"Length of data: {len(data)}")
    elif developer_mode == 1 and dev_mode_override == 1 and ghost_mode != 1: #add ghost mode here
        print("Developer mode override activated")
    elif developer_mode == 0 and dev_mode_override == 1 and ghost_mode != 1: #add ghost mode here
        print("Developer mode disabled")
    elif developer_mode == 0 and dev_mode_override == 0 and ghost_mode != 1: #add ghost mode here
        print("Developer mode disabled")

    data_in_processing = np.array(data)
    descriptor = ["Maximum value", "Minimum value", "Median value",
                  "Mean value", "Standard deviation",
                  "Quantiles", "Data variance", "Variance"]
    upgrades_integration = ["Skewness", "Kurtosis", "Mode"]
    enhanced_elements = ["Range"]
    augmented_stats_elements = ["Inter-quartile mean", "Semi-quartile range"]
    descriptor = descriptor + upgrades_integration + enhanced_elements + augmented_stats_elements
    siginicant_figures_recorded = 3
    data_analysis_element = [round(np.max(data_in_processing),siginicant_figures_recorded), round(np.min(data_in_processing),siginicant_figures_recorded), # 0 1
                             round(np.median(data_in_processing),siginicant_figures_recorded), round(np.mean(data_in_processing),siginicant_figures_recorded), # 2 3
                              round(statistics.stdev(data_in_processing),siginicant_figures_recorded), # 4 5
                            data_analysis_arm_QUANTILE_REFACTORING(statistics.quantiles(data_in_processing)), round(statistics.pvariance(data_in_processing),siginicant_figures_recorded), # 6 7
                             round(statistics.variance(data_in_processing),siginicant_figures_recorded)] # 8

    data_analysis_upgrades = [stats.skew(data_in_processing),stats.kurtosis(data_in_processing), stats.mode(data_in_processing)]
    data_analysis_experimental_enhancements = [round(max(data_in_processing) - min(data_in_processing), siginicant_figures_recorded)]
    augmented_stats_results = [advanced_augmentations(data_in_processing, "IQM"), advanced_augmentations(data_in_processing, "Semi-quartile range")]
    data_analysis_element = data_analysis_element + data_analysis_upgrades + data_analysis_experimental_enhancements + augmented_stats_results
    if developer_mode == 1 and ghost_mode != 1:
        print(".-.-.")
        print(f"Data catagory: {catagory}")
        for i in range(len(data_analysis_element)):
            print(f"{descriptor[i]}:{data_analysis_element[i]}")
        print(".-.-.")
    if ghost_mode != 1:
        print("----")
        print("Processing completed")
        print("----")
    if len(descriptor) == len(data_analysis_element):
        dataframe_generator(descriptor, data_analysis_element, catagory, purpose, developer_mode, metric, transmutation="Athena")
    print()