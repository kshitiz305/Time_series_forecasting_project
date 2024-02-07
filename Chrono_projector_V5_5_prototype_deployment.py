from supporting_functions import dataframe_generator, data_analysis_suite, diagostics_check, column_analyzer
from supporting_functions import data_extermination, error_metric

"""
Notes:
- Add the black wall nan clearing on every list and array as they enter every single function. Nan values will not be tolerated for any processing 
- Encountering circular import errors. Deal with that at a later date
"""

#imports --------------------------------------------------------------
import pandas as pd
import datetime
import time
import os
import math
import requests
import threading
import sklearn.metrics
import numpy as np
from statsmodels.graphics.tukeyplot import results
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import scipy.stats as stats
from pmdarima.arima import auto_arima
import time
import statistics
import matplotlib.pyplot as plt

#Supporting functions

def contains_nan(arr):
    assert(type(arr) == list or type(arr) == np.ndarray)
    nan_found = 0
    for i, value in enumerate(arr):
        if str(value) == 'nan':
            nan_found += 1
            break

    if nan_found == 1:
        return 1  # Return 1 if nan was found

    return 0  # Return 0 if no nan is found

def data_visualization(data, column_name, purpose, forecast, metric, delay, ghost_mode, stat_analysis_mode, enhanced_diagnostic_mode, developer_mode):
    data_clone = black_wall_protocol(data.copy(),"anomaly clearing",stat_analysis_mode,enhanced_diagnostic_mode,developer_mode, purpose, metric, dates=None)
    autodownload_confirmation = 1
    if autodownload_confirmation != 1:
        confirmation_reset = 1
        autodownload_confirmation = confirmation_reset

    # Initial data visualization
    plt.plot(data_clone)
    plt.xlabel(f"Current {purpose} data points")
    plt.ylabel(f"{purpose} in {metric}")
    plt.title(f"{purpose} Current Data({metric})")
    plt.grid(True)
    file_name_pre = f"Inital {purpose.lower()} in {metric}"
    if autodownload_confirmation == 1 and ghost_mode != 1: #add ghost mode here
        plt.savefig(file_name_pre)
        plt.close()
        print("Initial chart has been auto-downloaded")
        print(f"File name: {file_name_pre}")

    if delay == 1:
        time.sleep(1.5)

    # Forecast data visualization
    plt.plot(forecast)
    plt.xlabel(f"{column_name} forecast data trajectory")
    plt.ylabel(f"{purpose} in {metric}")
    plt.title(f"{purpose} forecast Data({metric})")
    plt.legend()
    plt.grid(True)
    file_name_post = f"Forecasted {purpose.lower()} for {column_name} in {metric}"
    if autodownload_confirmation == 1 and ghost_mode != 1:
        plt.savefig(file_name_post)
        plt.close()
        print("Forecasted chart has been auto-downloaded")
        print(f"File name: {file_name_post}")
    else:
        plt.show()
        plt.close()

    data_clone.clear()
    del(data_clone)

def black_wall_protocol(input, mutation_protocols, stat_analysis_mode, enhanced_diagnostic_mode, developer_mode, purpose, metric, dates):
    def black_ice_protocol(input):
        assert(type(input) == list or type(input) == np.ndarray)
        cleared = []
        if type(input) != np.ndarray:
            input = np.array(input)

        input = input.flatten()
        for x in range(len(input)):
            if not math.isnan(input[x]):
                cleared.append(input[x])
            elif type(input[x]) == str:
                continue
            elif type(input[x]) != int or type(input[x]) != float:
                continue

        return cleared

    def outlier_modifier(input_data):
        assert (type(input_data) == list or type(input_data) == np.ndarray)
        if type(input_data) != np.ndarray:
            input_data = np.array(input_data)

        dev_mode_lockdown = 1
        dev_mode = 0
        threshold = np.max(input_data) * 0.85

        data_deletion_protocol_deployment = 1
        # Resets to default disabling dev mode
        if dev_mode != 0 and dev_mode_lockdown == 1:
            dev_mode = 0

        source = black_ice_protocol(input_data)
        original_values = []
        modified_values = []
        transformed_values_outcomes = []

        if dev_mode == 1 and stat_analysis_mode == 1:
            print("<---->")
            print(f"Source data type: {type(source)}")
            print(f"Maximum value: {max(source)}")
            print(f"Minimum value: {min(source)}")
            if enhanced_diagnostic_mode == 1:
                print(f"Range: {max(source) - min(source)}")
            print(f"Threshold: {threshold}")
            print("<---->")

        for elements in source:
            transformer = 2.5 * math.sqrt(elements) * 4.8
            if elements >= threshold:
                transformed_values_outcomes.append(round(transformer, 3))
                original_values.append(elements)
                modified_values.append(round(transformer, 3))
            elif elements < threshold:
                transformed_values_outcomes.append(elements)

        if dev_mode == 1:
            if len(original_values) != 0 and len(modified_values) != 0:
                if len(original_values) == len(modified_values):
                    print("<|-----|>")
                    for pair in range(len(original_values)):
                        print(f"Original value: {original_values[pair]} | Modified: {modified_values[pair]}")
                    print("<|-----|>")

        original_values.clear()
        modified_values.clear()
        if dev_mode == 1:
            cat = ["Original values", "Modified values"]
            deleted_lists = [original_values, modified_values]

            if data_deletion_protocol_deployment == 1:
                print()
                for elements_A1 in range(len(deleted_lists)):
                    try:
                        del (deleted_lists[elements_A1])
                        print(deleted_lists[elements_A1][0])
                    except:
                        print(f"{cat[elements_A1]} has been successfully cleared and deleted.")
                print()

        # Create dataframes of pre and post conversion values
        # apply dataframe generators here at this point

        if data_deletion_protocol_deployment == 1:
            if original_values and modified_values:
                del original_values
                del modified_values

        return transformed_values_outcomes

    # ---
    cleared = ["Anomaly clearing", "Sanitation", "Sync"]
    if mutation_protocols not in cleared:
        return "Protocol does not exist on record"
    # ---
    # Just clears data of nans and thats it.
    elif mutation_protocols == "Anomaly clearing" and dates == None:
        # Notes
        # - No issues. Working exactly as planned.
        # - Some streamline is an option but fuctionally there are no issues.
        # - Any upgardes are welcome but not a requirement
        assert (type(input) == list or type(input) == np.ndarray)
        nan_detected = 0
        cleared = black_ice_protocol(input)
        if nan_detected != 0:
            print(f"Number of nan values: {nan_detected}")
        return cleared
    # --- Prepares the data for sarimax input
    elif mutation_protocols == "Sanitation" and dates == None:
        assert (type(input) == list or type(input) == np.ndarray)
        return_mode = 1
        if input:
            if len(input) == 0:
                print(f"Length of '{mutation_protocols}' data input is {len(input)}")
                return input

        training_data_applied = input
        nan_cleared_traning_data = black_ice_protocol(training_data_applied)
        # prepares it for machine learning deployment
        ml_ready = nan_cleared_traning_data
        # cleans the data as a failsafe just in case any undesirable elements are still present like nan values for example
        ml_proccessing_data = np.array(ml_ready)

        #Prepare the outlier handling functionality
        sarimax_compatible_output = outlier_modifier(ml_proccessing_data)

        if return_mode == 1:
            return sarimax_compatible_output
    # Sycronizer
    elif mutation_protocols == "Sync" and len(dates) > 0:
        if len(dates) > 0:
            data = input
            cleaned_data = []
            date_data = dates
            cleaned_date_data = []

            for value, date in zip(data, date_data):
                if not np.isnan(value):
                    cleaned_data.append(value)
                    cleaned_date_data.append(date)

            cleaned_data = np.array(cleaned_data)
            cleaned_date_data = np.array(cleaned_date_data)
            return cleaned_date_data, cleaned_data
        else:
            quit("Syncronizer failed to syncronize data because it there is an empty dates column or the dates column does not exist.")



# Critical functions--------------------------------------------

def specilized_stat_analysis(input_data, forecast_clone, developer_mode,enhanced_diagnostic_mode, metric, purpose): # isolate mankendall test and sens graph
    import pymannkendall as mk
    def specilized_analysis(input_data, developer_mode, purpose, metric, status):
        #isolate 10 years of data

        mk_results_output = mk.original_test(input_data)

        catagory = f"Specilized analysis for {status}"
        descriptions = ["Trend", "H", "P-value", "Z", "Sens slope"]
        results = [mk_results_output.trend, mk_results_output.h,
                   mk_results_output.p, mk_results_output.z, mk_results_output.slope]

        if developer_mode == 1:
            for result in range(len(descriptions)):
                print(f"{descriptions[result]} {results[result]}")

        dataframe_generator(descriptions, results, catagory, purpose, developer_mode, metric, "Athena")

    assert(type(input_data) == list or type(input_data) == np.ndarray)
    assert(type(forecast_clone) == list or type(forecast_clone) == np.ndarray)
    assert(type(developer_mode) == int)
    assert(type(enhanced_diagnostic_mode) == int)
    assert(type(metric) == str)

    if developer_mode == 1:
        c1_info = ["Historic data type: ", f"Historic data fragment(Full data length:{len(input_data)}): ",
                   f"Forecast data fragment(Full data length:{len(forecast_clone)})"]
        c2_details = [type(input_data), input_data[:10], forecast_clone[:10]]
        print("-----")
        for element in range(len(c1_info)):
            print(f"{c1_info[element]}: {c2_details[element]}")
        print("-----")
        print()

    activation = 1
    #mann kendal test, sens slope
    man_kendal_test_cat = "Mann kendal test data"
    sens_test_cat = "Sens test data"
    if activation == 0:
        return "Specilized statistical analysis dormant"
    elif activation == 1:
        status = ["Historic", "Projected"]
        historic_data = input_data
        forecasted_data = forecast_clone
        if __name__ == "__main__":
            t1 = threading.Thread(target=specilized_analysis(historic_data, developer_mode, purpose, metric, status[0]))
            t2 = threading.Thread(target=specilized_analysis(forecasted_data, developer_mode, purpose, metric, status[1]))

            t1.start(), t2.start()

            t1.join(), t2.join()
    else:
        return "Activation failure, double check that the activation variable is set to 1"

def data_extractor_protocol(data_input, forecast_clone):
    assert(type(data_input) == list or type(data_input) == np.ndarray)
    assert(type(forecast_clone) == list or type(forecast_clone) == np.ndarray)

    if forecast_clone:
        if len(forecast_clone) != 0:
            print("Forecast clone present")
            print(f"Length of forecast data: {len(forecast_clone)}")
            print(f"Cloned forecast data fragment: {forecast_clone[:10]}")

    print("Extractor function dormant")

def time_series_analysis(data, purpose, developer_mode, metric, delay, stat_analysis_mode, tsf_dev_mode_override,
                         data_view_mode, column_name, enhanced_diagnostic_mode, ghost_mode, forecast_horizon_selected, cloned_forecast, confirmation_activation):
    def forecast_mechanism_isolation(current_data, sarimax_params, forecast_horizon, forecast, cycles, confirmation_activation,
                                     developer_mode, purpose, metric):
        assert(type(current_data) == list or type(current_data) == np.ndarray or type(current_data) == dict)
        assert(type(forecast_horizon) == int)
        assert(type(forecast) == list or type(forecast) == np.ndarray)
        assert(type(cycles) == int)
        initial_cycles = cycles
        updated_cycles = 0
        # noinspection PyTypeChecker
        data = black_wall_protocol(current_data, "sanitation",stat_analysis_mode,
                                   enhanced_diagnostic_mode, developer_mode, purpose, metric, dates=None)

        for element in range(0, len(data) - forecast_horizon, forecast_horizon):
            updated_cycles += 1
            if updated_cycles == 1:
                print("Forecast initiated...")

            current_data = data[element:element + forecast_horizon]
            # forecast fitting
            model = SARIMAX(current_data, order=sarimax_params)
            model_fit = model.fit()
            # perform forecast on current horizon
            forecast_section = model_fit.forecast(steps=forecast_horizon)
            modified_forecast_selection = []
            for element in forecast_section:
                modified_forecast_selection.append(round(element,4))
            forecast_section = modified_forecast_selection
            forecast.extend(forecast_section)

        if enhanced_diagnostic_mode == 1:
            df_cat = "Forecast cycles completed"
            c1 = ["Initial", "Post-completion"]
            c2 = [initial_cycles, updated_cycles]
            dataframe_generator(c1,c2,df_cat,purpose,developer_mode, metric, transmutation="Athena")
    def sarimax_data_acquisition(list_injected):
        model = auto_arima(list_injected, start_p=1, start_q=1, max_p=3,
                           max_q=3, m=12, start_P=0, d=1, D=1, trace=False,
                           error_action='ignore', seasonal=True, suppress_warnings=True,
                           stepwise=True)
        return model.order

    assert (type(data) == list or type(data) == np.ndarray or type(data) == dict)
    assert(type(purpose) == str)
    assert(type(metric) == str)
    assert(stat_analysis_mode == 1)
    assert(type(developer_mode) == int and type(enhanced_diagnostic_mode) == int)
    assert(type(ghost_mode) == int)

    ratio = 0.85
    train_data = data[:int(len(data) * ratio)]
    test_data = data[int(len(data) * ratio):]
    data_cat = "Initial data"

    # Find SARIMAX parameters
    sarimax_params = sarimax_data_acquisition(train_data)

    # Select forecast horizon
    forecast_horizon = forecast_horizon_selected
    forecast = []
    forecast_cycles = 0
    # Perform forecasting
    print("Forecasting protocols initiating...")
    time.sleep(2)
    # Developer Mode Check
    if developer_mode == 1 and ghost_mode != 1:
        if tsf_dev_mode_override == 0:
            print("....")
            print(f"Initial cycle count: {forecast_cycles}")
            print("....")
        else:
            print("Developer mode override protocol active")

    if forecast_horizon == 20 and confirmation_activation == 1:
        forecast_mechanism_isolation(current_data = data, sarimax_params=sarimax_params, forecast_horizon=forecast_horizon,
                                     forecast=forecast, cycles = forecast_cycles, confirmation_activation=confirmation_activation, developer_mode= developer_mode, purpose=purpose, metric=metric)
    else:
        forecast_mechanism_isolation(current_data=data, sarimax_params=sarimax_params,
                                     forecast_horizon=forecast_horizon,
                                     forecast=forecast, cycles = forecast_cycles, confirmation_activation=confirmation_activation,
                                     developer_mode= developer_mode, purpose=purpose, metric=metric)

    # Developer Mode Check
    if developer_mode == 1  and ghost_mode != 1: #add ghost mode here
        if tsf_dev_mode_override == 0:
            print("....")
            print(f"Forecast cycles: {forecast_cycles}")
            print("....")
        else:
            print("Developer mode override protocol active")

    forecast = [data[-1]] + forecast
    forecast_datapoints = []
    for element in forecast:
        forecast_datapoints.append(element)
        cloned_forecast.append(element)

    print("-------------")
    if data_view_mode == 1 and ghost_mode != 1: #add ghost mode here
        print("Forecasted data fragment: ", forecast[:25])
    elif data_view_mode == 0 and ghost_mode != 1: #add ghost mode here
        print("Data viewing disabled.")

    forecast_cat = "Forecast"

    if delay == 1:
        time.sleep(3)

    if __name__ == "__main__":

        t1 = threading.Thread(target=diagostics_check(forecast, forecast_cat,developer_mode, tsf_dev_mode_override, ghost_mode,purpose, metric))
        t2 = threading.Thread(target=diagostics_check(data, data_cat,developer_mode,tsf_dev_mode_override,ghost_mode, purpose, metric))
        t3 = threading.Thread(target=data_analysis_suite(forecast, purpose, forecast_cat, developer_mode, metric, ghost_mode))
        t4 = threading.Thread(target=data_analysis_suite(data, purpose,data_cat, developer_mode, metric, ghost_mode))
        t5 = threading.Thread(target=data_visualization(data, column_name, purpose, forecast, metric, delay, ghost_mode, enhanced_diagnostic_mode, developer_mode,purpose))
        t6 = threading.Thread(target=dataframe_generator(forecast_datapoints, forecast, forecast_cat, purpose, developer_mode, metric, transmutation="Hephestus"))

        processes = [t1, t2, t3, t4, t5, t6]
        custom_thread_deployment = [t3, t4, t5, t6]

        for threads_A1 in processes:
            threads_A1.start()

        if enhanced_diagnostic_mode == 1 and enhanced_diagnostic_mode == 1:
            # start the processes if diagnostic mode is active
            for threads_A2 in processes:
                threads_A2.join()
        elif stat_analysis_mode == 1 and enhanced_diagnostic_mode != 1:
            # t1 and t2 are excluded
            for c_threads in custom_thread_deployment:
                c_threads.join()
        elif enhanced_diagnostic_mode == 1:
            for threads_A3 in processes:
                threads_A3.join()
        else:
            # t1 and t2 are excluded
            for threads_A2 in custom_thread_deployment:
                threads_A2.join()




overall_start = datetime.datetime.now()


project_name = "Chrono projector"

#status displayer(Initiate: 1, disable: 0)
status_verifier = 1

#ghost mode(Initiate: 1, disable: 0)
ghost_mode = 0
if ghost_mode == 0:
    print("Ghost mode inactive")
if ghost_mode == 1:
    updated_dev_mode_status = 0
    developer_mode = updated_dev_mode_status
    tsf_dev_mode_override = updated_dev_mode_status

if ghost_mode != 1:
    version = 5.0
    print("---------")
    print(f"Current version: {version}")
    print("---------")
    print()

ignore_warnings = 1
if ignore_warnings == 1:
    import warnings
    warnings.filterwarnings("ignore")
    print("Warnings are disabled")

if ghost_mode != 1:
    print(".-.-.-.-.-.")
#Real world deployment mode
rw_deployment = 0
#Developer mode(Initiate: 1, disable: 0)
developer_mode = 1
#View mode(Initiate: 1, disable: 0)
data_view_mode = 0
#Time Series Forecasting function developer mode override(Intiate: 1, disable: 0)
tsf_dev_mode_override = 0
#time delay mode(Initiate: 1, disable: 0)
delay = 0
#enhanced diagnostics mode(Initiate: 1, disable: 0)
enhanced_diagnostic_mode = 1

#---------------
#Status checks

#developer mode
if developer_mode == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Developer mode active")

#enhanced diagnostic mode checker
if enhanced_diagnostic_mode == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Enhanced diagnostics mode active")

#statistical analysis mode permanently active
stat_analysis_mode = 1

if stat_analysis_mode == 0:
    default_stat_analysis_state = 1
    stat_analysis_mode = default_stat_analysis_state
elif stat_analysis_mode != 1 or stat_analysis_mode != 0:
    failsafe_reset = 1
    stat_analysis_mode = failsafe_reset

#delay trigger checker
if delay == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Delay protocols active")

#sarimax dev mode override checker
if tsf_dev_mode_override == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Chrono Projector dev mode override protocol active")

#data view mode checker
if data_view_mode == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Data view mode active")
#---------------
if ghost_mode != 1:
    print("Ghost mode disabled")

experimental_blocker = 0

if rw_deployment == 1:
    ghost_mode = 1
    developer_mode = 0
    enhanced_diagnostic_mode = 0
    tsf_dev_mode_override = 0
    status_verifier = 0
    experimental_blocker = 1
print(".-.-.-.-.-.")
time.sleep(1.5)

real_world_mass_data_testing = [
    r"C:\Users\Tomy\PycharmProjects\Experiment - 7\Chrono Projector\Real world deployment prototypes\Real world data sets\Mass data sets\3499853.csv",
    r"C:\Users\Tomy\PycharmProjects\Experiment - 7\Chrono Projector\Real world deployment prototypes\Real world data sets\Mass data sets\Bale data.csv",
    r"C:\Users\Tomy\PycharmProjects\Experiment - 7\Chrono Projector\Real world deployment prototypes\Real world data sets\Mass data sets\SOUTHERN REGION DATA SET-Compatibility modified - SOUTHERN REGION DATA SET- ONE TABLE VS.6.csv"
]


#-----------------------------------------------------------------------------------
#Inputs
cleared_status = ["Enabled", "Disabled"]
print(f"API modes usable: {cleared_status}")
api_status = input("Selected mode: ")
if api_status not in cleared_status and api_status == "Experimental":
    exp_status = 0
    if developer_mode == 1 and enhanced_diagnostic_mode == 1 and ghost_mode == 0:
        if experimental_blocker == 0:
            # Trigger experimetal mode that extends it to 75 years of forecast
            confirm_trigger = 8008
            print(f"Experimental status: {exp_status}. Get ready to have some unhinged fun")
            print("Warning!! Not usable for the ordinary user. Are you sure you want to proceed?")
            confirm_code = int(input("Type the confirm trigger: "))
            if confirm_code == confirm_trigger:
                #deploy sarimax with a custom heavily modified time series
                pass

else:
    while api_status not in cleared_status:
        print("-----------------------------------------")
        print(f"Authorized modes: {cleared_status}")
        api_status = input("Please use authorized modes: ")
        print("-----------------------------------------")

if api_status == "Enabled":
    KS = 0
    api_override_kill_switch = 1
    if api_override_kill_switch == 1:
        KS += 1
        print(f"Api kill-switch status: {KS}")
        quit("API enabled process killed.")

elif api_status == "Disabled":
    # input mechanism for a file input
    """
    file_path = input("File path to source data: ")
    source_path = fr"{file_path}"

    if not os.path.exists(source_path):
        print("File does not exist at the specified path. Please check the path and try again.")
    else:
        print("Please input the correct path.")
        # while loop that does not let you leave until you give a path to a file that exists
    """
    #Triggered during events where there is no data to aquire from satellites

    print("*---*")
    # Load the time series data
    file_path = real_world_mass_data_testing[2] #replace with file path prompt at a later date
    if len(file_path) == 0:
        quit("No input file detected")

    data = pd.read_csv(file_path)
    column_analyzer(data, developer_mode, ghost_mode, enhanced_diagnostic_mode)

    # temperature target column
    target_TEMP_column_name =input("Temperature target column name: ")
    target_TEMP_column = data[target_TEMP_column_name].to_numpy()

    #Rainfall column
    target_PERCP_column_name = input("Percipitation target column name: ")
    target_PERCP_column = data[target_PERCP_column_name].to_numpy()

    #Humidity column
    target_humidity_column_name = input("Humidity target column name: ")
    target_humidity_column = data[target_humidity_column_name].to_numpy()

    # Date column name
    date_column_name = input("Date column name: ")
    date_copy_01 = data[date_column_name].to_numpy()
    date_copy_02 = []
    date_copy_03 = []
    #Cloning the date data on 2 other lists before they get converted to numpy arrays post-cloning
    for date_data in date_copy_01:
        date_copy_02.append(date_data)
        date_copy_03.append(date_data)
    #Cloned dates for rainfall and humidity
    date_copy_02 = np.array(date_copy_02)
    date_copy_03 = np.array(date_copy_03)

    if developer_mode == 1 and ghost_mode != 1:
        print("Combined data analysis")
        cat_stores = ["Temperature", "Percipitation/rainfall", "Humidity","Date clone 1", "Date clone 2", "Data clone 3"]
        data_stores = [target_TEMP_column, target_PERCP_column, target_humidity_column,date_copy_01, date_copy_02, date_copy_03]
        if len(data_stores) == len(cat_stores):
            for element in range(len(cat_stores)):
                print("---")
                print(f"{cat_stores[element]} data type: {type(data_stores[element])}")
                print(f"{cat_stores[element]} data volume: {len(data_stores[element])}")
                if type(data_stores[element]) == np.ndarray:
                    print(f"{cat_stores[element]} data shape: {data_stores[element].shape}")
                    print(f"{cat_stores[element]} dimensions detected: {data_stores[element].ndim}")
                    if data_stores[element].ndim > 1:
                        print(f"WARNING! {cat_stores[element]} has {data_stores[element].ndim} dimensions")
        print("---")
        cat_stores_A1 = ["Temperature", "Percipitation/rainfall", "Date clone 1", "Date clone 2"]
        data_stores_A1 = [target_TEMP_column, target_PERCP_column, date_copy_01, date_copy_02, date_copy_03]
        detection_triggers = []
        if len(cat_stores_A1) == len(data_stores_A1):
            # Use a for loop mechanism to log nan values detection algorithm
            for element_A1 in range(len(data_stores_A1)):
                detection_analysis = contains_nan(data_stores_A1[element_A1])
                detection_triggers.append(detection_analysis)

            for element_A2 in range(len(detection_triggers)):
                if detection_triggers[element_A2] == 1:
                    print(f"NAN values detected in {cat_stores_A1[element_A2]} array")
        print()

    temperature_cloned_forecast = []
    percipitation_cloned_forecast = []
    print("Preparing offline inputs...")
    temp_metrics_availible = ["°C", "°F"]
    percp_metrics_availible = ["mm"]
    humidity_metrics_availible = ["%"]

    purpose_temp = input("Type of temperature data: ")
    purpose_percp = input("Type of percipitation data: ")
    purpose_humidity = input("Type of humidity data: ")

    print("Note: The default is °C. You can convert to °F if you choose to.")
    temperatue_metric = input(f"Temperature metric being applied in the data forcasting(Compatible metrics{temp_metrics_availible}): ")
    percipitation_metric = input(f"Percipitation metric applied to data forecasting(Compatible metrics{percp_metrics_availible}): ")
    humidity_metric = input(f"Humidity metric applied to data forecasting(Compatible metrics{humidity_metrics_availible}): ")
    # Synchronization of dates and data columns process
    #Temperature
    synced_temp_dates, synced_temp_data = black_wall_protocol(target_TEMP_column, "Sync", stat_analysis_mode,enhanced_diagnostic_mode,
                                                              developer_mode,purpose_temp,temperatue_metric,date_copy_01)
    #Rainfall
    synced_rainfall_dates, synced_rainfall_data = black_wall_protocol(target_PERCP_column,"Sync",stat_analysis_mode,enhanced_diagnostic_mode,
                                                                      developer_mode, purpose_percp, percipitation_metric, date_copy_02)
    #Humidity
    sync_humidity_dates, synced_humidity_data = black_wall_protocol(target_humidity_column, "Sync", stat_analysis_mode, enhanced_diagnostic_mode, developer_mode,
                                                                    purpose_humidity, humidity_metric, date_copy_03)

    # Add °F converter if they choose to use °F
    fahrenheit_conversion = []
    if temperatue_metric == "°F":
        for temperatures in range(len(synced_temp_data)):
            converted_temperature = (synced_temp_data[temperatures] * 9 / 5) + 32
            fahrenheit_conversion.append(converted_temperature)
        synced_temp_data = fahrenheit_conversion
        if developer_mode == 1:
            print("Conversion to fahrenheit completed.")
            time.sleep(2.5)

    # Code breakers if nan values are detected. If this breaks your code you have nan values. Easy to understand.
    temp_func_integration_S1 = np.array(synced_temp_data)
    percp_func_integration_S1 = np.array(synced_rainfall_data)
    humidity_func_integration_S1 = np.array(synced_humidity_data)

    sync_temp_nan_status = contains_nan(temp_func_integration_S1)
    if sync_temp_nan_status == 1:
        quit("nan value detected in percipitation data post-sync")
    else:
        dataframe_generator(synced_temp_dates, temp_func_integration_S1, "Raw temperature data post-sync",
                            purpose_temp,developer_mode,temperatue_metric, "Isolator")

    sync_percp_nan_status = contains_nan(percp_func_integration_S1)
    if sync_percp_nan_status == 1:
        quit("nan value detected in temperature data post-sync")
    else:
        dataframe_generator(synced_rainfall_dates, percp_func_integration_S1, "Raw percipitation data post-sync",
                            purpose_percp, developer_mode, percipitation_metric, "Isolator")

    sync_humidity_nan_status = contains_nan(humidity_func_integration_S1)
    if sync_humidity_nan_status == 1:
        quit("nan value detected in humidity data post-sync")
    else:
        dataframe_generator(sync_humidity_dates,humidity_func_integration_S1,"Raw humidity data post-sync",
                            purpose_humidity, developer_mode, humidity_metric, "Isolator")

    #forecast horizon
    years_selectable = [5, 10, 15, 20]
    confirmation_activation = 0
    print(f"Available forecast horizons,  {years_selectable}")
    forecast_horizon_selected = int(input("Forecast horizon: "))
    if forecast_horizon_selected not in years_selectable:
        while forecast_horizon_selected not in years_selectable:
            print("Please input compatible forecast horizon")
            forecast_horizon_selected = int(input("Forecast horizon:"))
    elif forecast_horizon_selected == 20:
        print("Confirmation is required for continuing with a 20 year forecast.")
        confirmation = int(input("Do you confirm this 20 year forecast time frame?(press 1 to confirm, 0 to abort forecast): "))
        if confirmation == 1:
            confirmation_activation += 1
        else:
            quit(f"{project_name} disabled due to user not confirming 20 year forecast horizon.")

    #verification of parameters
    print("-............................-")
    time_frame_S = [5, 10, 15, 20]
    print("Verification of temperature data: ")
    # Start puting seperations between developer mode active or inactive
    verification_c1 = ["Temperature purpose", "Temperature metric", "Rainfall purpose", "Rainfall metric", "Forecast horizon selected"]
    verification_c2 = [purpose_temp, temperatue_metric, purpose_percp, percipitation_metric, forecast_horizon_selected]
    dev_c1_update = [0]
    developer_C1 = verification_c1 + dev_c1_update
    dev_c2_update = [0, 1]
    developer_C2 = verification_c2 + dev_c2_update
    if len(verification_c1) == len(verification_c2):
        print()
        print("Please verify perameters...") # prints out parameters in a way that is adjustable and not using too many lines
        for element in range(len(verification_c1)):
            print(f"{verification_c1[element]}: {verification_c2[element]}")
        print()
    else:
        print("length mismatch between the verification lists.")
        if developer_mode == 1 and rw_deployment == 0 and ghost_mode == 0:
            print(f"Verificication c1: {len(verification_c1)} | verification c2: {len(verification_c2)}")

    if developer_mode == 1:
        if len(developer_C1) == len(developer_C2):
            # Add specific dev mode specifics into the for loop
            pass

    print()
    print("|------------------------------------------------------|")
    file_confirmation = input(
        "Do you confirm that the file and column used is the correct file and column? 1 to confirm, 0 to deny: ")
    print()
    deployment_confirmation = input("Confirm SARIMAX deployment. 1 to confirm deployment, 0 to disable SARIMAX: ")
    print("|------------------------------------------------------|")
    deployment = 0
    if deployment_confirmation == '1' and file_confirmation == '1':
        deployment += 1

    if deployment == 1:
        print("Deployment status [ENABLED]")
        print(f"{project_name} deployment in...")
        countdown = [5, 4, 3, 2, 1]
        for cnt in countdown:
            print(cnt)
            time.sleep(1)
        print("...........................")
        # Replace the delivery system entirely
        logged_data_sources = [temp_func_integration_S1, percp_func_integration_S1, humidity_func_integration_S1]
        logged_purposes = [purpose_temp, purpose_percp, purpose_humidity]
        logged_metrics = [temperatue_metric, percipitation_metric, humidity_metric]
        logged_forecast_horizon = forecast_horizon_selected

        # Insert into a custom deployment class that deploys multicore/multithread hybrid
        # Sequence -> time series forecasting, specilized statistical analysis, error metric, possible other experimental components if possible
        # Deploy sarimax first, multithread everything else


    else:
        print("Deployment status [DISABLED]")