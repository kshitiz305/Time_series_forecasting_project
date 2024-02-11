from Multi_core_processor import Chrono_projector_deployment
from critical_functions import contains_nan, black_wall_protocol
from supporting_functions import dataframe_generator, data_analysis_suite, diagostics_check, column_analyzer
from supporting_functions import data_extermination, error_metric

"""Notes: - Add the black wall nan clearing on every list and array as they enter every single function. Nan values 
will not be tolerated for any processing - Encountering circular import errors. Deal with that at a later date"""

# imports --------------------------------------------------------------
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
import datetime
from pmdarima.arima import auto_arima
import time
import statistics
import matplotlib.pyplot as plt


# Supporting functions


overall_start = datetime.datetime.now()

project_name = "Chrono projector"

# status displayer(Initiate: 1, disable: 0)
status_verifier = 1

# ghost mode(Initiate: 1, disable: 0)
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
# Real world deployment mode
rw_deployment = 0
# Developer mode(Initiate: 1, disable: 0)
developer_mode = 1
# View mode(Initiate: 1, disable: 0)
data_view_mode = 0
# Time Series Forecasting function developer mode override(Intiate: 1, disable: 0)
tsf_dev_mode_override = 0
# time delay mode(Initiate: 1, disable: 0)
delay = 0
# enhanced diagnostics mode(Initiate: 1, disable: 0)
enhanced_diagnostic_mode = 1

# ---------------
# Status checks

# developer mode
if developer_mode == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Developer mode active")

# enhanced diagnostic mode checker
if enhanced_diagnostic_mode == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Enhanced diagnostics mode active")

# statistical analysis mode permanently active
stat_analysis_mode = 1

if stat_analysis_mode == 0:
    default_stat_analysis_state = 1
    stat_analysis_mode = default_stat_analysis_state
elif stat_analysis_mode != 1 or stat_analysis_mode != 0:
    failsafe_reset = 1
    stat_analysis_mode = failsafe_reset

# delay trigger checker
if delay == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Delay protocols active")

# sarimax dev mode override checker
if tsf_dev_mode_override == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Chrono Projector dev mode override protocol active")

# data view mode checker
if data_view_mode == 1 and ghost_mode != 1:
    if status_verifier == 1:
        print("Data view mode active")
# ---------------
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
    r".\SOUTHERN REGION DATA SET-Compatibility modified - SOUTHERN REGION DATA SET- ONE TABLE VS.6.csv"
]

# -----------------------------------------------------------------------------------
# Inputs
cleared_status = ["Enabled", "Disabled"]
print(f"API modes usable: {cleared_status}")
api_status = 'Disabled'#input("Selected mode: ")
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
                # deploy sarimax with a custom heavily modified time series
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
    # Triggered during events where there is no data to aquire from satellites

    print("*---*")
    # Load the time series data
    file_path = real_world_mass_data_testing[2]  # replace with file path prompt at a later date
    if len(file_path) == 0:
        quit("No input file detected")

    data = pd.read_csv(file_path)
    column_analyzer(data, developer_mode, ghost_mode, enhanced_diagnostic_mode)

    # temperature target column
    target_TEMP_column_name = 'Tmp. max.'#input("Temperature target column name: ")
    target_TEMP_column = data[target_TEMP_column_name].to_numpy()

    # Rainfall column
    target_PERCP_column_name = 'Prc.'#input("Percipitation target column name: ")
    target_PERCP_column = data[target_PERCP_column_name].to_numpy()

    # Humidity column
    target_humidity_column_name = 'Rel. Hum.'#input("Humidity target column name: ")
    target_humidity_column = data[target_humidity_column_name].to_numpy()

    # Date column name
    date_column_name = 'Date'#input("Date column name: ")
    date_copy_01 = data[date_column_name].to_numpy()
    date_copy_02 = []
    date_copy_03 = []
    # Cloning the date data on 2 other lists before they get converted to numpy arrays post-cloning
    for date_data in date_copy_01:
        date_copy_02.append(date_data)
        date_copy_03.append(date_data)
    # Cloned dates for rainfall and humidity
    date_copy_02 = np.array(date_copy_02)
    date_copy_03 = np.array(date_copy_03)

    if developer_mode == 1 and ghost_mode != 1:
        print("Combined data analysis")
        cat_stores = ["Temperature", "Percipitation/rainfall", "Humidity", "Date clone 1", "Date clone 2",
                      "Data clone 3"]
        data_stores = [target_TEMP_column, target_PERCP_column, target_humidity_column, date_copy_01, date_copy_02,
                       date_copy_03]
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
    humidity_cloned_forecast = []
    print("Preparing offline inputs...")
    temp_metrics_availible = ["°C", "°F"]
    percp_metrics_availible = ["mm"]
    humidity_metrics_availible = ["%"]

    purpose_temp = '°C' #input("Type of temperature data: ")
    purpose_percp = 'mm'#input("Type of percipitation data: ")
    purpose_humidity = '%'#input("Type of humidity data: ")

    print("Note: The default is °C. You can convert to °F if you choose to.")
    temperatue_metric = '°C'#input(
        #f"Temperature metric being applied in the data forcasting(Compatible metrics{temp_metrics_availible}): ")
    percipitation_metric = 'mm'#input(
        #f"Percipitation metric applied to data forecasting(Compatible metrics{percp_metrics_availible}): ")
    humidity_metric = '%'#input(
        #f"Humidity metric applied to data forecasting(Compatible metrics{humidity_metrics_availible}): ")
    # Synchronization of dates and data columns process
    # Temperature
    synced_temp_dates, synced_temp_data = black_wall_protocol(target_TEMP_column, "Sync", stat_analysis_mode,
                                                              enhanced_diagnostic_mode,
                                                              developer_mode, purpose_temp, temperatue_metric,
                                                              date_copy_01)
    # Rainfall
    synced_rainfall_dates, synced_rainfall_data = black_wall_protocol(target_PERCP_column, "Sync", stat_analysis_mode,
                                                                      enhanced_diagnostic_mode,
                                                                      developer_mode, purpose_percp,
                                                                      percipitation_metric, date_copy_02)
    # Humidity
    sync_humidity_dates, synced_humidity_data = black_wall_protocol(target_humidity_column, "Sync", stat_analysis_mode,
                                                                    enhanced_diagnostic_mode, developer_mode,
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
                            purpose_temp, developer_mode, temperatue_metric, "Isolator")

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
        dataframe_generator(sync_humidity_dates, humidity_func_integration_S1, "Raw humidity data post-sync",
                            purpose_humidity, developer_mode, humidity_metric, "Isolator")

    # forecast horizon
    years_selectable = [5, 10, 15, 20]
    confirmation_activation = 0
    print(f"Available forecast horizons,  {years_selectable}")
    forecast_horizon_selected = 10#int(input("Forecast horizon: "))
    if forecast_horizon_selected not in years_selectable:
        while forecast_horizon_selected not in years_selectable:
            print("Please input compatible forecast horizon")
            forecast_horizon_selected = int(input("Forecast horizon:"))
    elif forecast_horizon_selected == 20:
        print("Confirmation is required for continuing with a 20 year forecast.")
        confirmation = int(
            input("Do you confirm this 20 year forecast time frame?(press 1 to confirm, 0 to abort forecast): "))
        if confirmation == 1:
            confirmation_activation += 1
        else:
            quit(f"{project_name} disabled due to user not confirming 20 year forecast horizon.")

    # verification of parameters
    print("-............................-")
    time_frame_S = [5, 10, 15, 20]
    print("Verification of temperature data: ")
    # Start puting seperations between developer mode active or inactive
    verification_c1 = ["Temperature purpose", "Temperature metric", "Rainfall purpose", "Rainfall metric",
                       "Forecast horizon selected"]
    verification_c2 = [purpose_temp, temperatue_metric, purpose_percp, percipitation_metric, forecast_horizon_selected]
    dev_c1_update = [0]
    developer_C1 = verification_c1 + dev_c1_update
    dev_c2_update = [0, 1]
    developer_C2 = verification_c2 + dev_c2_update
    if len(verification_c1) == len(verification_c2):
        print()
        print(
            "Please verify perameters...")  # prints out parameters in a way that is adjustable and not using too many lines
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
    file_confirmation = '1'#input(
        # "Do you confirm that the file and column used is the correct file and column? 1 to confirm, 0 to deny: ")
    print()
    deployment_confirmation = '1'#input("Confirm SARIMAX deployment. 1 to confirm deployment, 0 to disable SARIMAX: ")
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
        logged_date_sources = [synced_temp_dates,synced_rainfall_dates,sync_humidity_dates]
        logged_purposes = [purpose_temp, purpose_percp, purpose_humidity]
        logged_metrics = [temperatue_metric, percipitation_metric, humidity_metric]
        logged_forecast_horizon = forecast_horizon_selected

        # Insert into a custom deployment class that deploys multicore/multithread hybrid
        # Sequence -> time series forecasting, specilized statistical analysis, error metric, possible other experimental components if possible
        # Deploy sarimax first, multithread everything else

        if __name__ == "__main__":
            Chrono_projector_deployment(data_sources = logged_data_sources, date_sources =logged_date_sources,
                    purposes = logged_purposes, metrics = logged_metrics, forecast_horizon=logged_forecast_horizon,tsf_dev_override= tsf_dev_mode_override, developer_mode = developer_mode,
                     enhanced_diagnostics_mode = enhanced_diagnostic_mode, ghost_mode=ghost_mode, single_core_deployment_status= 0,
                                        stat_analysis_mode = stat_analysis_mode , delay = delay , data_view_mode=data_view_mode,
                                        column_name=logged_metrics,confirmation_activation=confirmation_activation).deploy_multicore_processing()
            pass

    else:
        print("Deployment status [DISABLED]")
