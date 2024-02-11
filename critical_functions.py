import math
import threading
import time
from pmdarima.arima import auto_arima
import numpy as np
# from Chrono_projector_V5_5_prototype_deployment import black_wall_protocol
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from supporting_functions import dataframe_generator, data_analysis_suite, diagostics_check, column_analyzer


#critical functions --------------------------------------------------------------
def extreme_weather_probability_analysis(data_points, cloned_forecast,developer_mode, stat_analysis_mode, enhanced_diagnostic_mode, ghost_mode, data_view_mode):
    #Premise: Analysis of pre and post forecast data for detecting the probability of a drought occuring
    assert(type(data_points) == list or type(data_points) == np.ndarray)
    assert(type(cloned_forecast) == list or type(cloned_forecast) == np.ndarray)

    print("Extreme weather probability function dormant")
#........................................

def contains_nan(arr):
    assert (type(arr) == list or type(arr) == np.ndarray)
    nan_found = 0
    for i, value in enumerate(arr):
        if str(value) == 'nan':
            nan_found += 1
            break

    if nan_found == 1:
        return 1  # Return 1 if nan was found

    return 0  # Return 0 if no nan is found


def data_visualization(data, column_name, purpose, forecast, metric, delay, ghost_mode, stat_analysis_mode,
                       enhanced_diagnostic_mode, developer_mode):
    data_clone = black_wall_protocol(data.copy(), "Anomaly clearing", stat_analysis_mode, enhanced_diagnostic_mode,
                                     developer_mode, purpose, metric, dates=None)
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
    if autodownload_confirmation == 1 and ghost_mode != 1:  # add ghost mode here
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
    del (data_clone)


def black_wall_protocol(input, mutation_protocols, stat_analysis_mode, enhanced_diagnostic_mode, developer_mode,
                        purpose, metric, dates):
    def black_ice_protocol(input):
        assert (type(input) == list or type(input) == np.ndarray)
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
        #TODo:Error in the input check [input.size() if it is a numpy]
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

        # Prepare the outlier handling functionality
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
            quit(
                "Syncronizer failed to syncronize data because it there is an empty dates column or the dates column does not exist.")


# Critical functions--------------------------------------------

def specilized_stat_analysis(input_data, forecast_clone, developer_mode, enhanced_diagnostic_mode, metric,
                             purpose):  # isolate mankendall test and sens graph
    import pymannkendall as mk
    def specilized_analysis(input_data, developer_mode, purpose, metric, status):
        # isolate 10 years of data

        mk_results_output = mk.original_test(input_data)

        catagory = f"Specilized analysis for {status}"
        descriptions = ["Trend", "H", "P-value", "Z", "Sens slope"]
        results = [mk_results_output.trend, mk_results_output.h,
                   mk_results_output.p, mk_results_output.z, mk_results_output.slope]

        if developer_mode == 1:
            for result in range(len(descriptions)):
                print(f"{descriptions[result]} {results[result]}")

        dataframe_generator(descriptions, results, catagory, purpose, developer_mode, metric, "Athena")

    assert (type(input_data) == list or type(input_data) == np.ndarray)
    assert (type(forecast_clone) == list or type(forecast_clone) == np.ndarray)
    assert (type(developer_mode) == int)
    assert (type(enhanced_diagnostic_mode) == int)
    assert (type(metric) == str)

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
    # mann kendal test, sens slope
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
            t2 = threading.Thread(
                target=specilized_analysis(forecasted_data, developer_mode, purpose, metric, status[1]))

            t1.start(), t2.start()

            t1.join(), t2.join()
    else:
        return "Activation failure, double check that the activation variable is set to 1"


def data_extractor_protocol(data_input, forecast_clone):
    assert (type(data_input) == list or type(data_input) == np.ndarray)
    assert (type(forecast_clone) == list or type(forecast_clone) == np.ndarray)

    if forecast_clone:
        if len(forecast_clone) != 0:
            print("Forecast clone present")
            print(f"Length of forecast data: {len(forecast_clone)}")
            print(f"Cloned forecast data fragment: {forecast_clone[:10]}")

    print("Extractor function dormant")


def time_series_analysis(data, purpose, developer_mode, metric, delay, stat_analysis_mode, tsf_dev_mode_override,
                         data_view_mode, column_name, enhanced_diagnostic_mode, ghost_mode, forecast_horizon_selected,
                         cloned_forecast, confirmation_activation):
    def forecast_mechanism_isolation(current_data, sarimax_params, forecast_horizon, forecast, cycles,
                                     confirmation_activation,
                                     developer_mode, purpose, metric):
        assert (type(current_data) == list or type(current_data) == np.ndarray or type(current_data) == dict)
        assert (type(forecast_horizon) == int)
        assert (type(forecast) == list or type(forecast) == np.ndarray)
        assert (type(cycles) == int)
        initial_cycles = cycles
        updated_cycles = 0
        # noinspection PyTypeChecker
        # data = black_wall_protocol(current_data, "Sanitation", stat_analysis_mode,
        #                            enhanced_diagnostic_mode, developer_mode, purpose, metric, dates=None)
        data = current_data
        for element in range(0, forecast_horizon * 12, forecast_horizon):
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
                modified_forecast_selection.append(round(element, 4))
            forecast_section = modified_forecast_selection
            forecast.extend(forecast_section)

        if enhanced_diagnostic_mode == 1:
            df_cat = "Forecast cycles completed"
            c1 = ["Initial", "Post-completion"]
            c2 = [initial_cycles, updated_cycles]
            dataframe_generator(c1, c2, df_cat, purpose, developer_mode, metric, transmutation="Athena")

    def sarimax_data_acquisition(list_injected):
        # model = auto_arima(list_injected, start_p=1, start_q=1, max_p=3,
        #                    max_q=3, m=12, start_P=0, d=1, D=1, trace=False,
        #                    error_action='ignore', seasonal=True, suppress_warnings=True,
        #                    stepwise=True)
        model = auto_arima(list_injected, start_p=5, start_q=5, max_p=9,
                           max_q=9, m=12, start_P=0, d=1, D=1, trace=False,
                           error_action='ignore', seasonal=True, suppress_warnings=True,
                           stepwise=True)
        return model.order

    assert (type(data) == list or type(data) == np.ndarray or type(data) == dict)
    assert (type(purpose) == str)
    assert (type(metric) == str)
    assert (stat_analysis_mode == 1)
    assert (type(developer_mode) == int and type(enhanced_diagnostic_mode) == int)
    assert (type(ghost_mode) == int)

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
        forecast_mechanism_isolation(current_data=data, sarimax_params=sarimax_params,
                                     forecast_horizon=forecast_horizon,
                                     forecast=forecast, cycles=forecast_cycles,
                                     confirmation_activation=confirmation_activation, developer_mode=developer_mode,
                                     purpose=purpose, metric=metric)
    else:
        forecast_mechanism_isolation(current_data=data, sarimax_params=sarimax_params,
                                     forecast_horizon=forecast_horizon,
                                     forecast=forecast, cycles=forecast_cycles,
                                     confirmation_activation=confirmation_activation,
                                     developer_mode=developer_mode, purpose=purpose, metric=metric)

    # Developer Mode Check
    if developer_mode == 1 and ghost_mode != 1:  # add ghost mode here
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
    if data_view_mode == 1 and ghost_mode != 1:  # add ghost mode here
        print("Forecasted data fragment: ", forecast[:25])
    elif data_view_mode == 0 and ghost_mode != 1:  # add ghost mode here
        print("Data viewing disabled.")

    forecast_cat = "Forecast"

    if delay == 1:
        time.sleep(3)

    # if __name__ == "__main__":
    if True:
        data_visualization(data, column_name, purpose, forecast, metric, delay, ghost_mode,
                           enhanced_diagnostic_mode, developer_mode, purpose)
        t1 = threading.Thread(
            target=diagostics_check(forecast, forecast_cat, developer_mode, tsf_dev_mode_override, ghost_mode, purpose,
                                    metric))
        t2 = threading.Thread(
            target=diagostics_check(data, data_cat, developer_mode, tsf_dev_mode_override, ghost_mode, purpose, metric))
        t3 = threading.Thread(
            target=data_analysis_suite(forecast, purpose, forecast_cat, developer_mode, metric, ghost_mode))
        t4 = threading.Thread(target=data_analysis_suite(data, purpose, data_cat, developer_mode, metric, ghost_mode))
        t6 = threading.Thread(
            target=dataframe_generator(forecast_datapoints, forecast, forecast_cat, purpose, developer_mode, metric,
                                       transmutation="Hephestus"))

        processes = [t1, t2, t3, t4, t6]
        custom_thread_deployment = [t3, t4, t6]

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


    # ---

def drought_detector(historic_data, forecast_data): #experimental stage
    # purpose: Detect periods of low rainfall for extended periods of time
    # results are not definitive for how many droughts that occured
    def scanner(data_source, fragment, drought_events_logged, drought_threshold):
        assert(type(data_source) == list or type(data_source) == np.ndarray)
        assert (type(fragment) == list or type(fragment) == np.ndarray)
        assert(type(drought_threshold) == int)
        assert(type(drought_events_logged) == int)
        data_source = black_wall_protocol(np.array(data_source), "Anomaly clearing")
        fragment = np.array(fragment)
        logged_events = 0

        source_analysis = [max(data_source), np.mean(data_source)]
        fragment_analysis = [max(fragment)]
        if source_analysis[0] == 0 and fragment_analysis[0] == 0:
            return "Both max values for drought detector are zero"
        average = (source_analysis[0] + fragment_analysis[0]) / 2
        gap = abs(source_analysis[0] - fragment_analysis[0])
        percentage_difference = (gap/average) * 100

        if percentage_difference >= drought_threshold:
            logged_events += 1

        updated_events_log = logged_events
        drought_events_logged = updated_events_log

    def search_protocol(input_array, drought_events_logged, drought_threshold):
        for element in range(input_array-11):
            fragment_collected =  black_wall_protocol(input_array[element:element+18], "Anomaly clearing")
            # checks what is currently there at that point
            scanner(input_array, fragment_collected, drought_events_logged)
            # nested for loop that stores a fragment and then  calls the scanner function

    current_drought_events = 0
    forecasted_drought_events = 0
    drought_threshold = 85

    if __name__ == "__main__":
        t1 = threading.Thread(target=search_protocol(historic_data, current_drought_events, drought_threshold))
        t2 = threading.Thread(target=search_protocol(forecast_data, forecasted_drought_events, drought_threshold))

        t1.start(), t2.start()

        t1.join(), t2.join()
    description_column_c1 = ["Pre-forecast", "Post-forecast"]

    #dataframe_generator()





