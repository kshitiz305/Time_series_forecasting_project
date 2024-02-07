# Multithreading protocol here
import numpy as np

from Chrono_projector_V5_5_prototype_deployment import time_series_analysis, specilized_stat_analysis
from supporting_functions import error_metric

import threading
import datetime

def temperature_core(data_input, date_input, purpose, metric, forcast_horizon, tsf_dev_override, enhanced_diagnostic_mode,
                     stat_analysis_mode, delay ,developer_mode, data_view_mode, column_name ,ghost_mode, confirmation_activation):
    assert(type(data_input) == list or type(data_input) == np.ndarray)
    assert(type(date_input) == list or type(date_input) == np.ndarray)
    assert(type(purpose) == str)
    assert(type(metric) == str)
    assert(type(forcast_horizon) == int)
    assert(type(tsf_dev_override) == int)
    assert(type(enhanced_diagnostic_mode) == int)
    assert(type(developer_mode) == int)
    assert(type(ghost_mode) == int)

    if __name__ == "__main__":
        # Call any function that does not need multithreading before the threads are declared
        forecast_external_clone = []
        time_series_analysis(data_input, purpose,developer_mode, metric,delay, stat_analysis_mode, tsf_dev_override, data_view_mode, column_name,
                             enhanced_diagnostic_mode, ghost_mode, forcast_horizon, forecast_external_clone, confirmation_activation)

        # Creating the threads
        thread_01 = threading.Thread(target=specilized_stat_analysis(data_input, forecast_external_clone,
                                                                     developer_mode,enhanced_diagnostic_mode, metric, purpose))
        thread_02 = threading.Thread(target=error_metric(data_input, forecast_external_clone,purpose, metric))

        #Starters
        threads_in_processing = [thread_01, thread_02]
        for threads_A1 in threads_in_processing:
            threads_A1.start()
        #Joiners
        for threads_A2 in threads_in_processing:
            threads_A2.join()
    # Add a multithread processing script here
    print("Temperature core deployed")

def humidity_core(data_input, date_input, purpose, metric, forcast_horizon, tsf_dev_override, enhanced_diagnostic_mode, developer_mode, ghost_mode):
    assert (type(data_input) == list or type(data_input) == np.ndarray)
    assert (type(date_input) == list or type(date_input) == np.ndarray)
    assert (type(purpose) == str)
    assert (type(metric) == str)
    assert (type(forcast_horizon) == int)
    assert (type(tsf_dev_override) == int)
    assert (type(enhanced_diagnostic_mode) == int)
    assert (type(developer_mode) == int)
    assert (type(ghost_mode) == int)

    if __name__ == "__main__":
        # Call any function that does not need multithreading before the threads are declared
        forecast_external_clone = []
        simulated_sarimax_protocol()

        # Creating the threads
        thread_01 = threading.Thread(target=thread_01())
        thread_02 = threading.Thread(target=thread_02())

        #Starters
        threads_in_processing = [thread_01, thread_02]
        for threads_A1 in threads_in_processing:
            threads_A1.start()
        #Joiners
        threads_to_join = [thread_01, thread_02, thread_03]
        for threads_A2 in threads_to_join:
            threads_A2.join()

    print("Humidity core deployed")

def rainfall_core(data_input, date_input, purpose, metric, forcast_horizon, tsf_dev_override, enhanced_diagnostic_mode, developer_mode, ghost_mode):
    assert (type(data_input) == list or type(data_input) == np.ndarray)
    assert (type(date_input) == list or type(date_input) == np.ndarray)
    assert (type(purpose) == str)
    assert (type(metric) == str)
    assert (type(forcast_horizon) == int)
    assert (type(tsf_dev_override) == int)
    assert (type(enhanced_diagnostic_mode) == int)
    assert (type(developer_mode) == int)
    assert (type(ghost_mode) == int)
    def thread_01():
        print("---")
        print(f"Thread 1 starting")
        # Perform some task
        print(f"Thread 1 finishing")
        print("---")

    def thread_02():
        print("---")
        print(f"Thread 2 starting")
        # Perform some task
        print(f"Thread 2 finishing")
        print("---")

    def thread_03():
        print("---")
        print(f"Thread 3 starting")
        # Perform some task
        print(f"Thread 3 finishing")
        print("---")

    if __name__ == "__main__":
        # Call any function that does not need multithreading before the threads are declared
        forecast_external_clone = []
        simulated_sarimax_protocol()

        # Creating the threads
        thread_01 = threading.Thread(target=thread_01())
        thread_02 = threading.Thread(target=thread_02())
        thread_03 = threading.Thread(target=thread_03())

        # Starters
        threads_in_processing = [thread_01, thread_02, thread_03]
        for threads_A1 in threads_in_processing:
            threads_A1.start()
        # Joiners
        threads_to_join = [thread_01, thread_02, thread_03]
        for threads_A2 in threads_to_join:
            threads_A2.join()

    print("Rainfall core deployed")

simulation_test = 1
if simulation_test == 1:
    if __name__ == "__main__":
        # Call any function that does not need multithreading before the threads are declared
        start = datetime.datetime.now()
        simulated_sarimax_protocol()

        # Creating the threads
        thread_01 = threading.Thread(target=thread_01())
        thread_02 = threading.Thread(target=thread_02())
        thread_03 = threading.Thread(target=thread_03())

        #Starters
        threads_in_processing = [thread_01, thread_02, thread_03]
        for threads_A1 in threads_in_processing:
            threads_A1.start()
        #Joiners
        threads_to_join = [thread_01, thread_02, thread_03]
        for threads_A2 in threads_to_join:
            threads_A2.join()
        end = datetime.datetime.now()
        processing_time = end - start
        print(processing_time)
