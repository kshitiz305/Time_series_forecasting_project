import threading
import numpy as np
from Chrono_projector_V5_5_prototype_deployment import black_wall_protocol

#critical functions --------------------------------------------------------------
def extreme_weather_probability_analysis(data_points, cloned_forecast,developer_mode, stat_analysis_mode, enhanced_diagnostic_mode, ghost_mode, data_view_mode):
    #Premise: Analysis of pre and post forecast data for detecting the probability of a drought occuring
    assert(type(data_points) == list or type(data_points) == np.ndarray)
    assert(type(cloned_forecast) == list or type(cloned_forecast) == np.ndarray)

    print("Extreme weather probability function dormant")
#........................................


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





