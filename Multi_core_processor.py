from Multi_thread_processor import temperature_core, humidity_core, rainfall_core

import multiprocessing
import time
import datetime
import numpy as np
from statistics import mean


class Chrono_projector_deployment:
    def __init__(self, data_sources, date_sources, purposes, metrics, forecast_horizon, developer_mode,
                 enhanced_diagnostics_mode, ghost_mode, single_core_deployment_status):
        assertion_protocols = 1
        if assertion_protocols == 1:
            assert (type(data_sources) == list or type(data_sources) == np.ndarray)
            assert (type(date_sources) == list or type(date_sources) == np.ndarray)
            assert (type(purposes) == list or type(purposes) == np.ndarray)
            assert (type(metrics) == list or type(metrics) == np.ndarray)
            assert (type(forecast_horizon) == int)
            assert (type(developer_mode) == int)
            assert (type(enhanced_diagnostics_mode) == int)
            assert (type(ghost_mode) == int)
            assert (type(single_core_deployment_status) == int)

        self.data_sources = data_sources
        self.date_sources = date_sources
        self.purposes = purposes
        self.metrics = metrics
        self.forecast_horizon = forecast_horizon
        self.developer_mode = developer_mode
        self.enhanced_diagnostics_mode = enhanced_diagnostics_mode
        self.ghost_mode = ghost_mode
        self.single_core_deployment_status = single_core_deployment_status


    def deploy_multicore_processing(self, data_sources, date_sources, purposes, metrics, forecast_horizon, developer_mode,
                                    enhanced_diagnostics_mode, ghost_mode, tsf_dev_override):
        if __name__ == "__main__":
            temperature_relation = [data_sources[0], date_sources[0], purposes[0], metrics[0]]
            rainfall_relation = [data_sources[1], date_sources[1], purposes[1], metrics[1]]
            humidity_relation = [data_sources[2], date_sources[2], purposes[2], metrics[2]]

            core_1 = multiprocessing.Process(target=temperature_core(temperature_relation[0], temperature_relation[1], temperature_relation[2],temperature_relation[3], forecast_horizon,
                                                                     tsf_dev_override, enhanced_diagnostics_mode, developer_mode, ghost_mode))
            core_2 = multiprocessing.Process(target=humidity_core(rainfall_relation[0], rainfall_relation[1], rainfall_relation[2],rainfall_relation[3], forecast_horizon,
                                                                     tsf_dev_override, enhanced_diagnostics_mode, developer_mode, ghost_mode))
            core_3 = multiprocessing.Process(target=rainfall_core(humidity_relation[0], humidity_relation[1], humidity_relation[2],humidity_relation[3], forecast_horizon,
                                                                     tsf_dev_override, enhanced_diagnostics_mode, developer_mode, ghost_mode))

            tri_core_processes = [core_1, core_2, core_3]
            # Initialization time
            start_time = datetime.datetime.now()

            # Starter
            for core in tri_core_processes:
                core.start()
            # Joiner
            for cores in tri_core_processes:
                cores.join()

            # End time
            end_time = datetime.datetime.now()

            processing_time = end_time - start_time
            print(processing_time)



