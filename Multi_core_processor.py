from Multi_thread_processor import temperature_core, humidity_core, rainfall_core

import multiprocessing
import time
import datetime
import numpy as np
from statistics import mean


class Chrono_projector_deployment:
    def __init__(self, data_sources, date_sources, purposes, metrics, forecast_horizon,tsf_dev_override, enhanced_diagnostics_mode,
                 stat_analysis_mode,delay , developer_mode,data_view_mode,column_name, ghost_mode
                    ,confirmation_activation,single_core_deployment_status,):
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
        self.tsf_dev_override = tsf_dev_override
        self.stat_analysis_mode = stat_analysis_mode
        self.delay = delay
        self.column_name = column_name
        self.confirmation_activation = confirmation_activation
        self.data_view_mode = data_view_mode


    # def deploy_multicore_processing(self, data_sources, date_sources, purposes, metrics, forecast_horizon, developer_mode,
    #                                 enhanced_diagnostics_mode, ghost_mode, tsf_dev_override):
    def deploy_multicore_processing(self):
        # if __name__ == "__main__":
        if True:
            temperature_relation = [self.data_sources[0], self.date_sources[0], self.purposes[0], self.metrics[0]]
            rainfall_relation = [self.data_sources[1], self.date_sources[1], self.purposes[1], self.metrics[1]]
            humidity_relation = [self.data_sources[2], self.date_sources[2], self.purposes[2], self.metrics[2]]

            core_1 = multiprocessing.Process(target=temperature_core,args = (temperature_relation[0], temperature_relation[1], temperature_relation[2],temperature_relation[3], self.forecast_horizon,
                                                                     self.tsf_dev_override, self.enhanced_diagnostics_mode, self.stat_analysis_mode, self.delay ,self.developer_mode,
                                                                     self.data_view_mode,self.column_name,self.ghost_mode,self.confirmation_activation))
            core_2 = multiprocessing.Process(target=humidity_core,args = (rainfall_relation[0], rainfall_relation[1], rainfall_relation[2],rainfall_relation[3], self.forecast_horizon,
                                                                  self.tsf_dev_override, self.enhanced_diagnostics_mode,
                                                                  self.stat_analysis_mode, self.delay,
                                                                  self.developer_mode,
                                                                  self.data_view_mode, self.column_name,
                                                                  self.ghost_mode, self.confirmation_activation))
            core_3 = multiprocessing.Process(target=rainfall_core,args = (humidity_relation[0], humidity_relation[1], humidity_relation[2],humidity_relation[3], self.forecast_horizon,
                                                                  self.tsf_dev_override, self.enhanced_diagnostics_mode,
                                                                  self.stat_analysis_mode, self.delay,
                                                                  self.developer_mode,
                                                                  self.data_view_mode, self.column_name,
                                                                  self.ghost_mode, self.confirmation_activation))

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



