'''
Created on 23 mars 2024

@author: aletard
'''

#----------------------------------------------------------------#
#                                                                #
#                    External imports                            #
#                                                                #
#----------------------------------------------------------------#

import time
import numpy as np
import random as rd

#----------------------------------------------------------------#
#                                                                #
#                    Packages imports                            #
#                                                                #
#----------------------------------------------------------------#

from Src.utils.repository_manager import RepositoryManager as RM
from Src.data_management.data_loader import DataLoader as DL
from Src.Reporting.report_generator import ReportGenerator
from Src.Reporting.results_storer import ResultStorer

from Src.algorithms.EGreedy import EGreedy
from Src.algorithms.Random import Random
from Src.algorithms.UCB import UCB
from Src.algorithms.LinUCB import LinUCB
from Src.algorithms.TSBudget import TSBudget

#----------------------------------------------------------------#
#                                                                #
#                    Global variables                            #
#                                                                #
#----------------------------------------------------------------#

# ... No global variables defined ...

#----------------------------------------------------------------#
#                                                                #
#                    Abstract Classes                            #
#                                                                #
#----------------------------------------------------------------#

#----------------------------------------------------------------#
#                                                                #
#                    Functions & Classes                         #
#                                                                #
#----------------------------------------------------------------#

class Simulator():
    
    def __init__(self):
        
        print("Initializing Simulator")
        
        self.dataset_name = "04-Poker"
        self.datas = self.data_extraction()
        self.horizon = 30000
        self.budget = 50000
        self.costs = self.initialize_costs()

        user_id = rd.choice(self.datas["contexts"]["context_id"])
        user_context = self.context_formatter(self.datas["contexts"][self.datas["contexts"]['context_id'] == user_id])

        # Using TompsonSamplingBudget
        self.algorithm = TSBudget(self.datas["arms"], self.costs, self.budget)
        
        self.results = ResultStorer(self.horizon)
        self.reporter = ReportGenerator(RM.create_repository_with_timestamp("../Output"), \
                                        (self.dataset_name, self.horizon, self.algorithm.name))
        
        self.life_sign_delay = (300, 5000)
        self.selected_arms_list = []
        
        #-----------------------
        
    def run_simulation(self):

        print("starting simulation")        
        
        self.results.start_time = time.time()
        for iteration in range(self.horizon):
            if self.algorithm.remaining_budget <= 0:
                print("Budget exhausted, ending simulation.")
                self.results.end_time = time.time()
                self.sign_life(iteration-1)
                return 0

            user_id = rd.choice(self.datas["contexts"]["context_id"])
            user_context = self.context_formatter(self.datas["contexts"][self.datas["contexts"]['context_id'] == user_id])
            observed_value = self.datas["results"][self.datas["results"]["context_id"] == user_id].copy()
                            
            selected_arms = self.algorithm.run(observed_value, user_context)
            if selected_arms is not None:
                self.algorithm.update(observed_value)
                self.results.update_measures(iteration, observed_value)
                self.selected_arms_list.append(selected_arms)
            
            if (time.time() - self.results.start_time >= self.life_sign_delay[0]) or (iteration % self.life_sign_delay[1] == 0):
                self.sign_life(iteration)
            
        self.results.end_time = time.time()
        self.end_sign()

    #-----------------------

    def data_extraction(self):

        rss_path = RM.get_absolute_from_relative_path(f"../Resources/bandit_datasets/{self.dataset_name}")
        files_to_load = RM.get_files_in_directory(rss_path)
        files_path = []
        for file in files_to_load:
            files_path.append(f"{rss_path}/{file}")
            
        return DL.load_multiple_files(files_path)

    #-----------------------

    def context_formatter(self, context):
        
        try:
            context = context.drop(["context_id"], axis=1)        
        except:
            print("Error on context formatting")
        
        nb_dimensions = context.shape[1]
        context = np.array(context)
        user_context = context.reshape(nb_dimensions, )
        
        return user_context

    #-----------------------

    def initialize_costs(self):
        num_arms = len(self.datas["arms"])
        costs = np.linspace(1, 10, num=num_arms)
        rd.shuffle(costs)
        print(f"costs: {costs}")
        return costs
    #-----------------------

    def sign_life(self, iteration):

        sign_life_message = f"\nSimulator has been running for {round(time.time() - self.results.start_time, 3)} seconds. \n" + \
                               f"Currently going for iteration {iteration}, latest accuracy value : {round(self.results.algorithm_performance['accuracy'][iteration],3)}," + \
                               f" cumulated regrets: {round(self.results.algorithm_performance['cumulated_regrets'][iteration],3)}.\n\n"

        self.reporter.log_generator(sign_life_message)
        
    #-----------------------
        
    def end_sign(self):

        end_message = f"\nSimulation correctly ended. \n The simulation has been running for {round(self.results.end_time - self.results.start_time, 3)} seconds. \n" + \
                        f"The simulation included {self.horizon} iterations, latest accuracy value : {round(self.results.algorithm_performance['accuracy'][self.horizon-1],3)}," + \
                        f" cumulated regrets: {round(self.results.algorithm_performance['cumulated_regrets'][self.horizon-1],3)}.\n\n"

        self.reporter.log_generator(end_message)

    #-----------------------------------------------------------------------------------------
