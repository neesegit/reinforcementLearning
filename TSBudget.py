import random
import numpy as np
import pandas as pd

class TSBudget():

    def __init__(self, arms=None, cost=None, budget=None):
        
        self.ground_arms = arms
        self.arms_pool = self.ground_arms.copy()
        self.name = "Tompson Sampling with Budget"
        
        self.arms_payoff_vectors = {"Success" : np.zeros(len(self.ground_arms)),
                                    "Failures" : np.zeros(len(self.ground_arms))}
        
        self.arm_chosen = None
        self.threshold = 4
        self.budget = budget  # Total budget constraint
        self.remaining_budget = budget
        
        self.arm_costs = cost

    def run(self, observed_value, user_context=None):

        self.init_choice(observed_value)
        self.arm_chosen = self.choose_action()
        
        return self.arm_chosen

    def init_choice(self, observation):

        self.arm_chosen = -1
        self.arms_pool = self.ground_arms[self.ground_arms["arm_id"].isin(observation["arm_id"])]
        self.arms_pool.reset_index(inplace=True)

    def choose_action(self):

        if self.remaining_budget <= 0:
            return None

        expected_payoff = np.zeros(len(self.arms_pool['arm_id'])) - 1
        for i, arm in enumerate(self.arms_pool['arm_id']):
            arm_pos = self.ground_arms.index[self.ground_arms["arm_id"] == arm]
            expected_payoff[i] = self.beta(self.arms_payoff_vectors["Success"][arm_pos],
                                           self.arms_payoff_vectors["Failures"][arm_pos] + 1)

        affordable_arms = [i for i, arm in enumerate(self.arms_pool['arm_id']) if self.arm_costs[arm] <= self.remaining_budget]
        
        if not affordable_arms:
            return None

        affordable_payoffs = expected_payoff[affordable_arms]
        best_arm_index = affordable_arms[np.argmax(affordable_payoffs)]
        best_arm = self.arms_pool["arm_id"][best_arm_index]
        
        return best_arm

    def beta(self, S, F):
        return (S + 1) / (S + F + 2)

    def evaluate(self, observation):

        reward = 0
        feedback = observation["feedback"][observation["arm_id"] == self.arm_chosen].iloc[0]
        if feedback >= self.threshold:
            reward = 1

        return reward

    def update(self, observation):

        observed_reward = self.evaluate(observation)
        if observed_reward == 1:
            self.arms_payoff_vectors["Success"][self.arm_chosen] += 1
        else:
            self.arms_payoff_vectors["Failures"][self.arm_chosen] += 1
        
        self.remaining_budget -= self.arm_costs[self.arm_chosen]
