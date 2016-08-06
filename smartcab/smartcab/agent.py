import itertools
import numpy as np
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    TRAFFIC_LIGHT_STATES = [True, False]  # True if light is green, otherwise False.
    AUTOMOBILE_TRAFFIC_STATES = [True, False]  # True if car in intersection.
    WAYPOINT_STATES = ["left", "right", "forward"]  # The next waypoint direction.

    # The state space is a 5-tuple: (TrafficLight, LeftTraffic, ForwardTraffic, RightTraffic, NextWaypoint)
    STATE_SPACE = list(itertools.product(TRAFFIC_LIGHT_STATES,
                                    AUTOMOBILE_TRAFFIC_STATES,
                                    AUTOMOBILE_TRAFFIC_STATES,
                                    AUTOMOBILE_TRAFFIC_STATES,
                                    WAYPOINT_STATES))

    def __init__(self, env, alpha, gamma, epsilon):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = alpha
        self.one_minus_alpha = 1.0 - alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = dict()
        for s in self.STATE_SPACE:
            action_values = dict()
            for w in self.env.valid_actions:
                action_values[w] = float(0)
            self.q_values[s] = action_values


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.state = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'] == 'green',
                      inputs['left'] is None,
                      inputs['right'] is None,
                      inputs['oncoming'] is None,
                      self.next_waypoint)

        # TODO: Select action according to your policy
        if random.uniform(0.0, 1.0) < self.epsilon:
            action = random.choice(self.env.valid_actions)
        else:
            available_actions = self.q_values[self.state]
            action = max(available_actions, key=available_actions.get)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        collapsed_utility_values = [[z[1] for z in self.q_values[y].items()] for y in self.q_values]
        max_value = max(itertools.chain(*collapsed_utility_values))
        estimated_utility = reward + self.gamma * max_value
        updated_utility = self.calculate_utility_update(self.q_values[self.state][action], estimated_utility)
        self.q_values[self.state][action] = updated_utility

    def calculate_utility_update(self, current_utility, observed_utility):
        return self.one_minus_alpha * current_utility + self.alpha * observed_utility


def run(alpha, gamma, epsilon, num_trials):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, alpha=alpha, gamma=gamma, epsilon=epsilon)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=num_trials)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    return e.num_success / float(num_trials)


if __name__ == '__main__':
    # param_values = np.arange(0,1,0.2)
    # param_permutations = itertools.product(param_values, param_values, param_values)
    # results = dict()

    # for p in param_permutations:
    #     r = list()
    #     for x in range(20):
    #         r.append(run(p[0], p[1], p[2], 100))
    #     results[p] = np.mean(r)
    #
    # print 'Best result was: {}, where it\'s (alpha, gamma, epsilon).'.format(max(results.items(), key=lambda x: x[1]))

    print run(0.6, 0, 0.2, 100)
