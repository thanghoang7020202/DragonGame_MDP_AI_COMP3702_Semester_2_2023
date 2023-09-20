from typing import Dict, Tuple
from game_env import GameEnv
from game_state import GameState
from itertools import product
import numpy as np
import random
import copy

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each of the method stubs below. You may add additional methods and/or classes to this file if you 
wish. You may also create additional source files and import to this file if you wish.

COMP3702 Assignment 2 "Dragon Game" Support Code

Last updated by njc 30/08/23
"""


class Solver:
    # VI and PI implementations are inspired from https://github.com/comp3702/tutorial07
    def __init__(self, game_env: GameEnv):
        self.game_env = game_env
        self.list_of_states = []
        self.policy = {} 
        self.converged = None
        # Policy Iteration
        self.state_values = None
        self.t_model = None
        self.state_pair = None
        self.la_policy = None
        self.exit_state = None
        #
        # TODO: Define any class instance variables you require (e.g. dictionary mapping state to VI value) here.
        #
        pass

    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1, 2, 3, 4, 5]

    # === Value Iteration ==============================================================================================
    def get_all_gem_status_combinations(self, n_gems):
        """
        Get all possible gem status combinations.
        :return: list of gem status combinations
        """
        res = [ele for ele in product(range(0,2), repeat = n_gems)]
        return res
    
    def get_all_obstacle(self):
        """
        Get all possible obstacle combinations.
        :return: list of obstacle combinations
        """
        obstacle_list = []
        for row in range(self.game_env.n_rows):
            for col in range(self.game_env.n_cols):
                if self.game_env.grid_data[row][col] == self.game_env.SOLID_TILE:
                    obstacle_list.append((row,col))
                #if self.game_env.grid_data[row][col] == self.game_env.LAVA_TILE:
                #    obstacle_list.append((row,col))
        return obstacle_list
        
    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        #
        # TODO: Implement any initialisation for Value Iteration (e.g. building a list of states) here. You should not
        #  perform value iteration in this method.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        res = self.get_all_gem_status_combinations(self.game_env.n_gems)
        self.exit_state = GameState(-1,-1, tuple([1 for g in self.game_env.gem_positions]))
        obstacle_list = self.get_all_obstacle()
        self.list_of_states = list(GameState(row, col, res[i]) for row in range(self.game_env.n_rows) \
                              for col in range(self.game_env.n_cols) \
                                for i in range (len(res)) if (row,col) not in obstacle_list) + [self.exit_state]
        
        self.state_values = {state: 0 for state in self.list_of_states}
        enumActions = list(self.game_env.ACTIONS)
        self.policy = {state: self.game_env.WALK_LEFT for state in self.list_of_states}
        pass

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Value Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.converged
   
    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        #
        # TODO: Implement code to perform a single iteration of Value Iteration here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        old_state_values = copy.deepcopy(self.state_values)
        self.policy = dict()
        max_diff = -float('inf')
        for s in self.list_of_states:
            action_values = dict()
            for a in self.game_env.ACTIONS: # compute the total for every valid actions
                action_value = 0
                if self.game_env.perform_action(s, a)[0] == False:
                    continue
                for (next_state, reward, prob) in get_transition_outcomes_restricted(self.game_env, s, a): # get_transition_outcomes returns a list of tuples
                    
                    action_value += prob * (reward + self.game_env.gamma * self.state_values[next_state])
                action_values[a] = action_value
            
            self.state_values[s] = max(action_values.values())
            #print(new_state_values[s])
            self.policy[s] = dict_argmax(action_values)

        self.converged = check_convergence_vi(self, old_state_values)

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: GameState):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        #
        # TODO: Implement code to return the value V(s) for the given state (based on your stored VI values) here. If a
        #  value for V(s) has not yet been computed, this function should return 0.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.state_values[state]
        pass

    def vi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored VI values) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.policy[state]
        pass

   
    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        #
        # TODO: Implement any initialisation for Policy Iteration (e.g. building a list of states) here. You should not
        #  perform policy iteration in this method. You should assume an initial policy of always move FORWARDS.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        
        res = self.get_all_gem_status_combinations(self.game_env.n_gems)
        self.exit_state = GameState(-1,-1, tuple([1 for g in self.game_env.gem_positions]))
        obstacle_list = self.get_all_obstacle()
        self.list_of_states = list(GameState(row, col, res[i]) for row in range(self.game_env.n_rows) \
                              for col in range(self.game_env.n_cols) \
                                for i in range (len(res)) if (row,col) not in obstacle_list) + [self.exit_state]

        self.state_values = {s: 0 for s in self.list_of_states}
        enumActions = list(self.game_env.ACTIONS)
        self.policy = {s: np.random.choice(enumActions) for s in self.list_of_states}

        # r model (lin alg)
        r_model = np.zeros([len(self.list_of_states), len(self.game_env.ACTIONS)])

        # t model (lin alg)
        t_model = np.zeros([len(self.list_of_states), len(self.game_env.ACTIONS), len(self.list_of_states)])

        for i, s in enumerate(self.list_of_states):
            #valid_action = get_valid_actions(self.game_env, s)
            for j, a in enumerate(self.game_env.ACTIONS):
                if self.game_env.perform_action(s, a)[0] == False:
                    # continue to next valid action
                    continue
                for next_state, reward, prob in get_transition_outcomes_restricted(self.game_env, s, a):
                    t_model[i][j][self.list_of_states.index(next_state)] += prob
                    r_model[i][j] += prob * reward
        self.r_model = r_model
        self.t_model = t_model

        # lin alg policy (pi), arbitrarily initialise to 0 (UP) for all states
        
        self.la_policy = np.zeros(len(self.list_of_states), dtype=int)

        self.converged = False
        pass

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        #
        # TODO: Implement code to check if Policy Iteration has reached convergence here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        return self.converged
        pass

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        #
        # TODO: Implement code to perform a single iteration of Policy Iteration (evaluation + improvement) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        v_pi = self.policy_evaluation()
        self.policy_improvement(v_pi)
        pass
    
    def policy_evaluation(self):
        # use linear algebra for policy evaluation
        # V^pi = R + gamma T^pi V^pi
        # (I - gamma * T^pi) V^pi = R
        # Ax = b; A = (I - gamma * T^pi),  b = R

        # indices of every state
        state_numbers = np.array(range(len(self.list_of_states)))
        # index into t_model to select only entries where a = pi(s)
        t_pi = self.t_model[state_numbers, self.la_policy]
        r_pi = self.r_model[state_numbers, self.la_policy]
        # solve for V^pi(s) using linear algebra
        values = np.linalg.solve(np.identity(len(self.list_of_states)) - (self.game_env.gamma * t_pi), r_pi)
        # convert V^pi(s) vector to dict and return
        #print(values)
        self.state_values = {s: values[i] for i, s in enumerate(self.list_of_states)}
        #new_policy = {s: self.list_of_states[self.la_policy[i]] for i, s in enumerate(self.list_of_states)}
        return self.state_values  

    def policy_improvement(self, values):
        new_policy = {s: self.list_of_states[self.la_policy[i]] for i, s in enumerate(self.list_of_states)}
    
        policy_changed = False
        # loop over each state, and improve the policy using 1-step lookahead and the values from policy_evaluation

        for s in self.list_of_states:
            action_values = dict()
            #actions = get_valid_actions(self.game_env, s)
            for a in self.game_env.ACTIONS: # compute the total for every valid actions
                if self.game_env.perform_action(s, a)[0] == False:
                    # continue to next valid action
                    continue

                action_value = 0
                for (next_state, reward, prob) in get_transition_outcomes_restricted(self.game_env, s, a):
                    action_value += prob * (reward + (self.game_env.gamma * self.state_values[next_state])) 
                    
                action_values[a] = action_value

            # update policy
            new_policy[s] = dict_argmax(action_values)

        self.converged = convergence_check(self, new_policy)
        self.policy = new_policy

        for i, s in enumerate(self.list_of_states):
            self.la_policy[i] = list(self.game_env.ACTIONS).index(new_policy[s])

        return self.converged

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before pi_is_converged
            if self.pi_is_converged():
                break


    def pi_select_action(self, state: GameState):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        #
        # TODO: Implement code to return the optimal action for the given state (based on your stored PI policy) here.
        #
        # In order to ensure compatibility with tester, you should avoid adding additional arguments to this function.
        #
        enumActions = list(self.game_env.ACTIONS)
        return enumActions[self.la_policy[self.list_of_states.index(state)]]
        pass

    # === Helper Methods ===============================================================================================
    #
    #
    # TODO: Add any additional methods here
    #
    #
def check_convergence_vi(self, new_state_values: Dict[Tuple[int, int], float]) -> bool:
        #print(self.state_values )
        #print(new_state_values)
        differences = [abs(self.state_values[state] - new_state_values[state]) for state in self.list_of_states]
        
        return max(differences) < self.game_env.epsilon
def convergence_check(self, new_policy: Dict[Tuple[int, int], int]) -> bool:
    return self.policy == new_policy


def get_reward(game_env, state): # the next action will be edit later
    '''
    get the reward for the state
    return the reward, policy
    '''
    valid_action = get_valid_actions(game_env, state)
    best_q = -float('inf') # initialize to smallest possible value
    for a in valid_action: # compute the total for every valid actions
        total = 0
        if game_env.perform_action(state, a)[0] == False:
            # continue to next valid action
            continue
        for (next_state, reward, prob) in get_transition_outcomes_restricted(game_env, state, a): # get_transition_outcomes returns a list of tuples
            if next_state.row == -1:
                break
            total += prob * reward 
        if total > best_q:
            best_q = total
    return  best_q


def dict_argmax(dictionary: Dict):
    #print(dictionary)
    max_value = max((dictionary.values())) # TODO handle multiple keys with the same max value
    
    for key, value in dictionary.items():
        if value == max_value:
            return key


def get_valid_actions(game_env, state):
    """
    Get valid actions for the given state.
    :return: list of valid actions
    """
    
    valid_actions = []
    
    if game_env.grid_data[state.row + 1][state.col] in game_env.WALK_JUMP_ALLOWED_TILES:
        valid_actions.append(game_env.WALK_LEFT)
        valid_actions.append(game_env.WALK_RIGHT)
        valid_actions.append(game_env.JUMP)
    if game_env.grid_data[state.row + 1][state.col] in game_env.GLIDE_DROP_ALLOWED_TILES:
        valid_actions.append(game_env.GLIDE_LEFT_1)
        valid_actions.append(game_env.GLIDE_LEFT_2)
        valid_actions.append(game_env.GLIDE_LEFT_3)
        valid_actions.append(game_env.GLIDE_RIGHT_1)
        valid_actions.append(game_env.GLIDE_RIGHT_2)
        valid_actions.append(game_env.GLIDE_RIGHT_3)
        valid_actions.append(game_env.DROP_1)
        valid_actions.append(game_env.DROP_2)
        valid_actions.append(game_env.DROP_3)
    return valid_actions

#<CITATION: COMP3702 SUPPLIED CODE FOR TRANSITION FUNCTION, NJC>
def get_transition_outcomes_restricted(game_env, state, action):
    """
    This method assumes (state, action) is a valid combination.

    :param game_env: GameEnv instance
    :param state: current state
    :param action: selected action
    :return: list of (next_state, immediate_reward, probability) tuples
    """
    reward = -1 * GameEnv.ACTION_COST[action]
    remaining_prob = 1.0
    outcomes = []

    # absorbing state
    exit_state = GameState(-1,-1, tuple([1 for g in game_env.gem_positions]))
    if  state == exit_state or game_env.is_solved(state) or game_env.is_game_over(state):
        outcomes.append((exit_state, 0, 1.0))
        return outcomes

    max_glide1_outcome = max(game_env.glide1_probs.keys())
    max_glide2_outcome = max(game_env.glide2_probs.keys())
    max_glide3_outcome = max(game_env.glide3_probs.keys())

    # handle each action type separately
    if action in GameEnv.WALK_ACTIONS:
        # set movement direction
        if action == GameEnv.WALK_LEFT:
            move_dir = -1
        else:
            move_dir = 1

        # walk on normal walkable tile (super charge case not handled)
        # if on ladder, handle fall case
        if game_env.grid_data[state.row + 1][state.col] == GameEnv.LADDER_TILE and \
                game_env.grid_data[state.row + 2][state.col] not in GameEnv.COLLISION_TILES:
            next_row, next_col = state.row + 2, state.col
            # check if a gem is collected or goal is reached
            next_gem_status, _ = check_gem_collected_or_goal_reached(game_env, next_row, next_col, state.gem_status)
            outcomes.append((GameState(next_row, next_col, next_gem_status), reward, game_env.ladder_fall_prob))
            remaining_prob -= game_env.ladder_fall_prob

        next_row, next_col = state.row, state.col + move_dir
        # check for collision or game over
        next_row, next_col, collision, is_terminal = \
            check_collision_or_terminal(game_env, next_row, next_col, row_move_dir=0, col_move_dir=move_dir)

        # check if a gem is collected or goal is reached
        next_gem_status, _ = check_gem_collected_or_goal_reached(game_env, next_row, next_col, state.gem_status)

        if collision:
            # add any remaining probability to current state
            outcomes.append((GameState(next_row, next_col, next_gem_status),
                             reward - game_env.collision_penalty, remaining_prob))
        elif is_terminal:
            # add any remaining probability to current state
            outcomes.append((GameState(next_row, next_col, next_gem_status),
                             reward - game_env.game_over_penalty, remaining_prob))
        else:
            outcomes.append((GameState(next_row, next_col, next_gem_status), reward, remaining_prob))

    elif action == GameEnv.JUMP:
        # jump on normal walkable tile (super jump case not handled)

        #next_row, next_col = state.row - 1, state.col
        max_jump_height = 1
        min_jump_height = 1
        if game_env.grid_data[state.row + 1][state.col] == game_env.SUPER_JUMP_TILE:
            #print("enter super jump mode")
            min_jump_height = 2
            max_jump_height = 5
        for d in range(min_jump_height, max_jump_height + 1): #loop from 1 to 5
            next_row = state.row - d
            next_col = state.col
            # check for collision or game over
            next_row, next_col, collision, is_terminal = \
                check_collision_or_terminal(game_env, next_row, next_col, row_move_dir=-1, col_move_dir=0)

            # check if a gem is collected or goal is reached
            next_gem_status, _ = check_gem_collected_or_goal_reached(game_env, next_row, next_col, state.gem_status)

            if collision:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.collision_penalty, remaining_prob))
                break
            if is_terminal:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.game_over_penalty, remaining_prob))
                break

            # if this state is a possible outcome, add to list
            if d in game_env.super_jump_probs.keys():
                outcomes.append((GameState(next_row, next_col, next_gem_status), reward, game_env.super_jump_probs[d]))
                remaining_prob -= game_env.super_jump_probs[d]
            else:
                outcomes.append((GameState(next_row, next_col, next_gem_status), reward, remaining_prob))
                break

    elif action in GameEnv.GLIDE_ACTIONS:
        # glide on any valid tile
        # select probabilities to sample move distance
        if action in {GameEnv.GLIDE_LEFT_1, GameEnv.GLIDE_RIGHT_1}:
            probs = game_env.glide1_probs
            max_outcome = max_glide1_outcome
        elif action in {GameEnv.GLIDE_LEFT_2, GameEnv.GLIDE_RIGHT_2}:
            probs = game_env.glide2_probs
            max_outcome = max_glide2_outcome
        else:
            probs = game_env.glide3_probs
            max_outcome = max_glide3_outcome

        # set movement direction
        if action in {GameEnv.GLIDE_LEFT_1, GameEnv.GLIDE_LEFT_2, GameEnv.GLIDE_LEFT_3}:
            move_dir = -1
        else:
            move_dir = 1

        # add each possible movement distance to set of outcomes
        next_row, next_col = state.row + 1, state.col
        for d in range(0, max_outcome + 1):
            next_col = state.col + (move_dir * d)
            # check for collision or game over
            next_row, next_col, collision, is_terminal = \
                check_collision_or_terminal_glide(game_env, next_row, next_col, row_move_dir=0, col_move_dir=move_dir)

            # check if a gem is collected or goal is reached
            next_gem_status, _ = check_gem_collected_or_goal_reached(game_env, next_row, next_col, state.gem_status)

            if collision:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.collision_penalty, remaining_prob))
                break
            if is_terminal:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.game_over_penalty, remaining_prob))
                break

            # if this state is a possible outcome, add to list
            if d in probs.keys():
                outcomes.append((GameState(next_row, next_col, next_gem_status), reward, probs[d]))
                remaining_prob -= probs[d]

    elif action in GameEnv.DROP_ACTIONS:
        # drop on any valid tile
        next_row, next_col = state.row, state.col

        drop_amount = {GameEnv.DROP_1: 1, GameEnv.DROP_2: 2, GameEnv.DROP_3: 3}[action]

        # drop until drop amount is reached
        for d in range(1, drop_amount + 1):
            next_row = state.row + d

            # check for collision or game over
            next_row, next_col, collision, is_terminal = \
                check_collision_or_terminal_glide(game_env, next_row, next_col, row_move_dir=1, col_move_dir=0)

            # check if a gem is collected or goal is reached
            next_gem_status, _ = check_gem_collected_or_goal_reached(game_env, next_row, next_col, state.gem_status)

            if collision:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.collision_penalty, 1.0))
                break
            if is_terminal:
                # add any remaining probability to current state
                outcomes.append((GameState(next_row, next_col, next_gem_status),
                                 reward - game_env.game_over_penalty, 1.0))
                break

            if d == drop_amount:
                outcomes.append((GameState(next_row, next_col, next_gem_status), reward, 1.0))

    else:
        assert False, '!!! Invalid action given to perform_action() !!!'

    return outcomes


def check_collision_or_terminal(game_env, row, col, row_move_dir, col_move_dir):
    """
    Checks for collision with solid tile, or entering lava tile. Returns resulting next state (after bounce back if
    colliding), and booleans indicating if collision or game over has occurred.
    :return: (next_row,  next_col, collision (True/False), terminal (True/False))
    """
    terminal = False
    collision = False
    # check for collision condition
    if (not 0 <= row < game_env.n_rows) or (not 0 <= col < game_env.n_cols) or \
            game_env.grid_data[row][col] in GameEnv.COLLISION_TILES:
        row -= row_move_dir     # bounce back to previous position
        col -= col_move_dir     # bounce back to previous position
        collision = True
    # check for game over condition
    elif game_env.grid_data[row][col] == GameEnv.LAVA_TILE:
        terminal = True

    return row, col, collision, terminal


def check_collision_or_terminal_glide(game_env, row, col, row_move_dir, col_move_dir):
    """
    Checks for collision with solid tile, or entering lava tile for the special glide case (player always moves down by
    1, even if collision occurs). Returns resulting next state (after bounce back if colliding), and booleans indicating
    if collision or game over has occurred.
    :return: (next_row,  next_col, collision (True/False), terminal (True/False))
    """
    # variant for checking glide actions - checks row above as well as current row
    terminal = False
    collision = False
    # check for collision condition
    if (not 0 <= row < game_env.n_rows) or (not 0 <= col < game_env.n_cols) or \
            game_env.grid_data[row][col] in GameEnv.COLLISION_TILES or \
            game_env.grid_data[row - 1][col] in GameEnv.COLLISION_TILES:
        row -= row_move_dir     # bounce back to previous position
        col -= col_move_dir     # bounce back to previous position
        collision = True
    # check for game over condition
    elif game_env.grid_data[row][col] == GameEnv.LAVA_TILE or \
            game_env.grid_data[row - 1][col] == GameEnv.LAVA_TILE:
        terminal = True

    return row, col, collision, terminal


def check_gem_collected_or_goal_reached(game_env, row, col, gem_status):
    """
    Checks if the new row and column contains a gem, and returns an updated gem status. Additionally returns a flag
    indicating whether the goal state has been reached (all gems collected and player at exit).
    :return: new gem_status, solved (True/False)
    """
    is_terminal = False
    # check if a gem is collected (only do this for final position of charge)
    if (row, col) in game_env.gem_positions and \
            gem_status[game_env.gem_positions.index((row, col))] == 0:
        gem_status = list(gem_status)
        gem_status[game_env.gem_positions.index((row, col))] = 1
        gem_status = tuple(gem_status)
    # check for goal reached condition (only do this for final position of charge)
    elif row == game_env.exit_row and col == game_env.exit_col and \
            all(gs == 1 for gs in gem_status):
        is_terminal = True
    return gem_status, is_terminal

