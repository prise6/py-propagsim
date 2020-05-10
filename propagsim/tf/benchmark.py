# from .classes import State, Agent, Cell, Transitions, Map
from classes import Map
from time import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


N_PERIODS = 3
N_MOVES_PER_PERIOD = 4
AVG_P_MOVE = .5 / N_MOVES_PER_PERIOD
# N_AGENTS = 700000
N_AGENTS = 7000
# N_INFECTED_AGENTS_START = int(N_AGENTS / 40)
PROP_INFECTED_AGENTS_START = 1 / 10
# N_SQUARES_AXIS = 125
N_SQUARES_AXIS = 40
AVG_AGENTS_HOME = 2.2
N_HOME_CELLS = int(N_AGENTS / AVG_AGENTS_HOME)
PROP_PUBLIC_CELLS = 1 / 70  # there is one public place for 70 people in France
N_CELLS = int(N_HOME_CELLS + N_AGENTS * PROP_PUBLIC_CELLS)
DSCALE = 30
AVG_UNSAFETY = .5


def get_alpha_beta(min_value, max_value, mean_value):
    """ for the duration on a state, draw from a beta distribution with parameter alpha and beta """
    x = (mean_value - min_value) / (max_value - min_value)
    z = 1 / x - 1
    a, b = 2, 2 * z
    return a, b


def draw_beta(min_value, max_value, mean_value, n_values, round=False):
    """ draw `n_values` values between `min_value` and `max_value` having 
    `mean_value` as (asymptotical) average"""
    a, b = get_alpha_beta(min_value, max_value, mean_value)

    durations = tfp.distributions.Beta(a,b).sample(n_values) * (max_value - min_value) + min_value
    if round:
        durations = tf.math.round(durations)
    durations = tf.reshape(durations, [-1, 1])
    durations = tf.cast(durations, tf.float16)
    return durations

# =========== States ==============

# region
unique_state_ids = tf.cast(tf.range(0, 7), tf.uint32)
unique_contagiousities = tf.constant([0, .9, .8, .1, .05, 0, 0])
unique_sensitivities = tf.cast(tf.constant([1, 0, 0, 0, 0, 0, 0]), tf.float32)
unique_severities = tf.constant([0, .1, .8, 1, 1, 1, 0])
# endregion

# ========= Transitions ==========

# region
# For people younger than 15yo
transitions_15 = tf.constant([[1, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0.5, 0, 0, 0, 0.5], 
                            [0, 0, 0, 0.3, 0, 0, 0.7],
                            [0, 0, 0, 0, 0.3, 0, 0.7],
                            [0, 0, 0, 0, 0, 0.5, 0.5],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1]])

# For people younger between 15 and 44yo
transitions_15_44 = tf.constant([[1, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0.5, 0, 0, 0, 0.5], 
                                [0, 0, 0, 0.3, 0, 0, 0.7],
                                [0, 0, 0, 0, 0.3, 0, 0.7],
                                [0, 0, 0, 0, 0, 0.5, 0.5],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]])

# For people younger between 45 and 64yo
transitions_45_64 = tf.constant([[1, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0.5, 0, 0, 0, 0.5], 
                                [0, 0, 0, 0.3, 0, 0, 0.7],
                                [0, 0, 0, 0, 0.3, 0, 0.7],
                                [0, 0, 0, 0, 0, 0.5, 0.5],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]])

# For people younger between 65 and 75yo
transitions_65_74 = tf.constant([[1, 0, 0, 0, 0, 0, 0], 
                                [0, 0, 0.5, 0, 0, 0, 0.5], 
                                [0, 0, 0, 0.3, 0, 0, 0.7],
                                [0, 0, 0, 0, 0.3, 0, 0.7],
                                [0, 0, 0, 0, 0, 0.5, 0.5],
                                [0, 0, 0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 0, 0, 1]])

# For people younger >= 75yo
transitions_75 = tf.constant([[1, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0.5, 0, 0, 0, 0.5], 
                            [0, 0, 0, 0.3, 0, 0, 0.7],
                            [0, 0, 0, 0, 0.3, 0, 0.7],
                            [0, 0, 0, 0, 0, 0.5, 0.5],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1]])

transitions = [transitions_15, transitions_15_44, transitions_45_64, transitions_65_74, transitions_75]
transitions = tf.stack(transitions, axis=2)
print(f'Shape of transitions: {transitions.shape}')
# endregion

# =========== Agents ================

# region
agent_ids = tf.cast(tf.range(0, N_AGENTS), tf.int32)
print(f'Shape of agent_ids: {agent_ids.shape}')

prop_population = tf.convert_to_tensor([[.17, .35, .3, .1, .08]])
transitions_ids = tf.random.categorical(tf.math.log(prop_population), N_AGENTS)
transitions_ids = tf.squeeze(transitions_ids)
print(f'Shape of transitions_ids: {transitions_ids.shape}')

home_cell_ids = tf.random.uniform((N_AGENTS,), minval=0, maxval=N_HOME_CELLS, dtype=tf.int32)
print(f'Shape of home_cell_ids: {home_cell_ids.shape}')

p_moves = draw_beta(0, 1, AVG_P_MOVE, N_AGENTS)
print(f'Shape of p_moves: {p_moves.shape}')

durations_healthy = durations_dead = durations_recovered = tf.cast(tf.ones(shape=(N_AGENTS, 1)) * -1, tf.float16)
print(f'Shape of durations_healthy: {durations_healthy.shape}')
durations_asymptomatic = draw_beta(1, 14, 5, N_AGENTS, True)
durations_mild = draw_beta(5, 10, 7, N_AGENTS, True)
durations_hospital = draw_beta(1, 8, 4, N_AGENTS, True)
durations_reanimation = draw_beta(15, 30, 21, N_AGENTS, True)

durations = [durations_healthy, durations_asymptomatic, durations_mild,
             durations_hospital, durations_reanimation, durations_dead, durations_recovered]
durations = tf.stack(durations, axis=1)
durations = tf.squeeze(durations)
print(f'Shape of durations: {durations.shape}')

current_state_ids = tfp.distributions.Binomial(total_count=1, probs=PROP_INFECTED_AGENTS_START).sample(N_AGENTS)
current_state_ids = tf.cast(current_state_ids, tf.int32) #fixfor gather
current_state_ids = tf.squeeze(current_state_ids)
print(current_state_ids)
print(f'Shape of current_state_ids: {current_state_ids.shape}')
print(f'Sum of current_state_ids (state 0 or 1): {tf.reduce_sum(current_state_ids, axis=-1)}')

current_state_durations = tf.zeros((N_AGENTS,))
current_state_durations = tf.cast(current_state_durations, tf.int32)
print(f'Shape of current_state_durations: {current_state_durations.shape}')

least_state_ids = tf.ones((N_AGENTS,))
print(f'Shape of least_state_ids: {least_state_ids.shape}')

# endregion


# ========== Cells ==============

# region
cell_ids = tf.cast(tf.range(0, N_CELLS), tf.uint32)
print(f'Shape of cell_ids: {cell_ids.shape}')

positions_x = tf.random.uniform(minval=0, maxval=N_SQUARES_AXIS, shape=(N_CELLS, 1))
positions_y = tf.random.uniform(minval=0, maxval=N_SQUARES_AXIS, shape=(N_CELLS, 1))
print(f'Shape of positions_x: {positions_x.shape}')

positions = tf.concat([positions_x, positions_y], axis=1)
print(f'Shape of positions: {positions.shape}')

attractivities = tf.random.uniform(shape=(N_CELLS,))
print(f'Shape of attractivities: {attractivities.shape}')
indices = tf.reshape(tf.range(0, N_HOME_CELLS), shape=(N_HOME_CELLS, 1))
attractivities = tf.tensor_scatter_nd_update(attractivities, indices, tf.zeros(N_HOME_CELLS))


unsafeties = tf.squeeze(draw_beta(0, 1, AVG_UNSAFETY, N_CELLS))
print(f'Shape of unsafeties: {unsafeties.shape}')
unsafeties = tf.tensor_scatter_nd_update(unsafeties, indices, tf.cast(tf.zeros(N_HOME_CELLS) + 1, tf.float16))

# endregion

# ========== Map ==============

# region
map = Map(cell_ids, attractivities, unsafeties, positions_x, positions_y, unique_state_ids,
          unique_contagiousities, unique_sensitivities, unique_severities, transitions, agent_ids, home_cell_ids, p_moves, least_state_ids,
          current_state_ids, current_state_durations, durations, transitions_ids, dscale=DSCALE, current_period=0, verbose=0)
# endregion

# Test pour mode graph:
# @tf.function
def benchmark():
    stats = {}
    t_start = time()

    for i in range(N_PERIODS):
        print(f'starting period {i}...')
        t0 = time()
        for j in range(N_MOVES_PER_PERIOD):
            t_ = time()
            map.make_move()
            print(f'move computed in {time() - t_}s')
        map.forward_all_cells()
        states_ids, state_numbers = map.get_states_numbers()
        states_ids, state_numbers = states_ids.numpy(), state_numbers.numpy()
        stats[i] = {states_ids[k]: state_numbers[k] for k in range(len(states_ids))}
        print(f'period {i} computed in {time() - t0}s')

    print(f'duration: {time() - t_start}s')
    print(stats)


if __name__ == '__main__':
    benchmark()
