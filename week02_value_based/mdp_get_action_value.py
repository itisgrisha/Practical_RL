
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    return sum(
        prob * (mdp.get_reward(state, action, state_dash) + gamma * state_values[state_dash]) 
        for state_dash, prob in mdp.get_next_states(state, action).items()
    )
