class Config:
    # Grid and obstacles
    GRID_SIZE = 10
    NUM_OBSTACLES = 10

    # Q-learning parameters
    ALPHA = 0.7        # Learning rate
    EPSILON = 0.2      # Exploration probability
    GAMMA = 0.9        # Discount factor

    # Rewards
    WIN = 1000
    ARREST = -100
    STEP = -2

    # Simulation
    EPISODES = 50
    STEP_DELAY = 1/60  # seconds

    # Cop stochastic parameters
    COP_TIMING_RANGE = (1, 1)     # Steps before cop moves
    COP_DISTANCE_RANGE = (1, 1)   # Distance cop moves each step
