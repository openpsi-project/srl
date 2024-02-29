from legacy.algorithm.ppo import *
# from legacy.algorithm.dagger import *
from legacy.algorithm.q_learning import *
try:
    from legacy.algorithm.muzero import *
except ImportError:
    print("Import Mu-Zero algorithm has failed. Possibly due to missing c-MCTS module.")
