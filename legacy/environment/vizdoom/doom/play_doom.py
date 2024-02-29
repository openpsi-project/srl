import sys

from .doom_gym import VizdoomEnv
from .doom_utils import doom_env_by_name, make_doom_env_impl, make_doom_multiplayer_env


def main():
    env_name = "doom_duel_bots"
    env = make_doom_env_impl(doom_env_by_name(env_name), custom_resolution="1280x720")

    # cfg.num_agents = 1
    # cfg.num_bots = 7
    # # cfg.num_humans = 1
    # env = make_doom_multiplayer_env(doom_env_by_name(env_name), cfg=cfg, custom_resolution='1280x720')

    return VizdoomEnv.play_human_mode(env, skip_frames=2, num_actions=15)


if __name__ == "__main__":
    sys.exit(main())
