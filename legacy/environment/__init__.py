"""Legacy environments are registered safely.
"""
from api.environment import register

register("atari", "AtariEnvironment", "legacy.environment.atari.atari_env")
register("dmlab", "DMLabEnvironment", "legacy.environment.dmlab.dmlab_env")
register("football", "FootballEnvironment", "legacy.environment.google_football.gfootball_env")
register("gym_mujoco", "GymMuJoCoEnvironment", "legacy.environment.gym_mujoco.gym_mujoco_env")
register("hanabi", "HanabiEnvironment", "legacy.environment.hanabi.hanabi_env")
register("single-agent-hanabi", "SingleAgentHanabiEnvironment", "legacy.environment.hanabi.hanabi_env")
register("hide_and_seek", "HideAndSeekEnvironment", "legacy.environment.hide_and_seek.hns_env")
register("overcooked", "OvercookedEnvironment", "legacy.environment.overcooked.overcooked_env")
register("smac", "SMACEnvironment", "legacy.environment.smac.smac_env")
register("vizdoom", "VizDoomEnvironment", "legacy.environment.vizdoom.vizdoom_env")
