from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import starcraft2_ai.agents as agents
import starcraft2_ai.platform_settings as platform_settings
from pysc2.env import run_loop
from pysc2.env import sc2_env

import sys
from absl import flags


FLAGS = flags.FLAGS


def run_agent(map_name, agent):
    steps = 2000
    step_mul = 8
    with sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(
                    screen=64,
                    minimap=64)),
            visualize=True,
            step_mul=step_mul,
            game_steps_per_episode=steps * step_mul) as env:
        run_loop.run_loop([agent], env, steps)


if __name__ == "__main__":
    FLAGS(sys.argv)
    run_agent(map_name='Simple128', agent=agents.TerranBasicAgent())
