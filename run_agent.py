from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import starcraft2_ai.agents as agents
import minigame_agents as ma
import platform_settings
from pysc2.env import run_loop
from pysc2.env import sc2_env

import sys
from absl import flags


class AgentInstance(object):
    def __init__(self, map_name, agent):
        FLAGS = flags.FLAGS
        flags.DEFINE_string('f', '', 'kernel')
        FLAGS(sys.argv)
        self.map_name = map_name
        self.agent = agent
        self.screen_size = 256
        self.minimap_size = 64

    def run_agent(self):
        steps = 2000
        step_mul = 8
        with sc2_env.SC2Env(
                save_replay_episodes=1,
                replay_dir=platform_settings.get_replay_dir(),
                map_name=self.map_name,
                players=[sc2_env.Agent(sc2_env.Race.terran),
                    sc2_env.Bot(sc2_env.Race.protoss,
                    difficulty='VeryEasy')],
                agent_interface_format=sc2_env.AgentInterfaceFormat(
                    feature_dimensions=sc2_env.Dimensions(
                        screen=self.screen_size,
                        minimap=self.minimap_size),
                    camera_width_world_units=144),
                visualize=True,
                step_mul=step_mul,
                game_steps_per_episode=steps * step_mul) as env:
            run_loop.run_loop([self.agent], env, steps)


if __name__ == "__main__":
    AgentInstance(map_name='Simple128', agent=agents.TerranBasicAgent()).run_agent()
