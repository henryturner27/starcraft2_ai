from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import starcraft2_ai.agents as agents
import starcraft2_ai.minigame_agents as ma
import starcraft2_ai.platform_settings as platform_settings
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


    def run_agent(self):
        steps = 2000
        step_mul = 8
        with sc2_env.SC2Env(
                save_replay_episodes=1,
                replay_dir='C:/Users/Henry/Dropbox/jupyter_projects/starcraft2_ai/replays',
                map_name=self.map_name,
                players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.protoss, difficulty='VeryEasy')],
                agent_interface_format=sc2_env.AgentInterfaceFormat(
                    feature_dimensions=sc2_env.Dimensions(
                        screen=84,
                        minimap=84)),
                visualize=True,
                step_mul=step_mul,
                game_steps_per_episode=steps * step_mul) as env:
            run_loop.run_loop([self.agent], env, steps)


if __name__ == "__main__":
    AgentInstance(map_name='Simple128', agent=agents.TerranBasicAgent()).run_agent()
