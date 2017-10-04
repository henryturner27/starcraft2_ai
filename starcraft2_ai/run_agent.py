from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import starcraft2_ai.agents as agents
import starcraft2_ai.platform_settings as platform_settings
from pysc2.env import run_loop
from pysc2.env import sc2_env

import sys
import gflags as flags


FLAGS = flags.FLAGS


class RunAgent(object):

    def __init__(self, map_name, agent):
        FLAGS(sys.argv)
        self.map_name = map_name
        self.agent = agent
        self.replay_dir = platform_settings.replay_dir

    def run_agent(self):
        steps = 2000
        step_mul = 8
        with sc2_env.SC2Env(
                map_name=self.map_name,
                screen_size_px=(84, 84),
                minimap_size_px=(64, 64),
                agent_race='T',
                save_replay_steps=steps * step_mul,
                visualize=True,
                replay_dir=self.replay_dir + self.map_name,
                step_mul=step_mul,
                game_steps_per_episode=steps * step_mul) as env:
            agent = self.agent
            run_loop.run_loop([agent], env, steps)


if __name__ == "__main__":
    RunAgent(map_name='Simple128', agent=agents.TerranBasicAgent()).run_agent()
