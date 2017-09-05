#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a random agent for a few steps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import starcraft2_ai.agents as agents
from pysc2.env import run_loop
from pysc2.env import sc2_env


class RunAgent(object):

    def __init__(self, map_name, agent):
        self.map_name = map_name
        self.agent = agent

    def run_agent(self):
        steps = 100
        step_mul = 8
        with sc2_env.SC2Env(
                map_name=self.map_name,
                screen_size_px=(84, 84),
                minimap_size_px=(64, 64),
                agent_race='T',
                save_replay_steps=steps * step_mul,
                replay_dir='/Users/turnerh27/Dropbox (Personal)/jupyter_projects/starcraft2_ai/replays/'+self.map_name,
                step_mul=step_mul,
                game_steps_per_episode=steps * step_mul) as env:
            agent = self.agent
            run_loop.run_loop([agent], env, steps)


if __name__ == "__main__":
    RunAgent(map_name='CollectMineralsAndGas', agent=agents.CollectMinerals()).run_agent()
