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
"""A base agent to write custom scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
_BACKGROUND = 0
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_MINERAL_FIELD = 341
_VESPENE_GEYSER = 342
_MAX_REFINERIES = 2
_SCV = 45
_COMMAND_CENTER = 18
_BARRACKS = 21
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_SCREEN = actions.FUNCTIONS.select_rect.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_ALL_TYPE = [2]
_SELECT_UNIT = actions.FUNCTIONS.select_unit.id
_CONTROL_GROUP =actions.FUNCTIONS.select_control_group.id
_SET_CONTROL_GROUP = [1]
_SELECT_CONTROL_GROUP = [0]
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SMART_SCREEN = actions.FUNCTIONS.Smart_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_HARVEST_GATHER_SCREEN = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_RALLY_WORKERS = actions.FUNCTIONS.Rally_Workers_screen.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_QUEUED = [1]
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

# def window_avg(array, s):
#     final_list = []
#     for row in range(len(array)):
#         temp_list = []
#         for value_index in range(len(array[row])):
#             row_array = array[row]
#             column_array = array[:, value_index]
#             if value_index-s <= 0:
#                 prepended_zeros_row = np.zeros(s-value_index, dtype='int')
#                 row_array = np.insert(row_array, 0, prepended_zeros_row)
#             if row-s <= 0:
#                 prepended_zeros_column = np.zeros(s-row, dtype='int')
#                 column_array = np.insert(column_array, 0, prepended_zeros_column)
#             if value_index+s+1 > len(array):
#                 row_array = np.append(row_array, np.zeros(abs(s-value_index)+len(prepended_zeros_row)-1, dtype='int'))
#             if row+s+1 > len(array):
#                 column_array = np.append(column_array, np.zeros(abs(s-row)+len(prepended_zeros_column)-1, dtype='int'))
#             window_row = row_array[(value_index-s+len(prepended_zeros_row)) : (value_index+s+len(prepended_zeros_row)+1)]
#             window_column = column_array[(row-s+len(prepended_zeros_column)) : (row+s+len(prepended_zeros_column)+1)]
#             temp_list.append((window_row.sum()+window_column.sum())/((s*4)+1))
#         final_list.append(temp_list)
#     return np.asarray(final_list)


def find_farthest(array):
    final_list = []
    item_locations = []
    x_coords, y_coords = array.nonzero()
    zipped = zip(x_coords, y_coords)
    for coords in zipped:
        item_locations.append(coords)
    for row in range(len(array)):
        temp_list = []
        for value_index in range(len(array[row])):
            dist_list = []
            for coords in item_locations:
                dist_list.append(np.linalg.norm(np.asarray([row, value_index]) - np.asarray([coords])))
            temp_list.append(np.min(dist_list))
        final_list.append(temp_list)
    return np.unravel_index(np.asarray(final_list).argmax(), (84,84))


def find_farthest_with_helper_coords(array, helper_coords):
    final_list = []
    item_locations = []
    x_coords, y_coords = array.nonzero()
    zipped = zip(x_coords, y_coords)
    for coords in zipped:
        item_locations.append(coords)
    for row in range(len(helper_coords)):
        temp_list = []
        for value_index in range(len(helper_coords[row])):
            dist_list = []
            for coords in item_locations:
                dist_list.append(np.linalg.norm(np.asarray([row, value_index]) - np.asarray([coords])))
            temp_list.append(np.min(dist_list))
        final_list.append(temp_list)
    return np.unravel_index(np.asarray(final_list).argmax(), (84,84))


def count_nonzero(array):
  final_list = []
  for row in range(len(array)):
    temp_list = []
    for value_index in range(len(array[row])):
      row_array = array[row]
      column_array = array[:, value_index]
      row_count = np.count_nonzero(row_array)
      column_count = np.count_nonzero(column_array)
      temp_list.append(row_count + column_count)
    final_list.append(temp_list)
  return np.asarray(final_list)

def window_avg(array, size):
    temp_array = array
    final_list = []
    for row in range(len(temp_array)):
        temp_list = []
        for value_index in range(len(temp_array[row])):
            summation = temp_array[np.clip(row-size, 0, len(temp_array)+1):np.clip(row+size+1, 0, len(temp_array)+1), np.clip(value_index-size, 0, len(temp_array)+1):np.clip(value_index+size+1, 0, len(temp_array)+1)]
            avg = summation.sum()/(((size*2)+1)**2)
            temp_list.append(avg)
        final_list.append(temp_list)
    return np.asarray(final_list)





class BaseAgent(object):
  """A base agent to write custom scripted agents."""

  def setup(self, obs_spec, action_spec):
    self.reward = 0
    self.episodes = 0
    self.steps = 0
    self.obs_spec = obs_spec
    self.action_spec = action_spec

  def reset(self):
    self.episodes += 1
    self.target = None
    self.gas_target = None
    self.first_refinery_location = None
    self.second_refinery_location = None
    self.refineries_built = 0
    self.workers_on_ref_one = 0
    self.workers_on_ref_two = 0
    self.assigned_worker_to_gas = 0
    self.rally_set = False
    self.workers_mining = False
    self.idle_worker_selected = False
    self.initial_worker_selection = False
    self.freeze_sd_building = 0
    self.freeze_ref_worker_assignment = 0
    self.barracks_to_build = 1
    self.barracks_built = 0
    self.freeze_barracks_selection = 0
    self.building_locations = []


  def step(self, obs):
    self.steps += 1
    self.reward += obs.reward
    return actions.FunctionCall(0, [])


class CollectMinerals(BaseAgent):
  """An agent specifically designed for solving the Collect MineralsAndGas map."""

  def __init__(self):
    self.target = None

  def step(self, obs):
    super(CollectMinerals, self).step(obs)
    if self.initial_worker_selection:
      if _SELECT_IDLE_WORKER not in obs.observation["available_actions"]:
        if not self.rally_set:
          if _RALLY_WORKERS not in obs.observation["available_actions"]:
            unit_type = obs.observation["screen"][_UNIT_TYPE]
            command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
            command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [command_center[0], command_center[1]]])
          else:
            self.rally_set = True                       
            return actions.FunctionCall(_RALLY_WORKERS, [_NOT_QUEUED, self.target])
        else:
          try:
            if (obs.observation["player"][3] < obs.observation["player"][4]) & (obs.observation["player"][1] >= 50) & (obs.observation["player"][6] < 30) & (obs.observation["build_queue"][0][6] >= 90) & (len(obs.observation["build_queue"]) < 2):
              return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])
            else:
                return actions.FunctionCall(_NO_OP, [])
          except IndexError:
            if _TRAIN_SCV not in obs.observation["available_actions"]:
              return actions.FunctionCall(_NO_OP, [])
            else:
              return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])
      unit_type = obs.observation["screen"][_UNIT_TYPE]
      minerals_observed = (unit_type == _MINERAL_FIELD)
      processed_neutral_y, processed_neutral_x = (window_avg(minerals_observed, 5) > 0.9).nonzero()
      command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
      command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
      zipped_minerals = zip(processed_neutral_x, processed_neutral_y)
      mineral_locations = []
      for location in zipped_minerals:
        mineral_locations.append(location)
      closest, min_dist = None, None
      for p in mineral_locations:
        dist = np.linalg.norm(np.array(command_center) - np.array(p))
        if not min_dist or dist < min_dist:
          closest, min_dist = p, dist
      if self.target == None:
        self.target = [int(closest[0]), int(closest[1])]
      self.workers_mining = True
      return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.target])
    else:
      if not self.initial_worker_selection:
        self.initial_worker_selection = True
        return actions.FunctionCall(_SELECT_SCREEN, [[0], [0, 0], [83, 83]])
      else:
        return actions.FunctionCall(_NO_OP, [])


class CollectMineralsAndGas(BaseAgent):
  """An agent specifically designed for solving the Collect MineralsAndGas map."""

  def step(self, obs):
    super(CollectMineralsAndGas, self).step(obs)
    #initialize
    if self.initial_worker_selection:

      # add in better building placement logic

      #set workers to mining
      if not self.workers_mining:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        minerals_observed = (unit_type == _MINERAL_FIELD)
        processed_neutral_y, processed_neutral_x = (window_avg(minerals_observed, 2) > 0.99).nonzero()
        command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
        command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
        zipped_minerals = zip(processed_neutral_x, processed_neutral_y)
        mineral_locations = []
        for location in zipped_minerals:
          mineral_locations.append(location)
        closest, min_dist = None, None
        for p in mineral_locations:
          dist = np.linalg.norm(np.array(command_center) - np.array(p))
          if not min_dist or dist < min_dist:
            closest, min_dist = p, dist
        if self.target == None:
          self.target = [int(closest[0]), int(closest[1])]
        self.workers_mining = True
        print('smart screen 172')
        return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.target])
      
      #set rally for workers
      if not self.rally_set:
        if _RALLY_WORKERS not in obs.observation["available_actions"]:
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
          command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
          print('select point 179')
          return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [command_center[0], command_center[1]]])
        else:
          self.rally_set = True
          print('rally workers 182')
          return actions.FunctionCall(_RALLY_WORKERS, [_NOT_QUEUED, self.target])

      #move idle workers back to mining minerals
      if _SELECT_IDLE_WORKER in obs.observation["available_actions"]:
      	if not self.idle_worker_selected:
      	  self.idle_worker_selected = True
      	  print('select idle worker 198')
      	  return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
      	elif self.idle_worker_selected:
      	  self.idle_worker_selected = False
      	  print('smart screen 215')
      	  return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.target])

      #train SCVs
      if (obs.observation["player"][3] < obs.observation["player"][4]) & (obs.observation["player"][1] >= 50) & (obs.observation["player"][6] < 30) & (_TRAIN_SCV in obs.observation["available_actions"]):
        if (len(obs.observation["build_queue"]) == 1) & ((obs.observation["player"][6]) + len(obs.observation["build_queue"]) < 30):
          if (obs.observation["build_queue"][0][6] >= 90):
            print('train scv 422')
            return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])
        elif (len(obs.observation["build_queue"]) == 0) & (_TRAIN_SCV in obs.observation["available_actions"]) & ((obs.observation["player"][6]) + len(obs.observation["build_queue"]) < 30):
          print('train scv 425')
          return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])

      #build supply depots
      if ((obs.observation["player"][4] - obs.observation["player"][3]) <= 3)  & (obs.observation["player"][1] >= 100) & (_BUILD_SUPPLY_DEPOT not in obs.observation["available_actions"]) & (self.steps >= self.freeze_sd_building + 50):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        scv_observed = (unit_type == _SCV)
        processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.99).nonzero()
        scv_mass_top_left = [processed_neutral_x.mean()-3, processed_neutral_y.mean()-3]
        scv_mass_bottom_right = [processed_neutral_x.mean()+3, processed_neutral_y.mean()+3]
        print('select some scvs 235')
        return actions.FunctionCall(_SELECT_SCREEN, [_NOT_QUEUED, scv_mass_top_left, scv_mass_bottom_right])
      elif ((obs.observation["player"][4] - obs.observation["player"][3]) <= 3)  & (obs.observation["player"][1] >= 100) & (_BUILD_SUPPLY_DEPOT in obs.observation["available_actions"]) & (self.steps >= self.freeze_sd_building + 50):
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        background = (player_relative == 0)
        ####testing blocking out known placements
        for building in self.building_locations:
          background[np.clip(building[0]-4, 0, len(background)):np.clip(building[0]+4, 0, len(background)), np.clip(building[1]-4, 0, len(background)):np.clip(building[1]+4, 0, len(background))] = 0
        ####
        ####random placement
        # processed_freespace_x, processed_freespace_y = (window_avg(background, 4) > 0.99).nonzero()
        # random_placement = random.choice(range(len(processed_freespace_x)))
        # freespace = [int(processed_freespace_x[random_placement]), int(processed_freespace_y[random_placement])]
        ####
        ####fixed placement
        freespace = np.unravel_index(window_avg(background, 12).argmax(), (84,84))
        ####
        self.freeze_sd_building = self.steps
        self.building_locations.append(freespace)
        print('freespace', freespace)
        print('build supply depot 345')
        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [[0], freespace])

      #build refineries
      if ((obs.observation["player"][3]) >= 24)  & (obs.observation["player"][1] >= 75) & (_BUILD_REFINERY in obs.observation["available_actions"]) & (self.refineries_built < 2):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        gas_observed = (unit_type == _VESPENE_GEYSER)
        processed_neutral_y, processed_neutral_x = (window_avg(gas_observed, 3) > 0.9).nonzero()
        command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
        command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
        zipped_gas = zip(processed_neutral_x, processed_neutral_y)
        gas_locations = []
        for location in zipped_gas:
          gas_locations.append(location)
        closest, min_dist = None, None
        for p in gas_locations:
          dist = np.linalg.norm(np.array(command_center) - np.array(p))
          if not min_dist or dist < min_dist:
            closest, min_dist = p, dist
        self.gas_target = [int(closest[0]), int(closest[1])]
        self.second_refinery_location = self.gas_target
        print('build refinery 257')
        self.refineries_built += 1
        self.freeze_ref_worker_assignment = self.steps
        return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, self.gas_target])
      elif ((obs.observation["player"][3]) >= 18)  & (obs.observation["player"][1] >= 75) & (_BUILD_REFINERY in obs.observation["available_actions"]) & (self.refineries_built < 1):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        gas_observed = (unit_type == _VESPENE_GEYSER)
        processed_neutral_y, processed_neutral_x = (window_avg(gas_observed, 3) > 0.99).nonzero()
        command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
        command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
        zipped_gas = zip(processed_neutral_x, processed_neutral_y)
        gas_locations = []
        for location in zipped_gas:
          gas_locations.append(location)
        closest, min_dist = None, None
        for p in gas_locations:
          dist = np.linalg.norm(np.array(command_center) - np.array(p))
          if not min_dist or dist < min_dist:
            closest, min_dist = p, dist
        self.gas_target = [int(closest[0]), int(closest[1])]
        self.first_refinery_location = self.gas_target
        print('build refinery 276')
        self.refineries_built += 1
        self.freeze_ref_worker_assignment = self.steps
        return actions.FunctionCall(_BUILD_REFINERY, [_NOT_QUEUED, self.gas_target])

      #put workers on refineries
      if (self.refineries_built == 1) & (self.workers_on_ref_one < 2) & (_MOVE_SCREEN in obs.observation["available_actions"]) & (len(obs.observation['single_select']) == 1) & (self.steps >= self.freeze_ref_worker_assignment + 75) & (self.steps >= self.assigned_worker_to_gas + 15):
        if obs.observation['single_select'][0][0] == 45:
          self.workers_on_ref_one += 1
          self.assigned_worker_to_gas = self.steps
          print('set scv to refinery 290')
          return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.first_refinery_location])
      if (self.refineries_built == 1) & (self.workers_on_ref_one < 2) & (_MOVE_SCREEN in obs.observation["available_actions"]) & (self.steps >= self.freeze_ref_worker_assignment + 75):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        scv_observed = (unit_type == _SCV)
        processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.99).nonzero()
        scv_mass = [processed_neutral_x.mean(), processed_neutral_y.mean()]
        print('select one scv 286')
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, scv_mass])
      if (self.refineries_built == 2) & (self.workers_on_ref_two < 2) & (_MOVE_SCREEN in obs.observation["available_actions"]) & (len(obs.observation['single_select']) == 1) & (self.steps >= self.freeze_ref_worker_assignment + 75) & (self.steps >= self.assigned_worker_to_gas + 15):
        if obs.observation['single_select'][0][0] == 45:
          self.workers_on_ref_two += 1
          self.assigned_worker_to_gas = self.steps
          print('set scv to refinery 300')
          return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.second_refinery_location])
      if (self.refineries_built == 2) & (self.workers_on_ref_two < 2) & (_MOVE_SCREEN in obs.observation["available_actions"]) & (self.steps >= self.freeze_ref_worker_assignment + 75):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        scv_observed = (unit_type == _SCV)
        processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.99).nonzero()
        scv_mass = [processed_neutral_x.mean(), processed_neutral_y.mean()]
        print('select one scv 296')
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, scv_mass])

      #if nothing to do then cycle through buildings and workers
      _alternator = self.steps % 2
      if _alternator == 0:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
        command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [command_center[0], command_center[1]]])
      elif _alternator == 1:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        scv_observed = (unit_type == _SCV)
        processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.99).nonzero()
        scv_mass_top_left = [processed_neutral_x.mean()-3, processed_neutral_y.mean()-3]
        scv_mass_bottom_right = [processed_neutral_x.mean()+3, processed_neutral_y.mean()+3]
        return actions.FunctionCall(_SELECT_SCREEN, [_NOT_QUEUED, scv_mass_top_left, scv_mass_bottom_right])

    else:
      if not self.initial_worker_selection:
        self.initial_worker_selection = True
        print('select screen 239')
        return actions.FunctionCall(_SELECT_SCREEN, [[0], [0, 0], [83, 83]])
      else:
        return actions.FunctionCall(_NO_OP, [])


class BuildMarines(BaseAgent):
  """An agent designed for solving the BuildMarines map."""

  def step(self, obs):
    super(BuildMarines, self).step(obs)
    #initialize
    if self.initial_worker_selection:

      # add in better building placement logic - THIS IS PRIORITY ONE, BECAUSE THEN I CAN SAVE COORDS OF BUILDINGS THAT ARE PLACED
      # need to assign control groups and iterate through those using the alternator
      # barracks count number should be controlled by number in control group

      #set workers to mining
      if not self.workers_mining:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        minerals_observed = (unit_type == _MINERAL_FIELD)
        processed_neutral_y, processed_neutral_x = (window_avg(minerals_observed, 2) > 0.9).nonzero()
        command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
        command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
        zipped_minerals = zip(processed_neutral_x, processed_neutral_y)
        mineral_locations = []
        for location in zipped_minerals:
          mineral_locations.append(location)
        closest, min_dist = None, None
        for p in mineral_locations:
          dist = np.linalg.norm(np.array(command_center) - np.array(p))
          if not min_dist or dist < min_dist:
            closest, min_dist = p, dist
        if self.target == None:
          self.target = [int(closest[0]), int(closest[1])]
        self.workers_mining = True
        print('smart screen 172')
        return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.target])
      
      #set rally for workers
      if not self.rally_set:
        if _RALLY_WORKERS not in obs.observation["available_actions"]:
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
          command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
          print('select point 179')
          return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [command_center[0], command_center[1]]])
        else:
          self.rally_set = True
          print('rally workers 182')
          return actions.FunctionCall(_RALLY_WORKERS, [_NOT_QUEUED, self.target])

      #set control group for command center to 5
      if ((obs.observation["control_groups"][5][0]) == 0):
        if (_RALLY_WORKERS in obs.observation["available_actions"]):
          print('set command center to control group 5')
          return actions.FunctionCall(_CONTROL_GROUP, [_SET_CONTROL_GROUP, [5]])
        else:
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
          command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
          print('select point 179')
          return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [command_center[0], command_center[1]]])

      #move idle workers back to mining minerals
      if _SELECT_IDLE_WORKER in obs.observation["available_actions"]:
        if not self.idle_worker_selected:
          self.idle_worker_selected = True
          print('select idle worker 198')
          return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
        elif self.idle_worker_selected:
          self.idle_worker_selected = False
          print('smart screen 215')
          return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.target])

      #train SCVs
      if (obs.observation["player"][3] < obs.observation["player"][4]) & (obs.observation["player"][1] >= 50) & (obs.observation["player"][6] < 16) & (_TRAIN_SCV in obs.observation["available_actions"]):
        if (len(obs.observation["build_queue"]) == 1) & ((obs.observation["player"][6]) + len(obs.observation["build_queue"]) < 16):
          if (obs.observation["build_queue"][0][6] >= 90):
            print('train scv 422')
            return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])
        elif (len(obs.observation["build_queue"]) == 0) & (_TRAIN_SCV in obs.observation["available_actions"]) & ((obs.observation["player"][6]) + len(obs.observation["build_queue"]) < 16):
          print('train scv 425')
          return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])

      #build supply depots
      if ((obs.observation["player"][4] - obs.observation["player"][3]) <= 3)  & (obs.observation["player"][1] >= 100) & (_BUILD_SUPPLY_DEPOT not in obs.observation["available_actions"]) & (self.steps >= self.freeze_sd_building + 50):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        scv_observed = (unit_type == _SCV)
        processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.99).nonzero()
        scv_mass_top_left = [processed_neutral_x.mean()-3, processed_neutral_y.mean()-3]
        scv_mass_bottom_right = [processed_neutral_x.mean()+3, processed_neutral_y.mean()+3]
        print('select some scvs 235')
        return actions.FunctionCall(_SELECT_SCREEN, [_NOT_QUEUED, scv_mass_top_left, scv_mass_bottom_right])
      elif ((obs.observation["player"][4] - obs.observation["player"][3]) <= 3)  & (obs.observation["player"][1] >= 100) & (_BUILD_SUPPLY_DEPOT in obs.observation["available_actions"]) & (self.steps >= self.freeze_sd_building + 50):
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        background = (player_relative == 0)
        ####testing blocking out known placements
        for building in self.building_locations:
          background[np.clip(building[0]-4, 0, len(background)):np.clip(building[0]+4, 0, len(background)), np.clip(building[1]-4, 0, len(background)):np.clip(building[1]+4, 0, len(background))] = 0
        ####
        ####random placement
        # processed_freespace_x, processed_freespace_y = (window_avg(background, 4) > 0.99).nonzero()
        # random_placement = random.choice(range(len(processed_freespace_x)))
        # freespace = [int(processed_freespace_x[random_placement]), int(processed_freespace_y[random_placement])]
        ####
        ####fixed placement
        freespace = np.unravel_index(window_avg(background, 12).argmax(), (84,84))
        ####
        self.freeze_sd_building = self.steps
        self.building_locations.append(freespace)
        print('build supply depot 544')
        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [[0], freespace])

      #build barracks
      if (len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 2) > 0.99).nonzero()[0])) < self.barracks_to_build) & (obs.observation["player"][1] >= 150) & (obs.observation["player"][3] >= 13) & (_BUILD_BARRACKS in obs.observation["available_actions"]):
        print('len barracks coords', str(len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0]))))
        print('barracks coords', str(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0])))
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        background = (player_relative == 0)
        for building in self.building_locations:
          background[np.clip(building[0]-4, 0, len(background)):np.clip(building[0]+4, 0, len(background)), np.clip(building[1]-4, 0, len(background)):np.clip(building[1]+4, 0, len(background))] = 0
        ####
        ####random placement
        # processed_freespace_x, processed_freespace_y = (window_avg(background, 4) > 0.99).nonzero()
        # random_placement = random.choice(range(len(processed_freespace_x)))
        # freespace = [int(processed_freespace_x[random_placement]), int(processed_freespace_y[random_placement])]
        ####
        ####fixed placement
        freespace = np.unravel_index(window_avg(background, 12).argmax(), (84,84))
        ####
        self.building_locations.append(freespace)
        print('build barracks 472')
        return actions.FunctionCall(_BUILD_BARRACKS, [[0], freespace])

      #set barracks to control group
      if len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 2) > 0.99).nonzero()[0])) > obs.observation["control_groups"][4][1]:
        if (len(obs.observation["multi_select"]) == len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 2) > 0.99).nonzero()[0]))):
          unit_list = []
          for unit in obs.observation["multi_select"]:
            if unit[0] != _BARRACKS:
              return actions.FunctionCall(_SELECT_UNIT, [[3], [unit[0]]])
            else:
              unit_type = obs.observation["screen"][_UNIT_TYPE]
          barracks = (unit_type == _BARRACKS)

          ###### DEBUGGING - it doesn't select all barracks for some reason
          print('len barracks coords', str(len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0]))))
          print('barracks coords', str(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0])))
          #np.savetxt('unit_type_array.txt', unit_type, delimiter=',')
          #np.savetxt('barracks_array.txt', barracks, delimiter=',')
          #test_x, test_y = barracks.nonzero()
          #print(str(np.argmax(window_avg(barracks, 3))))
          processed_barracks_x, processed_barracks_y = (window_avg(barracks, 2) > 0.99).nonzero()
          zipped_barracks = list(zip(processed_barracks_x, processed_barracks_y))
          #print(str(zipped_barracks))
          selected_barracks = zipped_barracks[0]
          #print(str(selected_barracks))

          print(str(zipped_barracks))
          ############end debugging

          print('select all barracks 519')
          return actions.FunctionCall(_SELECT_POINT, [[_ALL_TYPE], [selected_barracks[0], selected_barracks[1]]])

          print(str('multi select len:'), str(len(obs.observation["multi_select"])))
          print(str('multi select:'), str(obs.observation["multi_select"]))
          print('len barracks coords', str(len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0]))))
          print('barracks coords', str(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0])))
          print('set barracks to control group 4')
          return actions.FunctionCall(_CONTROL_GROUP, [_SET_CONTROL_GROUP, [4]])
        elif (len(obs.observation["multi_select"] != len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0])))):
          unit_type = obs.observation["screen"][_UNIT_TYPE]
          barracks = (unit_type == _BARRACKS)

          ###### DEBUGGING - it doesn't select all barracks for some reason
          print('len barracks coords', str(len(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0]))))
          print('barracks coords', str(list((window_avg(obs.observation["screen"][_UNIT_TYPE] == _BARRACKS, 6) > 0.99).nonzero()[0])))
          # np.savetxt('unit_type_array.txt', unit_type, delimiter=',')
          # np.savetxt('barracks_array.txt', barracks, delimiter=',')
          #test_x, test_y = barracks.nonzero()
          #print(str(np.argmax(window_avg(barracks, 3))))
          processed_barracks_x, processed_barracks_y = (window_avg(barracks, 2) > 0.99).nonzero()
          zipped_barracks = list(zip(processed_barracks_x, processed_barracks_y))
          #print(str(zipped_barracks))
          selected_barracks = zipped_barracks[0]
          #print(str(selected_barracks))

          print(str(zipped_barracks))
          ############end debugging

          print('select all barracks 548')
          return actions.FunctionCall(_SELECT_POINT, [[0], [selected_barracks[0], selected_barracks[1]]])

      #train marines
      if (obs.observation["player"][3] < obs.observation["player"][4]) & (obs.observation["player"][1] >= 50) & (_TRAIN_MARINE in obs.observation["available_actions"]):
        if (len(obs.observation["build_queue"]) == 1):
          if (obs.observation["build_queue"][0][6] >= 90):
            print('train marine 463')
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        elif (len(obs.observation["build_queue"]) == 0) & (_TRAIN_MARINE in obs.observation["available_actions"]):
          return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        else:
          return actions.FunctionCall(_NO_OP, [])

      #if nothing to do then cycle through control groups and workers
      if (obs.observation["control_groups"][4][1]) > 0:
        _ = 3
      else:
        _ = 2
      _cycler = self.steps % _

      if _cycler == 0:
        return actions.FunctionCall(_CONTROL_GROUP, [_SELECT_CONTROL_GROUP, [5]])
      elif _cycler == 1:
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        scv_observed = (unit_type == _SCV)
        processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.8).nonzero()
        scv_mass_top_left = [processed_neutral_x.mean()-5, processed_neutral_y.mean()-5]
        scv_mass_bottom_right = [processed_neutral_x.mean()+5, processed_neutral_y.mean()+5]
        return actions.FunctionCall(_SELECT_SCREEN, [_NOT_QUEUED, scv_mass_top_left, scv_mass_bottom_right])
      elif _cycler == 2:
        return actions.FunctionCall(_CONTROL_GROUP, [_SELECT_CONTROL_GROUP, [4]])

    else:
      if not self.initial_worker_selection:
        self.initial_worker_selection = True
        print('select screen 239')
        return actions.FunctionCall(_SELECT_SCREEN, [[0], [0, 0], [83, 83]])
      else:
        return actions.FunctionCall(_NO_OP, [])

      
