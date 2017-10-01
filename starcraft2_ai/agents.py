from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_UNIT_DENSITY = features.SCREEN_FEATURES.unit_density.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_MM_SELECTED = features.MINIMAP_FEATURES.selected.index
_MM_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index
_MM_CAMERA = features.MINIMAP_FEATURES.camera.index
_MM_HEIGHT = features.MINIMAP_FEATURES.height_map.index
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
_CONTROL_GROUP = actions.FUNCTIONS.select_control_group.id
_SET_CONTROL_GROUP = [1]
_SELECT_CONTROL_GROUP = [0]
_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
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
_SMART_MINIMAP = actions.FUNCTIONS.Smart_minimap.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_QUEUED = [1]
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


def _layer_string(layer):
    return '\n'.join(' '.join(str(v) for v in row) for row in layer)

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
            summation = temp_array[
                        np.clip(row - size, 0, len(temp_array) + 1):np.clip(row + size + 1, 0, len(temp_array) + 1),
                        np.clip(value_index - size, 0, len(temp_array) + 1):np.clip(value_index + size + 1, 0,
                                                                                    len(temp_array) + 1)]
            avg = summation.sum() / (((size * 2) + 1) ** 2)
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

    def step(self, obs):
        self.steps += 1
        self.reward += obs.reward
        return actions.FunctionCall(0, [])

class TerranBasicAgent(BaseAgent):
    """An agent that plays Terran and builds marines."""

    def reset(self):
        self.builder_scvs_selected = False
        self.army_selected = False
        self.production_buildings_selected = False
        self.initial_base = None
        self.initial_scv_loc = None
        self.first_refinery_location = None
        self.second_refinery_location = None
        self.refineries_built = 0
        self.workers_on_ref_one = 0
        self.workers_on_ref_two = 0
        self.assigned_worker_to_gas = 0
        self.rally_set = False
        self.production_rally_set = False
        self.workers_mining = False
        self.idle_worker_selected = False
        self.initial_worker_selection = False
        self.freeze_sd_building = 0
        self.freeze_ref_worker_assignment = 0
        self.barracks_to_build = 3
        self.barracks_built = 0
        self.freeze_barracks_selection = 0
        self.building_locations = []
        self.supply_depot_locations = []
        self.barracks_locations = []
        self.front_ramp = None

    def step(self, obs):
        super(TerranBasicAgent, self).step(obs)
        # initialize
        while True:

            ### NOTES FOR ADDITIONAL LOGIC TO ADD:
            # start adding in logic based on absolute position using the minimap instead of relative position using the screen - need this for gas mining and building locations
                # all build screen actions should have a check to make sure the screen is centered in the correct spot
            # once number of marines exceeds threshold... ATTACK!
            # build supply depots based on food growth per step

            # set control group for command center to 5, define initial command center location, and assess minimap heights
            if ((obs.observation["control_groups"][5][0]) == 0):
                if (_RALLY_WORKERS in obs.observation["available_actions"]):
                    base_y, base_x = (obs.observation["minimap"][_MM_SELECTED] == 1).nonzero()
                    self.initial_base = [int(base_x.mean()), int(base_y.mean())]
                    base_height = obs.observation["minimap"][_MM_HEIGHT][self.initial_base[1], self.initial_base[0]]
                    height_map = obs.observation["minimap"][_MM_HEIGHT]
                    heights, counts = np.unique(height_map[height_map > 0], return_counts=True)
                    useful_heights = heights[np.argwhere(counts > (len(height_map.flatten()) * 0.05))].flatten()
                    useful_heights = useful_heights[useful_heights.argsort()[::-1]]
                    inbetween_heights = (height_map < useful_heights[np.argwhere(useful_heights == base_height)]) & (
                        height_map > useful_heights[(np.argwhere(useful_heights == base_height) + 1)])
                    ramp_y, ramp_x = (window_avg(inbetween_heights, 1) > 0.7).nonzero()
                    ramp_inds = zip(ramp_y, ramp_x)
                    ramp_ind_list = []
                    for loc in ramp_inds:
                        ramp_ind_list.append(np.asarray(loc))
                    absolute_positions = []
                    for ind in ramp_ind_list:
                        surrounding = height_map[ind[0] - 1:ind[0] + 2, ind[1] - 1:ind[1] + 2]
                        relative_y, relative_x = (surrounding == base_height).nonzero()
                        relative = zip(relative_y - 1, relative_x - 1)
                        for loc in relative:
                            abs_pos = ind + loc
                            absolute_positions.append(abs_pos)
                    closest, min_dist = None, None
                    for p in absolute_positions:
                        dist = np.linalg.norm(np.array(self.initial_base) - np.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    self.front_ramp = closest
                    print('set command center to control group 5')
                    return actions.FunctionCall(_CONTROL_GROUP, [_SET_CONTROL_GROUP, [5]])
                else:
                    unit_type = obs.observation["screen"][_UNIT_TYPE]
                    command_center_y, command_center_x = (unit_type == _COMMAND_CENTER).nonzero()
                    command_center = [int(command_center_x.mean()), int(command_center_y.mean())]
                    print('select point 179')
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, [command_center[0], command_center[1]]])

            # define initial scv location
            if (obs.observation["minimap"][_MM_CAMERA][self.initial_base[1], self.initial_base[0]] == 1) & \
                    (self.initial_scv_loc is None):
                if len(obs.observation["multi_select"]) < 2:
                    unit_type = obs.observation["screen"][_UNIT_TYPE]
                    scv_observed = (unit_type == _SCV)
                    processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.8).nonzero()
                    scv_mass_top_left = [processed_neutral_x.mean() - 5, processed_neutral_y.mean() - 5]
                    scv_mass_bottom_right = [processed_neutral_x.mean() + 5, processed_neutral_y.mean() + 5]
                    self.builder_scvs_selected = True
                    print('select some scvs 162')
                    return actions.FunctionCall(_SELECT_SCREEN, [_NOT_QUEUED, scv_mass_top_left, scv_mass_bottom_right])
                else:
                    selected_scvs = obs.observation["minimap"][_MM_SELECTED]
                    selected_scv_y, selected_scv_x = (selected_scvs == 1).nonzero()
                    selected_scv_x = selected_scv_x.mean()
                    selected_scv_y = selected_scv_y.mean()
                    print('initial scv location assigned 178')
                    self.initial_scv_loc = [int(selected_scv_x), int(selected_scv_y)]

            # move idle workers back to mining minerals
            if _SELECT_IDLE_WORKER in obs.observation["available_actions"]:
                if not self.idle_worker_selected:
                    self.idle_worker_selected = True
                    print('select idle worker 198')
                    return actions.FunctionCall(_SELECT_IDLE_WORKER, [_NOT_QUEUED])
                if obs.observation["minimap"][_MM_CAMERA][self.initial_scv_loc[1], self.initial_scv_loc[0]] != 1:
                    print('move camera to scv line 159')
                    return actions.FunctionCall(_MOVE_CAMERA, [self.initial_scv_loc])
                if (self.idle_worker_selected) & \
                        (obs.observation["minimap"][_MM_CAMERA][self.initial_scv_loc[1], self.initial_scv_loc[0]] == 1):

                    self.idle_worker_selected = False
                    unit_type = obs.observation["screen"][_UNIT_TYPE]
                    minerals_observed = (unit_type == _MINERAL_FIELD)
                    processed_neutral_y, processed_neutral_x = (window_avg(minerals_observed, 2) > 0.9).nonzero()
                    random_mineral_selection = np.random.choice(range(len(processed_neutral_x)))
                    random_minerals = [int(processed_neutral_x[random_mineral_selection]), int(processed_neutral_y[random_mineral_selection])]
                    print('smart screen 215')
                    return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, random_minerals])

            # train SCVs
            if (obs.observation["player"][3] < obs.observation["player"][4]) & (obs.observation["player"][1] >= 50) & (
                        obs.observation["player"][6] < 22) & (_TRAIN_SCV in obs.observation["available_actions"]):
                if (len(obs.observation["build_queue"]) == 1) & (
                                (obs.observation["player"][6]) + len(obs.observation["build_queue"]) < 22):
                    if (obs.observation["build_queue"][0][6] >= 90):
                        print('train scv 422')
                        return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])
                elif (len(obs.observation["build_queue"]) == 0) & (
                            _TRAIN_SCV in obs.observation["available_actions"]) & (
                                (obs.observation["player"][6]) + len(obs.observation["build_queue"]) < 22):
                    print('train scv 425')
                    return actions.FunctionCall(_TRAIN_SCV, [_NOT_QUEUED])

            # build supply depots
            if ((obs.observation["player"][4] - obs.observation["player"][3]) <= 3) & \
                    (self.builder_scvs_selected is True) & (obs.observation["player"][1] >= 100) & \
                    ((self.steps >= self.freeze_sd_building + 150) | (len(self.building_locations) == 0)):

                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                background = (player_relative == 0)
                ####testing blocking out known placements
                for building in self.building_locations:
                    background[
                    np.clip(building[1] - 5, 0, len(background)):np.clip(building[1] + 5, 0, len(background)),
                    np.clip(building[0] - 5, 0, len(background)):np.clip(building[0] + 5, 0, len(background))] = 0
                freespace = None
                if len(self.supply_depot_locations) == 0:
                    #### random placement
                    processed_freespace_y, processed_freespace_x = (window_avg(background, 7) > 0.99).nonzero()
                    random_placement = np.random.choice(range(len(processed_freespace_x)))
                    freespace = [int(processed_freespace_x[random_placement]), int(processed_freespace_y[random_placement])]
                else:
                    #### nearby placement
                    processed_freespace_y, processed_freespace_x = (window_avg(background, 7) > 0.99).nonzero()
                    zipped_freespace = zip(processed_freespace_x, processed_freespace_y)
                    freespace_locations = []
                    for location in zipped_freespace:
                        freespace_locations.append(location)
                    closest, min_dist = None, None
                    for p in freespace_locations:
                        dist = np.linalg.norm(np.array(self.barracks_locations[-1]) - np.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    freespace = closest
                self.freeze_sd_building = self.steps
                self.supply_depot_locations.append(freespace)
                self.building_locations.append(freespace)
                print('build supply depot 544')
                return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [[0], freespace])

            # build refineries
            if ((obs.observation["player"][3]) >= 24) & (obs.observation["player"][1] >= 75) & \
                    (self.builder_scvs_selected is True) & (self.refineries_built < 2):
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
            elif ((obs.observation["player"][3]) >= 18) & (obs.observation["player"][1] >= 75) & \
                (self.builder_scvs_selected is True) & (self.refineries_built < 1):
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

            # put workers on refineries
            if (self.refineries_built == 1) & (self.workers_on_ref_one < 2) & (
                        _MOVE_SCREEN in obs.observation["available_actions"]) & (
                        len(obs.observation['single_select']) == 1) & (
                        self.steps >= self.freeze_ref_worker_assignment + 75) & (
                        self.steps >= self.assigned_worker_to_gas + 15):
                if obs.observation['single_select'][0][0] == 45:
                    self.workers_on_ref_one += 1
                    self.assigned_worker_to_gas = self.steps
                    print('set scv to refinery 290')
                    return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.first_refinery_location])
            if (self.refineries_built == 1) & (self.workers_on_ref_one < 2) & (
                        _MOVE_SCREEN in obs.observation["available_actions"]) & (
                        self.steps >= self.freeze_ref_worker_assignment + 75):
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                scv_observed = (unit_type == _SCV)
                processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.99).nonzero()
                scv_mass = [processed_neutral_x.mean(), processed_neutral_y.mean()]
                print('select one scv 286')
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, scv_mass])
            if (self.refineries_built == 2) & (self.workers_on_ref_two < 2) & (
                        _MOVE_SCREEN in obs.observation["available_actions"]) & (
                        len(obs.observation['single_select']) == 1) & (
                        self.steps >= self.freeze_ref_worker_assignment + 75) & (
                        self.steps >= self.assigned_worker_to_gas + 15):
                if obs.observation['single_select'][0][0] == 45:
                    self.workers_on_ref_two += 1
                    self.assigned_worker_to_gas = self.steps
                    print('set scv to refinery 300')
                    return actions.FunctionCall(_SMART_SCREEN, [_NOT_QUEUED, self.second_refinery_location])
            if (self.refineries_built == 2) & (self.workers_on_ref_two < 2) & (
                        _MOVE_SCREEN in obs.observation["available_actions"]) & (
                        self.steps >= self.freeze_ref_worker_assignment + 75):
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                scv_observed = (unit_type == _SCV)
                processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.99).nonzero()
                scv_mass = [processed_neutral_x.mean(), processed_neutral_y.mean()]
                print('select one scv 296')
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, scv_mass])

            # build barracks
            if (self.barracks_built < self.barracks_to_build) & \
                    (obs.observation["player"][1] >= 150) & (obs.observation["player"][3] >= 13) & \
                    (_BUILD_BARRACKS in obs.observation["available_actions"]) & (self.builder_scvs_selected is True):

                player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
                background = (player_relative == 0)
                for building in self.building_locations:
                    background[
                    np.clip(building[1] - 7, 0, len(background)):np.clip(building[1] + 7, 0, len(background)),
                    np.clip(building[0] - 7, 0, len(background)):np.clip(building[0] + 7, 0, len(background))] = 0
                freespace = None
                if len(self.barracks_locations) == 0:
                    #### random placement
                    processed_freespace_y, processed_freespace_x = (window_avg(background, 7) > 0.99).nonzero()
                    random_placement = np.random.choice(range(len(processed_freespace_x)))
                    freespace = [int(processed_freespace_x[random_placement]), int(processed_freespace_y[random_placement])]
                else:
                    #### nearby same building placement
                    processed_freespace_y, processed_freespace_x = (window_avg(background, 7) > 0.99).nonzero()
                    zipped_freespace = zip(processed_freespace_x, processed_freespace_y)
                    freespace_locations = []
                    for location in zipped_freespace:
                        freespace_locations.append(location)
                    closest, min_dist = None, None
                    for p in freespace_locations:
                        dist = np.linalg.norm(np.array(self.barracks_locations[-1]) - np.array(p))
                        if not min_dist or dist < min_dist:
                            closest, min_dist = p, dist
                    freespace = closest
                self.building_locations.append(freespace)
                self.barracks_locations.append(freespace)
                self.freeze_barracks_selection = self.steps
                self.barracks_built += 1
                print('build barracks 472')
                return actions.FunctionCall(_BUILD_BARRACKS, [[0], freespace])

            # set barracks to control group
            if (obs.observation["control_groups"][4][1] < self.barracks_built) & (self.barracks_built == 1) & \
                (self.steps > self.freeze_barracks_selection + 50):
                # if (obs.observation["multi_select"].any() != _BARRACKS):
                    # unit_type = None
                # for unit in obs.observation["multi_select"]:
                if obs.observation["single_select"][0][0] != _BARRACKS:
                    print('select all barracks 621')
                    return actions.FunctionCall(_SELECT_POINT, [[2], self.barracks_locations[0]])
                else:
                    self.production_rally_set = False
                    print('set barracks to control group 427')
                    return actions.FunctionCall(_CONTROL_GROUP, [_SET_CONTROL_GROUP, [4]])

            if (obs.observation["control_groups"][4][1] < self.barracks_built) & (self.barracks_built > 1) & \
                    (self.steps > self.freeze_barracks_selection + 50):
                if len(obs.observation['multi_select']) == 0:
                    return actions.FunctionCall(_SELECT_POINT, [[2], self.barracks_locations[0]])
                else:
                    for unit in obs.observation["multi_select"]:
                        if unit[0] != _BARRACKS:
                            print('select all barracks 636')
                            return actions.FunctionCall(_SELECT_POINT, [[2], self.barracks_locations[0]])
                        else:
                            self.production_rally_set = False
                            print('set barracks to control group 427')
                            return actions.FunctionCall(_CONTROL_GROUP, [_SET_CONTROL_GROUP, [4]])

            # rally barracks to nearest ramp
            if (self.production_buildings_selected is True) & (self.production_rally_set is False):
                self.production_rally_set = True
                return actions.FunctionCall(_SMART_MINIMAP, [_NOT_QUEUED, [self.front_ramp[1], self.front_ramp[0]]])

            # train marines
            if (obs.observation["player"][3] < obs.observation["player"][4]) & (obs.observation["player"][1] >= 50) & (
                        _TRAIN_MARINE in obs.observation["available_actions"]):
                if (len(obs.observation["build_queue"]) == 1):
                    if (obs.observation["build_queue"][0][6] >= 90):
                        print('train marine 463')
                        return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
                elif (len(obs.observation["build_queue"]) == 0) & (
                            _TRAIN_MARINE in obs.observation["available_actions"]):
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
                else:
                    return actions.FunctionCall(_NO_OP, [])

            # if nothing to do then cycle through control groups, workers, and army
            cgs = 2
            if obs.observation["control_groups"][4][1] > 0:
                cgs = 3
            if obs.observation["player"][8] > 0:
                cgs = 4
            cycler = self.steps % cgs

            if cycler == 0:
                self.army_selected = False
                self.builder_scvs_selected = False
                self.production_buildings_selected = False
                return actions.FunctionCall(_CONTROL_GROUP, [_SELECT_CONTROL_GROUP, [5]])
            elif cycler == 1:
                if (obs.observation["minimap"][_MM_CAMERA][self.initial_scv_loc[1], self.initial_scv_loc[0]] == 1) & \
                        (self.initial_scv_loc is not None):
                    self.army_selected = False
                    self.production_buildings_selected = False
                    unit_type = obs.observation["screen"][_UNIT_TYPE]
                    scv_observed = (unit_type == _SCV)
                    processed_neutral_y, processed_neutral_x = (window_avg(scv_observed, 1) > 0.8).nonzero()
                    scv_mass_top_left = [processed_neutral_x.mean() - 5, processed_neutral_y.mean() - 5]
                    scv_mass_bottom_right = [processed_neutral_x.mean() + 5, processed_neutral_y.mean() + 5]
                    self.builder_scvs_selected = True
                    return actions.FunctionCall(_SELECT_SCREEN, [_NOT_QUEUED, scv_mass_top_left, scv_mass_bottom_right])
                else:
                    return actions.FunctionCall(_MOVE_CAMERA, [self.initial_scv_loc])
            elif cycler == 2:
                self.army_selected = False
                self.builder_scvs_selected = False
                self.production_buildings_selected = True
                return actions.FunctionCall(_CONTROL_GROUP, [_SELECT_CONTROL_GROUP, [4]])
            elif cycler == 3:
                self.builder_scvs_selected = False
                self.production_buildings_selected = False
                self.army_selected = True
                return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])
