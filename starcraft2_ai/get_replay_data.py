from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import sys
import gflags as flags

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point

from s2clientprotocol import sc2api_pb2 as sc_pb

import starcraft2_ai.platform_settings as platform_settings

FLAGS = flags.FLAGS


class Config(object):
    """Holds the configuration options."""

    def __init__(self, replay_file, map_name):
        # Environment.
        FLAGS(sys.argv)
        self.max_steps = 2000
        self.replay_name = map_name + '/' + replay_file
        self.player_id = 1
        self.map_name = map_name
        self.screen_unit_type_id = features.SCREEN_FEATURES.unit_type.index
        self.screen_height_map_id = features.SCREEN_FEATURES.height_map.index
        self.screen_visibility_id = features.SCREEN_FEATURES.visibility_map.index
        self.screen_creep_id = features.SCREEN_FEATURES.creep.index
        self.screen_power_id = features.SCREEN_FEATURES.power.index
        self.screen_player_id = features.SCREEN_FEATURES.player_id.index
        self.screen_player_relative_id = features.SCREEN_FEATURES.player_relative.index
        self.screen_selected_id = features.SCREEN_FEATURES.selected.index
        self.screen_hit_points_id = features.SCREEN_FEATURES.unit_hit_points.index
        self.screen_energy_id = features.SCREEN_FEATURES.unit_energy.index
        self.screen_shields_id = features.SCREEN_FEATURES.unit_shields.index
        self.screen_unit_density_id = features.SCREEN_FEATURES.unit_density.index
        self.mm_height_map_id = features.MINIMAP_FEATURES.height_map.index
        self.mm_visibility_id = features.MINIMAP_FEATURES.visibility_map.index
        self.mm_creep_id = features.MINIMAP_FEATURES.creep.index
        self.mm_camera_id = features.MINIMAP_FEATURES.camera.index
        self.mm_player_id =features.MINIMAP_FEATURES.player_id.index
        self.mm_player_relative_id = features.MINIMAP_FEATURES.player_relative.index
        self.mm_selected_id = features.MINIMAP_FEATURES.selected.index
        self.screen_size_px = (84, 84)
        self.minimap_size_px = (64, 64)
        self.camera_width = 24
        self.random_seed = 42

        self.interface = sc_pb.InterfaceOptions(raw=True, score=True,
                                                feature_layer=sc_pb.SpatialCameraSetup(width=self.camera_width))
        resolution = point.Point(*self.screen_size_px)
        resolution.assign_to(self.interface.feature_layer.resolution)
        minimap_resolution = point.Point(*self.minimap_size_px)
        minimap_resolution.assign_to(self.interface.feature_layer.minimap_resolution)


def _layer_string(layer):
    return '\n'.join(' '.join(str(v) for v in row) for row in layer)


class GameController(object):
    """Wrapper class for interacting with the game in play/replay mode."""

    def __init__(self, config):
        """Constructs the game controller object.

        Args:
            config: Interface configuration options.
        """
        self._config = config
        self._sc2_proc = None
        self._controller = None
        self.replay_dir = platform_settings.replay_dir

        self._initialize()

    def _initialize(self):
        """Initialize play/replay connection."""
        run_config = run_configs.get()
        self._map_inst = maps.get(self._config.map_name)
        self._map_data = run_config.map_data(self._map_inst.path)
        run_config.replay_dir = self.replay_dir
        self._replay_data = run_config.replay_data(self._config.replay_name)

        self._sc2_proc = run_config.start()
        self._controller = self._sc2_proc.controller
        self.replay_info = self.controller.replay_info(self._replay_data)

    def start_replay(self, replay_data):
        start_replay = sc_pb.RequestStartReplay(
                replay_data=replay_data,
                map_data=self._map_data,
                options=self._config.interface,
                disable_fog=True,
                observed_player_id=self._config.player_id)
        self._controller.start_replay(start_replay)

    @property
    def controller(self):
        return self._controller

    def close(self):
        """Close the controller connection."""
        if self._controller:
            self._controller.quit()
            self._controller = None
        if self._sc2_proc:
            self._sc2_proc.close()
            self._sc2_proc = None

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class GetObsAndActs(object):

    def __init__(self, map_name, replay_file):

        self.map_name = map_name
        self.replay_file = replay_file


    def _get_replay_data(self, controller, config):
        """Runs a replay to get the replay data."""
        f = features.Features(controller.game_info())

        rl_data = []
        while True:
            o = controller.observe()
            obs = f.transform_obs(o.observation)

            # stop observations when game ends
            if o.player_result:
                break

            available_actions = f.available_actions(o.observation)
            exec_actions = []
            for ac in o.actions:
                try:
                    action = f.reverse_action(ac)
                except ValueError:
                    action = features.actions.FunctionCall(0, [])  # no_op
                exec_actions.append(action)

            step = o.observation.game_loop

            screen_unit_type = obs['screen'][config.screen_unit_type_id]
            screen_unit_type_frame = np.asarray(screen_unit_type)
            screen_height_map = obs['screen'][config.screen_height_map_id]
            screen_height_map_frame = np.asarray(screen_height_map)
            screen_visibility = obs['screen'][config.screen_visibility_id]
            screen_visibility_frame = np.asarray(screen_visibility)
            screen_creep = obs['screen'][config.screen_creep_id]
            creep_frame = np.asarray(screen_creep)
            screen_power = obs['screen'][config.screen_power_id]
            screen_power_frame = np.asarray(screen_power)
            screen_player_id = obs['screen'][config.screen_player_id]
            screen_player_id_frame = np.asarray(screen_player_id)
            screen_player_relative = obs['screen'][config.screen_player_relative_id]
            screen_player_relative_frame = np.asarray(screen_player_relative)
            screen_selected = obs['screen'][config.screen_selected_id]
            screen_selected_frame = np.asarray(screen_selected)
            screen_hit_points = obs['screen'][config.screen_hit_points_id]
            screen_hit_points_frame = np.asarray(screen_hit_points)
            screen_energy = obs['screen'][config.screen_energy_id]
            screen_energy_frame = np.asarray(screen_energy)
            screen_shields = obs['screen'][config.screen_shields_id]
            screen_shields_frame = np.asarray(screen_shields)
            screen_unit_density = obs['screen'][config.screen_unit_density_id]
            screen_unit_density_frame = np.asarray(screen_unit_density)

            mm_height_map = obs['minimap'][config.mm_height_map_id]
            mm_height_map_frame = np.asarray(mm_height_map)
            mm_visibility = obs['minimap'][config.mm_visibility_id]
            mm_visibility_frame = np.asarray(mm_visibility)
            mm_creep = obs['minimap'][config.mm_creep_id]
            mm_creep_frame = np.asarray(mm_creep)
            mm_camera = obs['minimap'][config.mm_camera_id]
            mm_camera_frame = np.asarray(mm_camera)
            mm_player_id = obs['minimap'][config.mm_player_id]
            mm_player_id_frame = np.asarray(mm_player_id)
            mm_player_relative = obs['minimap'][config.mm_player_relative_id]
            mm_player_relative_frame = np.asarray(mm_player_relative)
            mm_selected = obs['minimap'][config.mm_selected_id]
            mm_selected_frame = np.asarray(mm_selected)

            player_minerals = obs['player'][1]
            player_gas = obs['player'][2]
            player_food_used = obs['player'][3]
            player_food_cap = obs['player'][4]
            player_food_used_by_army = obs['player'][5]
            player_food_used_by_workers = obs['player'][6]
            player_idle_worker_count = obs['player'][7]
            player_army_count = obs['player'][8]
            player_warp_gate_count = obs['player'][9]
            player_larva_count = obs['player'][10]

            rl_data.append([step, available_actions, screen_unit_type_frame, screen_height_map_frame,
                            screen_visibility_frame, creep_frame, screen_power_frame, screen_player_id_frame,
                            screen_player_relative_frame, screen_selected_frame, screen_hit_points_frame,
                            screen_energy_frame, screen_shields_frame, screen_unit_density_frame,
                            mm_height_map_frame, mm_visibility_frame, mm_creep_frame, mm_camera_frame,
                            mm_player_id_frame, mm_player_relative_frame, mm_selected_frame,
                            player_minerals, player_gas, player_food_used, player_food_cap, player_food_used_by_army,
                            player_food_used_by_workers, player_idle_worker_count, player_army_count,
                            player_warp_gate_count, player_larva_count, exec_actions])

            controller.step()

        return rl_data

    def do_it(self):
        config = Config(map_name=self.map_name, replay_file=self.replay_file)

        with GameController(config) as game_controller:

            game_controller.start_replay(game_controller._replay_data)

            rl_data = self._get_replay_data(game_controller.controller, config)

            return rl_data


if __name__ == '__main__':
    GetObsAndActs(
        map_name='Simple128', replay_file='Simple128_2017-10-07-20-54-57.SC2Replay').do_it()
