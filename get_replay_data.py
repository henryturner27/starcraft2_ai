"""
Get unit_type observational data and the actions taken from a replay
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from future.builtins import range   # pylint: disable=redefined-builtin
# import six
# import sys

from pysc2 import maps
from pysc2 import run_configs
# from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import point
from pysc2.tests import utils

from pysc2.lib import basetest
from pysc2.lib import app
from s2clientprotocol import sc2api_pb2 as sc_pb
# from pysc2.tests import replay_obs_test


class Config(object):
    """Holds the configuration options."""

    def __init__(self):
        # Environment.
        self.player_id = 1
        self.map_name = 'CollectMineralsAndGas'
        self.unit_type_id = features.SCREEN_FEATURES.unit_type.index
        # self.num_observations = 3000
        self.screen_size_px = (32, 32)
        self.minimap_size_px = (32, 32)
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

        self._initialize()

    def _initialize(self):
        """Initialize play/replay connection."""
        run_config = run_configs.get()
        self._map_inst = maps.get(self._config.map_name)
        self._map_data = run_config.map_data(self._map_inst.path)
        run_config.replay_dir = '/Applications/Starcraft II/Replays/'
        self._replay_data = run_config.replay_data('CollectMineralsAndGas/CollectMineralsAndGas_2017-08-31-22-24-49.SC2Replay')

        self._sc2_proc = run_config.start()
        self._controller = self._sc2_proc.controller
        self.replay_info = self.controller.replay_info(self._replay_data)

    def start_replay(self, replay_data):
        start_replay = sc_pb.RequestStartReplay(
                replay_data=replay_data,
                map_data=self._map_data,
                options=self._config.interface,
                disable_fog=False,
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

    def _get_replay_data(self, controller, config):
        """Runs a replay to get the replay data."""
        f = features.Features(controller.game_info())

        observations = {}
        # for _ in range(config.num_observations):
        while True:
            o = controller.observe()
            obs = f.transform_obs(o.observation)

            # stop observations when game ends
            if o.player_result:
                break

            if o.actions:
                action = f.reverse_action(o.actions[0])
                print(action)
            else:
                action = 'no action'
                print(action)

            unit_type = obs['screen'][config.unit_type_id]
            observations[o.observation.game_loop] = unit_type

            # print(_layer_string(unit_type))

            controller.step()

        return observations

    def do_it(self):
        config = Config()

        with GameController(config) as game_controller:

            game_controller.start_replay(game_controller._replay_data)

            observations = self._get_replay_data(game_controller.controller, config)

            return observations


if __name__ == '__main__':
    GetObsAndActs().do_it()
