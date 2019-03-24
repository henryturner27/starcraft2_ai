from platform import system
from os import getlogin, environ, makedirs
import json


def get_replay_dir():
    replay_dir = None

    if system() == 'Windows':
        dropbox_json_file = open('C:\\Users\\' + getlogin() + '\\AppData\\Local\\Dropbox\\info.json')
        dropbox_info = dropbox_json_file.read()
        dropbox_path = json.loads(dropbox_info)['personal']['path']
        replay_dir = dropbox_path + '\\jupyter_projects\\starcraft2_ai\\replays\\'
        dropbox_json_file.close()
    elif system() == 'Darwin':
        dropbox_json_file = open('/Users/' + getlogin() + '/.dropbox/info.json')
        dropbox_info = dropbox_json_file.read()
        dropbox_path = json.loads(dropbox_info)['personal']['path']
        replay_dir = dropbox_path + '/jupyter_projects/starcraft2_ai/replays/'
        dropbox_json_file.close()
    else:
    	print('Not using Windows or Mac OS')
    	makedirs(str(environ['HOME'] + '/replay_data'), exist_ok=True)
    	replay_dir = str(environ['HOME'] + '/replay_data')
    return replay_dir
