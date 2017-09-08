from platform import system
from os import getlogin
import json

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
    print('Dropbox location not found. Saving replays to default location.')
