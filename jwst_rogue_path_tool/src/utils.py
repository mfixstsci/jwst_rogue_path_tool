import json
import os

__location__ = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_config():
    """Return a dictionary that holds the contents of the ``jwql``
    config file.

    Returns
    -------
    settings : dict
        A dictionary that holds the contents of the config file.
    """
    config_file_location = os.path.join(__location__, 'jwst_rogue_path_tool', 'config.json')

    # Make sure the file exists
    if not os.path.isfile(config_file_location):
        raise FileNotFoundError('The jwst_rogue_path_tool requires a configuration file '
                                'to be placed within the main directory. '
                                'This file is missing.')

    with open(config_file_location, 'r') as config_file_object:
        try:
            # Load it with JSON
            settings = json.load(config_file_object)
        except json.JSONDecodeError as e:
            # Raise a more helpful error if there is a formatting problem
            raise ValueError('Incorrectly formatted config.json file. '
                             'Please fix JSON formatting: {}'.format(e))

    return settings