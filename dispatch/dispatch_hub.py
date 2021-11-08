from data_preprocessing.init_dataset import init
from dispatch.offline import offline
from dispatch.online import online


def dispatch(script_params, config):

    dataset = init(selected_cue_set=config.id)

    if script_params.offline_mode:
        offline(script_params, config, dataset)
    elif script_params.online_mode:
        online(script_params, config, dataset)
