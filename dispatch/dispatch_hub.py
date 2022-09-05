import hypotheses.hypothesis_one as hypothesis_one
import hypotheses.hypothesis_two as hypothesis_two
import hypotheses.hypothesis_three as hypothesis_three
import hypotheses.hypothesis_four as hypothesis_four
import hypotheses.hypothesis_five as hypothesis_five
import hypotheses.hypothesis_six as hypothesis_six
import modules.session_analysis as session_analysis
from utility.logger import get_logger
import numpy as np


def dispatch(config):
    np.random.seed(159)

    if config.module_choice == 1:
        hypothesis_one.run(config)
    elif config.module_choice == 2:
        hypothesis_two.run(config)
    elif config.module_choice == 3:
        for sub in range(10):
            hypothesis_three.run(config, sub)
    elif config.module_choice == 4:
        hypothesis_four.run(config)
    elif config.module_choice == 5:
        hypothesis_five.run(config)
    elif config.module_choice == 6:
        hypothesis_six.run(config)
    elif config.module_choice == 7:
        session_analysis.run(config)
    else:
        get_logger().info('No valid module selected - exiting')
        exit()
