import hypotheses.hypothesis_one as hypothesis_one
import hypotheses.hypothesis_two as hypothesis_two
import hypotheses.hypothesis_three as hypothesis_three
import hypotheses.hypothesis_four as hypothesis_four
import hypotheses.hypothesis_five as hypothesis_five
import hypotheses.hypothesis_six as hypothesis_six
from utility.logger import get_logger


def dispatch(config):

    if config.hypothesis_choice == 1:
        hypothesis_one.run(config)
    elif config.hypothesis_choice == 2:
        hypothesis_two.run(config)
    elif config.hypothesis_choice == 3:
        hypothesis_three.run(config)
    elif config.hypothesis_choice == 4:
        hypothesis_four.run(config)
    elif config.hypothesis_choice == 5:
        hypothesis_five.run(config)
    elif config.hypothesis_choice == 6:
        hypothesis_six.run(config)
    else:
        get_logger().info('No valid hypothesis selected - exiting')
        exit()
