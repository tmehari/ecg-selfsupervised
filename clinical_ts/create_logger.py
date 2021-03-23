import logging 


def create_logger(name):
    ch = logging.StreamHandler()
    BEIGE, VIOLET, OKBLUE, ANTHRAZIT, ENDC = [
        '\033[32m', '\033[35m', '\033[94m', '\033[90m', '\033[0m']

    ch.setFormatter(logging.Formatter(BEIGE+'%(asctime)s '+VIOLET+'%(name)s:' +
                                      OKBLUE+'%(lineno)s '+ANTHRAZIT+'%(levelname)s: '+ENDC+' %(message)s'))
    logger = logging.getLogger(name)
    logger.addHandler(ch)
    return logger
