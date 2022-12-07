import logging

# logging utils

def get_root_logger(fname=None, file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler(f"{fname}.txt", mode='w')
        handler.setFormatter(format)
        logger.addHandler(handler)

    logger.addHandler(logging.StreamHandler())
    
    return logger