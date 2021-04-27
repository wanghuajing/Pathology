import logging

a=1
logging.basicConfig(level=logging.DEBUG,format='%(levelname)s: %(message)s')

logging.info(f'\na:{a}\nb:{a}')
