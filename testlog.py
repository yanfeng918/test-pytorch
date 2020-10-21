import logging
FORMAT = "%(asctime)s %(thread)d %(message)s"
logging.basicConfig(level=logging.INFO,format=FORMAT)
#
#
#
# logging.warning("{}".format("adsf"))

log = logging.getLogger(__name__)
log.info('start data preparing...')