import logging
class Logger:
    def __init__(self, nombre, nivel=logging.DEBUG):
        self.logger = logging.getLogger(nombre)
        self.logger.setLevel(nivel)
        formato = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.consola = logging.StreamHandler()
        self.consola.setFormatter(formato)
        self.logger.addHandler(self.consola)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)