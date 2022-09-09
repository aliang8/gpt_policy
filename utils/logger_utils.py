import os
import copy
import json
import logging
from pygments import highlight
from pygments.style import Style
from pygments.token import Token
from pygments.lexers import JsonLexer, Python3Lexer
from pygments.formatters import TerminalFormatter, Terminal256Formatter

from termcolor import colored
from pprint import pprint
from omegaconf import DictConfig, OmegaConf


def print_cfg(cfg):
    print(colored("config:"))
    cfg_str = json.dumps(OmegaConf.to_object(cfg), indent=4, sort_keys=True)
    print(highlight(cfg_str, JsonLexer(), TerminalFormatter()))


class Colors:
    grey = "\x1b[0;37m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[1;31m"
    purple = "\x1b[1;35m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    reset = "\x1b[0m"
    blink_red = "\x1b[5m\x1b[1;31m"


class CustomFormatter(logging.Formatter):
    def __init__(self, auto_colorized=True, custom_format: str = None):
        super(CustomFormatter, self).__init__()
        self.auto_colorized = auto_colorized
        self.custom_format = custom_format
        self.FORMATS = self.define_format()
        if auto_colorized and custom_format:
            print(
                "WARNING: Ignoring auto_colorized argument because you provided a custom_format"
            )

    def define_format(self):

        if self.auto_colorized:

            format_prefix = (
                f"{Colors.purple}%(asctime)s{Colors.reset} "
                f"{Colors.blue}%(name)s{Colors.reset} "
                f"{Colors.light_blue}(%(filename)s:%(lineno)d){Colors.reset} "
            )

            format_suffix = "%(levelname)s - %(message)s"

            return {
                logging.DEBUG: format_prefix
                + Colors.green
                + format_suffix
                + Colors.reset,
                logging.INFO: format_prefix
                + Colors.grey
                + format_suffix
                + Colors.reset,
                logging.WARNING: format_prefix
                + Colors.yellow
                + format_suffix
                + Colors.reset,
                logging.ERROR: format_prefix
                + Colors.red
                + format_suffix
                + Colors.reset,
                logging.CRITICAL: format_prefix
                + Colors.blink_red
                + format_suffix
                + Colors.reset,
            }

        else:
            if self.custom_format:
                _format = self.custom_format
            else:
                _format = f"%(asctime)s %(name)s (%(filename)s:%(lineno)d) %(levelname)s - %(message)s"
            return {
                logging.DEBUG: _format,
                logging.INFO: _format,
                logging.WARNING: _format,
                logging.ERROR: _format,
                logging.CRITICAL: _format,
            }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 0
        logger.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        ch.setFormatter(CustomFormatter())

        logger.addHandler(ch)
    return logger


def save_log(dout, progress, total, stage, **kwargs):
    """
    logging a method json for besteffort mode and jobs monitoring on Alex's machine
    """
    info_path = os.path.join(dout, "info.json")
    info_dicts = []
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info_dicts = json.load(f)
    info_dict = {"stage": stage, "progress": progress, "total": total}
    info_dict.update(kwargs)
    info_dicts.append(info_dict)
    with open(info_path, "w") as f:
        json.dump(info_dicts, f)


def update_log(dout, stage, update, **kwargs):
    """
    updating a method json for monitoring on Alex's machine
    """
    assert update in ("increase", "rewrite")
    info_path = os.path.join(dout, "info.json")
    assert os.path.exists(info_path)
    with open(info_path) as f:
        info_dicts = json.load(f)
    info_dict = copy.deepcopy([el for el in info_dicts if el["stage"] == stage][-1])
    # update the values
    for key, value in kwargs.items():
        assert key in info_dict
        new_value = value + info_dict[key] if update == "increase" else value
        info_dict[key] = new_value
    # decide what to do with the list with updated values
    if info_dicts[-1]["stage"] == stage:
        # rewrite the values
        info_dicts[-1] = info_dict
    else:
        # append a new list element
        info_dicts.append(info_dict)
    # dump to the disk
    with open(info_path, "w") as f:
        json.dump(info_dicts, f)
