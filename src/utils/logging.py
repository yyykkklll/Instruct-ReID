from __future__ import absolute_import
import os
import sys
from pathlib import Path

from .osutils import mkdir_if_missing


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            # 如果 fpath 是相对路径，相对于项目根目录
            ROOT_DIR = Path(__file__).parent.parent.parent  # 从 src/utils/ 到 TextGuidedReID/
            fpath = os.path.join(ROOT_DIR, fpath) if not os.path.isabs(fpath) else fpath
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()