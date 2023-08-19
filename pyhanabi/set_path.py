import os
import sys


def append_sys_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # rela = os.path.join(root, "build", "rela")
    # print(rela)
    # assert os.path.exists(rela), rela
    # if rela not in sys.path:
    #     sys.path.append(rela)

    build = os.path.join(root, "build")
    assert os.path.exists(build)
    if build not in sys.path:
        sys.path.append(build)


append_sys_path()
