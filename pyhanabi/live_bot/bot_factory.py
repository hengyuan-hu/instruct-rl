# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from agent import PiklAgent

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_root = os.path.join(root, "models")
print(model_root)

BotFactory = {
    "Bot-Color": lambda: PiklAgent(
        os.path.join(
            model_root, "icml/iql_color/iql1_pkc_load_pikl_lambda0.15_seeda_num_epoch50/model0.pthw"
        )
    ),
    "Bot-Rank": lambda: PiklAgent(
        os.path.join(
            model_root, "icml/iql_rank/iql1_pkr_load_pikl_lambda0.15_seeda_num_epoch50/model0.pthw"
        )
    ),
}
