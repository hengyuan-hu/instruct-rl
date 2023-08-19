# Language Instructed Reinforcement Learning for Human-AI Coordination

This is the code for [Language Instructed Reinforcement Learning for Human-AI Coordination](https://arxiv.org/abs/2304.07297) (ICML 2023).

The code has been tested with PyTorch 2.0.1

## Get Started

Clone the repo with `--recursive` to include submodules
```bash
git clone --recursive git@github.com:hengyuan-hu/instruct-rl.git
```

Dependencies
```bash
pip install tdqm scipy matplotlib 'transformers[torch]'
pip install openai
```

## The Say-Select Experiments

```bash
cd say-select

# train instruct-rl policies using default hyper-parameters
python train.py

# train vanilla rl policies
python train.py --lmd 0
```

## The Hanabi Experiments

### Prepare

First build the C++ part of the repo if you want to train/evaluate models
```bash
# under the root folder of the repo, compile
make

# Run this line before running any training code to prevent tensor operations
# from using single thread as our code uses multi-threading internally to run
# large number of environments in parallel
# Add it to your bashrc for convenience
export OMP_NUM_THREADS=1
```

Download pretrained OBL models and fully trained models used in ICML paper.
```bash
# under instruct-rl root directory
pip install gdown
gdown https://drive.google.com/uc\?id\=1Kko7L9zdS6ywCUs6VIaeCY32kGU1RZrz
unzip models.zip
```

Or directly download from Google Drive: `https://drive.google.com/file/d/1Kko7L9zdS6ywCUs6VIaeCY32kGU1RZrz/view?usp=drive_link`

### Run the code

Then go to the `pyhanabi` folder to run the code.
```bash
cd pyhanabi
```

Generate the language observations and language descriptions of the possible actions.
```bash
python gen_all_langs.py
```

The `openai_api.py` file contains code to evaluate prompts using `openai-api`. Please check
that file for detailed instructions. That file is deisgned to run interactively using VSCode's
[Python interactive window](https://code.visualstudio.com/docs/python/jupyter-support-py).
Pre-generated prior policies used in the paper are stored in `pyhanabi/openai`.

To train the model, run
```bash
export OMP_NUM_THREADS=1  # if you have not put this into bashrc
# ppo, the config uses color-instruction
python ppo_main.py --config configs/ppo.yaml

# iql,  the config uses color-instruction
python r2d2_main.py --config configs/iql.yaml
```

### Additional resources

To evaluate a trained model
```bash
# inside pyhanabi folder
python tools/eval_model.py --weight1 ../models/icml/iql_rank/iql1_pkr_load_pikl_lambda0.15_seeda_num_epoch50/model0.pthw
```

To examine the condtional action matrix
```bash
python tools/action_matrix.py --model ../models/icml/iql_color/iql1_pkc_load_pikl_lambda0.15_seeda_num_epoch50/model0.pthw
```

To train a belief model
```bash
python train_belief.py --policy ../models/icml/iql_color/iql1_pkc_load_pikl_lambda0.15_seeda_num_epoch50/model0.pthw
```

To run sparta for the fast adaptation experiments in the appendix
```bash
python sparta.py
```

To host a bot online so that people can play with it.
```bash
cd live_bot
pip install websocket-client requests
python main.py --name Bot-Color --login_name Bot-Something --password agoodpassword
```
