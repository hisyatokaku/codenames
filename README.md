# Codenames

Code for playing Codenames agent.

## Requirements
Code is written in Python3 (anaconda3-4.3.1) and requires gensim (3.1.0)


Using the pre-trained word2vec vectors will also require downloading the binary file from https://code.google.com/p/word2vec/

## Running selfplay
```
python main.py --setting selfplay --exname experiment_name
```
--setting: setting name. You can define the parameters and paths in Config/setting.exp
--exname: experiment name. This character is concatenated to the name of directory for log.
--n: word2vec noise. noise is sampled from gaussian distribution with mean=0.0, std=noise.

## Parameters in setting.exp
Required:
- cards: (str) cards which is played on field. filename(str) in cards/ 
- embed: (str) pretrained embedding in models/ 
- spywtable: (str) word table path. word table is used for effective computation. (can be extra but required for now)
- spyrwords: (str) restricted vocabulary for spymaster. if empty, the vocabulary with size 300,000 which was used in training word2vec will be used. (not recommended)
- enable_wv_noise: (int) 0 or 1. you can execute noised selfplay by giving 1 to this entry.

Extra:
- test: deprecated. do not have to specify.
- wv_noise_path: (str) store noise vector as pkl file when you execute noised selfplay.
- spywrlist: will be deprecated. stores clue score calculated by spymaster for each turn. 


## output files
normally:

```
logs/%m%d%Y_%H%M_experiment_name/:
  red_spy_player.log
  blue_spy_player.log
  exp_setting.log
  field.log

models/
  word_table.pkl ... similarity table(size:[|V|, 25]) between vocabulary words and field cards. Once you create this, you can reuse it and save computation time.
```

if you specify noised selfplay, the following will be added to above.:
```
logs/%m%d%Y_%H%M_experiment_name/:
  red_metrics.json ... calculated metrics value for red teams.
  blue_metrics.json ... same as the above for blue teams.
```

