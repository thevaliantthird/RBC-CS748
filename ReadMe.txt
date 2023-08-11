Copy paste the following files in place of the fianchetto_bot,Strangefish_bot,multiprocessing_strategies files and run the following commands after activating 
the virtualenv. 

python scripts/buffer.py {Path to the game you want to replay}
python scripts/buffer_replay.py {Path to game you want to replay}

For example Path to game: python scripts/buffer.py /home/puranjay/Fianchetto_IIT_Bombay/Fianchetto_fast/655504.json

The replay buffer will show the best move scores and top board probabilities for every position of the game.

-------------------------------------------------------------------------------------------------------------------------------------------------------------
Winrate_Fianchetto.ipynb was used to find the win percentage against the various bots(on the RBC server).
NewOpponentModel.ipynb was used to train the weights of the lc0 network 


