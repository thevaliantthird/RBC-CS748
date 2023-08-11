import traceback
from datetime import datetime
from reconchess import history
import chess
import argparse
from reconchess import LocalGame, play_local_game
from reconchess.bots.trout_bot import TroutBot
import json
from Fianchetto_fast_tournament_2022 import Fianchetto_fastBot_buffer as Fianchetto_fastBot_buffer
from Fianchetto_fast_tournament_2022 import Strangefish_buffer as Strangefish_buffer
from Fianchetto_fast_tournament_2022.strategies import multiprocessing_strategies as ms
from strangefish.strategies import multiprocessing_strategies



#h=history.GameHistory.from_file('/home/puranjay/Fianchetto_IIT_Bombay/Fianchetto_fast/579052.json')


def main(move_white_list,move_black_list,sense_white_list,sense_black_list):
    white_bot_name, black_bot_name = 'Fianchetto_fast', 'StrangeFish'

    game = LocalGame()
    # game = LocalGame(seconds_per_player=3600)
    # game = LocalGame(seconds_per_player=18000)
    with open('sample_moves.json','w')as outfile:
            outfile.truncate()#json.dump({}, outfile)

    try:
        print(*ms.create_strategy())        
        x=Fianchetto_fastBot_buffer(move_white_list,sense_white_list,*ms.create_strategy())
        y=Strangefish_buffer(move_black_list,sense_black_list)
        
        winner_color, win_reason, history = play_local_game(
            
            x,
            y,
            game=game
        )
        print(x.allfen)
        winner = 'Draw' if winner_color is None else chess.COLOR_NAMES[winner_color]
    except:
        traceback.print_exc()
        game.end()

        winner = 'ERROR'
        history = game.get_game_history()

    print('Game Over!')
    print('Winner: {}!'.format(winner))

    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    replay_path = '{}-{}-{}-{}-{}.json'.format(white_bot_name, black_bot_name, winner, win_reason, timestamp)
    with open("sample_fen.json", "w") as outfile:
        json.dump(x.allfen, outfile)
    # replay_path = '{}-{}-{}-{}.json'.format(white_bot_name, black_bot_name, winner, timestamp)

    print('Saving replay to {}...'.format(replay_path))
    history.save(replay_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Allows you to watch a saved match.')
    parser.add_argument('history_path', help='Path to saved Game History file.')
    args = parser.parse_args()


    h = history.GameHistory.from_file(args.history_path)
    #h=history.GameHistory.from_file('/home/puranjay/Fianchetto_IIT_Bombay/Fianchetto_fast/579052.json')
    move_black_list=[];sense_black_list=[]
    move_white_list=[];sense_white_list=[]
    for turn in list(h.turns(False)):
        if h.has_move(turn):
            move_black_list.append(h.taken_move(turn))
            #board=h.truth_board_before_move(turn)
        if h.has_sense(turn):
            sense_black_list.append(h.sense(turn))
            # print(turn,h.sense(turn),h.has_move(turn),h.has_sense(turn))

    for turn in list(h.turns(True)):
        print(h.requested_move(turn))
        if h.has_move(turn):
            move_white_list.append(h.taken_move(turn))
            #board=h.truth_board_before_move(turn)
        if h.has_sense(turn):
            sense_white_list.append(h.sense(turn))
    
    main(move_white_list,move_black_list,sense_white_list,sense_black_list)


"""
export STOCKFISH_EXECUTABLE=/home/puranjay/Fianchetto_IIT_Bombay/strangefish-master/stockfish_14_x64
export PYTHONPATH=/home/puranjay/Fianchetto_IIT_Bombay/Fianchetto_fast/:$PYTHONPATH
export PYTHONPATH=/home/puranjay/Fianchetto_IIT_Bombay/strangefish-master/:$PYTHONPATH
python3 scripts/buffer.py
"""