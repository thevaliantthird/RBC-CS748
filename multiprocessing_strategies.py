import json
# import multiprocessing as mp
import random
from collections import defaultdict
from dataclasses import dataclass
from time import sleep, time
from typing import List, Set, Callable, Tuple, DefaultDict
from functools import partial

# from scripts import r_vs_r

import chess.engine
import numpy as np
from reconchess import Square
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from Fianchetto_fast_tournament_2022.utilities import SEARCH_SPOTS, stockfish, simulate_move, simulate_sense, simulate_sense_prime, get_next_boards_and_capture_squares, \
    generate_rbc_moves, generate_moves_without_opponent_pieces as all_rbc_moves, force_promotion_to_queen, capture_square_of_move
from Fianchetto_fast_tournament_2022.utilities.rbc_move_score import calculate_score, ScoreConfig, calculate_score_rollout_d1, calculate_score_rollout_d2, calculate_score_rollout_end
from Fianchetto_fast_tournament_2022.utilities.player_logging import create_sub_logger
from reconchess.utilities import move_actions
from termcolor import colored
from Fianchetto_fast_tournament_2022.strategies.threat_evaluation import RBC_threat_evaluation

SCORE_ROUNDOFF = 1e-5
move_ind = 0
store = []
Qs = []
msi = []
esr = []
msr = []
moves_dict={}
# prev_time_per_board = [2/800]
# logger = mp.log_to_stderr()
# logger.setLevel(mp.SUBDEBUG)

BOARD_SET_LIMIT = 50_000
# board_set_explosion_count = 0

@dataclass
class DynamicFactorConfig:
    mean_score_factor: float = 0.2  # relative contribution of a move's average outcome on its compound score
    min_score_factor: float = 0.2  # relative contribution of a move's worst outcome on its compound score
    max_score_factor: float = 0.6  # relative contribution of a move's best outcome on its compound score


@dataclass
class MoveConfig:
    mean_score_factor: float = 0.7  # relative contribution of a move's average outcome on its compound score
    min_score_factor: float = 0.3  # relative contribution of a move's worst outcome on its compound score
    max_score_factor: float = 0.0  # relative contribution of a move's best outcome on its compound score
    threshold_score_factor: float = 0.05  # fraction below best compound score in which any move will be considered
    sense_by_move: bool = False  # Use bonus score to encourage board set reduction by attempted moves
    force_promotion_queen: bool = True  # change all pawn-promotion moves to choose queen, otherwise it's often a knight


@dataclass
class TimeConfig:
    turns_to_plan_for: int = 30  # fixed number of turns over which the remaining time will be divided
    min_time_for_turn: float = 1.0  # minimum time to allocate for a turn
    time_for_sense: float = 0.8  # fraction of turn spent in choose_sense
    time_for_move: float = 0.2  # fraction of turn spent in choose_move
    max_batch: int = 1024
    num_boards_per_sec: int = 800 #per process
    calc_time_per_board: float = (2/num_boards_per_sec)  # starting time estimate for move score calculation
    board_limit_for_belief: int = num_boards_per_sec*20
    increment: int = 5 # number of seconds added after each move

# Add a hash method for chess.Board objects so that they can be tested for uniqueness. For our purposes, a unique EPD
#  string is adequate; it contains piece positions, castling rights, and en-passant square, but not turn counters.
chess.Board.__hash__ = lambda self: hash(self.epd(en_passant='xfen'))

# Function for playing RandomBot vs RandomBot rollout games using multiprocessing
def rollout_func(board_fen, temp):
    return r_vs_r.main(board_fen)

# Create a cache key for the requested board and move (keyed based on the move that would result from that request)
def make_cache_key(board):
    if type(board) == str:
        return board
    return board.epd(en_passant='xfen')


# Each "worker" runs a StockFish engine and waits for requested scores
# def worker(request_queue, response_queue, score_config, num_threads):
#     logger = create_sub_logger('lc0_queue_worker')
#     b = stockfish.create_engine()
#     count = 0
#     while True:
#         if not request_queue.empty():
#             (board, moves) = request_queue.get()
#             print('got request')

#             # exit()
#             try:
#                 score = calculate_score(board_epd=board, b=b, moves=moves, score_config=score_config)
#                 # print('Q:',score[0])
#                 Qs.append(score[0])
#                 # print('type', type(score))
#                 # print(score)
#             except Exception as e:
#                 print('Got in Exception:', e)
#                 logger.error('lc0 engine died while analysing (%s).',
#                              board)
#                 # If the analysis crashes the engine, something went really wrong. This tends to happen when the
#                 #  board is not valid, but that is meant to be filtered in calculate_score. Just in case, do not
#                 #  re-query the engine, instead assign the move a conservative score (here: as though into check).
#                 response_queue.put({make_cache_key(board): score_config.into_check_score})
#                 print('put response')
#                 b = stockfish.create_engine()
#             else:
#                 response_queue.put({make_cache_key(board): score})
#                 print('put response')
#         else:
#             count += 1
#             if count == 1000:
#                 print('sleeping')
#                 count = 0
#             sleep(0.001)


def create_strategy(
        gpu_id: int = 0,
        op_name: str = "",
        dynamic_config: DynamicFactorConfig = DynamicFactorConfig(),
        move_config: MoveConfig = MoveConfig(),
        score_config: ScoreConfig = ScoreConfig(),
        time_config: TimeConfig = TimeConfig(),
        board_weight_90th_percentile: float = 1000,
        boards_per_centipawn: int = 800,
        num_workers: int = 1,
        num_threads: int = None,

        checkmate_sense_override: bool = True,
        while_we_wait_extension: bool = True,
        load_cache_data: bool = False,
        rc_disable_pbar: bool = False
)\
        -> Tuple[Callable[[DefaultDict[str, float], bool, List[Square], List[chess.Move], float, float], Square],
                # Callable[[DefaultDict[str, float], bool, List[Square], List[chess.Move], float], Square],
                 # Callable[[DefaultDict[str, float], bool, List[chess.Move], float], chess.Move],
                 Callable[[DefaultDict[str, float], bool, List[chess.Move], float, float], chess.Move],
                 Callable[[DefaultDict[str, float], bool], None],
                 Callable[[None], None]]:
    """
    Constructs callable functions corresponding to input configurations for parallelized decision-impact based sensing
    and compound score based moving decisions.

    Before sensing, all possible moves are scored on each board by Stockfish with a set of heuristics for evaluating
    board states unique to RBC, then each move is ranked based on a weighted-average score and on best- and worst-case
    scores among possible boards. Move scores are computed for sub-sets of boards corresponding to each possible sense
    result, and sensing choices are made to maximize the expected change in move scores before and after the sense
    result. Move scores are re-computed based on the observed sense result and the highest-scored move is made.
    Additionally, both sense and move strategies have small score incentives to reduce the set of possible boards. When
    time does not allow all possible boards to be evaluated, a random sample is taken.

    :param move_config: A dataclass of parameters which determine the move strategy's compound score
    :param score_config: A dataclass of parameters which determine the centi-pawn score assigned to a board's strength
    :param time_config: A dataclass of parameters which determine how time is allocated between turns

    :param board_weight_90th_percentile: The centi-pawn score associated with a 0.9 weight in the board set
    :param boards_per_centipawn: The scaling factor for combining decision-impact and set-reduction sensing

    :param num_workers: The number of StockFish engines to create for scoring moves
    :param num_threads: The number of threads for StockFish engine configuration (config skipped if None)

    :param checkmate_sense_override: A bool which toggles the corner-case sensing strategy for winning from checkmate
    :param while_we_wait_extension: A bool that toggles the scoring of boards that could be reached two turns ahead

    :param load_cache_data: A bool that tells whether to "warm up" the cache from a file of pre-calculated scores
    :param rc_disable_pbar: A bool which turns off tqdm progress bars if True

    :return: A tuple of callable functions (sense, move, ponder, exit)
    """
    # global get_next_boards_and_capture_squares

    logger = create_sub_logger('multiprocessing_strategies')
    logger.debug('Creating new instance of multiprocessing strategies.')

    # Initialize a list to store calculation time data for dynamic time management
    score_calc_times = []
    # prev_time_per_board = [time_config.calc_time_per_board]
    bkd = stockfish.create_engine(time_config.max_batch, gpu_id)
    

    # Estimate calculation time based on data stored so far this game (and a provided starting datum)
    def calc_time_per_eval() -> float:
        n0 = 1
        t0 = time_config.calc_time_per_board * n0
        # total_num = n0 + sum(n for n, t in score_calc_times)
        total_num = n0 + len(score_calc_times)
        total_time = t0 + sum(score_calc_times)
        return total_time / total_num

    # Determine how much of the remaining time should be spent on (the rest of) the current turn.
    def allocate_time(seconds_left: float, fraction_turn_passed: float = 0):
        turns_left = time_config.turns_to_plan_for - fraction_turn_passed  # account for previous parts of turn
        equal_time_split = seconds_left / turns_left
        return max(equal_time_split, time_config.min_time_for_turn)

    # Convert a board strength score into a probability for use in weighted averages (here using the logistic function)
    def weight_board_probability(score):
        return 1 / (1 + np.exp(-2 * np.log(3) / board_weight_90th_percentile * score))

    # If requested, pre-load the board/move score cache from a file
    if load_cache_data:
        logger.debug('Loading cached scores from file.')
        with open('strangefish/score_cache.json', 'r') as file:
            score_data = json.load(file)
        score_cache = score_data['cache']
        boards_in_cache = set(score_data['boards'])
    else:
        score_cache = dict()
        boards_in_cache = set()

    # Create the multiprocessing queues for communication with multiple StockFish engines
    # request_queue = mp.Queue()
    # response_queue = mp.Queue()

    # Memoized calculation of the score associated with one move on one board
    def memo_calc_score(board: chess.Board, moves: List[chess.Move]):
        key = make_cache_key(board)
        if key in score_cache:
            return score_cache[key]

        if not moves:
            moves = move_actions(board)
            if not chess.Move.null() in moves:
                moves.append(chess.Move.null())
        request_queue.put((key, moves))
        return None


    def memo_calc_set(board_set_tuples, fian_color, turn_num, pool):
        """
        Takes list of tuple of (board_epd, moves (can be None))
        and returns list of tuple of (q, s)
        """
        # global prev_time_per_board
        # t0=time()
        if type(board_set_tuples) != dict:
            board_set_tuples = {k:v for k, v in board_set_tuples}
        results, cached_asked_moves = calculate_score(bkd, board_set_tuples, score_cache, time_config, fian_color, modifier, turn_num, pool, score_config)
        # results, cached_asked_moves = calculate_score_rollout_d1(bkd, board_set_tuples, score_cache, time_config, fian_color, pool, score_config)
        # results, cached_asked_moves = calculate_score_rollout_d2(bkd, board_set_tuples, score_cache, time_config, fian_color, pool, score_config)
        # results, cached_asked_moves = calculate_score_rollout_end(bkd, board_set_tuples, score_cache, time_config, fian_color, pool, score_config)
        [boards_in_cache.add(board_epd) for board_epd in results.keys()]
        # cst=time()-t0
        # for board_epd, response in results.items():
        #     (q, s, m) = response
        #     asked_moves = cached_asked_moves.get(board_epd, None)
        #
        #     if asked_moves:
        #         # print(colored('Note: Asked moves ordered different', 'green'))
        #         # print(asked_moves)
        #         # print(m)
        #         mapping = {i:m.index(simulate_move(chess.Board(board_epd), move) or chess.Move.null()) for i, move in enumerate(asked_moves)}
        #         # p = [p[mapping[i]] for i in range(len(p))]
        #         s = [s[mapping[i]] for i in range(len(asked_moves))]
        #
        #         final_results[board_epd] = (q, s)
        #     else:
        #         final_results[board_epd] = (q, s)

        # t1 = time() - t0
        # prev_time_per_board += [t1 / len(board_set_tuples)]

        # logger.debug(f'cst={cst}, ft={ft} | on memo calc set\n')
        return results

    def memo_calc_set_move(board_set_tuples, fian_color, turn_num,  pool):
        
        if type(board_set_tuples) != dict:
            board_set_tuples = {k:v for k, v in board_set_tuples}
        results, cached_asked_moves = calculate_score(bkd, board_set_tuples, score_cache, time_config, fian_color, modifier, turn_num, pool, score_config)
        # results, cached_asked_moves = calculate_score_rollout_d1(bkd, board_set_tuples, score_cache, time_config, fian_color, pool, score_config)
        # results, cached_asked_moves = calculate_score_rollout_d2(bkd, board_set_tuples, score_cache, time_config, fian_color, pool, score_config)
        # results, cached_asked_moves = calculate_score_rollout_end(bkd, board_set_tuples, score_cache, time_config, fian_color, pool, score_config)
        [boards_in_cache.add(board_epd) for board_epd in results.keys()]
        return results


    # # Handler for requested scores. Filters for unique requests, then gets cached or calculated results.
    # def memo_calc_set(requests):
    #
    #     filtered_requests = {board: moves
    #                          for board, moves in requests}
    #
    #     start = time()
    #     print('memo set started')
    #
    #     results = {make_cache_key(board):
    #                memo_calc_score(board, moves)
    #                for board, moves in filtered_requests.items()}
    #
    #     print('results:')
    #     print(results)
    #
    #     num_new = sum(1 for score in results.values() if score is None)
    #
    #     while any(score is None for score in results.values()):
    #         response = response_queue.get()
    #         print('got final response')
    #         # print('response:', response)
    #         score_cache.update(response)
    #         results.update(response)
    #         print(results)
    #
    #     print('out of score none check while')
    #     duration = time() - start
    #     if num_new:
    #         score_calc_times.append((num_new, duration))
    #
    #     final_results = {}
    #     for board_epd, response in results.items():
    #         q, s, m = response
    #         asked_moves = filtered_requests[chess.Board(board_epd)]
    #         if asked_moves:
    #             if set(asked_moves) != set(m):
    #                 print(colored('Error: Asked moves different', 'red'))
    #                 exit()
    #             if asked_moves != m:
    #                 print(colored('Note: Asked moves ordered different', 'green'))
    #                 mapping = {i:m.index(move) for i, move in enumerate(asked_moves)}
    #                 # p = [p[mapping[i]] for i in range(len(p))]
    #                 s = [s[mapping[i]] for i in range(len(s))]
    #
    #                 final_results[board_epd] = (q, s)
    #             else:
    #                 final_results[board_epd] = response[:-1]
    #         else:
    #             final_results[board_epd] = response[:-1]
    #
    #     print('returning final_results from memo set')
    #     return final_results

    # Add a new board to the cache (evaluate the board's strength and relative score for every possible move).
    def cache_board(boards_epd, fian_color, turn_num):
        # board.turn = not board.turn
        memo_calc_set(boards_epd, fian_color, turn_num,  None)
        # board.turn = not board.turn
        # memo_calc_set([(board, move, -op_score) for move in generate_moves_without_opponent_pieces(board)])

    # Randomly sample from the board set, but also include all of the boards which are already in the cache.
    def cache_favored_random_sample(board_set: DefaultDict[str, float], sample_size):
        board_sample = defaultdict(float)
        uncached_boards = defaultdict(float)
        for board in board_set:
            if board in boards_in_cache:
                board_sample[board] = board_set[board]
            else:
                uncached_boards[board] = board_set[board]

        if len(uncached_boards) == 0:
            return board_sample

        probs = np.array(list(uncached_boards.values()))
        probs /= np.sum(probs)
        
        sample_keys = np.random.choice(list(uncached_boards.keys()), size=min(len(board_set) - len(board_sample), sample_size), replace=False, p=probs)

        for key in sample_keys:
            board_sample[key] = board_set[key]

        return board_sample

    # Randomly sample from the board set, but also include all of the boards which are already in the cache.
    def prob_random_sample(board_set: DefaultDict[str, float], sample_size):
        probs = np.array(list(board_set.values()))
        probs /= np.sum(probs)
        
        sample_keys = np.random.choice(list(board_set.keys()), size=min(len(board_set), sample_size), replace=False, p=probs)

        board_sample = {key: board_set[key] for key in sample_keys}
        return board_sample

    # Randomly choose one board from the board set, excluding boards which are already in the cache.
    def choose_uncached_board(board_set: DefaultDict[str, float]):
        uncached_boards = set(board_set.keys()) - boards_in_cache
        # return random.choice(tuple(uncached_boards)) if uncached_boards else None
        return uncached_boards if uncached_boards else None

    # Create and start the requested number of StockFish "worker" processes
    # workers = [mp.Process(target=worker, args=(request_queue, response_queue, score_config, num_threads)) for _ in range(num_workers)]
    # for process in workers:
    #     process.start()

    piece_diff = 5
    piece_value_threshold = 5
    rollout_games = 100
    move_threshold_factor = 30

    # add increment to sense step and move step in same ratio as time is split
    #add_increment = 0
    #print('Added increment? :', add_increment)

    # change this to change the subsampling parameter
    fast_divide_factor_dictionary = {'StrangeFish2': 1,
                                     'Oracle': 1,
                                     'JKU-CODA': 1,
                                     'DynamicEntropy': 1,
                                     'ROOKie': 1,
                                     'trout': 1,
                                     'attacker': 1,
                                     'random': 1,
                                     'Marmot': 1}

    if op_name in fast_divide_factor_dictionary.keys():
        value = fast_divide_factor_dictionary[op_name]
        # sample optimum value with 2/3rd probability, with 1/3rd probability choose value 10 or 100
        # fast_divide_factor = random.sample([value, value, value, value, 10, 100], 1)[0]
        fast_divide_factor = value 
        print("Opponent name is ", op_name)
    else:
        print("Opponent name not in list")
        fast_divide_factor = 1

    print("fast_divide_factor", fast_divide_factor)
    
    modifier = 0
    if op_name == 'Oracle' or op_name == 'trout' or op_name == 'JKU-CODA':
        modifier = 1
    elif op_name == 'StrangeFish2' or op_name == 'Kevin':
        modifier = 2
    elif op_name[:2] == "Ch":
        modifier = 3

    if op_name[:2] == "Ch":
        move_config.min_score_factor = 0.99
        move_config.mean_score_factor = 0.01 
    elif op_name == 'StrangeFish2':
        move_config.min_score_factor = 0.7
        move_config.mean_score_factor = 0.3 
    elif op_name == 'Kevin' or op_name == 'JKU-CODA':
        move_config.min_score_factor = 0.8
        move_config.mean_score_factor = 0.2
    elif op_name == 'ROOKie' or op_name == "uccchess" or op_name == 'Marmot':
        move_config.min_score_factor = 0.9
        move_config.mean_score_factor = 0.1
        
    # strong_opponent_list = ['StrangeFish2', 'Oracle', 'ROOKie']
    # strong_opponent =  (op_name in strong_opponent_list)
    
    # print('Strong Opponent? :', strong_opponent)

    fast_threshold_factor = 10
    
    # BOARD_SET_LIMIT = 50_000
    # board_set_explosion_count = 0

    BSR_sense_x1=5000
    BSR_sense_y1=1
    BSR_sense_x2=50000
    BSR_sense_y2=50

    BSR_move_x1=500
    BSR_move_y1=1
    # BSR_move_x2=BOARD_SET_LIMIT
    BSR_move_y2=100

    def white_position(fen):
        white_piece_count=0
        for char in fen:
            if(char=='P'):
               white_piece_count+=1 
            elif(char=='B'):
               white_piece_count+=3
            elif(char=='N'):
               white_piece_count+=3
            elif(char=='R'):
               white_piece_count+=5
            elif(char=='Q'):
               white_piece_count+=9
        return white_piece_count

    def black_position(fen):
        black_piece_count=0
        for char in fen:
            if(char=='p'):
               black_piece_count-=1 
            elif(char=='b'):
               black_piece_count-=3
            elif(char=='n'):
               black_piece_count-=3
            elif(char=='r'):
               black_piece_count-=5
            elif(char=='q'):
               black_piece_count-=9
        return black_piece_count

    def piece_score(fen):
        advantage=0
        for char in fen:
            if(char==' ' or char=='w'):
                return advantage
            elif(char=='p'):
               advantage-=1 
            elif(char=='b'):
               advantage-=3
            elif(char=='n'):
               advantage-=3
            elif(char=='r'):
               advantage-=5
            elif(char=='q'):
               advantage-=9
            elif(char=='P'):
               advantage+=1 
            elif(char=='B'):
               advantage+=3
            elif(char=='N'):
               advantage+=3
            elif(char=='R'):
               advantage+=5
            elif(char=='Q'):
               advantage+=9
        return advantage

    def BSR_factor_sense(board_set_size):
        if board_set_size<=BSR_sense_x1:
            return BSR_sense_y1
        elif board_set_size>=BSR_sense_x2:
            return BSR_sense_y2
        else:
            return BSR_sense_y1+((BSR_sense_y2-BSR_sense_y1)*((board_set_size-BSR_sense_x1)/(BSR_sense_x2-BSR_sense_x1)))

    # def BSR_factor_move(board_set_size):
      #  if board_set_size<=BSR_move_x1:
      #      return BSR_move_y1
      #  else:
      #      return BSR_move_y1+((BSR_move_y2-BSR_move_y1)*((board_set_size-BSR_move_x1)/(BSR_move_x2-BSR_move_x1)))


    def sense_strategy(board_set: DefaultDict[str, float], our_color: bool,
                       sense_actions: List[Square], moves: List[chess.Move],
                       seconds_left: float, pool, turn_num: float):
        """
        Choose a sense square to maximize the expected effect on move scores (to best inform the next move decision).

        This strategy randomly samples from the current board set, then weights the likelihood of each board being the
        true state by an estimate of the opponent's position's strength. All possible moves are scored on these boards,
        and the combinations of scores for each possible sense result (since each sense would validate/invalidate
        different boards) are calculated. The sense square is chosen to maximize the change in move scores from before
        to after the sense.

        Centipawn points are also added per board for an expected board set size reduction by a sense choice. If the
        board set size is large enough, this becomes the dominant decision-making influence.

        Finally, a corner case is added to pinpoint the opponent's king in cases where we are (nearly) sure that their
        king is in check on all possible boards.
        """
        # print(colored(f'board set size: {len(board_set)}', 'red'))
        t0 = time()
        global msr
        global move_ind
        global store
        global BOARD_SET_LIMIT
        global board_set_explosion_count
        # global prev_time_per_board
        move_ind += 1
        global msi, esr
        # print('movelist')
        # print(moves)
        # Don't sense if there is nothing to learn from it
        if len(board_set) == 1:
            return None

        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')
        print('Time Left Before Sensing:',seconds_left)
        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')
        # file = open("output_time.txt", "a")
        # file.write("Time Left Before Sensing: " + str(seconds_left) + "\n")
        # file.close()

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = allocate_time(seconds_left)

        # if add_increment:
           # time_for_turn += time_config.increment

        time_for_phase = time_for_turn * time_config.time_for_sense
        time_per_move = calc_time_per_eval()
        time_per_board = time_per_move #* len(moves)
        # time_per_board = prev_time_per_board[-1]
        sample_size = max(num_workers, int(time_for_phase / time_per_board))
        sample_size = max(1, int(sample_size / fast_divide_factor))
        # if sample_size > BOARD_SET_LIMIT:
          #  board_set_explosion_count += 1
          #  if board_set_explosion_count > 1:
          #      BOARD_SET_LIMIT = 0.1*BOARD_SET_LIMIT
          #      print(BOARD_SET_LIMIT)

        # sample_size = min(sample_size, BOARD_SET_LIMIT)

        # if(turn_num < move_threshold_factor):
        #     sample_size = min(sample_size, 2*fast_threshold_factor)
        # else:
        #     sample_size = min(sample_size, fast_threshold_factor)
        # sample_size = min(sample_size, fast_threshold_factor)

        logger.debug('In sense phase with %.2f seconds left. Allocating %.2f seconds for this turn and %.2f seconds '
                     'for this sense step. Estimating %.4f seconds per calc over %d moves is %.4f seconds per '
                     'board score so we have time for %d boards.',
                     seconds_left, time_for_turn, time_for_phase, time_per_move,
                     len(moves), time_per_board, sample_size)

        # Initialize some parameters for tracking information about possible sense results
        num_occurances = defaultdict(lambda: defaultdict(float))
        weighted_probability = defaultdict(lambda: defaultdict(float))
        total_weighted_probability = 0
        sense_results = defaultdict(lambda: defaultdict(set))
        sense_possibilities = defaultdict(lambda: defaultdict(set))
        king_locations = defaultdict(lambda: defaultdict(set))

        # Get a random sampling of boards from the board set
        # board_sample = cache_favored_random_sample(board_set, sample_size)
        board_sample = prob_random_sample(board_set, sample_size)
        # Initialize arrays for board and move data (dictionaries work here, too, but arrays were faster)
        board_sample_weights = np.zeros(len(board_sample))
        move_scores = np.zeros([len(moves), len(board_sample)])

        logger.debug('Sampled %d boards out of %d for sensing.', len(board_sample), len(board_set))
        
        # Get board position strengths before move for all boards in sample (to take advantage of parallel processing)
        bst = time()-t0
        board_score_reqs = {}
        for board_epd in board_sample:
            board = chess.Board(board_epd)
            if board.turn!=our_color:
                print(colored("ss - color problem","blue"))
            board.turn = our_color
            board_score_reqs[board.epd(en_passant='xfen')] = moves
        board_score_dict = memo_calc_set(board_score_reqs, our_color, turn_num, pool)
###################################################################################################################
        queen_squares = defaultdict(float)
        total_queen_squares = 0
        king_squares = defaultdict(float)
        total_king_squares = 0
###################################################################################################################
#------------------------------------------------------------------------------------------------------------------
        sense_advantage_array = []
        sense_average_advantage = 0.0
#------------------------------------------------------------------------------------------------------------------
        # board_sample_weights = softmax([-(board_score_dict[make_cache_key(board)][0]) for board in board_sample])
        bet = time()-t0
        my_score_sum = 0
        threat_score_arr = np.zeros(len(board_sample))
#------------------------------------------------------------------------------------------------------------------
        # if len(board_sample) <= 5000:
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print(f'-----Using Threat Avoidance Scores for sensing-----Number of boards = {len(board_sample)}-----')
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
        # else:    
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print(f'---Not Using Threat Avoidance Scores for sensing---Number of boards = {len(board_sample)}---')
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')

        for num_board, (board_epd, board_prob) in enumerate(tqdm(board_sample.items(), disable=rc_disable_pbar,
                                                   desc=f'{chess.COLOR_NAMES[our_color]} '
                                                        'Calculating choose_sense scores '
                                                        f'{len(moves)} moves in {len(board_set)} boards',
                                                   unit='boards')):
            board = chess.Board(board_epd)
            board.turn = our_color
            board_epd_p = make_cache_key(board)


#------------------------------------------------------------------------------------------------------------------
          #  if len(board_sample) <= 5000:
          #      threat_board = board.copy()
          #      threat_board.turn = not our_color

          #      threat_score_arr[num_board] = RBC_threat_evaluation(threat_board)
            
#------------------------------------------------------------------------------------------------------------------            
#------------------------------------------------------------------------------------------------------------------
            sense_advantage_array.append(piece_score(board.fen()))
#------------------------------------------------------------------------------------------------------------------
            opp_queen=board.pieces(chess.QUEEN, not our_color)
            for square in opp_queen:
                attacks_queen=board.attackers(our_color,square)
                if len(attacks_queen)>0:
                    queen_squares[square]+=1
                total_queen_squares+=1
            opp_king=board.pieces(chess.KING, not our_color)
            for square in opp_king:
                attacks_king=board.attackers(our_color,square)
                if len(attacks_king)>0:
                    king_squares[square]+=1
                total_king_squares+=1
###################################################################################################################
            # print('results:', board_score_dict[make_cache_key(board)])
            my_score, all_moves_score = board_score_dict[board_epd_p]
            my_score_sum += my_score
            # op_score = -my_score #assumption
            # board_sample_weights[num_board] = weight_board_probability(op_score)
            board_sample_weights[num_board] = board_prob
            total_weighted_probability += board_sample_weights[num_board]

            # board.turn = our_color
            # boards_in_cache.add(board_epd_p)  # Record that this board (and all moves) are in our cache

            # move_score_dict = get_move_prob(board, moves)  # Score all moves

            # Place move scores into array for later logical indexing
            move_scores[:, num_board] = all_moves_score

            # Gather information about sense results for each square on each board (and king locations)
            for square in SEARCH_SPOTS:
                sense_result = simulate_sense_prime(board, square)
                # sense_result = simulate_sense(board, square)
                num_occurances[square][sense_result] += 1
                weighted_probability[square][sense_result] += board_sample_weights[num_board]
                # sense_results[board_epd][square] = sense_result
                sense_possibilities[square][sense_result].add(num_board)
                king_locations[square][sense_result].add(board.king(not our_color))

#------------------------------------------------------------------------------------------------------------------
        sense_average_advantage = np.average(np.array(sense_advantage_array))
#------------------------------------------------------------------------------------------------------------------

        # Take a different strategy if we are sure they are in checkmate (the usual board weight math fails there)
        if checkmate_sense_override and abs(my_score_sum - len(board_sample)) < 1e-5:
            logger.debug("All scores indicate checkmate, therefore sensing based on king location.")
            num_king_squares = {square: np.mean([len(n) for n in king_locations[square].values()])
                                for square in SEARCH_SPOTS}
            min_num_king_squares = min(num_king_squares.values())
            sense_choice = random.choice([square for square, n in num_king_squares.items()
                                          if n == min_num_king_squares])
            return sense_choice

        # Calculate the mean, min, and max scores for each move across the board set (or at least the random sample)
        full_set_mean_scores = (np.average(move_scores, axis=1, weights=board_sample_weights))
        full_set_min_scores = (np.min(move_scores, axis=1))
        full_set_max_scores = (np.max(move_scores, axis=1))
        # full_set_threat_scores = (np.average(threat_score_arr, weights=board_sample_weights))

        # Find the expected change in move scores caused by any sense choice
        sense_impact = defaultdict(lambda: defaultdict(float))
        for square in tqdm(SEARCH_SPOTS, disable=rc_disable_pbar,
                           desc=f'{chess.COLOR_NAMES[our_color]} Evaluating sense impacts '
                                f'for {len(board_set)} boards', unit='squares'):
            possible_results = sense_possibilities[square]
            for sense_result in possible_results:
                if len(possible_results) > 1:
                    # subset_index = [i for i, board_epd in enumerate(board_sample)
                    #                 if sense_result == sense_results[board_epd][square]]
                    subset_index = list(possible_results[sense_result])
                    subset_move_scores = move_scores[:, subset_index]
                    subset_board_weights = board_sample_weights[subset_index]


                    # Calculate the mean, min, and max scores for each move across the board sub-set
                    sub_set_mean_scores = (np.average(subset_move_scores, axis=1, weights=subset_board_weights))
                    sub_set_min_scores = (np.min(subset_move_scores, axis=1))
                    sub_set_max_scores = (np.max(subset_move_scores, axis=1))

                    # Subtract the full set scores from the sub-set scores (and take the absolute value)
                    change_in_mean_scores = np.abs(sub_set_mean_scores - full_set_mean_scores)
                    change_in_min_scores = np.abs(sub_set_min_scores - full_set_min_scores)
                    change_in_max_scores = np.abs(sub_set_max_scores - full_set_max_scores)

                    # Combine the mean, min, and max changes in scores based on the config settings
                    change_in_compound_score = (
                        change_in_mean_scores * move_config.mean_score_factor +
                        change_in_min_scores * move_config.min_score_factor +
                        change_in_max_scores * move_config.max_score_factor
                    )
                    
                    
#------------------------------------------------------------------------------------------------------------------
    
           #         if len(board_sample) <= 5000:
           #             threat_scores = threat_score_arr[subset_index]
           #             sub_set_threat_scores = (np.average(threat_scores, weights=subset_board_weights))
           #             change_in_threat_score = np.abs(sub_set_threat_scores - full_set_threat_scores)
           #             change_in_compound_score = change_in_compound_score + change_in_threat_score

#------------------------------------------------------------------------------------------------------------------
            
                    # if(turn_num < move_threshold_factor):
                    #     change_in_compound_score = (
                    #         change_in_mean_scores * move_config.mean_score_factor +
                    #         change_in_min_scores * move_config.min_score_factor +
                    #         change_in_max_scores * move_config.max_score_factor
                    #     )
                    # else:
                    #     change_in_compound_score = (
                    #         change_in_mean_scores * dynamic_config.mean_score_factor +
                    #         change_in_min_scores * dynamic_config.min_score_factor +
                    #         change_in_max_scores * dynamic_config.max_score_factor
                    #     )
#------------------------------------------------------------------------------------------------------------------
                    # if((our_color and sense_average_advantage < (-1*piece_diff)) or ((not our_color) and sense_average_advantage > piece_diff)):
                    #     change_in_compound_score = (
                    #         change_in_mean_scores * dynamic_config.mean_score_factor +
                    #         change_in_min_scores * dynamic_config.min_score_factor +
                    #         change_in_max_scores * dynamic_config.max_score_factor
                    #     )
                    # else:
                    #     change_in_compound_score = (
                    #         change_in_mean_scores * move_config.mean_score_factor +
                    #         change_in_min_scores * move_config.min_score_factor +
                    #         change_in_max_scores * move_config.max_score_factor
                    #     )
#------------------------------------------------------------------------------------------------------------------


                    # The impact of this sense result is the resulting average change in absolute value of move scores
                    sense_impact[square][sense_result] = float(np.mean(change_in_compound_score))

                else:
                    sense_impact[square][sense_result] = 0

        # Calculate the expected mean change in centipawn score for each sense square
        mean_sense_impact = {
            square:
                sum([sense_impact[square][sense_result] * weighted_probability[square][sense_result]
                     for sense_result in sense_possibilities[square]]) / total_weighted_probability
            for square in SEARCH_SPOTS
        }
###################################################################################################################
        majority_count=0
        majority_square=-1
        for square, count in queen_squares.items():
            if(count>majority_count):
                majority_count=count
                majority_square=square
        if(majority_square>0):
            if(majority_count>=(total_queen_squares/2)):
                if (chess.square_file(majority_square)==0):
                    majority_square+=1
                if (chess.square_file(majority_square)==7):
                    majority_square-=1
                if (chess.square_rank(majority_square)==0):
                    majority_square+=8
                if (chess.square_rank(majority_square)==7):
                    majority_square-=8
                mean_sense_impact[majority_square]+=(abs((min(max(mean_sense_impact.values()),40)-(mean_sense_impact[majority_square])))*(6/10))
###################################################################################################################
###################################################################################################################
        majority_count=0
        majority_square=-1
        for square, count in king_squares.items():
            if(count>majority_count):
                majority_count=count
                majority_square=square
        if(majority_square>0):
            if(majority_count>=(total_king_squares/2)):
                if (chess.square_file(majority_square)==0):
                    majority_square+=1
                if (chess.square_file(majority_square)==7):
                    majority_square-=1
                if (chess.square_rank(majority_square)==0):
                    majority_square+=8
                if (chess.square_rank(majority_square)==7):
                    majority_square-=8
                mean_sense_impact[majority_square]+=(abs((min(max(mean_sense_impact.values()),40)-(mean_sense_impact[majority_square])))*(6/10))
###################################################################################################################
        # Also calculate the expected board set reduction for each sense square (scale from board sample to full set)
        expected_set_reduction = {
            square:
                len(board_set) *
                (1 - (1 / len(board_sample) / total_weighted_probability) *
                 sum([num_occurances[square][sense_result] * weighted_probability[square][sense_result]
                      for sense_result in sense_possibilities[square]]))
            for square in SEARCH_SPOTS
        }

        # Combine the decision-impact and set-reduction estimates
        for square in SEARCH_SPOTS:
            msi.append(mean_sense_impact[square])
            esr.append(expected_set_reduction[square])
        sense_score = {square:
                       mean_sense_impact[square] + (BSR_factor_sense(len(board_set)))*(expected_set_reduction[square] / boards_per_centipawn)
                       # mean_sense_impact[square] + 1.5*expected_set_reduction[square]
                       for square in SEARCH_SPOTS}

        max_sense_score = max(sense_score.values())
        sense_choice = random.choice([square for square, score in sense_score.items()
                                      if abs(score - max_sense_score) < SCORE_ROUNDOFF])

        print("sense_score")
        for square, score in sense_score.items():
            print(square,score)

        print(f'got sense choice {sense_choice}')
        ft = time()-t0
        # with open('/home/rbc/reconchess/Fianchetto_fast/timelog.txt', 'a') as f:
        #     f.write(f'bst={bst}, bet={bet}, ft={ft} | turn={move_ind} on sense\n')
        logger.debug(f'bst={bst}, bet={bet-bst}, ft={ft-bet} | turn={move_ind} on sense\n')
        return sense_choice
        # return 56

    def move_strategy(board_set: DefaultDict[str, float], our_color: bool,
                      moves: List[chess.Move],
                      seconds_left: float, pool, turn_num: float):
        """
        Choose the move with the maximum score calculated from a combination of mean, min, and max possibilities.

        This strategy randomly samples from the current board set, then weights the likelihood of each board being the
        true state by an estimate of the opponent's position's strength. Each move is scored on each board, and the
        resulting scores are assessed together by looking at the worst-case score, the average score, and the best-case
        score. The relative contributions of these components to the compound score are determined by a config object.
        If requested by the config, bonus points are awarded to moves based on the expected number of boards removed
        from the possible set by attempting that move. Deterministic move patterns are reduced by randomly choosing a
        move that is within a few percent of the maximum score.
        """
        # print(colored(f'board set size: {len(board_set)}', 'red'))
        # sleep(10)
        t0 = time()
        global msr
        global move_ind
        global store
        # global prev_time_per_board
        # global board_set_explosion_count
        # global BOARD_SET_LIMIT
        # global prev_time_per_board
        # move_ind += 1
        store.append({})

        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')
        print('Time Left Before Moving:',seconds_left)
        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------------')

        # file = open("output_time.txt", "a")
        # file.write("Time Left Before Moving: " + str(seconds_left) + "\n")
        # file.close()

        # Allocate remaining time and use that to determine the sample_size for this turn
        time_for_turn = allocate_time(seconds_left, fraction_turn_passed=1-time_config.time_for_move)
        
        # if add_increment:
           # time_for_turn += time_config.increment

        time_for_phase = time_for_turn * time_config.time_for_move
        time_per_move = calc_time_per_eval()
        time_per_board = time_per_move #* len(moves)
        # time_per_board = prev_time_per_board[-1]
        sample_size = max(1, int(time_for_phase / time_per_board))
        sample_size = max(1, int(sample_size / fast_divide_factor))
        # sample_size = min(sample_size, BOARD_SET_LIMIT)

        # if(turn_num < move_threshold_factor):
        #     sample_size = min(sample_size, 2*fast_threshold_factor)
        # else:
        #     sample_size = min(sample_size, fast_threshold_factor)
        # sample_size = min(sample_size, fast_threshold_factor)

        logger.debug('In move phase with %.2f seconds left. Allowing up to %.2f seconds for this move step. '
                     'Estimating %.4f seconds per calc over %d moves is %.4f seconds per '
                     'board score so we have time for %d boards.',
                     seconds_left, time_for_turn, time_per_move,
                     len(moves), time_per_board, sample_size)

        moves_against_attacker = [chess.Move.from_uci("b1c3"),chess.Move.from_uci("c3e4"),chess.Move.from_uci("e4d6"),chess.Move.from_uci("d6e8")]
        moves_against_strangefish_w = [chess.Move.from_uci("g1f3"), chess.Move.from_uci("b2b3"), chess.Move.from_uci("d2d3")]
        moves_against_strangefish_b = [chess.Move.from_uci("g7g6"), chess.Move.from_uci("d7d6"), chess.Move.from_uci("b7b6")]

        if (op_name == 'StrangeFish2' or op_name[:2] == "Ch" or op_name == "Kevin") and turn_num == 1:  
            index = random.sample([0,1,2], 1)[0]
            if our_color:
                return moves_against_strangefish_w[index]
            else:
                return moves_against_strangefish_b[index]

        # if op_name == 'attacker' and our_color == True:
           # return moves_against_attacker[turn_num-1]
        
        if op_name == 'ROOKie' and turn_num == 1:
            index = random.sample([0,1,2], 1)[0]
            if not our_color:
                return moves_against_strangefish_b[index]


        move_scores = defaultdict(list)
        weighted_sum_move_scores = defaultdict(float)

        # Initialize some parameters for tracking information about possible move results
        num_occurances = defaultdict(lambda: defaultdict(int))
        weighted_probability = defaultdict(lambda: defaultdict(float))
        move_possibilities = defaultdict(set)
        total_weighted_probability = 0

        # Get a random sampling of boards from the board set
        board_sample = cache_favored_random_sample(board_set, sample_size)
        print(f'Sampled {len(board_sample)} boards out of {len(board_set)} for moving.')

        bst = time()-t0

        # Get board position strengths before move for all boards in sample (to take advantage of parallel processing)
        board_score_reqs = {}
        for board_epd in board_sample:
            board = chess.Board(board_epd)
            if board.turn!=our_color:
                print(colored("ms - color problem","blue"))
            board.turn = our_color
            # board_score_reqs.append((board.epd(en_passant='xfen'), moves))
            board_score_reqs[board.epd(en_passant='xfen')] = moves
            # oppboard = chess.Board(board_epd)
            # oppboard.turn = not our_color
            # board_score_reqs.append((oppboard, None))
        
        board_score_dict = memo_calc_set(board_score_reqs, our_color, turn_num, pool)
        # board_score_dict = memo_calc_set_move(board_score_reqs, our_color, pool)

        # if(turn_num<move_threshold_factor):
            # board_score_dict = memo_calc_set(board_score_reqs, our_color, pool)
        # else:
            # time_before = time()
            # board_score_dict = memo_calc_set_move(board_score_reqs, our_color, pool)            
            # time_after = time() - time_before
            # file = open("output_time.txt", "a")
            # file.write("Time taken : " + str(time_after) + "\n")
            # file.close()

        # def check_score_wrapper(board):
        #     board = chess.Board(board)
        #     board.turn = our_color
        #     if board.is_check():
        #         return 1
        #     else:
        #         return -(board_score_dict[make_cache_key(board)][0])

#------------------------------------------------------------------------------------------------------------------
        move_advantage_array = []
        piece_value_array = []
        move_average_advantage = 0.0
        average_piece_value = 0.0
#------------------------------------------------------------------------------------------------------------------

        # board_sample_weights = softmax([check_score_wrapper(board) for board in board_sample])
        bet = time()-t0
        move_scores = np.zeros([len(moves), len(board_sample)])
        board_weight_arr = np.zeros(len(board_sample))
        board_epd_arr = {}

        for num_board, (board_epd, board_weight) in enumerate(tqdm(board_sample.items(), disable=rc_disable_pbar,
                              desc=f'{chess.COLOR_NAMES[our_color]} Calculating choose_move scores '
                                   f'{len(moves)} moves in {len(board_set)} boards', unit='boards')):
            board = chess.Board(board_epd)

            board.turn = our_color
            my_score, all_moves_score = board_score_dict[make_cache_key(board)]

            
            # if chess.Move.from_uci("f3e5") in moves:
            #      print("move2",num_board,moves.index(chess.Move.from_uci("f3e5")),all_moves_score[moves.index(chess.Move.from_uci("f3e5"))],"\n")
            
            
            # op_score = -my_score
            # board_weight = weight_board_probability(op_score)
            # total_weighted_probability += board_weight
            
            # print(num_board,board)

            # op_score = -my_score
            # board_weight = weight_board_probability(op_score)
            # total_weighted_probability += board_weight
            board_weight_arr[num_board] = board_weight
            board_epd_arr[num_board] = board_epd

            # board.turn = our_color
            boards_in_cache.add(board.epd(en_passant='xfen'))  # Record that this board (and all moves) are in our cache

            if chess.Move.from_uci("e5e8") in moves:
                print("move1:",num_board,moves.index(chess.Move.from_uci("e5e8")),all_moves_score[moves.index(chess.Move.from_uci("e5e8"))])
                print(board,board_weight_arr[num_board])
                print(all_moves_score)
                if all_moves_score[moves.index(chess.Move.from_uci("e5e8"))]==-160:
                    print(board,board_weight_arr[num_board])
                    # sleep(5)
                    
#------------------------------------------------------------------------------------------------------------------
            move_advantage_array.append(piece_score(board.fen()))
            if(our_color):
                piece_value_array.append(white_position(board.fen()))
            else:
                piece_value_array.append(black_position(board.fen()))
#------------------------------------------------------------------------------------------------------------------

            # move_score_dict = get_move_prob(board, moves)  # Score all moves

            # Place move scores into array for later logical indexing
            # move_scores[:, num_board] = all_moves_prob

            # Gather scores and information about move results for each requested move on each board
            move_scores[:, num_board] = all_moves_score
            # for it, move in enumerate(moves):
            #     score = all_moves_score[it]
            #     move_scores[move].append(score)
            #     weighted_sum_move_scores[move] += score * board_weight

                # sim_move = simulate_move(board, move) or chess.Move.null()
                # move_result = (sim_move, board.is_capture(sim_move))
                # move_possibilities[move].add(move_result)
                # num_occurances[move][move_result] += 1
                # weighted_probability[move][move_result] += board_weight

        # Combine the mean, min, and max possible scores based on config settings
        # compound_score = {move: (
        #     weighted_sum_move_scores[move] / total_weighted_probability * move_config.mean_score_factor +
        #     min(scores) * move_config.min_score_factor +
        #     max(scores) * move_config.max_score_factor
        # ) for (move, scores) in move_scores.items()}
        # board_weight_arr /= np.sum(board_weight_arr)

        compound_score = move_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
                       + move_config.min_score_factor * np.min(move_scores, axis=-1) \
                       + move_config.max_score_factor * np.max(move_scores, axis=-1)


#------------------------------------------------------------------------------------------------------------------
        move_average_advantage = np.average(np.array(move_advantage_array))
        average_piece_value = np.average(np.array(piece_value_array))

        for x in moves:#chess.Move.from_uci("a1a5") in moves:
            y=str(x)
            if y=='d5d2':
                print(x)
                print(np.average(move_scores, axis=-1, weights=board_weight_arr),"\n",np.min(move_scores, axis=-1),"\n",np.max(move_scores, axis=-1))
                if chess.Move.from_uci("d5d2") in moves:
                    print("compound_score=",compound_score[moves.index(chess.Move.from_uci("d5d2"))])
                    print(move_scores[:][moves.index(chess.Move.from_uci("d5d2"))])
                if chess.Move.from_uci("f3g5") in moves:
                    print("compound_score=",compound_score[moves.index(chess.Move.from_uci("f3g5"))])
                    print(move_scores[:][moves.index(chess.Move.from_uci("f3g5"))])
                
                print("board_weight=",board_weight_arr)
                print(move_config.mean_score_factor,move_config.min_score_factor,move_config.max_score_factor)
            
#------------------------------------------------------------------------------------------------------------------
        # try threat avoidance theory
        total_boards = len(board_epd_arr)
        # if total_boards <= 1000:


          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print(f'---------------------Trying Threat Avoidance, number of boards = {total_boards}---------------------')
          #  print('----------------------------------------------------------------------------------------')
          #  print('----------------------------------------------------------------------------------------')
            
          #  for i in tqdm(range(len(moves))):
          #      move = moves[i]
          #      scores = []

          #      for num_board, board_epd in board_epd_arr.items():
          #          new_board = chess.Board(board_epd)
          #          new_board.push(chess.Move.null() if simulate_move(new_board, move) is None else simulate_move(new_board, move))
          #          new_board.turn = not our_color

          #          score = RBC_threat_evaluation(new_board)

          #          scores += [score * board_weight_arr[num_board]]

            
          #      threat_scores[i] = (0.5*np.max(scores) + 0.3*np.average(scores))
                
          #  compound_score = list(np.subtract(compound_score, threat_scores))

#------------------------------------------------------------------------------------------------------------------
        # if(turn_num < move_threshold_factor):
        #     compound_score = move_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #        + move_config.min_score_factor * np.min(move_scores, axis=-1) \
        #        + move_config.max_score_factor * np.max(move_scores, axis=-1)
        # else:
        #     compound_score = dynamic_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #        + dynamic_config.min_score_factor * np.min(move_scores, axis=-1) \
        #        + dynamic_config.max_score_factor * np.max(move_scores, axis=-1)

#------------------------------------------------------------------------------------------------------------------

        # if((our_color and move_average_advantage < (-1*piece_diff)) or ((not our_color) and move_average_advantage > piece_diff)):
        #     compound_score = dynamic_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #        + dynamic_config.min_score_factor * np.min(move_scores, axis=-1) \
        #        + dynamic_config.max_score_factor * np.max(move_scores, axis=-1)        
        # else:
        #     compound_score = move_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #        + move_config.min_score_factor * np.min(move_scores, axis=-1) \
        #        + move_config.max_score_factor * np.max(move_scores, axis=-1)

#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------

        # if((our_color and average_piece_value < piece_value_threshold) or ((not our_color) and ((-1)*average_piece_value) < piece_value_threshold)):
        #     if((our_color and move_average_advantage < (-1*piece_diff)) or ((not our_color) and move_average_advantage > piece_diff)):
        #         compound_score = dynamic_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #            + dynamic_config.min_score_factor * np.min(move_scores, axis=-1) \
        #            + dynamic_config.max_score_factor * np.max(move_scores, axis=-1)        
        #     else:
        #         compound_score = move_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #            + move_config.min_score_factor * np.min(move_scores, axis=-1) \
        #            + move_config.max_score_factor * np.max(move_scores, axis=-1)
        # else:
        #     compound_score = move_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #        + move_config.min_score_factor * np.min(move_scores, axis=-1) \
        #        + move_config.max_score_factor * np.max(move_scores, axis=-1)


        # if((our_color and average_piece_value < piece_value_threshold) or ((not our_color) and ((-1)*average_piece_value) < piece_value_threshold)):
        #     compound_score = dynamic_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #            + dynamic_config.min_score_factor * np.min(move_scores, axis=-1) \
        #            + dynamic_config.max_score_factor * np.max(move_scores, axis=-1)
        # else:
        #     compound_score = move_config.mean_score_factor * np.average(move_scores, axis=-1, weights=board_weight_arr) \
        #        + move_config.min_score_factor * np.min(move_scores, axis=-1) \
        #        + move_config.max_score_factor * np.max(move_scores, axis=-1)


#------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------


        store[-1]['#boards'] = len(board_set)
        store[-1]['#sample'] = len(board_sample)

        # Add centipawn points to a move based on an estimate of the board set reduction caused by that move
        # for move in compound_score:
        #     msr.append(1 / boards_per_centipawn *
        #                       (1 - (1 / len(board_sample)) *
        #                        sum([num_occurances[move][move_result] * weighted_probability[move][move_result]
        #                             for move_result in move_possibilities[move]])))
        # if move_config.sense_by_move:
        #     compound_score = {move: score + 1 / boards_per_centipawn *
        #                       (1 - (1 / len(board_sample)) *
        #                        sum([num_occurances[move][move_result] * weighted_probability[move][move_result]
        #                             for move_result in move_possibilities[move]]))
        #                       for move, score in compound_score.items()}

        # Determine the minimum score a move needs to be considered
        highest_score = np.max(compound_score)
        threshold_score = highest_score - abs(highest_score) * move_config.threshold_score_factor
        print(highest_score, threshold_score)
        # threshold_score = highest_score * move_config.threshold_score_factor


        # Create a list of all moves which scored above the threshold
        # store[-1]['compound_score'] = {k:v for k, v in compound_score.items() if v >= threshold_score}
        move_options = [move for move, score in zip(moves, compound_score) if score >= threshold_score]
        # print(store[-1]['compound_score'])
        # # Eliminate move options which we know to be illegal (mainly for replay clarity)
        # move_options = [move for move in move_options
        #                 if move in {taken_move for taken_move, _ in move_possibilities[move]}]
        # Randomly choose one of the remaining moves
        
        moves = [move for _,move in sorted(zip(compound_score, moves), key=lambda pair: pair[0], reverse = True)]
        compound_score = sorted(compound_score, reverse = True)

        print('----------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------')
        print(f'move number: {turn_num}')
        print('----------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------')
        print(f'Top moves: {moves}')
        print('----------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------')
        print(f'Corresponding scores: {compound_score}')
        print('----------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------')

        print(move_options)
        

        turn=turn_num-1
        moves_dict[turn]=[]
        for move, score in zip(moves, compound_score):
            moves_dict[turn].append((score,str(move)))
        # print(moves_dict)
        with open('sample_moves.json','w')as outfile:
            json.dump(moves_dict, outfile)


        move_choice = chess.Move.null()        
        try:
            move_choice = random.choice(move_options)
        except Exception as e:
            print(e)
            print(len(board_set))
            if len(board_set) == 1:
                print(list(board_set)[0])
            print('compound_score:', compound_score)
            print('highest score:', highest_score)
            print('threshold_score:', threshold_score)
            print('move_options:', move_options)
            print('move_possibilities')
            print(move_possibilities)


        # move_choice = chess.Move.null()
        # move_choice_score = 0.0        
        # if len(move_options)==1:
        #     try:
        #         move_choice = random.choice(move_options)
        #     except Exception as e:
        #         print(e)
        #         print(len(board_set))
        #         if len(board_set) == 1:
        #             print(list(board_set)[0])
        #         print('compound_score:', compound_score)
        #         print('highest score:', highest_score)
        #         print('threshold_score:', threshold_score)
        #         print('move_options:', move_options)
        #         print('move_possibilities')
        #         print(move_possibilities)
        # else:
        #     # rollout_time = time()
        #     simulation_score = {}

        #     for move_to_play in move_options:
        #         simulation_score[move_to_play] = 0.0
        #         move_rollout_score = []

        #         for num_board, board_epd in board_epd_arr.items():
        #             board_1 = chess.Board(board_epd)
        #             board_1.push(chess.Move.null() if simulate_move(board_1,move_to_play) is None else simulate_move(board_1,move_to_play))
        #             board_1.turn = not our_color

        #             rollout_score = 0.0
        #             with Pool() as pool_rollout:
        #                 rollout_score = np.sum(list(pool_rollout.map(partial(rollout_func, board_1.fen()), range(rollout_games))))
        #             rollout_score /= rollout_games

        #             if not our_color:
        #                 rollout_score *= -1

        #             move_rollout_score.append(rollout_score*board_weight_arr[num_board])

        #         # simulation_score[move_to_play] = np.average(move_rollout_score)
        #         simulation_score[move_to_play] = move_config.mean_score_factor * np.average(move_rollout_score) \
        #                                         + move_config.min_score_factor * np.min(move_rollout_score)


        #         if(simulation_score[move_to_play] >= move_choice_score):
        #             move_choice =  move_to_play
        #             move_choice_score = simulation_score[move_to_play]
            
            # file = open("output_time.txt", "a")
            # file.write(f'Moves: {len(move_options)} Boards: {len(board_epd_arr)} Time: {time() - rollout_time}' + "\n")
            # file.close()
            
            # print(f'Moves: {len(move_options)} Boards: {len(board_epd_arr)} Time: {time() - rollout_time}')
            # print('----------------------------------------------------------------------------------------')
            # print('----------------------------------------------------------------------------------------')
            # print(f'Time taken in rollouts: {time() - rollout_time}')
            # print('----------------------------------------------------------------------------------------')
            # print('----------------------------------------------------------------------------------------')

        ############################################
        # assert abs(sum(compound_score.values()) - 1) < 1e-5, f"Move probs summing to {sum(compound_score.values())}"
        # move_options = sorted(compound_score.items(), key=lambda x: x[1], reverse=True)[:max(1, int(len(compound_score)/10))]
        # move_options = list(zip(*move_options))
        # move_choice = random.choices(move_options[0], weights=move_options[1])[0]
        ft = time()-t0
        # with open('/home/rbc/reconchess/Fianchetto_fast/timelog.txt', 'a') as f:
        #     f.write(f'bst={bst}, bet={bet}, ft={ft} | turn={move_ind} on move\n')

        logger.debug(f'bst={bst}, bet={bet-bst}, ft={ft-bet} | turn={move_ind} on move\n')

        # if chess.Move.from_uci("f1b6") in moves:
        #     print(move_choice,move_config.force_promotion_queen,force_promotion_to_queen(move_choice))
        #     sleep(10)

        return force_promotion_to_queen(move_choice) if move_config.force_promotion_queen else move_choice

    def while_we_wait(board_set: DefaultDict[str, float], our_color: bool, turn_num = 0):
        """
        Calculate scores for moves on next turn's boards. Store to cache for later processing acceleration.
        """
        if board_set:
            uncached_boards_epd = choose_uncached_board(board_set)
            cache_batch = {}
            while uncached_boards_epd and len(cache_batch) < time_config.max_batch:
                uncached_board_epd = uncached_boards_epd.pop()
                cache_batch[uncached_board_epd] = None

            # If there are still boards for next turn without scores calculated, calculate move scores for one
            if uncached_boards_epd:
                # board = chess.Board(uncached_board_epd)
                # print('caching uncached')
                cache_board(cache_batch, our_color, turn_num)

            # Otherwise, calculate move scores for a random board that could be reached in two turns
            elif while_we_wait_extension:
                cache_batch = {}

                board = chess.Board(random.choice(tuple(board_set)))

                for move1 in generate_rbc_moves(board):
                    next_board1 = board.copy()
                    next_board1.push(move1)
                    for move2 in generate_rbc_moves(next_board1):
                        if len(cache_batch) == time_config.max_batch:
                            break
                        next_board2 = next_board1.copy()
                        next_board2.push(move2)
                        if next_board2.king(chess.WHITE) is not None and next_board2.king(chess.BLACK) is not None:
                            cache_batch[make_cache_key(next_board2)] = None
                    if len(cache_batch) == time_config.max_batch:
                        break
                # board.push(random.choice(list(generate_rbc_moves(board))))
                # if board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None:
                cache_board(cache_batch, our_color, turn_num)

            else:
                sleep(0.001)

    def end_game():
        """
        Quit the StockFish engine instance(s) associated with this strategy once the game is over.
        """
        global msi
        global esr
        global msr
        global store
        # np.save('qarr.npy', Qs)
        np.save('msi', msi)
        np.save('esr', esr)
        np.save('msr', msr)
        np.save('store', store)
        logger.debug("During this game, averaged %.5f seconds per score using batch size %d",
                     calc_time_per_eval(), time_config.max_batch)

        # Shut down the StockFish "workers"
        # [process.terminate() for process in workers]
        # [process.join() for process in workers]


    # # Generate tuples of next turn's boards and capture squares for one current board
    # def get_next_boards_and_capture_squares(my_color, board_set):
    #     next_turn_boards = defaultdict(lambda: defaultdict(float))
    #     all_boards = board_set.keys()
    #     for board in all_boards:
    #         board.turn = not my_color
    #
    #     all_b_m = [(board, generate_rbc_moves(board)) for board in all_boards if board_set[board] > 0]
    #     scores = memo_calc_set(all_b_m)
    #
    #     # board_epd, prob = board_epd_prob
    #     # board = chess.Board(board_epd)
    #     # Calculate all possible opponent moves from this board state
    #     # board.turn = not my_color
    #     # pairs = []
    #     # moves = generate_rbc_moves(board)
    #     # my_score, all_moves_score = memo_calc_set([(board, moves)])[make_cache_key(board)]
    #
    #     for (i, (my_score, all_moves_score)) in enumerate(scores):
    #         board, moves = all_b_m[i]
    #         prob = board_set[board]
    #         if all_moves_score[-1] != 10:
    #             all_moves_prob = np.zeros(len(moves))
    #             all_moves_prob[:-1] = softmax(all_moves_score[:-1])
    #             all_moves_prob[-1] = 1/(len(moves)-1)
    #             norm = (len(moves)-1)/len(moves)
    #         else:
    #             all_moves_prob = softmax(all_moves_score)
    #             norm = 1
    #
    #         for i, move in enumerate(moves):
    #             if all_moves_prob[i] > 0:
    #                 next_board = board.copy()
    #                 next_board.push(move)
    #                 capture_square = capture_square_of_move(board, move)
    #                 next_epd = next_board.epd(en_passant='xfen')
    #                 # pairs.append((capture_square, next_epd, prob*(all_moves_prob[i]*norm)))
    #                 next_turn_boards[capture_square][next_epd] += prob*(all_moves_prob[i]*norm)
    #
    #     return next_turn_boards
    #
    #
    # # Expand one turn's boards into next turn's set by all possible moves. Store as dictionary keyed by capture square.
    # def populate_next_board_set(board_set: DefaultDict[str, float], my_color, pool=None, rc_disable_pbar: bool = False,
    #                             gnbcs: Callable = get_next_boards_and_capture_squares):
    #     next_turn_boards = defaultdict(lambda: defaultdict(float))
    #     # iter_boards = tqdm(board_set.items(), disable=rc_disable_pbar, unit='boards',
    #     #                    desc=f'{chess.COLOR_NAMES[my_color]} Expanding {len(board_set)} boards into new set')
    #     # all_pairs = (pool.imap_unordered(partial(gnbcs, my_color), iter_boards)
    #     #              if pool else map(partial(gnbcs, my_color), iter_boards))
    #
    #     next_turn_boards = gnbcs(my_color, board_set)
    #
    #     # for pairs in all_pairs:
    #     # for capture_square, next_epd, prob in pairs:
    #     #     if prob > 0:
    #     #         next_turn_boards[capture_square][next_epd] += prob
    #     return next_turn_boards

    def get_weight_for_scoring(board_set_size):
        # weight returned should never be 1
        # return 0.5
        param=int(board_set_size/1_000)
        # if param<1:
        #     return 0.99999
        # elif param<2:
        #     return 0.9999
        # elif param<3:
        #     return 0.999
        # if param<4:
        #     return 0.99
        if param<5:
            return 0.8
        elif param<20:
            return 0.5
        else:
            return 0

    # Generate tuples of next turn's boards and capture squares for one current board
    def populate_next_board_set(board_set: DefaultDict[str, float], my_color, pool=None, rc_disable_pbar: bool = False, turn_num = 0):
        t0=time()
        next_turn_boards = defaultdict(lambda: defaultdict(float))
        all_boards = []
        flag=0
        for board_epd, prob in board_set.items():
            board = chess.Board(board_epd)

            if board.turn==my_color:
                print(colored("pnbs - color problem","blue"))
            flag+=int(board.turn)

            board.turn = not my_color
            all_boards.append((board_epd, board, prob))

        all_b_m = {board.epd(en_passant='xfen'): list(generate_rbc_moves(board)) for board_epd, board, prob in all_boards if prob > 0}

        if flag!=0 and flag!=len(board_set):
            print(colored("pnbs - MULTI COLOR problem","green"))

        w = get_weight_for_scoring(len(all_b_m))

        if w != 0:
            scores = memo_calc_set(all_b_m, my_color, turn_num, pool)

        iter = ((board_epd, moves, all_boards[i][2], scores[board_epd] if w != 0 else None) for (i, (board_epd, moves)) in enumerate(all_b_m.items()))
        i = tqdm(
            iter,
            disable=rc_disable_pbar,
            desc=f'populating newer boards (speed should be high)',
            unit='boards',
        )

        all_pairs = (pool.imap_unordered(partial(get_next_boards_and_capture_squares, op_name, turn_num, w, len(all_b_m)), i)
                     if pool else map(partial(get_next_boards_and_capture_squares, op_name, turn_num, w, len(all_b_m)), i))

        for pairs in all_pairs:
            for capture_square, next_epd, prob in pairs:
                next_turn_boards[capture_square][next_epd] += prob

        logger.debug(f'Pop next board set speed = {len(board_set)/(time()-t0)} boards/s')
        return next_turn_boards



    # Return the callable functions so they can be used by StrangeFish
    return fast_divide_factor, modifier, sense_strategy, move_strategy, while_we_wait, end_game, populate_next_board_set, time_config.max_batch
