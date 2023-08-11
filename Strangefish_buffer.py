import random
from reconchess import *
from reconchess import history



class Strangefish_buffer(Player):
    def __init__(self,move_list,sense_list):
        self.move_number=0
        self.sense_number=0
        self.move_list=move_list
        self.sense_list=sense_list

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        pass

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        pass

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        self.sense_number+=1
        if self.sense_number<=len(self.sense_list):
                return self.sense_list[self.sense_number-1]
        return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        pass

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        self.move_number+=1
        if self.move_number<=len(self.move_list):
                return self.move_list[self.move_number-1]
        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        pass

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass