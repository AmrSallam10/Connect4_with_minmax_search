import gym
import random
import requests
import numpy as np
import argparse
import sys
import math
from gym_connect_four import ConnectFourEnv

ROW_COUNT = 6
COL_COUNT = 7
WINDOW_WIDTH = 4
PLAYER_PIECE = 1
SERVER_PIECE = -1 
EMPTY = 0
MAX_DEPTH = 4

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["amr2"] # TODO: fill this list with your stil-id's

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to opponent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarantee a win
   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

def student_move(board):
   col, _ = alpha_beta_decision(board, MAX_DEPTH, -float("inf"), float("inf"), True)
   return col

def get_available_moves(board):
   available_moves = []
   for c in range(COL_COUNT):
      if(is_valid_move(board, c)):
         available_moves.append(c)
   return available_moves

def is_valid_move(board, c):
   return board[0][c] == EMPTY

def place_piece(board, row, col, player):
   board[row][col] = player

def get_open_row(board, col):
    for r in range(ROW_COUNT-1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

def game_over(board):
    for row in range(ROW_COUNT):
        for col in range(COL_COUNT - 3):
            if board[row][col] == board[row][col + 1] == board[row][col + 2] == board[row][col + 3] and board[row][col] != EMPTY:
                return True
    for row in range(ROW_COUNT - 3):
        for col in range(COL_COUNT):
            if board[row][col] == board[row + 1][col] == board[row + 2][col] == board[row + 3][col] and board[row][col] != EMPTY:
                return True
    for row in range(ROW_COUNT - 3):
        for col in range(COL_COUNT - 3):
            if board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] == board[row + 3][col + 3] and board[row][col] != EMPTY:
                return True
    for row in range(ROW_COUNT - 3):
        for col in range(3, COL_COUNT):
            if board[row][col] == board[row + 1][col - 1] == board[row + 2][col - 2] == board[row + 3][col - 3] and board[row][col] != EMPTY:
                return True
    return False

def evaluate_board(board):
    score = 0

    # Check for a tie
    if len(get_available_moves(board)) == 0:
        return 0

    # Adding more advantage to the central column
    center_col = COL_COUNT//2
    center_array = [int(i) for i in list(board[:, center_col])]
    score += center_array.count(PLAYER_PIECE)*3 - center_array.count(SERVER_PIECE)*3
    
    # Check horizontal sequences
    for row in range(ROW_COUNT):
        for col in range(COL_COUNT - 3):
            seq = [board[row][col + i] for i in range(4)]
            score += evaluate_sequence(seq)

    # Check vertical sequences
    for col in range(COL_COUNT):
        for row in range(ROW_COUNT - 3):
            seq = [board[row + i][col] for i in range(4)]
            score += evaluate_sequence(seq)

    # Check diagonal (top left to bottom right) sequences
    for row in range(ROW_COUNT - 3):
        for col in range(COL_COUNT - 3):
            seq = [board[row + i][col + i] for i in range(4)]
            score += evaluate_sequence(seq)

    # Check diagonal (top right to bottom left) sequences
    for row in range(ROW_COUNT - 3):
        for col in range(3, COL_COUNT):
            seq = [board[row + i][col - i] for i in range(4)]
            score += evaluate_sequence(seq)
    
    return score

def evaluate_sequence(seq):
    score = 0
    if seq.count(PLAYER_PIECE) == 4:
        score = 1000
    elif seq.count(PLAYER_PIECE) == 3 and seq.count(EMPTY) == 1:
        score = 100
    elif seq.count(PLAYER_PIECE) == 2 and seq.count(EMPTY) == 2:
        score = 10
    elif seq.count(SERVER_PIECE) == 4:
        score = -1000
    elif seq.count(SERVER_PIECE) == 3 and seq.count(EMPTY) == 1:
        score = -100
    elif seq.count(SERVER_PIECE) == 2 and seq.count(EMPTY) == 2:
        score = -10
    return score

def alpha_beta_decision(board, depth, alpha, beta, maximizingPlayer):
   available_moves = get_available_moves(board)
   if depth == 0 or game_over(board):
        return (None, evaluate_board(board))

   if maximizingPlayer:
        value = -float("inf")
        column = random.choice(available_moves)
        for col in available_moves:
            row = get_open_row(board, col)
            b_copy = board.copy()
            place_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = alpha_beta_decision(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

   else:
      value = float("inf")
      column = random.choice(available_moves)
      for col in available_moves:
         row = get_open_row(board, col)
         b_copy = board.copy()
         place_piece(b_copy, row, col, SERVER_PIECE)
         new_score = alpha_beta_decision(b_copy, depth-1, alpha, beta, True)[1]
         if new_score < value:
            value = new_score
            column = col
         beta = min(beta, value)
         if alpha >= beta:
            break
      return column, value

def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(state) # TODO: change input here

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tried to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like
         print()
         print(state)
         print()
         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:
         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      for i in range (20):
         play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats by running the program with "--stats"

if __name__ == "__main__":
    main()
