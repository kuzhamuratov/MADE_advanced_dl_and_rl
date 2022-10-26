import sys
import gym

import random
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math


N_ROWS, N_COLS, N_WIN = 3, 3, 3


class TicTacToe(gym.Env):
    def __init__(self, n_rows=N_ROWS, n_cols=N_COLS, n_win=N_WIN):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_win = n_win

        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.gameOver = False
        self.boardHash = None
        # ход первого игрока
        self.curTurn = 1
        self.emptySpaces = None
        
        self.reset()

    def getEmptySpaces(self):
        if self.emptySpaces is None:
            res = np.where(self.board == 0)
            self.emptySpaces = np.array([ (i, j) for i,j in zip(res[0], res[1]) ])
        return self.emptySpaces

    def makeMove(self, player, i, j):
        self.board[i, j] = player
        self.emptySpaces = None
        self.boardHash = None

    def getHash(self):
        if self.boardHash is None:
            self.boardHash = ''.join(['%s' % (x+1) for x in self.board.reshape(self.n_rows * self.n_cols)])
        return self.boardHash

    def _check_terminal(self, cur_p):
        cur_marks = np.where(self.board == cur_p)
        for i,j in zip(cur_marks[0], cur_marks[1]):
            if i <= self.n_rows - self.n_win:
                if np.all(self.board[i:i+self.n_win, j] == cur_p):
                    return True
            if j <= self.n_cols - self.n_win:
                if np.all(self.board[i,j:j+self.n_win] == cur_p):
                    return True
            if i <= self.n_rows - self.n_win and j <= self.n_cols - self.n_win:
                if np.all(np.array([ self.board[i+k,j+k] == cur_p for k in range(self.n_win) ])):
                    return True
            if i <= self.n_rows - self.n_win and j >= self.n_win-1:
                if np.all(np.array([ self.board[i+k,j-k] == cur_p for k in range(self.n_win) ])):
                    return True
        return False
    
    def isTerminal(self):
        # проверим, не закончилась ли игра
        cur_win = self._check_terminal(self.curTurn)
        if cur_win:
                self.gameOver = True
                return self.curTurn
            
        if len(self.getEmptySpaces()) == 0:
            self.gameOver = True
            return 0

        self.gameOver = False
        return None

    def getWinner(self):
        # фактически запускаем isTerminal два раза для крестиков и ноликов
        if self._check_terminal(1):
            return 1
        if self._check_terminal(-1):
            return -1
        if len(self.getEmptySpaces()) == 0:
            return 0
        return None
    
    def printBoard(self):
        for i in range(0, self.n_rows):
            print('----'*(self.n_cols)+'-')
            out = '| '
            for j in range(0, self.n_cols):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('----'*(self.n_cols)+'-')

    def getState(self):
        return (self.getHash(), self.getEmptySpaces(), self.curTurn)

    def action_from_int(self, action_int):
        return ( int(action_int / self.n_cols), int(action_int % self.n_cols))

    def int_from_action(self, action):
        return action[0] * self.n_cols + action[1]
    
    def step(self, action):
        if self.board[action[0], action[1]] != 0:
            return self.getState(), -10, True, {}
        self.makeMove(self.curTurn, action[0], action[1])
        reward = self.isTerminal()
        self.curTurn = -self.curTurn
        return self.getState(), 0 if reward is None else reward, reward is not None, {}

    def reset(self):
        self.board = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.boardHash = None
        self.gameOver = False
        self.emptySpaces = None
        self.curTurn = 1



def evaluate_strategy_against_random(env, Q, player="crosses"):
    env.reset()
    done = False
    while not done:
        if env.curTurn == 1:
            s, actions = env.getHash(), env.getEmptySpaces()
            a = actions[Q[s][tuple(actions.T)].argmax()]
            _, reward, done, _ = env.step(a)
        else:
            s, actions = env.getHash(), env.getEmptySpaces()
            a = actions[np.random.randint(0, len(actions))]
            _, reward, done, _ = env.step(a)
    return reward if player == "crosses" else -reward


def get_action(Q, state, possible_actions, epsilon):
    if np.random.uniform(0, 1) > epsilon:
        return possible_actions[np.argmax(Q[state][tuple(possible_actions.T)])]
    else:
        return possible_actions[np.random.randint(0, len(possible_actions))]
    
def learn_episode(env, Q, alpha=0.05, eps=0.05, gamma=1.0):
    env.reset()
    s_crosses, actions_crosses, done = env.getHash(), env.getEmptySpaces(), False
    a_crosses = get_action(Q, s_crosses, actions_crosses, eps)
    
    next_s, reward, done, _ = env.step(a_crosses)
    
    s_naughts, actions_naughts, _ = next_s
    a_naughts = get_action(Q, s_naughts, actions_naughts, eps)
  

    while not done:
        if env.curTurn == 1:
            next_s, reward, done, _ = env.step(a_crosses)
            
            if reward == 1:
                Q[s_crosses][tuple(a_crosses)] = reward
            
            s_naughts_next, actions_naughts, _ = next_s
            
            if actions_naughts.any():
                a_naughts_next = get_action(Q, s_naughts_next, actions_naughts, eps)
            
            delta_tmp = -reward + gamma * np.max(Q[s_naughts_next]) - Q[s_naughts][tuple(a_naughts)]
            Q[s_naughts][tuple(a_naughts)] += alpha * delta_tmp
                          
            s_naughts, a_naughts = s_naughts_next, a_naughts_next
                
        else:
            next_s, reward, done, _ = env.step(a_naughts)
            
            if reward == -1:
                Q[s_naughts][tuple(a_naughts)] = reward * (-1)
                
            s_crosses_next, actions_crosses, _ = next_s
            
            if actions_crosses.any():
                a_crosses_next = get_action(Q, s_crosses_next, actions_crosses, eps)
            
            
            delta_tmp = reward + gamma * np.max(Q[s_crosses_next]) - Q[s_crosses][tuple(a_crosses)]
            Q[s_crosses][tuple(a_crosses)] += alpha * delta_tmp
                          
            s_crosses, a_crosses = s_crosses_next, a_crosses_next

