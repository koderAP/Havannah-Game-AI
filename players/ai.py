import time
import numpy as np
from helper import *
import random
import math
from collections import defaultdict
from typing import Tuple, List
from time import sleep


def is_corner(r, c, size):
    n = size
    return (r == 0 and c == 0) or \
           (r == 0 and c == n-1) or \
           (r == n-1 and c == 2*n-2) or \
           (r == 2*n-2 and c == n-1) or \
           (r == 2*n-2 and c == 0) or \
           (r == n-1 and c == 0)

def is_edge(r, c, size):
    n = size
    return (r == 0 or r == 2*n-2) or \
           (c == 0 or c == 2*n-2) or \
           (r + c == n-1) or \
           (r + c == 3*n-3)


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])  # Path compression
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1

class MCTSNode:
    def __init__(self, state: np.array, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.depth = parent.depth + 1 if parent else 0
        
        # RAVE statistics (action-based statistics across the tree)
        self.rave_visits = defaultdict(int)  # Number of times an action has been taken in a simulation
        self.rave_wins = defaultdict(int)    # Number of times an action led to a win in a simulation


    def is_fully_expanded(self, valid_moves):
        return len(self.children) == len(valid_moves)


class AIPlayer:
    def __init__(self, player_number: int, timer):
        """
        Initialize the AIPlayer with hybrid MCTS and Minimax strategies.
        """
        self.player_number = player_number
        self.opponent_number = 3 - player_number
        self.timer = timer
        self.max_simulations = 3000  # Number of MCTS simulations per move
        self.mcts_extension_probability = 1  # Probability to extend Minimax with MCTS
        self.C = 0.7 # Exploration constant for UCB1
        self.total_moves = 0
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.transposition_table = defaultdict(lambda: (None, 'EXACT', False, -1))
        self.opening_threshold = 6
        self.neighbors = set()
        self.random_play_depth = 9
        self.total_states = 0
        self.counter = 3
        self.time_limit = 7

    def open(self, state):
        """
        Opening moves for the AI.
        """
        n = state.shape[0]
        opening_moves = [(0, 1), (1, 0), (2, 2), (1, 3),  (0,2), (0,0),(0, n-2), (1, n-1), (2, n-3), (1, n-4)]
        return opening_moves

    def open1(self, state):
        n = state.shape[0]
        # opening_moves = [(6,3), (4,4), (3,6), (4,2), (3,0), (2,1), (2,5)]
        opening_moves = [(6,3),(3,6)]
        # return []
        # opening_moves = [(6,3), (3,6), (0,6),(3,0), (0,0)]
        return opening_moves
    
    def open2(self, state):
        n = state.shape[0]
        opening_moves = [(0,n-1),(n//2, n-1),(2,n-2), (3, n-1), (5, n-2),(5, n-1), (3,n-2), (1,n-2)]
        return opening_moves
    
    def open3(self, state):
        n = state.shape[0]
        opening_moves = [(0,0), (n//2, 0),(2,1),(n//2,1) ,(3,0), (3,1)]
        return opening_moves
    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Main method to get the best move by combining MCTS and Minimax.
        """

        valid_moves = get_valid_actions(state)

        # 1. Check if AI can win in one move
        for move in valid_moves:
            if check_win(self.simulate_move(state, move, self.player_number), move, self.player_number)[0]:
                return tuple(map(int, move))

        opponent_number = 3 - self.player_number

        # 2. Block opponent’s winning move
        for move in valid_moves:
            if check_win(self.simulate_move(state, move, opponent_number), move, opponent_number)[0]:
                return tuple(map(int, move))

        best_move = None
        max_win_count = 2  # Minimum win count threshold

        for move in valid_moves:
            new_state = self.simulate_move(state, move, self.player_number)
            win_count = self.count_winning_conditions(new_state, self.player_number)
            if win_count >= max_win_count:
                max_win_count = win_count
                best_move = move
                print("best move so far is", best_move)
                return tuple(map(int, best_move))

            l = neighbours_get(new_state, move)
            temp_win_count = 0
            for x in l:
                temp_win = self.count_winning_conditions(self.simulate_move(new_state, x, self.player_number), self.player_number)
                if temp_win >= 2:
                    temp_win_count += 2

            if temp_win_count >= max_win_count:
                max_win_count = temp_win_count
                best_move = move

        if best_move:
            print("SELF")
            return tuple(map(int, best_move))

        for move in valid_moves:
            new_state = self.simulate_move(state, move, opponent_number)
            win_count = self.count_winning_conditions(new_state, opponent_number)
            if win_count >= max_win_count:
                max_win_count = win_count
                best_move = move

            temp_win_count = 0
            l = neighbours_get(new_state, move)
            for x in l:
                temp_win = self.count_winning_conditions(self.simulate_move(new_state, x, opponent_number), opponent_number)
                if temp_win >= 2:
                    temp_win_count += temp_win

            if temp_win_count > max_win_count:
                max_win_count = temp_win_count
                best_move = move

        if best_move:
            return tuple(map(int, best_move))

        self.neighbors = self.all_valid_neighbors(state)

        # handle 4x4
        if state.shape[0] == 7:
            for move in self.open1(state):
                if state[move[0]][move[1]] == 0:
                    return tuple(map(int, move))

        elif self.total_moves < self.opening_threshold:
            l = self.open2(state)
            for move in l:
                if state[move[0]][move[1]] == 0:
                    self.total_moves += 1
                    return tuple(map(int, move))

        self.neighbors = self.all_valid_neighbors(state)

        self.total_states = 0
        start_time = time.time()
        best_move = None
        root = MCTSNode(state)
        for _ in range(self.max_simulations):
            if time.time() - start_time > self.time_limit:
                print(_)
                break

            node = self.select(root)
            if not node.is_fully_expanded(valid_moves):
                node = self.expand(node)

            # Simulate and retrieve the result and action sequence
            result, action_sequence = self.simulate(node)

            # Backpropagate result and RAVE statistics
            self.backpropagate(node, result, action_sequence)

        best_child = max(root.children, key=lambda child: child.wins / child.visits, default=None)
        if best_child:
            best_move = best_child.move

        return tuple(map(int, best_move)) if best_move else tuple(map(int, random.choice(valid_moves)))


    def get_board_hash(self, state: np.array) -> int:
        """
        Get a hash value for the board state.
        """
        return hash(state.tobytes())
    
    def minimax(self, state: np.array, depth: int, alpha: float, beta: float, maximizing_player: bool) -> int:
        """
        Minimax algorithm with Alpha-Beta pruning and transposition table.
        """
        state_hash = self.get_board_hash(state)
        if state_hash in self.transposition_table:
            return self.transposition_table[state_hash]

        if depth == 0:
            return self.heuristic(state)

        valid_moves = self.neighbors
    

        if maximizing_player:
            max_eval = -float('inf')
            for move in valid_moves:
                new_state = self.simulate_move(state, move, self.player_number)
                if check_win(new_state, move, self.player_number)[0]:
                    return float('inf')
                eval_value = self.minimax(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break  # Pruning
            self.transposition_table[state_hash] = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            opponent_number = 3 - self.player_number
            for move in valid_moves:
                new_state = self.simulate_move(state, move, opponent_number)
                if check_win(new_state, move, opponent_number)[0]:
                    return -float('inf')
                eval_value = self.minimax(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break  # Pruning
            self.transposition_table[state_hash] = min_eval
            return min_eval
        


    


    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Traverse the tree using UCB1 to find the most promising node.
        """
        while node.children:
            node = max(node.children, key=lambda child: self.ucb1(child))
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expand the tree by creating a new child for an untried move.
        """

        valid_moves = self.neighbors
        if len(valid_moves) == 0:
            return node
        unexplored_moves = [move for move in valid_moves if move not in [child.move for child in node.children]]
        move = random.choice(unexplored_moves)
        new_state = self.simulate_move(node.state, move, self.player_number)
        child_node = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def simulate(self, node: MCTSNode) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Simulate a random playout from the current node and return the result and the sequence of moves taken.
        """
        current_state = node.state.copy()
        current_player = self.player_number
        valid_moves = get_valid_actions(current_state)
        valid_moves = set(valid_moves)
        action_sequence = []  # Track actions taken during the simulation
        
        for _ in range(self.random_play_depth):  
            if not valid_moves:
                break
            move = random.choice(list(valid_moves))
            action_sequence.append(move)
            valid_moves.remove(move)
            current_state = self.simulate_move(current_state, move, current_player)
            current_player = 3 - current_player

            if check_win(current_state, move, self.player_number)[0]:
                return float('inf'), action_sequence  # AI wins
            elif check_win(current_state, move, self.opponent_number)[0]:
                return -float('inf'), action_sequence  # Opponent wins
                
        return self.heuristic(current_state), action_sequence


    def backpropagate(self, node: MCTSNode, result: int, action_sequence: List[Tuple[int, int]]):
        """
        Backpropagate the result of the simulation up the tree.
        Also, update RAVE statistics for actions in the simulation.
        """
        while node:
            node.visits += 1
            
            if result == 1:  # Win for the AI
                node.wins += 1
            elif result == -1:  # Loss for the AI
                node.wins -= 1
            
            # Update RAVE statistics for all actions that occurred during the simulation
            for move in action_sequence:
                node.rave_visits[move] += 1
                if result == 1:
                    node.rave_wins[move] += 1

            node = node.parent


    def ucb1(self, node: MCTSNode) -> float:
        """
        Calculate UCB1 score for a node with RAVE incorporated.
        """
        if node.visits == 0:
            return float('inf')

        # Standard UCB1 score
        exploitation = node.wins / node.visits
        exploration = self.C * math.sqrt(math.log(node.parent.visits) / node.visits)

        # Calculate RAVE score for the node's move
        move = node.move
        if node.rave_visits[move] > 0:
            rave_exploitation = node.rave_wins[move] / node.rave_visits[move]
            rave_exploration = math.sqrt(math.log(node.parent.visits) / node.rave_visits[move])
        else:
            rave_exploitation = 0
            rave_exploration = float('inf')

        # Blending UCB1 and RAVE with β as a balancing parameter
        beta = node.rave_visits[move] / (node.visits + node.rave_visits[move] + 1)
        # Decay factor based on depth
        depth_factor = 1 / (node.visits + node.depth + 1)
        blended_score = (1 - beta * depth_factor) * (exploitation + exploration) + beta * depth_factor * (rave_exploitation + rave_exploration)
        return (1 - beta * depth_factor) * (exploitation + exploration) + beta * depth_factor * (rave_exploitation + rave_exploration)


    def heuristic(self, state: np.array) -> int:



        player_number = self.player_number
        opponent_number = self.opponent_number

        player_wins = self.count_winning_conditions(state, player_number)
        opponent_wins = self.count_winning_conditions(state, opponent_number)

        player_group_size = 0
        opponent_group_size = 0
        player_edge_control = 0
        opponent_edge_control = 0
        player_virtual_connections = 0
        opponent_virtual_connections = 0


        player_virtual_connections = self.calculate_virtual_connections(state, player_number)
        opponent_virtual_connections = self.calculate_virtual_connections(state, opponent_number)

        opponent_edge_control = self.calculate_edge_control(state, opponent_number)

        opponent_ring_threat = self.detect_ring_threat(state, opponent_number)
        our_ring_threat = self.detect_ring_threat(state, player_number)

        # Heuristic scoring:
        # 1. Heavily reward player winning conditions (goal of the game).
        # 2. Reward large group sizes (larger groups are stronger).
        # 3. Encourage forming virtual connections (potential winning strategy).
        # 4. Reward control of edges and corners (critical in games like Havannah).
        # 5. Penalize for opponent's ring threats (defensive aspect).
    

        return (1000 * player_wins + player_group_size + 
                200 * player_virtual_connections - 
                5000 * opponent_ring_threat + 100*our_ring_threat) - (
                2000 * opponent_wins + 10 * opponent_group_size + 
                300 * opponent_virtual_connections + 50 * opponent_edge_control)

    def count_winning_conditions(self, state: np.array, player_number: int) -> int:
        """
        Count potential winning conditions for the given player.
        """
        win_count = 0
        valid_moves = get_valid_actions(state)
        for move in valid_moves:
            if check_win(self.simulate_move(state, move, player_number), move, player_number)[0]:
                win_count += 1
        return win_count
    
    
    def calculate_virtual_connections(self, board: np.array, player: int, max_depth: int = 3) -> int:
        size = len(board)
        union_find = UnionFind(size * size)
        virtual_connection_score = 0

        # Step 1: Union adjacent stones of the same player
        for r in range(size):
            for c in range(size):
                if board[r][c] == player:
                    neighbors = neighbours_get(board, (r, c))
                    for nr, nc in neighbors:
                        if board[nr][nc] == player:
                            union_find.union(r * size + c, nr * size + nc)

        # Step 2: Check for potential virtual connections
        def depth_limited_search(r, c, depth, visited):
            nonlocal virtual_connection_score
            if depth == 0:
                return

            neighbors = neighbours_get(board, (r, c))
            neighbor_components = set()
            
            for nr, nc in neighbors:
                if board[nr][nc] == player:
                    neighbor_components.add(union_find.find(nr * size + nc))
                elif board[nr][nc] == 0 and (nr, nc) not in visited:  # Empty cell and not visited
                    visited.add((nr, nc))  # Mark as visited
                    depth_limited_search(nr, nc, depth - 1, visited)  # Recur for depth

            # If there's more than one unique component, we can form a virtual connection
            if len(neighbor_components) > 1:
                virtual_connection_score += len(neighbor_components) - 1
                if is_edge(r, c, size) or is_corner(r, c, size):
                    virtual_connection_score += 2  # Bonus for edge/corner control

        # Start depth-limited search from each empty cell
        for r in range(size):
            for c in range(size):
                if board[r][c] == 0:  # Only start from empty cells
                    visited = set()  # Reset visited for each empty cell
                    visited.add((r, c))  # Mark the starting position
                    depth_limited_search(r, c, max_depth, visited)

        return virtual_connection_score





    def calculate_edge_control(self, state: np.array, player_number: int) -> int:
        """
        Calculate the control of edges and corners by counting stones near the edges.
        This function prioritizes control of strategic locations such as edges or corners.
        """
        n = state.shape[0]
        edge_control_score = 0
        
        # Check the stones placed at the edge or near corners
        for i in range(n):
            for j in range(n):
                if state[i, j] == player_number:
                    if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                        edge_control_score += 1  # Stones placed at the edge
                    if (i == 0 and j == 0) or (i == 0 and j == n - 1) or (i == n - 1 and j == 0) or (i == n - 1 and j == n - 1):
                        edge_control_score += 2  # Extra points for corner control

        return edge_control_score


    def detect_ring_threat(self, state: np.array, opponent_number: int) -> int:
        """
        Detect if the opponent is close to forming a ring and penalize accordingly.
        """
        ring_threat_count = 0
        # valid_moves = get_valid_actions(state)
        valid_moves = self.neighbors
        
        # Check if the opponent is close to forming a ring
        for move in valid_moves:
            if check_win(self.simulate_move(state, move, opponent_number), move, opponent_number)[0]:
                ring_threat_count += 1

        if ring_threat_count > 1:
            ring_threat_count = 1000
        return ring_threat_count


    def neighbours_get(board: np.array, move: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get neighboring cells for a given move.
        """
        row, col = move
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        neighbors = []

        n = board.shape[0]
        for dr, dc in directions:
            if 0 <= (row + dr) < n and 0 <= (col + dc) < n:
                neighbors.append((row + dr, col + dc))

        return neighbors


    def all_valid_neighbors(self, state: np.array) -> List[Tuple[int, int]]:
        """
        Get all valid neighboring cells.
        """
        valid_neighbors = set()
        n = state.shape[0]
        for i in range(n):
            for j in range(n):
                if state[i, j] == 0:
                    valid_neighbors.add((i, j))
                    l1 = neighbours_get(state, (i, j))
                    valid_neighbors.update(l1)
                    for x in l1:
                        l2 = neighbours_get(state, x)
                        valid_neighbors.update(l2)

        return valid_neighbors

    # def update_neighbors(self, state, move, valid_moves):
    #     """
    #     Update the set of neighbors for future calculations.
    #     """
    #     x = neighbours_get(state, move)
    #     self.neighbors.update(x)
    #     self.neighbors = {n for n in self.neighbors if n in valid_moves}

    def simulate_move(self, state: np.array, move: Tuple[int, int], player_number: int) -> np.array:
        """
        Simulate the move by placing the player's stone on the board.
        """
        new_state = state.copy()
        new_state[move[0], move[1]] = player_number
        self.neighbors.update(neighbours_get(new_state, move))
        return new_state
    



def neighbours_get(board, move):
    """
    Get neighboring cells for a given move.
    """
    l = []
    for move in get_neighbours(board.shape[0], move):
        if board[move] == 0: 
            l.append(move)

    return l


def neighbours_get1(borad, move):
    """
    Get neighboring cells for a given move.
    """
    row, col = move
    l = [(0,1), (1,0), (1,1), (1,-1), (0,-1), (-1,0), (-1,-1), (-1,1)]
    neighbors = []

    n = borad.shape[0]
    for dr, dc in l:
        if  -1 < (dr + row) < n and -1 < (col + dc) < n :
            if borad[dr+row][dc+col] !=3:
                neighbors.append((row+dr, col+dc)) 

    return neighbors
