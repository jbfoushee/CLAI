from collections import deque

def solve_river_crossing():
    """
    Solves the classic river crossing puzzle.
    Returns the sequence of moves to transport fox, goose, and grain across the river.
    """
    
    # State representation: (farmer_side, fox_side, goose_side, grain_side)
    # 0 = left bank, 1 = right bank
    initial_state = (0, 0, 0, 0)
    goal_state = (1, 1, 1, 1)
    
    # Invalid states where items are left alone together
    def is_valid(state):
        farmer, fox, goose, grain = state
        # Fox and goose can't be alone together
        if fox == goose and fox != farmer:
            return False
        # Goose and grain can't be alone together
        if goose == grain and goose != farmer:
            return False
        return True
    
    # BFS to find shortest path
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    
    while queue:
        state, moves = queue.popleft()
        
        if state == goal_state:
            return moves
        
        farmer, fox, goose, grain = state
        next_side = 1 - farmer  # Toggle between 0 and 1
        
        # Try moving each item
        items = [
            ("fox", fox),
            ("goose", goose),
            ("grain", grain)
        ]
        
        for item_name, item_side in items:
            if item_side == farmer:  # Can only move items on farmer's side
                # Create new state
                new_state = list(state)
                new_state[0] = next_side  # Move farmer
                
                if item_name == "fox":
                    new_state[1] = next_side
                elif item_name == "goose":
                    new_state[2] = next_side
                elif item_name == "grain":
                    new_state[3] = next_side
                
                new_state = tuple(new_state)
                
                if is_valid(new_state) and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, moves + [item_name]))
    
    return None  # No solution found

# Run and display results
solution = solve_river_crossing()
if solution:
    print("Solution found! Sequence of moves:")
    for i, move in enumerate(solution, 1):
        print(f"{i}. Transport {move} across the river")
    print(f"\nTotal moves: {len(solution)}")
else:
    print("No solution exists")