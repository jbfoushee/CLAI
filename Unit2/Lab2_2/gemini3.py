from collections import deque

def solve_river_crossing():
    # State: (farmer, fox, goose, grain) - 0 for start side, 1 for finish side
    initial_state = (0, 0, 0, 0)
    goal_state = (1, 1, 1, 1)

    def is_valid(state):
        f, x, g, n = state
        # Fox and goose alone
        if x == g and x != f: return False
        # Goose and grain alone
        if g == n and g != f: return False
        return True

    queue = deque([(initial_state, [])])
    visited = {initial_state}

    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            return path

        f, x, g, n = current_state
        next_f = 1 - f

        # Possible moves: Farmer alone, or carrying one of the items
        moves = [
            ((next_f, x, g, n), "Farmer crosses alone"),
            ((next_f, next_f, g, n), "Farmer carries the fox") if x == f else None,
            ((next_f, x, next_f, n), "Farmer carries the goose") if g == f else None,
            ((next_f, x, g, next_f), "Farmer carries the grain") if n == f else None
        ]

        for move in moves:
            if move:
                state, description = move
                if is_valid(state) and state not in visited:
                    visited.add(state)
                    queue.append((state, path + [description]))

    return None

solution = solve_river_crossing()
if solution:
    for i, step in enumerate(solution, 1):
        print(f"{i}. {step}")
else:
    print("No solution found.")