from collections import deque

def solve_river_crossing():
    # State: (farmer, fox, goose, grain) - 0 for left, 1 for right
    initial = (0, 0, 0, 0)
    goal = (1, 1, 1, 1)

    def is_valid(state):
        f, x, g, n = state
        if x == g and f != x:
            return False  # Fox eats goose
        if g == n and f != g:
            return False  # Goose eats grain
        return True

    queue = deque([(initial, [])])
    visited = {initial}

    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path

        f, x, g, n = state
        next_f = 1 - f

        # Farmer crosses alone
        new_state = (next_f, x, g, n)
        if is_valid(new_state) and new_state not in visited:
            visited.add(new_state)
            queue.append((new_state, path + ["Farmer crosses alone"]))

        # Farmer takes fox
        if x == f:
            new_state = (next_f, next_f, g, n)
            if is_valid(new_state) and new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + ["Farmer takes fox"]))

        # Farmer takes goose
        if g == f:
            new_state = (next_f, x, next_f, n)
            if is_valid(new_state) and new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + ["Farmer takes goose"]))

        # Farmer takes grain
        if n == f:
            new_state = (next_f, x, g, next_f)
            if is_valid(new_state) and new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + ["Farmer takes grain"]))

    return None

solution = solve_river_crossing()
if solution:
    print("Solution found! Sequence of moves:")
    for i, move in enumerate(solution, 1):
        print(f"{i}. {move}")
else:
    print("No solution exists.")