# V1
LAN:
- Based on the `main_template()` function, wrote a runnable `main()` function.
- Wrote runnable `check_winner()` function in `utility_func.py`
- Implemented the feature of alternating turns between human and computer (random moves).
- Implemented the feature of highlighting the last move in red was .
- Implemented the feature of highlighting five-in-a-row in green when a win occurs.

# V2
CHEN:
- Modified the `check_winner（）` function, the logic of determining who goes first (black), and the function to highlight the winner, which finally resolved the bug that causing incorrect winning conditions when switching between black and white.

# V3
XIAO:
- Modified the comments and optimized some redundant code logic, reducing the amount of invalid code.

# V4
HUANG：
- Fixed the issue where the game would crash immediately after a player played black.
- Added a feature to prevent automatic exit after the game ends, making it easier to review for human players.
- Noting the rule that having more or fewer than five stones does not count as a win, modified the winning condition.