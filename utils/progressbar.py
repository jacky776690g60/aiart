from functools import cache
import math
from typing import List


class TermArtist:
    """
    A class containing different colors 
    and methods for printing to terminal.
    """
    WHITE = '\033[0m'
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ProgressBar:
    """
    A class for creating progress bar.\n
    Can only print one bar at a time.

    ## Compatability
    Tested on Python3.9.10
        - `Ubuntu`
        - `MacOS`
        - `Windows`
    """

    SOLID_LS = ['█', '■', '▮']
    EMPTY_LS = ['░', '⣀', '□', '▯']

    def __init__(self, total, start_value=0, bar_len=100, style: int = 0) -> None:
        self.start_value = start_value if start_value != 0 else 0 
        self.total       = total
        self.bar_len     = bar_len
        self.style       = style


    def draw(self, progress: float) -> None:
        """
        Print the progress to console based on condition

        ### Args
        - `progress` current values (usually starts with 0)
        - `total` total count
        - `bar_len` the length of the bar 
        """
        def pickStyle() -> tuple[str]:
            style = self.style
            if style == 0:    return TermArtist.WHITE,   ProgressBar.SOLID_LS[0],  TermArtist.WHITE,  ProgressBar.EMPTY_LS[1]
            elif style == 1:  return TermArtist.ORANGE,  ProgressBar.SOLID_LS[0],  TermArtist.BLUE,   ProgressBar.EMPTY_LS[1]
            elif style == 2:  return TermArtist.ORANGE,  ProgressBar.SOLID_LS[0],  TermArtist.CYAN,   ProgressBar.EMPTY_LS[1]
            elif style == 3:  return TermArtist.GREEN,   ProgressBar.SOLID_LS[0],  TermArtist.WHITE,  ProgressBar.EMPTY_LS[1]
            elif style == 4:  return TermArtist.WHITE,   ProgressBar.SOLID_LS[0],  TermArtist.GREEN,  ProgressBar.EMPTY_LS[1]
            elif style == 5:  return TermArtist.CYAN,    ProgressBar.SOLID_LS[0],  TermArtist.WHITE,  ProgressBar.EMPTY_LS[1]
            ##
            elif style == 6:  return TermArtist.WHITE,   ProgressBar.SOLID_LS[2],  TermArtist.WHITE,  ProgressBar.EMPTY_LS[2]
            elif style == 7:  return TermArtist.ORANGE,  ProgressBar.SOLID_LS[2],  TermArtist.BLUE,   ProgressBar.EMPTY_LS[2]
            elif style == 8:  return TermArtist.ORANGE,  ProgressBar.SOLID_LS[2],  TermArtist.CYAN,   ProgressBar.EMPTY_LS[2]
            elif style == 9:  return TermArtist.GREEN,   ProgressBar.SOLID_LS[2],  TermArtist.WHITE,  ProgressBar.EMPTY_LS[2]
            elif style == 10: return TermArtist.WHITE,   ProgressBar.SOLID_LS[2],  TermArtist.GREEN,  ProgressBar.EMPTY_LS[2]
            elif style == 11: return TermArtist.CYAN,    ProgressBar.SOLID_LS[2],  TermArtist.WHITE,  ProgressBar.EMPTY_LS[2]
            else:
                raise ValueError("Incorrect index for style. Enter (0 ~ 11)")

        FOREGROUND, SOLID_BAR, BACKGROUND, EMPTY_BAR = pickStyle()



        # ◼︎ normalize number to (0 ~ 100)
        scaled_progress = ((progress - self.start_value + 1) / (self.total - self.start_value)) * self.total

        # ◼︎ get percentage
        perc = 100 * (scaled_progress / self.total) if progress < self.total - 1 else 100
        perc_bar = int(perc * (self.bar_len / 100))
        try:
            bar = FOREGROUND + (SOLID_BAR * perc_bar) + \
                  BACKGROUND + (EMPTY_BAR * (self.bar_len - perc_bar))
            # print()
            print(f"\r{bar} {TermArtist.WHITE}: {perc:.2f}%", end="")
        except:
            print(" <- Progress bar went wrong here")
        





if __name__ == "__main__":
    """
    Below is an example for testing the
    progress bar. We can initialize 
    the bar by calling the function first.

    If we don't initialize it, it will
    skip 0.00%. 
    """
    numbers = [x * 5 for x in range(2000, 3000)]
    results = []

    for i, x in enumerate(numbers):
        results.append(math.factorial(x))
        ProgressBar.draw(i + 1, len(numbers), 30, 4)