def print_for_verbosity(verbosity: int, first_level: str = None, second_level: str = None):
    if verbosity <= 0:
        return
    print_first_level = first_level is not None and ((verbosity == 1) or (second_level is None))
    print_second_level = second_level is not None and verbosity == 2
    if print_first_level:
        print(first_level, end="")
    if print_second_level:
        print(second_level, end="")