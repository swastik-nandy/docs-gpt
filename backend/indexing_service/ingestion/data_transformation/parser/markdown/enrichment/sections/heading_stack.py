#--------------------------------------HEADING BUILDER-------------------------------------------

def update_heading_stack(stack, level, text):
    """
    Maintain proper heading hierarchy.

    Rules:
    - Same level → replace
    - Deeper level → append
    - Higher level → pop back then append

    Example:
    H2 -> H2        => replace
    H2 -> H3        => push
    H3 -> H2        => pop → replace
    """

    if not text:
        return stack

    #---------------- NORMALIZE ----------------

    if level is None:
        level = 1

    # Store as (level, text)
    stack = stack.copy()

    #---------------- FIRST HEADING ----------------

    if not stack:
        return [(level, text)]

    #---------------- POP UNTIL VALID ----------------

    while stack and stack[-1][0] >= level:
        stack.pop()

    #---------------- ADD CURRENT ----------------

    stack.append((level, text))

    return stack


def extract_path(stack):
    """
    Convert stack → clean path

    Input:
    [(2, "A"), (3, "B")]

    Output:
    ["A", "B"]
    """

    return [h[1] for h in stack]