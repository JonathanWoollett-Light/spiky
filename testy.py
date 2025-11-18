"""
Simple test: set breakpoint, execute code, read variable, done.
"""

import sys


def test_breakpoint_read_variable():
    """Set breakpoint in code, execute it, read a variable"""

    # State to capture the variable
    captured_value = None

    def trace_function(frame, event, arg):
        """Called on every line - capture the variable we want"""
        nonlocal captured_value

        if event == "line" and frame.f_code.co_name == "target_code":
            # Read the variable from the frame's local scope
            if "secret" in frame.f_locals:
                captured_value = frame.f_locals["secret"]

        return trace_function

    def target_code():
        """The code we want to inspect"""
        secret = 42  # Variable we want to read
        return secret * 2

    # Set the breakpoint by enabling trace
    sys.settrace(trace_function)

    # Execute the code
    result = target_code()

    # Stop tracing
    sys.settrace(None)

    # Check we captured the variable
    print(f"Captured secret value: {captured_value}")
    print(f"Function returned: {result}")
    assert captured_value == 42
    assert result == 84


if __name__ == "__main__":
    test_breakpoint_read_variable()
