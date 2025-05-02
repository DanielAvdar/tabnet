from typing import Any


def check_data_general(
    data: Any,
) -> None:
    """Check data format and values:
    - 1: data should be a numpy array
    - 2: data type should be only of the following: float, int, bool or there variants
    - 3: data should not be empty.

    Parameters
    ----------
    data : Any
        Object to check

    """


def model_input_data_check(data: Any) -> None:
    """Check data format and values:
    - 1: data should be check_data_general compatible
    - 2: shape should be 2D
    - 3: data should not contain NaN values
    - 4: data should not contain infinite values.


    Parameters
    ----------
    data : Any
        Object to check


    """
    check_data_general(data)
    # todo add here


def model_input_and_target_data_check(data: Any, target: Any) -> None:
    """Check data format and values:
    - 1: data should be check_data_general compatible
    - 2: target should be check_data_general compatible
    - 2: data model_input_data_check compatible
    - 3: data shape[0] should be equal to target shape[0].

    Parameters
    ----------
    data : Any
        Object to check
    target : Any
        Object to check

    """
    model_input_data_check(data)
    check_data_general(target)
    # todo add here
