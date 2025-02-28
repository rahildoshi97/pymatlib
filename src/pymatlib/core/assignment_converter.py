import numpy as np
import sympy as sp
import pystencils as ps
from typing import List, Dict, Tuple, Union
from pystencils.types.quick import Arr, Fp
from pystencils.types import create_type, PsType
from pystencils.sympyextensions.typed_sympy import CastFunc
from pymatlib.core.typedefs import Assignment, ArrayTypes


def type_mapping(str_type: str, length: int) -> Union[PsType, Arr]:
    """
    Maps a string representation of a type to a corresponding numpy or pystencils data type.

    Parameters:
        str_type (str): The string representation of the type (e.g., "double[]", "float[]").
        length (int): The length of the array for array types.

    Returns:
        Union[np.dtype, Arr]: The corresponding numpy or pystencils data type.

    Examples:
        >>> type_mapping("double[]", 5)
        PsArrayType(element_type=PsIeeeFloatType( width=64, const=True ), size=5, const=False)
        >>> type_mapping("float[]", 3)
        PsArrayType(element_type=PsIeeeFloatType( width=32, const=True ), size=3, const=False)
        >>> type_mapping("double", 1)
        dtype('float64')
    """
    if str_type == "double[]":
        return Arr(Fp(64, const=True), length)  # 64-bit floating point array
    elif str_type == "float[]":
        return Arr(Fp(32, const=True), length)  # 32-bit floating point array
    else:
        return create_type(str_type)
        # raise ValueError(f"Unsupported type string: {str_type}")


def assignment_converter(assignment_list: List[Assignment]) \
        -> Tuple[List[ps.Assignment], Dict[sp.Symbol, ps.TypedSymbol]]:
    """
    Converts a list of assignments from the Alloy class to pystencils-compatible format.

    Parameters:
        assignment_list (List[sp.Assignment]): A list of `Assignment` objects where each `Assignment` object should have:
            - lhs (sp.Symbol): The left-hand side of the assignment, which is a symbolic variable. It should have a `name` and a `type` attribute.
            - rhs (Union[tuple, sp.Expr]): The right-hand side of the assignment, which can be a tuple or a sympy expression.
            - lhs_type (str): A string representing the type of the left-hand side, indicating the data type (e.g., "double[]", "float[]").

    Returns:
        Tuple[List[ps.Assignment], Dict[sp.Symbol, ps.TypedSymbol]]:
            - A list of `pystencils.Assignment` objects, each representing an assignment in the pystencils format.
              This list includes assignments where the `rhs` is directly assigned or cast to the appropriate type.
            - A dictionary mapping each original `sp.Symbol` object (from the lhs) to its corresponding `pystencils.TypedSymbol`.
              The dictionary allows for easy retrieval of types and assignments.
    """
    subexp = []
    subs = {}

    for assignment in assignment_list:
        if assignment.lhs is None or assignment.rhs is None or assignment.lhs_type is None:
            raise ValueError("Invalid assignment: lhs, rhs, and lhs_type must not be None")

    for assignment in assignment_list:
        if isinstance(assignment.rhs, ArrayTypes):
            length = len(assignment.rhs)
        else:
            length = 1
        ps_type = type_mapping(assignment.lhs_type, length)
        data_symbol = ps.TypedSymbol(assignment.lhs.name, ps_type)

        if isinstance(ps_type, Arr):
            subexp.append(ps.Assignment(data_symbol, assignment.rhs))
        else:  # Cast rhs to the appropriate type when necessary
            subexp.append(ps.Assignment(data_symbol, CastFunc(assignment.rhs, ps_type)))

        subs[assignment.lhs] = data_symbol

    return subexp, subs
