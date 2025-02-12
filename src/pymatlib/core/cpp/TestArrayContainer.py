import numpy as np
from pystencils import fields
from pystencilssfg import SourceFileGenerator
from pymatlib.data.alloys.SS316L import SS316L
from pymatlib.core.interpolators import DoubleLookupArrayContainer


with SourceFileGenerator() as sfg:
    '''u = fields(f"u: double[2D]", layout='fzyx')
    mat = SS316L.create_SS316L(u.center())
    arr_container = DoubleLookupArrayContainer.from_material("SS316L", mat)
    sfg.generate(arr_container)'''


    custom_container = DoubleLookupArrayContainer("DoubleLookupTests",
                                       np.array([3243.15, 3253.15, 3263.15, 3273.15], dtype=np.float64),
                                       np.array([1.68e10, 1.69e10, 1.70e10, 1.71e10], dtype=np.float64))
    sfg.generate(custom_container)
