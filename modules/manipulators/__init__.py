from .base import (
    Base_Manipulator,
    Base_MTL_Manipulator
)

from .base_moe_altered import (
    MOE_Manipulator_Altered
)

from .ada import (
    ADA_Manipulator
)

from .mgda import (
    MGDA
)

from .pcgrad import (
    PCGrad
)

from .cagrad import (
    CAGrad
)

from .nashmtl import (
    NashMTL
)

from .famo import (
    FAMO
)

manipulator_map = {
    'stl': Base_Manipulator,
    'mix': Base_MTL_Manipulator,
    'ada': ADA_Manipulator,
    'mgda': MGDA,
    'pcgrad': PCGrad,
    'cagrad': CAGrad,
    'nashmtl': NashMTL,
    'famo': FAMO
}

# from .ls import (
#     Weight_Linear_Scalarization,
#     Weight_ScaleInvariant_Linear_Scalarization
# )