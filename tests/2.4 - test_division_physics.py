from importlib import reload
import goo
from goo.division import *
from goo.handler import *

reload(goo)
goo.reset_modules()
goo.reset_scene()

celltype = goo.SimpleType("cellA")
celltype.homo_adhesion_strength = 5000
cell = celltype.create_cell(name="cell", loc=(0, 0, 0), color=(0, 1, 1))
cell.stiffness = 5
cell.pressure = 5

sim = goo.Simulator(celltypes=[celltype])
sim.setup_world()
sim.add_handlers(
    
    [
        GrowthPIDHandler(target_volume=75),
        SizeDivisionHandler(BisectDivisionLogic, mu=60, sigma=10),  # TimeDivisionHandler
        AdhesionLocationHandler(),
        RemeshHandler(),
        RandomMotionHandler(distribution=ForceDist.CONSTANT, max_strength=0),
    ]
)

# Run simulation headless
# sim.run(65)