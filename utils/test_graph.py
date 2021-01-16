from graph import CrystalGraph
from pymatgen.core.structure import Structure

MoS2_structure = Structure.from_file("materials/MoS2_POSCAR")
MoS2_graph = CrystalGraph(MoS2_structure)
atomic_periods = MoS2_graph.get_atomic_periods
atomic_groups = MoS2_graph.get_atomic_groups
atomic_electronegativity = MoS2_graph.get_atomic_electronegativity
atomic_cov_rad = MoS2_graph.get_atomic_covalence_redius
print(atomic_cov_rad)
