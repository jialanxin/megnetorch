from graph import CrystalEmbedding
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

MoS2_structure = Structure.from_file("materials/MoS2_mp-2815_conventional_standard.cif")
MoS2_structure.make_supercell(2)
MoS2_graph = CrystalEmbedding(MoS2_structure)
atomic_periods = MoS2_graph.get_atomic_periods
atomic_groups = MoS2_graph.get_atomic_groups
atomic_electronegativity = MoS2_graph.get_atomic_electronegativity
atomic_cov_rad = MoS2_graph.get_atomic_covalence_radius
atomic_FIE = MoS2_graph.get_atomic_first_ionization_energy
atomic_electron_affinity = MoS2_graph.get_atomic_electron_affinity
padding = MoS2_graph.padding
positions = MoS2_graph.positions
atomic_weight = MoS2_graph.get_atomic_weight
mendeleev_no = MoS2_graph.get_mendeleev_no
valence_electron = MoS2_graph.get_valence_electrons
space_group = MoS2_graph.get_space_group_number()
cell_volume = MoS2_graph.get_cell_volume()
# raman_modes = MoS2_graph.get_raman_modes()
pos,itn = MoS2_graph.get_xrd()
input = MoS2_graph.convert_to_model_input()
print("end")