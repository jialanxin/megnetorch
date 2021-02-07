from graph import CrystalEmbedding
from pymatgen.core.structure import Structure

MoS2_structure = Structure.from_file("materials/MoS2_POSCAR")
MoS2_graph = CrystalEmbedding(MoS2_structure)
atomic_periods = MoS2_graph.get_atomic_periods
atomic_groups = MoS2_graph.get_atomic_groups
atomic_electronegativity = MoS2_graph.get_atomic_electronegativity
atomic_cov_rad = MoS2_graph.get_atomic_covalence_redius
atomic_valence_electron_number = MoS2_graph.get_valence_electron_number
atomic_FIE = MoS2_graph.get_atomic_first_ionization_energy
atomic_electron_affinity = MoS2_graph.get_atomic_electron_affinity
atomic_block = MoS2_graph.get_atomic_blocks
space_group = MoS2_graph.get_raman_mode_numbers()
padding = MoS2_graph.padding
positions = MoS2_graph.positions
input = MoS2_graph.convert_to_model_input()
print(input)
print("end")