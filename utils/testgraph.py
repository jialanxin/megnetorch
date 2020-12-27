from utils.graph import CrystalGraphWithAtomicFeatures
import pickle
with open("/home/jlx/megnetorch/Structures.pkl", "rb") as f:
    structures = pickle.load(f)
a_struct = structures[0]
graph = CrystalGraphWithAtomicFeatures(a_struct).encoded_atom_groups()