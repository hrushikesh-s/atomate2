import math
import numpy as np
import warnings

from typing import Dict, List, Optional, Union
from jobflow import job
from monty.serialization import loadfn, dumpfn

from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPStaticSet, VaspInputSet


# from atomate.utils.utils import get_logger

from atomate2.utils.log import initialize_logger


# logger = initialize_logger(__name__)

__all__ = [
    "supercell_maker",
    "struct_to_supercell",
    "get_rattled_structures",
    "run_static_calculations",

]

from ase.io import read
from ase.build import find_optimal_cell_shape
from ase.build import make_supercell
from pymatgen.io.ase import AseAtomsAdaptor

from ase import atoms
from pathlib import Path
from atomate2.vasp.jobs.base import BaseVaspMaker
from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import Flow, Response, job
import ase
from typing import Union
from dataclasses import dataclass, field
from atomate2.vasp.jobs.core import (
    RelaxMaker,
    MPStaticMaker,
    StaticMaker,
)

@job
def struct_to_supercell(
    structure: Structure,
    supercell_matrix_kwargs:Optional[List[List[int]]] = None,
        ):
    """
    Generate a supercell from a structure.
    Args:
        structure (Structure): input structure
        supercell_matrix_kwargs (dict): kwargs for make_supercell
    Returns:
        supercell_structure (Structure): supercell structure
    """

    # structure = read('/global/homes/h/hrushi99/atomate2_workflows/hiphive/Ba/555/job_2023-04-11-01-10-35-712778-78860/POSCAR')
    if supercell_matrix_kwargs is not None:
        q = supercell_matrix_kwargs
    else: 
        q = [[4, 0, 0],[0, 4, 0],[0, 0, 4]]      

    # Convert to ASE atoms
    atoms = AseAtomsAdaptor.get_atoms(structure)

    # Create supercell
    supercell = make_supercell(atoms, q)

    # Convert back to pymatgen structure
    supercell_structure = AseAtomsAdaptor.get_structure(supercell)
    
    
    # return supercell
    return supercell_structure


from hiphive.structure_generation import (generate_mc_rattled_structures)

@job
def get_rattled_structures(
    supercell,
    alat:Optional[float] = None,
    n_structures:Optional[int] = None,
    rattle_std:Optional[float] = None,
    min_distance:Optional[float] = None
    ):
    """
    Generate a supercell from a structure.
    Args:
        supercell (Structure): input structure
        alat (float): lattice constant
        n_structures (int): number of structures
        rattle_std (float): standard deviation of rattle
        min_distance (float): minimum distance between atoms
    Returns:
        structures_pymatgen ([Structure]): list of Structures
    """
    alat = 4.2564840000000004
    n_structures = 5
    rattle_std = 0.01
    min_distance = 0.4 * alat

    # Convert to ASE atoms
    supercell_ase = AseAtomsAdaptor.get_atoms(supercell)
    structures_ase = generate_mc_rattled_structures(supercell_ase, n_structures, 0.25*rattle_std, min_distance, n_iter=20)

    # Convert back to pymatgen structure
    structures_pymatgen = []
    for atoms in structures_ase:
        structure_i = AseAtomsAdaptor.get_structure(atoms)
        structures_pymatgen.append(structure_i)

    return structures_pymatgen


@job
def run_static_calculations(
    rattled_structures: list[Structure],
    prev_vasp_dir: Union[str, Path, None] = None,
    MPstatic_maker: BaseVaspMaker = field(default_factory=MPStaticMaker),
    # static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
):
    """
    Run static calculations.

    Note, this job will replace itself with N static calculations, where N is
    the number of deformations.

    Args:
        rattled_structures : list of Structures
            A pymatgen Structure object.
        prev_vasp_dir : str or Path or None
            A previous VASP directory to use for copying VASP outputs.
        MPstatic_maker : BaseVaspMaker
            A VaspMaker object to use for generating VASP inputs.
    Returns:
        Response
    """

    all_jobs = []
    all_jobs_output = []
    
    for i, structure in enumerate(rattled_structures):
            # Load the atoms object from a file or create it manually
            print(structure)
            static = MPstatic_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            # static = static_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
            static.name += " {}".format(i)
            all_jobs.append(static)
            all_jobs_output.append(static.output)


    static_flow = Flow(jobs=all_jobs, output=all_jobs_output)
    static_flow.name += " 999"

    return Response(replace=static_flow)

@job
def collect_perturbed_structures(
     structure: Structure,
     supercell: Structure,
     supercell_matrix: list[list[int]],
     rattled_structures: list[Structure]
    ):
    """
    Generate a supercell from a structure.
    Args:
        structure (Structure): input structure
        supercell (Structure): supercell structure
        supercell_matrix (list[list[int]]): supercell matrix
        rattled_structures (list[Structure]): list of Structures
    Returns:
        None
    """

    structure_data = {
            "structure": structure,
            "supercell_structure": supercell,
            "supercell_matrix": supercell_matrix,
        }

    dumpfn(rattled_structures, "perturbed_structures.json") 
    dumpfn(structure_data, "structure_data.json")
    
    return None