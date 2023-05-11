from __future__ import annotations

import os
import glob
import math
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Union
import numpy as np
from monty.serialization import loadfn, dumpfn
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
from jobflow import job, Flow, Maker

from ase.io import read
from hiphive.utilities import get_displacements

from pymatgen.core.structure import Structure, IStructure
from pymatgen.io.vasp.sets import MPStaticSet, VaspInputSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.sets import MPStaticSet, VaspInputSet

from atomate2.common.jobs.utils import structure_to_conventional, structure_to_primitive
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import (
    RelaxMaker,
    MPStaticMaker,
    StaticMaker,
    LatticeDynamicsRelaxMaker,
    TightRelaxMaker
)
from atomate2.vasp.flows.core import (
    DoubleRelaxMaker
    )
from atomate2.vasp.jobs.hiphive import (
    # supercell_maker,
    # get_perturbed_structure_wf
    struct_to_supercell,
    get_rattled_structures,
    run_static_calculations,
    collect_perturbed_structures
)
from atomate2.utils.utils import get_logger
from atomate2.vasp.analysis.lattice_dynamics_1 import (
    get_lattice_dynamics_wf, vasp_to_db_params
)
from atomate2.vasp.analysis.lattice_dynamics_2 import FIT_METHOD
from atomate2.utils.utils import get_logger
from atomate2.vasp.config import DB_FILE, SHENGBTE_CMD, VASP_CMD
from atomate2.vasp.analysis.lattice_dynamics_2 import (
    FIT_METHOD,
    MESH_DENSITY,
    IMAGINARY_TOL,
    T_QHA,
    T_KLAT,
    T_RENORM,
    RENORM_METHOD,
    RENORM_NCONFIG,
    RENORM_MAX_ITER,
    RENORM_CONV_THRESH,
    DISP_CUT,
    SEPERATE_FIT,
    T_THERMAL_CONDUCTIVITY
)

from atomate2.vasp.analysis.lattice_dynamics_3 import (
    FitForceConstantsFW,
    # LatticeThermalConductivityFW,
    # RenormalizationFW,
)
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.sets import MPStaticSet, VaspInputSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import (
    CubicSupercellTransformation,
)

from atomate2.vasp.analysis.lattice_dynamics_4 import (
    # CollectPerturbedStructures,
    # ForceConstantsToDb,
    RunHiPhive,
    RunHiPhiveRenorm,
    RunShengBTE,
    # ShengBTEToDb,
    # LatticeThermalConductivity
    )


__all__ = ["HiphiveMaker"]


@dataclass
class HiphiveMaker(Maker):
    """
    This workflow will calculate interatomic force constants and vibrational
    properties using the hiPhive package.
    A summary of the workflow is as follows:
    0. Structure relaxtion
    1. Calculate a supercell transformation matrix that brings the
       structure as close as cubic as possible, with all lattice lengths
       greater than 5 nearest neighbor distances.
    2. Perturb the atomic sites for each supercell using a Monte Carlo
       rattle procedure. The atoms are perturbed roughly according to a
       normal deviation. A number of standard deviation perturbation distances
       are included. Multiple supercells may be generated for each perturbation
       distance.
    3. Run static VASP calculations on each perturbed supercell to calculate
       atomic forces.
    4. Aggregate the forces and conduct the fit atomic force constants using
       the minimization schemes in hiPhive.
    5. Output the interatomic force constants, and phonon band structure and
       density of states to the database.
    6. Optional: Perform phonon renormalization at finite temperature - useful
       when unstable modes exist 
    7. Optional: Solve the lattice thermal conductivity using ShengBTE and
       output to the database.
    
    Args
    ----------
    name : str
        Name of the flows produced by this maker.
    """
    name: str = "hiphive"
    # relax_maker: BaseVaspMaker = field(default_factory=RelaxMaker)
    relax_maker: BaseVaspMaker = field(default_factory=LatticeDynamicsRelaxMaker)
    MPstatic_maker: BaseVaspMaker = field(default_factory=MPStaticMaker)
    static_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    bulk_relax_maker: BaseVaspMaker = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )

    # static_maker: BaseVaspMaker = field(default_factory=ElectronPhononSetGenerator) 

    def make(
        self,
        structure: Structure,
        bulk_modulus: float,
        fit_method:str,
        separate_fit: bool,
        disp_cut: float,
        cutoffs: Optional[List[List[float]]],
        prev_vasp_dir: str | Path | None = None,
        vasp_input_set: Optional[VaspInputSet] = None,
        copy_vasp_outputs: bool = False,
        supercell_matrix_kwargs: Optional[dict] = None,
        num_supercell_kwargs: Optional[dict] = None,
        perturbed_structure_kwargs: Optional[dict] = None,
        calculate_lattice_thermal_conductivity: bool = True,
        # thermal_conductivity_temperature: Union[float, Dict] = T_KLAT,
        renormalize: bool =	True,
        # renormalize: bool =	False,
        renormalize_temperature: Union[float, List, Dict] = T_RENORM,
        renormalize_method: str = RENORM_METHOD,
        renormalize_nconfig: int = RENORM_NCONFIG,
        renormalize_conv_thresh: float = RENORM_CONV_THRESH,
        renormalize_max_iter: int = RENORM_MAX_ITER,
        renormalize_thermal_expansion_iter: bool = False,
        mesh_density: float = MESH_DENSITY,
        shengbte_cmd: str = SHENGBTE_CMD,
        # thermal_conductivity_temperature=None,
        thermal_conductivity_temperature: Union[float, List, Dict] = T_THERMAL_CONDUCTIVITY,
        imaginary_tol: float = None,
        temperature_qha: float = None,
    ):
        """
        Make flow to calculate the phonon properties.

        Parameters
        ----------
        structure : .Structure
            A pymatgen structure. Please start with a structure
            that is nearly fully optimized as the internal optimizers
            have very strict settings!
        prev_vasp_dir : str or Path or None
            A previous vasp calculation directory to use for copying outputs.
        
        ______OTHER PARAMETERS WILL BE ADDED SOON TO THE DOCSTRING______   
        """
        jobs = []
        outputs = []

        ##### 1. Relax the structure
        bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        jobs.append(bulk)
        outputs.append(bulk.output)
        print(bulk.output.structure)
        structure = bulk.output.structure
        prev_vasp_dir = bulk.output.dir_name
        
        # region

        ##### 2. Calculate the supercell transformation matrix that brings the structure as close as cubic as possible, with all lattice lengths greater than 5 nearest neighbor distances.
        # # Read POSCAR file
        # # structure = read('/global/homes/h/hrushi99/atomate2_workflows/hiphive/Ba/555/job_2023-04-11-01-10-35-712778-78860/POSCAR')
        # # structure = read('/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/1_relax/CONTCAR.relax2')
        # structure = IStructure.from_file('/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/1_relax/CONTCAR.relax2')
        
        supercell = struct_to_supercell(structure)
        supercell.name += " {}".format(21)
        jobs.append(supercell)
        outputs.append(supercell.output)

        ##### 3.  Generate rattled structures using Monte Carlo perturbation technique
        rattled_structures = get_rattled_structures(supercell=supercell.output)
        rattled_structures.name += " {}".format(1)
        jobs.append(rattled_structures)
        outputs.append(rattled_structures.output) 

        #endregion

        ##### 4.  Run Static calculations on each rattled structure
        # vasp_static_calcs = self.MPstatic_maker.make(rattled_structures.output[0], prev_vasp_dir=prev_vasp_dir)
        # # structure = IStructure.from_file('/global/u2/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-03-22-20-01-753635-49605/POSCAR_Ca_supercell')
        # # vasp_static_calcs = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        # jobs.append(vasp_static_calcs)
        # outputs.append(vasp_static_calcs)
        # print(vasp_static_calcs.output.structure)
        # structure = vasp_static_calcs.output.structure
        # prev_vasp_dir = vasp_static_calcs.output.dir_name

        vasp_static_calcs = run_static_calculations(
            rattled_structures.output,
            prev_vasp_dir=prev_vasp_dir,
            MPstatic_maker=self.MPstatic_maker,
            # static_maker=self.static_maker
        )
        jobs.append(vasp_static_calcs)
        outputs.append(vasp_static_calcs.output)
        # print(f'vasp_static_calcs.output: {vasp_static_calcs.output}')


        #### 5.  Save the "structure.json", "perturbed_structures.json", & "perturbed_forces.json" and 
        ####    Hiphive Fitting using the above saved "json" files and input data supplied by the "user"
        q = [[2, 0, 0],[0, 2, 0],[0, 0, 2]] 
        json_saver = collect_perturbed_structures(
        # json_saver_rattled_struct, json_saver_forces, json_saver_struct = collect_perturbed_structures(
            structure,
            supercell.output,
            q,
            rattled_structures.output,
            vasp_static_calcs.output,
            # len(rattled_structures.output),
        )
        json_saver.name += " {}".format(1)
        jobs.append(json_saver)
        outputs.append(json_saver.output)


        # all_structures = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-03-22-20-01-753635-49605/perturbed_structures.json")
        # all_forces = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-03-22-20-01-753635-49605/perturbed_forces.json")
        # structure_data = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-03-22-20-01-753635-49605/structure_data.json")


        # all_structures = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-08-06-03-49-043950-46388/perturbed_structures.json")
        # # all_forces = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-08-06-03-49-043950-46388/perturbed_forces.json")
        # # all_forces = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-08-06-03-49-043950-46388/perturbed_forces_new.json")
        # all_forces = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-11-05-57-10-839791-62532/perturbed_forces_new.json")
        # structure_data = loadfn("/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-08-06-03-49-043950-46388/structure_data.json")

        # ##### 6. Hiphive Fitting using the above saved "json" files and input data supplied by the "user" 
        # print(cutoffs)
        # print(separate_fit)
        # print(fit_method)
        # print(disp_cut)

        fw_fit_force_constant = RunHiPhive(
            cutoffs=cutoffs,
            separate_fit=separate_fit,
            fit_method=fit_method,
            disp_cut=disp_cut,
            bulk_modulus=bulk_modulus,
            temperature_qha=temperature_qha,
            imaginary_tol=imaginary_tol,

            # rattled_structures=json_saver.output[0],
            # forces=json_saver.output[1],
            # structure_data=json_saver.output[2],
            
            # rattled_structures=all_structures,
            # forces=all_forces,
            # structure_data=structure_data,

            prev_dir_struct=json_saver.output[3],
        ) 
        fw_fit_force_constant.name += " {}".format(1)
        jobs.append(fw_fit_force_constant)
        outputs.append(fw_fit_force_constant.output)   



        #### 7. Renormalization (pass_inputs like bulk modulus)
        if renormalize:
            renorm_force_constants = RunHiPhiveRenorm(
                # renorm_temp=temperature,
                # renorm_method=renorm_method,
                # nconfig=renorm_nconfig,
                # conv_thresh=renorm_conv_thresh,
                # max_iter=renorm_max_iter,
                # renorm_TE_iter=renorm_TE_iter,
                # bulk_modulus=bulk_modulus
                renorm_temp=renormalize_temperature,
                renorm_method=renormalize_method,
                nconfig=renormalize_nconfig,
                conv_thresh=renormalize_conv_thresh,
                max_iter=renormalize_max_iter,
                renorm_TE_iter=renormalize_thermal_expansion_iter,
                bulk_modulus=bulk_modulus,
                prev_dir_hiphive=fw_fit_force_constant.output[4],
                # prev_dir_hiphive=fw_fit_force_constant.output.dir_name,
                prev_dir_struct=json_saver.output[3]
                # prev_dir_struct=json_saver.output.dir_name
                )  
            renorm_force_constants.name += " {}".format(1)
            jobs.append(renorm_force_constants)
            outputs.append(renorm_force_constants.output) 

        

        #### 8. Lattice thermal conductivity calculation
        if calculate_lattice_thermal_conductivity:
            if renormalize:
                # Because of the way ShengBTE works, a temperature array that is not
                # equally spaced out (T_step) requires submission for each temperature
                for t,T in enumerate(renormalize_temperature):
                    if T == 0:
                        continue
                    # fw_lattice_conductivity = LatticeThermalConductivityFW(
                    #     db_file=db_file,
                    #     shengbte_cmd=shengbte_cmd,
                    #     renormalized=True,
                    #     )
                    run_shengbte = RunShengBTE(
                        shengbte_cmd=shengbte_cmd,
                        renormalized=True,
                        temperature=T,
                        structure_data=json_saver.output[2],
                        # structure_data=structure_data,
                        # control_kwargs=shengbte_control_kwargs,
                        )
                    # if shengbte_fworker:
                    #     fw_lattice_conductivity.spec["_fworker"] = shengbte_fworker
                    # wf.append_wf(
                    #     Workflow.from_Firework(fw_lattice_conductivity), [wf.fws[-(t+1)].fw_id]
                    #     )
            else:
                # fw_lattice_conductivity = LatticeThermalConductivity(
                #     # db_file=db_file,
                #     shengbte_cmd=shengbte_cmd,
                #     renormalized=False,
                #     temperature=thermal_conductivity_temperature,
                #     )
                for t,T in enumerate(thermal_conductivity_temperature):
                    if T == 0:
                        continue

                    run_shengbte = RunShengBTE(
                        shengbte_cmd=shengbte_cmd,
                        renormalized=False,
                        temperature=T,
                        structure_data=json_saver.output[2],
                        # structure_data=structure_data,
                        )
                # if shengbte_fworker:
                #     fw_lattice_conductivity.spec["_fworker"] = shengbte_fworker
                # wf.append_wf(
                #     Workflow.from_Firework(fw_lattice_conductivity), [wf.fws[-1].fw_id]
                #     )

            run_shengbte.name += " {}".format(1)
            jobs.append(run_shengbte)
            outputs.append(run_shengbte.output) 
                    
        

        return Flow(jobs=jobs,
                    output=outputs,
                    name=self.name)
     