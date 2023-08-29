## Basic Python packages
from __future__ import annotations
from copy import deepcopy
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

## Jobflow packages
from jobflow import job, Flow, Maker

## Pymatgen packages
from pymatgen.core.structure import Structure

## Atomate2 packages
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import (
    # MPStaticMaker,
    TightRelaxMaker,
    StaticMaker
)
from atomate2.vasp.flows.core import (
    DoubleRelaxMaker
    )
from atomate2.vasp.jobs.hiphive import (
    run_static_calculations,
    collect_perturbed_structures,
    QualityControl,
    RunHiPhive,
    RunShengBTE,
    RunFCtoPDOS,
    RenormalizationFW,
    LatticeThermalConductivityFW
)
# from atomate2.vasp.config import (
#     DB_FILE,
#     SHENGBTE_CMD,
#     VASP_CMD,
#     PHONO3PY_CMD
# )
# from atomate2.vasp.analysis.lattice_dynamics_2 import (
#     FIT_METHOD,
#     MESH_DENSITY,
#     IMAGINARY_TOL,
#     T_QHA,
#     T_KLAT,
#     T_RENORM,
#     RENORM_METHOD,
#     RENORM_NCONFIG,
#     RENORM_MAX_ITER,
#     RENORM_CONV_THRESH,
#     T_THERMAL_CONDUCTIVITY
# )

# from atomate2.vasp.analysis.lattice_dynamics_3 import (
#     FitForceConstantsFW,
#     # LatticeThermalConductivityFW,
#     # RenormalizationFW,
# )
# from atomate2.vasp.analysis.lattice_dynamics_4 import (
#     # CollectPerturbedStructures,
#     # ForceConstantsToDb,
#     RunHiPhive,
#     # RunHiPhiveRenorm,
#     RunShengBTE,
#     # ShengBTEToDb,
#     # LatticeThermalConductivity
#     RenormalizationFW,
#     LatticeThermalConductivityFW,
#     RunPhono3py,
#     RunFCtoPDOS
#     )
from atomate2.vasp.powerups import update_user_incar_settings, update_user_potcar_functional
# from atomate2.vasp.sets.core import MPStaticSetGenerator
from atomate2.vasp.sets.core import StaticSetGenerator

__all__ = ["HiphiveMaker"]

__author__ = "Alex Ganose, Junsoo Park, Zhuoying Zhu, Hrushikesh Sahasrabuddhe"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov, zyzhu@lbl.gov, hpsahasrabuddhe@lbl.gov"

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
    name: str = "Lattice-Dynamics"
    task_document_kwargs : dict = field(default_factory=lambda: {"task_label": "dummy_label"})
    # my_custom_set = MPStaticSetGenerator(user_incar_settings={"ISMEAR": 0})
    # MPstatic_maker = MPStaticMaker(input_set_generator=my_custom_set)
    MPstatic_maker: BaseVaspMaker = field(default_factory=StaticMaker)
    # bulk_relax_maker = DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    bulk_relax_maker: BaseVaspMaker | None = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    

    IMAGINARY_TOL = 0.025  # in THz
    MESH_DENSITY = 100.0  # should always be a float 

    # Temperature for straight-up phonopy calculation of thermodynamic properties (free energy etc.)
    T_QHA = [i*100 for i in range(21)]
    # Temperature at which renormalization is to be performed
    # T_RENORM = [0,50,100,200,300,500,700,1000,1500]#[i*100 for i in range(0,16)]
    # T_RENORM = [0,300, 700, 1000, 1500]#[i*100 for i in range(0,16)]
    T_RENORM = [1500]#[i*100 for i in range(0,16)]
    # T_RENORM = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#[i*100 for i in range(0,16)]
    # Temperature at which lattice thermal conductivity is calculated
    # If renormalization is performed, T_RENORM overrides T_KLAT for lattice thermal conductivity
    # T_KLAT = {"t_min":100,"t_max":1500,"t_step":100} #[i*100 for i in range(0,11)]
    T_KLAT = 300 #[i*100 for i in range(0,11)]
    # T_THERMAL_CONDUCTIVITY = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#[i*100 for i in range(0,16)]
    T_THERMAL_CONDUCTIVITY = [0, 100, 200, 300]#[i*100 for i in range(0,16)]

    FIT_METHOD = "rfe" 
    RENORM_METHOD = 'pseudoinverse'
    RENORM_NCONFIG = 5 # Changed from 50
    RENORM_CONV_THRESH = 0.1 # meV/atom
    RENORM_MAX_ITER = 30 # Changed from 20
    SHENGBTE_CMD = "srun -n 32 ./ShengBTE 2>BTE.err >BTE.out"
    PHONO3PY_CMD = "phono3py --mesh 19 19 19 --fc3 --fc2 --br --dim 5 5 5"


    def make(
        self,
        mpid: str,
        structure: Structure,
        bulk_modulus: float,
        fit_method:Optional[str] = None,
        disp_cut: Optional[float] = None,
        cutoffs: Optional[List[List[float]]] = None,
        prev_vasp_dir: str | Path | None = None,
        supercell_matrix_kwargs: Optional[dict] = None,
        num_supercell_kwargs: Optional[dict] = None,
        perturbed_structure_kwargs: Optional[dict] = None,
        calculate_lattice_thermal_conductivity: bool = True,
        # thermal_conductivity_temperature: Union[float, Dict] = T_KLAT,
        renormalize: bool =	False,
        renormalize_temperature: Union[float, List, Dict] = T_RENORM,
        renormalize_method: str = RENORM_METHOD,
        renormalize_nconfig: int = RENORM_NCONFIG,
        renormalize_conv_thresh: float = RENORM_CONV_THRESH,
        renormalize_max_iter: int = RENORM_MAX_ITER,
        renormalize_thermal_expansion_iter: bool = False,
        mesh_density: float = MESH_DENSITY,
        shengbte_cmd: str = SHENGBTE_CMD,
        phono3py_cmd: str = PHONO3PY_CMD,
        thermal_conductivity_temperature: Union[float, List, Dict] = T_KLAT,
        imaginary_tol: float = IMAGINARY_TOL,
        temperature_qha: float = T_QHA,
        n_structures: float = None,
        rattle_std: float = None,
        prev_dir_hiphive: str | Path | None = None,
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
        loops = 1

        ##### 1. Relax the structure
        bulk = self.bulk_relax_maker.make(structure, prev_vasp_dir=prev_vasp_dir)
        jobs.append(bulk)
        outputs.append(bulk.output)
        print(bulk.output.structure)
        structure = bulk.output.structure
        prev_vasp_dir = bulk.output.dir_name
        # bulk.update_metadata({"tag": [f"mp_id={mpid}", f"relax_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})
        bulk.update_metadata({"tag": [f"mp_id=.....", f"relax_.....", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})
        

        ##### 2. Generates supercell, performs fixed displ rattling and runs static calculations
        vasp_static_calcs = run_static_calculations(
            structure = structure,
            supercell_matrix_kwargs = supercell_matrix_kwargs,
            n_structures = n_structures,
            rattle_std = rattle_std,
            loop = loops,
            prev_vasp_dir = prev_vasp_dir,
            MPstatic_maker = self.MPstatic_maker,
        )
        jobs.append(vasp_static_calcs)
        outputs.append(vasp_static_calcs.output)
        vasp_static_calcs.metadata.update({"tag": [f"mp_id={mpid}", f"vasp_static_calcs_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})


        ##### 3.  Save "structure_data_{loop}.json", "relaxed_structures.json", "perturbed_forces_{loop}.json" and "perturbed_structures_{loop}.json" files locally
        json_saver = collect_perturbed_structures(
            structure = structure,
            supercell_matrix_kwargs = supercell_matrix_kwargs,
            rattled_structures = vasp_static_calcs.output,
            forces = vasp_static_calcs.output,
            loop = loops
        )
        json_saver.name += f" {loops}"
        jobs.append(json_saver)
        outputs.append(json_saver.output)
        json_saver.metadata.update({"tag": [f"mp_id={mpid}", f"json_saver_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})

        # # loops = 2
        # ##### 4. Hiphive Fitting of FCPs upto 4th order
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GaP/job_2023-06-27-05-58-32-459504-31596" ## 1 config -- GaP -- Hrushi
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GaP/hiphive_Zhuoying/Zhuyoing_hiphive_atomate/files_requried_for_hiphive_fitting" ## 1 config -- GaP -- Zhuoying
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GaP/block_2023-06-26-23-59-11-936116/launcher_2023-07-16-07-04-35-443805/launcher_2023-07-16-07-09-53-648895" ## 1 config -- GaP -- Hrushi_new
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GaP/hiphive_Zhuoying/Hrushi_hiphive_atomate2/job_2023-07-16-07-37-10-718237-56216" ## 1 config -- GaP -- Hrushi_new_2
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GaP/block_2023-06-26-23-59-11-936116/launcher_2023-07-17-04-37-48-088293/launcher_2023-07-17-04-38-10-338257" ## 1 config -- GaP -- Hrushi_new_3 -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GaP/hiphive_Zhuoying/Hrushi_hiphive_atomate2/2,2,2-supercell/saved_files" ## 1 config -- 2*2*2 supercell -- GaP -- Hrushi_new_2
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GaP/hiphive_Zhuoying/Hrushi_hiphive_atomate2/without_seperate_fit_Hrushis_saved_files_rfe_no_random_seed_generator" ## 1 config -- GaP -- Hrushi_new -- no random seed -- FCS & Input files
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/InAs/block_2023-06-29-17-47-56-254158/launcher_2023-07-17-06-56-22-052563/launcher_2023-07-17-06-59-10-506460" ## 1 config -- InAs -- Hrushi_new -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/InAs/block_2023-06-29-17-47-56-254158/launcher_2023-06-29-20-01-14-574433/launcher_2023-06-29-20-19-25-612257" ## 1 config -- InAs -- Hrushi_new -- no random seed -- FCS & Input files
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/BeTe/block_2023-07-17-19-19-45-102785/launcher_2023-07-17-19-42-59-335823/launcher_2023-07-17-20-28-59-262540" ## 1 config -- BeTe -- Hrushi_new -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/BeTe/block_2023-07-17-19-19-45-102785/launcher_2023-07-17-23-18-57-616424/launcher_2023-07-17-23-20-23-417783" ## 1 config -- BeTe -- Hrushi_new_2 -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/GeTe/block_2023-07-17-19-10-55-595175/launcher_2023-07-18-04-53-48-142683/launcher_2023-07-18-04-59-44-574767" ## 1 config -- GeTe -- Hrushi_new -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/ZnSe/block_2023-07-18-06-40-57-244009/launcher_2023-07-18-07-38-53-344195/launcher_2023-07-18-07-39-24-231920" ## 1 config -- ZnSe -- Hrushi_new -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/ZnSe/without_seperate_fit_Hrushis_saved_files_rfe_no_random_seed_generator" ## 1 config -- ZnSe -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files 
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/ZnSe/manual_hiphive_fitting_one_anharmonic_0,1_cutoff_5,3,2_rfe" ## 1 config -- 1 anharmonic struct -- ZnSe -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/MgO/block_2023-07-18-07-48-42-230857/launcher_2023-07-18-10-12-34-524032/launcher_2023-07-18-10-13-12-862969" ## 1 config -- MgO -- Hrushi_new -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/MgO/without_seperate_fit_Hrushis_saved_files_rfe_no_random_seed_generator" ## 1 config -- MgO -- Hrushi_new -- no random seed -- FCS & Input files
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/block_2023-07-18-10-18-04-627432/launcher_2023-07-18-10-54-47-247213/launcher_2023-07-18-10-56-22-067687" ## 1 config -- Li2O -- Hrushi_new -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/block_2023-07-18-10-18-04-627432/launcher_2023-07-18-15-52-11-195765/launcher_2023-07-18-15-52-29-448168" ## 1 config -- Li2O -- Hrushi_new -- no random seed -- FCS & Input files
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/block_2023-07-18-10-18-04-627432/launcher_2023-07-26-18-46-46-526797/launcher_2023-07-26-18-48-24-247079" ## 1 config -- 3 harmonic structs -- Li2O -- Hrushi_new -- no random seed
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting" ## 1 config -- harmonic + anharmonic structs -- Li2O -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files 
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_only_harmonic" ## 1 config -- 3 harmonic structs -- Li2O -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files 
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_harmonic_0,008" ## 1 config -- 1 harmonic struct -- Li2O -- 0.008 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files 
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files 
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1_ardr" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- ardr
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1_bayesian-ridge" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- bayesian-ridge
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1_least-squares" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- least-squares
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1_omp" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- omp
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1_ridge" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- ridge
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1_split-bregman" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- split-bregman
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_one_anharmonic_0,1_cutoff_5,3,2_rfe" ## 1 config -- 1 anharmonic struct -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_MC_Rattle_std_0,01_cutoff_5,3,2_rfe" ## 1 config -- 3 MC Rattle -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_MC_Rattle_std_0,01_cutoff_5,3,2_rfe_only_harmonic" ## 1 config -- 3 MC Rattle -- Li2O -- 0.1 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe --only harmonics 2nd order
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_fixed_displ_Rattle_std_0,002_cutoff_5,3,2_rfe_3_configs" ## 3 config -- 1 Fixed displ -- Li2O -- 0.002 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_fixed_displ_Rattle_std_0,002_cutoff_5,3,2_rfe_20_configs" ## 20 config -- 1 Fixed displ -- Li2O -- 0.002 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_fixed_displ_Rattle_std_0,002_cutoff_5,3,2_rfe_20_configs_lattice_vector_9" ## 20 config -- 1 Fixed displ -- Li2O -- 0.002 -- supercell_lattice_vector 9 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe
        # # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/manual_hiphive_fitting_fixed_displ_Rattle_std_0,002_cutoff_5,3,2_rfe_20_configs_lattice_vector_666" ## 20 config -- 1 Fixed displ -- Li2O -- 0.002 -- supercell_lattice_vector 6*6*6 -- Hrushi_new -- no random seed -- manual hiphive fitting -- FCS & Input files -- cutoff 5,3,2 -- rfe
        fw_fit_force_constant = RunHiPhive(
            fit_method = fit_method,
            disp_cut = disp_cut,
            bulk_modulus = bulk_modulus,
            temperature_qha = temperature_qha,
            mesh_density = mesh_density,
            imaginary_tol = imaginary_tol,
            prev_dir_json_saver=json_saver.output[3],
            # prev_dir_json_saver=prev_dir_json_saver,
            loop = loops
        ) 
        fw_fit_force_constant.name += f" {loops}"
        jobs.append(fw_fit_force_constant)
        outputs.append(fw_fit_force_constant.output)   
        fw_fit_force_constant.metadata.update({"tag": [f"mp_id={mpid}", f"fw_fit_force_constant_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})
        
        fw_fc_pdos_pb_to_db = RunFCtoPDOS(
            renormalized = renormalize,
            renorm_temperature = renormalize_temperature,
            mesh_density = mesh_density,
            prev_dir_json_saver=fw_fit_force_constant.output[4],
            # prev_dir_json_saver = prev_dir_json_saver,
            loop = loops
        ) 
        fw_fc_pdos_pb_to_db.name += f" {loops}"
        jobs.append(fw_fc_pdos_pb_to_db)
        outputs.append(fw_fc_pdos_pb_to_db.output)   
        fw_fc_pdos_pb_to_db.metadata.update({"tag": [f"mp_id={mpid}", f"fw_fit_force_constant_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})
 

        # ##### 5. Quality Control Job to check if the desired Test RMSE is achieved, if not, then increase the number of structures -- Using "addintion" feature of jobflow  
        # # loops = 2
        # # n_structures+=2
        # loops+=1
        # n_structures+=1
        # print(f'Number of structures increased to {n_structures}')
        # print(f'loop = {loops}')
        # prev_dir_json_saver="/pscratch/sd/h/hrushi99/atomate2/Li2O/block_2023-07-18-10-18-04-627432/launcher_2023-07-18-14-07-29-351734/launcher_2023-07-18-14-12-24-962576"
        # error_check_job = QualityControl(
        #     # rmse_test = fw_fit_force_constant.output[5],
        #     rmse_test = 0.011,
        #     n_structures = n_structures,
        #     rattle_std = rattle_std,
        #     loop = loops,
        #     fit_method = fit_method,
        #     disp_cut = disp_cut,
        #     bulk_modulus = bulk_modulus,
        #     temperature_qha = temperature_qha,
        #     mesh_density = mesh_density,
        #     imaginary_tol = imaginary_tol,
        #     # prev_dir_json_saver = json_saver.output[3],
        #     prev_dir_json_saver = prev_dir_json_saver,
        #     prev_vasp_dir = prev_vasp_dir,
        #     supercell_matrix_kwargs = supercell_matrix_kwargs
        # )
        # error_check_job.name += f" {loops}"
        # jobs.append(error_check_job)
        # outputs.append(error_check_job.output)   
        # error_check_job.metadata.update({"tag": [f"error_check_job_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})


        # prev_dir_hiphive="/pscratch/sd/h/hrushi99/atomate2/Li2O/block_2023-07-18-10-18-04-627432/launcher_2023-07-27-17-40-32-100303/launcher_2023-07-27-17-42-08-878096/1_phono3py/block_2023-07-27-22-11-55-140470/launcher_2023-07-27-23-06-53-141489/launcher_2023-07-27-23-13-10-251503"
        # #### 6. Renormalization (pass_inputs like bulk modulus)
        # if renormalize:
        #     for temperature in renormalize_temperature:
        #         nconfig = renormalize_nconfig*(1+temperature//100)
        #         fw_renormalization = RenormalizationFW(
        #             temperature = temperature,
        #             renorm_method = renormalize_method,
        #             nconfig = nconfig,
        #             conv_thresh = renormalize_conv_thresh,
        #             max_iter = renormalize_max_iter,
        #             renorm_TE_iter = renormalize_thermal_expansion_iter,
        #             bulk_modulus = bulk_modulus,
        #             mesh_density = mesh_density,
        #             # prev_dir_hiphive = fw_fit_force_constant.output[4],
        #             prev_dir_hiphive = prev_dir_hiphive,
        #             # prev_dir_struct = json_saver.output[3]
        #             # prev_dir_struct = prev_dir_struct,
        #             loop = loops
        #             )  

        #         fw_renormalization.name += f" {temperature} {loops}"
        #         jobs.append(fw_renormalization)
        #         outputs.append(fw_renormalization.output) 

        # # #### 7. Lattice thermal conductivity calculation using phono3py
        # # if calculate_lattice_thermal_conductivity:
        # #     fw_lattice_conductivity = RunPhono3py(
        # #         # phono3py_cmd=phono3py_cmd,
        # #         renormalized=renormalize,
        # #         loop=loops,
        # #         # prev_dir_hiphive=prev_dir_hiphive,
        # #         prev_dir_hiphive=fw_fit_force_constant.output[4],
        # #         )
        # # else:
        # #     pass

        # # fw_lattice_conductivity.name += f" {loops}"
        # # jobs.append(fw_lattice_conductivity)
        # # outputs.append(fw_lattice_conductivity.output)  
        # # fw_lattice_conductivity.metadata.update({"tag": [f"mp_id={mpid}", f"fw_lattice_conductivity_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})


        # # loops = 2   
        # prev_dir_hiphive="/pscratch/sd/h/hrushi99/atomate2/compounds_with_computed_CTE/block_2023-07-28-07-46-39-700826/launcher_2023-07-28-17-32-03-049173/launcher_2023-07-28-17-32-27-055936"
        # #### 8. Lattice thermal conductivity calculation
        # if calculate_lattice_thermal_conductivity:
        #     if renormalize:
        #         temperatures = renormalize_temperature
        #     else:
        #         temperatures = thermal_conductivity_temperature
        #     # Because of the way ShengBTE works, a temperature array that is not
        #     # evenly spaced out (T_step) requires submission for each temperature
        #     if not renormalize:
        #         if type(temperatures)==dict:
        #             pass
        #         elif type(temperatures) in [list,np.ndarray]:
        #             assert all(np.diff(temperatures)==np.diff(temperatures)[0])
        #         fw_lattice_conductivity = LatticeThermalConductivityFW(
        #             # db_file=db_file,
        #             shengbte_cmd=shengbte_cmd,
        #             renormalized=renormalize,
        #             temperature=temperatures,
        #             loop=loops,
        #             prev_dir_hiphive=prev_dir_hiphive,
        #             # prev_dir_hiphive=fw_fit_force_constant.output[4],
        #         )
        #     else:
        #         push = 1
        #         for t,T in enumerate(temperatures):
        #             if T == 0:
        #                 push = 0
        #                 continue
        #             fw_lattice_conductivity = LatticeThermalConductivityFW(
        #                 # db_file=db_file,
        #                 shengbte_cmd=shengbte_cmd,
        #                 renormalized=renormalize,
        #                 temperature=T,
        #                 loop=loops,
        #                 prev_dir_hiphive=prev_dir_hiphive,
        #                 # prev_dir_hiphive=fw_fit_force_constant.output[4],
        #             )

        # fw_lattice_conductivity.name += f" {loops}"
        # jobs.append(fw_lattice_conductivity)
        # outputs.append(fw_lattice_conductivity.output)  
        # fw_lattice_conductivity.metadata.update({"tag": [f"fw_lattice_conductivity_{loops}", f"nConfigsPerStd={n_structures}", f"rattleStds={rattle_std}", f"dispCut={disp_cut}", f"supercell_matrix_kwargs={supercell_matrix_kwargs}", f"loop={loops}"]})


        return Flow(jobs=jobs,
                    output=outputs,
                    # name=self.name,
                    name=f"{mpid}_ShengBTE")
     

