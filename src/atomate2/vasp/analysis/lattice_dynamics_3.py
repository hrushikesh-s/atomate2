from typing import List, Optional, Union, Dict

import os

# from atomate.common.firetasks.glue_tasks import (
#     CopyFiles,
#     CopyFilesFromCalcLoc,
#     PassCalcLocs,
# )

# from atomate.vasp.config import SHENGBTE_CMD
from atomate2.vasp.analysis.lattice_dynamics_4 import (
    # CollectPerturbedStructures,
    # ForceConstantsToDb,
    RunHiPhive,
    # RunHiPhiveRenorm,
    # RunShengBTE,
    # ShengBTEToDb
    )

from fireworks import Firework
from jobflow import job

from hiphive import ClusterSpace, ForceConstants

__author__ = "Alex Ganose, Junsoo Park"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov"


class FitForceConstantsFW(Firework):
    """
    Compile perturbed supercell calculations and fit force constants
    using hiPhive.
    Args:
        parents: Parent(s) of this Firework.
        name: Name of this FW.
        db_file: Path to a db file.
        cutoffs: A list of cutoffs to trial. If None, a set of trial cutoffs
            will be generated based on the structure (default).
        separate_fit: Boolean to determine whether harmonic and anharmonic fitting
            are to be done separately (True) or in one shot (False)
        disp_cut: if separate_fit true, determines the mean displacement of perturbed
            structure to be included in harmonic (<) or anharmonic (>) fitting  
        bulk_modulus: in GPa, necessary for thermal expansion
        imaginary_tol: Tolerance used to decide if a phonon mode is
            imaginary, in THz.
        fit_method: Method used for fitting force constants. This can be
            any of the values allowed by the hiphive ``Optimizer`` class.
        mesh_density: The density of the q-point mesh used to calculate the
            phonon density of states. See the docstring for the ``mesh``
            argument in Phonopy.init_mesh() for more details.
        **kwargs: Other kwargs that are passed to Firework.__init__.
    """

    def __init__(
        # self,
        fit_method: str,
        separate_fit: bool,
        disp_cut: float,
        bulk_modulus: float,
        imaginary_tol: float,
        mesh_density: float,
        temperature_qha: float,
        cutoffs: Optional[List[List[float]]] = None,
        name="Fit Force Constants",
        parents: Optional[Union[Firework, List[Firework]]] = None,
        db_file: str = None,
        **kwargs
    ):
        # collect_structures = CollectPerturbedStructures()
        
        fit_force_constants = RunHiPhive(
            cutoffs=cutoffs,
            separate_fit=separate_fit,
            fit_method=fit_method,
            disp_cut=disp_cut,
            bulk_modulus=bulk_modulus,
            temperature_qha=temperature_qha,
            imaginary_tol=imaginary_tol,
        )
        # to_db = ForceConstantsToDb(
        #     db_file=db_file, renormalized=False, mesh_density=mesh_density, additional_fields={}
        # )
        # pass_locs = PassCalcLocs(name=name)

        # tasks = [collect_structures, fit_force_constants, to_db, pass_locs]
        # super().__init__(tasks, parents=parents, name=name, **kwargs)


# class LatticeThermalConductivityFW(Firework):
#     """
#     Calculate the lattice thermal conductivity using ShengBTE.
#     Args:
#         name: Name of this FW.
#         prev_calc_dir: Path to a directory containing the force constant
#             information. Will override ``parents`` when collecting the force
#             constants to run ShengBTE.
#         db_file: Path to a db file.
#         shengbte_cmd: The name of the shengbte executable to run. Supports
#             env_chk.
#         renormalized: boolean to denote whether force constants are from 
#             phonon renormalization (True) or directly from fitting (False)
#         temperature: The temperature to calculate the lattice thermal
#             conductivity for. Can be given as a single float, or a
#             dictionary with the keys "min", "max", "step".
#         shengbte_control_kwargs: Options to be included in the ShengBTE
#             control file.
#         **kwargs: Other kwargs that are passed to Firework.__init__.
#     """

#     def __init__(
#         self,
#         shengbte_cmd: str,
#         temperature: Union[float, int, dict],
#         renormalized: bool,
#         name="Lattice Thermal Conductivity",
#         prev_calc_dir: Optional[str] = None,
#         db_file: str = None,
#         shengbte_control_kwargs: Optional[dict] = None,
#         **kwargs
#         ):

#         # files needed to run ShengBTE
#         if renormalized: # must check if FORCE_CONSTANTS_2ND_{T}K can be copied individually
#             assert type(temperature) in [float,int]
#             files = [
#                 "structure_data.json",
#                 "FORCE_CONSTANTS_2ND_{}K".format(temperature),
#                 "FORCE_CONSTANTS_3RD"
#             ]
# #            temperature_copy = temperature[:]
# #            for t,T in enumerate(temperature_copy):
# #                try:
# #                    files = files.append("FORCE_CONSTANTS_2ND_{}K".format(T))
# #                except:
# #                    logger.info("FORCE_CONSTANTS_2ND_{}K is missing".format(T))
# #                    logger.info("Renormalization must have failed at {} K".format(T))
# #                    logger.info("Cannot calculate thermal conductivity at {} K".format(T))
# #                    temperature.remove(T)
#             if prev_calc_dir:
#                 copy_files = CopyFiles(from_dir=prev_calc_dir, filenames=files)
#             else:
#                 copy_files = CopyFilesFromCalcLoc(calc_loc='Renormalization', filenames=files)
#             os.system('mv FORCE_CONSTANTS_2ND_{}K FORCE_CONSTANTS_2ND'.format(temperature))

#         else: # only the default files are needed
#             files = [
#                 "structure_data.json",
#                 "FORCE_CONSTANTS_2ND",
#                 "FORCE_CONSTANTS_3RD",
#             ]
#             if prev_calc_dir:
#                 copy_files = CopyFiles(from_dir=prev_calc_dir, filenames=files)
#             else:
#                 copy_files = CopyFilesFromCalcLoc(calc_loc='Fit Force Constants', filenames=files)

#         run_shengbte = RunShengBTE(
#             shengbte_cmd=shengbte_cmd,
#             renormalized=renormalized,
#             temperature=temperature,
#             control_kwargs=shengbte_control_kwargs,
#             )
            
#         shengbte_to_db = ShengBTEToDb(db_file=db_file, additional_fields={})

#         tasks = [copy_files, run_shengbte, shengbte_to_db]
#         if renormalized:
#             super().__init__(tasks, name=name+' at {}K'.format(temperature), **kwargs)
#         else:
#             super().__init__(tasks, name=name, **kwargs)



# class RenormalizationFW(Firework):        
#     """
#     Performs temperature-dependent phonon renormalization to obtain phonons
#     at finite temperatures. Can be used to stabilize dynamically unstable modes
                                                                                                                                                                                                
#     Args:
#         name: Name of this FW.
#         prev_calc_dir: Path to a directory containing the force constant
#             information. Will override ``parents`` when collecting the force
#             constants to run ShengBTE.
#         db_file: Path to a db file.
#         temperature: The temperature to perform phonon renormalization at
#             Can be given as a single float, or a dictionary with the keys 
#             "min", "max", "step".
#         shengbte_control_kwargs: Options to be included in the ShengBTE
#             control file.
#         **kwargs: Other kwargs that are passed to Firework.__init__.
#     """
    
#     def __init__(
#         self,
#         temperature: Union[float, Dict], 
#         renorm_method: str,
#         renorm_nconfig: int,
#         renorm_conv_thresh: float,
#         renorm_max_iter: float,
#         renorm_TE_iter: bool,
#         bulk_modulus: float,
#         mesh_density: float,
#         name="Renormalization",
#         prev_calc_dir: Optional[str] = None,
#         db_file: str = None,
#         **kwargs
#         ):
        
#         # files needed to run renormalization
#         files = ["cluster_space.cs","parameters.txt","force_constants.fcs",
#                  "structure_data.json","fitting_data.json","phonopy_params.yaml"]

#         if prev_calc_dir:
#             copy_files = CopyFiles(from_dir=prev_calc_dir, filenames=files)
#         else:
#             copy_files = CopyFilesFromCalcLoc(calc_loc="Fit Force Constants", filenames=files)

#         renorm_force_constants = RunHiPhiveRenorm(
#             renorm_temp=temperature,
#             renorm_method=renorm_method,
#             nconfig=renorm_nconfig,
#             conv_thresh=renorm_conv_thresh,
#             max_iter=renorm_max_iter,
#             renorm_TE_iter=renorm_TE_iter,
#             bulk_modulus=bulk_modulus,
#             **kwargs
#             )        

#         to_db = ForceConstantsToDb(
#             db_file=db_file, renormalized=True, mesh_density=mesh_density, additional_fields={}
# 	)
#         pass_locs = PassCalcLocs(name=name)

#         tasks = [copy_files, renorm_force_constants, to_db, pass_locs]
#         super().__init__(tasks, name=name, **kwargs)