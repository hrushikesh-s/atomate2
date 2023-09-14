import pytest
from pymatgen.core import Lattice, Species, Structure

from atomate2.vasp.sets.core import StaticSetGenerator


@pytest.fixture(scope="module")
def struct_no_magmoms() -> Structure:
    """Dummy FeO structure with no magnetic moments defined."""
    return Structure(
        lattice=Lattice.cubic(3),
        species=["Fe", "O"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture(scope="module")
def struct_with_spin() -> Structure:
    """Dummy FeO structure with spins defined."""
    fe = Species("Fe2+", spin=4)
    o = Species("O2-", spin=0.63)

    return Structure(
        lattice=Lattice.cubic(3),
        species=[fe, o],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )


@pytest.fixture(scope="module")
def struct_with_magmoms(struct_no_magmoms) -> Structure:
    """Dummy FeO structure with magmoms defined."""
    struct = struct_no_magmoms.copy()
    struct.add_site_property("magmom", [4.7, 0.0])
    return struct


def test_user_incar_settings():
    structure = Structure([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ["H"], [[0, 0, 0]])

    # check to see if user incar settings (even when set to nonsensical values, as done
    # below) are always preserved.
    uis = {
        "ALGO": "VeryFast",
        "EDIFF": 1e-30,
        "EDIFFG": -1e-10,
        "ENAUG": 20000,
        "ENCUT": 15000,
        "GGA": "PE",
        "IBRION": 1,
        "ISIF": 1,
        "ISPIN": False,  # wrong type, should be integer (only 1 or 2)
        "LASPH": False,
        "ISMEAR": -2,
        "LCHARG": 50,  # wrong type, should be bool.
        "LMIXTAU": False,
        "LORBIT": 14,
        "LREAL": "On",
        "MAGMOM": {"H": 100},
        "NELM": 5,
        "NELMIN": 10,  # should not be greater than NELM
        "NSW": 5000,
        "PREC": 10,  # wrong type, should be string.
        "SIGMA": 20,
    }

    static_set_generator = StaticSetGenerator(user_incar_settings=uis)
    incar = static_set_generator.get_input_set(structure, potcar_spec=True).incar

    for key in uis:
        if isinstance(incar[key], str):
            assert incar[key].lower() == uis[key].lower()
        else:
            assert incar[key] == uis[key]


@pytest.mark.parametrize(
    "structure,user_incar_settings",
    [
        ("struct_no_magmoms", {}),
        ("struct_with_magmoms", {}),
        ("struct_with_spin", {}),
        ("struct_no_magmoms", {"MAGMOM": [3.7, 0.8]}),
        ("struct_with_magmoms", {"MAGMOM": [3.7, 0.8]}),
        ("struct_with_spin", {"MAGMOM": [3.7, 0.8]}),
    ],
)
def test_incar_magmoms_precedence(structure, user_incar_settings, request) -> None:
    """
    According to VaspInputGenerator._get_magmoms, the magmoms for a new input set are
    determined given the following precedence:

    1. user incar settings
    2. magmoms in input struct
    3. spins in input struct
    4. job config dict
    5. set all magmoms to 0.6

    Here, we use the StaticSetGenerator as an example, but any input generator that has
    an implemented get_incar_updates() method could be used.
    """
    structure = request.getfixturevalue(structure)

    input_gen = StaticSetGenerator(user_incar_settings=user_incar_settings)
    incar = input_gen.get_input_set(structure, potcar_spec=True).incar
    incar_magmom = incar["MAGMOM"]

    has_struct_magmom = structure.site_properties.get("magmom")
    has_struct_spin = getattr(structure.species[0], "spin", None) is not None

    if user_incar_settings:  # case 1
        assert incar_magmom == user_incar_settings["MAGMOM"]
    elif has_struct_magmom:  # case 2
        assert incar_magmom == structure.site_properties["magmom"]
    elif has_struct_spin:  # case 3
        assert incar_magmom == [s.spin for s in structure.species]
    else:  # case 4 and 5
        assert incar_magmom == [
            input_gen.config_dict["INCAR"]["MAGMOM"].get(str(s), 0.6)
            for s in structure.species
        ]
