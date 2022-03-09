import io
import json
import os

import numpy as np

# the file containing the database of materials
_materials_database_fn = os.path.join(os.path.dirname(__file__), "data/materials.json")

materials_absorption_table = {
    "anechoic": {"description": "Anechoic material", "coeffs": [1.0]},
}

materials_scattering_table = {
    "no_scattering": {"description": "No scattering", "coeffs": [0.0]},
}


with io.open(_materials_database_fn, "r", encoding="utf8") as f:
    materials_data = json.load(f)

    center_freqs = materials_data["center_freqs"]

    tables = {
        "absorption": materials_absorption_table,
        "scattering": materials_scattering_table,
    }

    for key, table in tables.items():
        for subtitle, contents in materials_data[key].items():
            for keyword, p in contents.items():
                table[keyword] = {
                    "description": p["description"],
                    "coeffs": p["coeffs"],
                    "center_freqs": center_freqs[: len(p["coeffs"])],
                }

class Material(object):
    """
    A class that describes the energy absorption and scattering
    properties of walls.

    Attributes
    ----------
    energy_absorption: dict
        A dictionary containing keys ``description``, ``coeffs``, and
        ``center_freqs``.
    scattering: dict
        A dictionary containing keys ``description``, ``coeffs``, and
        ``center_freqs``.

    Parameters
    ----------
    energy_absorption: float, str, or dict
        * float: The material created will be equally absorbing at all frequencies
            (i.e. flat).
        * str: The absorption values will be obtained from the database.
        * dict: A dictionary containing keys ``description``, ``coeffs``, and
            ``center_freqs``.
    scattering: float, str, or dict
        * float: The material created will be equally scattering at all frequencies
            (i.e. flat).
        * str: The scattering values will be obtained from the database.
        * dict: A dictionary containing keys ``description``, ``coeffs``, and
            ``center_freqs``.
    """

    def __init__(self, energy_absorption, scattering=None):

        # Handle the energy absorption input based on its type
        if isinstance(energy_absorption, (float, np.float32, np.float64)):
            # This material is flat over frequencies
            energy_absorption = {"coeffs": [energy_absorption]}

        elif isinstance(energy_absorption, str):
            # Get the coefficients from the database
            energy_absorption = dict(materials_absorption_table[energy_absorption])

        elif not isinstance(energy_absorption, dict):
            raise TypeError(
                "The energy absorption of a material can be defined by a scalar value "
                "for a flat absorber, a name refering to a material in the database, "
                "or a list with one absoption coefficients per frequency band"
            )

        if scattering is None:
            # By default there is no scattering
            scattering = 0.0

        if isinstance(scattering, (float, np.float32, np.float64)):
            # This material is flat over frequencies
            # We match the number of coefficients for the absorption
            if len(energy_absorption["coeffs"]) > 1:
                scattering = {
                    "coeffs": [scattering] * len(energy_absorption["coeffs"]),
                    "center_freqs": energy_absorption["center_freqs"],
                }
            else:
                scattering = {"coeffs": [scattering]}

        elif isinstance(scattering, str):
            # Get the coefficients from the database
            scattering = dict(materials_scattering_table[scattering])

        elif not isinstance(scattering, dict):
            # In all other cases, the material should be a dictionary
            raise TypeError(
                "The scattering of a material can be defined by a scalar value "
                "for a flat absorber, a name refering to a material in the database, "
                "or a list with one absoption coefficients per frequency band"
            )

        # Now handle the case where energy absorption is flat, but scattering is not
        if len(scattering["coeffs"]) > 1 and len(energy_absorption["coeffs"]) == 1:
            n_coeffs = len(scattering["coeffs"])
            energy_absorption["coeffs"] = energy_absorption["coeffs"] * n_coeffs
            energy_absorption["center_freqs"] = list(scattering["center_freqs"])

        # checks for `energy_absorption` dict
        assert isinstance(energy_absorption, dict), (
            "`energy_absorption` must be a "
            "dictionary with the keys "
            "`coeffs` and `center_freqs`."
        )
        assert "coeffs" in energy_absorption.keys(), (
            "Missing `coeffs` keys in " "`energy_absorption` dict."
        )
        if len(energy_absorption["coeffs"]) > 1:
            assert len(energy_absorption["coeffs"]) == len(
                energy_absorption["center_freqs"]
            ), (
                "Length of `energy_absorption['coeffs']` and "
                "energy_absorption['center_freqs'] must match."
            )

        # checks for `scattering` dict
        assert isinstance(scattering, dict), (
            "`scattering` must be a "
            "dictionary with the keys "
            "`coeffs` and `center_freqs`."
        )
        assert "coeffs" in scattering.keys(), (
            "Missing `coeffs` keys in " "`scattering` dict."
        )
        if len(scattering["coeffs"]) > 1:
            assert len(scattering["coeffs"]) == len(scattering["center_freqs"]), (
                "Length of `scattering['coeffs']` and "
                "scattering['center_freqs'] must match."
            )

        self.energy_absorption = energy_absorption
        self.scattering = scattering

    def is_freq_flat(self):
        """
        Returns ``True`` if the material has flat characteristics over
        frequency, ``False`` otherwise.
        """
        return (
            len(self.energy_absorption["coeffs"]) == 1
            and len(self.scattering["coeffs"]) == 1
        )

    @property
    def absorption_coeffs(self):
        """shorthand to the energy absorption coefficients"""
        return self.energy_absorption["coeffs"]

    @property
    def scattering_coeffs(self):
        """shorthand to the scattering coefficients"""
        return self.scattering["coeffs"]

    def resample(self, octave_bands):
        """resample at given octave bands"""
        self.energy_absorption = {
            "coeffs": octave_bands(**self.energy_absorption),
            "center_freqs": octave_bands.centers,
        }
        self.scattering = {
            "coeffs": octave_bands(**self.scattering),
            "center_freqs": octave_bands.centers,
        }

    @classmethod
    def all_flat(cls, materials):
        """
        Checks if all materials in a list are frequency flat

        Parameters
        ----------
        materials: list or dict of Material objects
            The list of materials to check

        Returns
        -------
        ``True`` if all materials have a single parameter, else ``False``
        """
        if isinstance(materials, dict):
            return all([m.is_freq_flat() for m in materials.values()])
        else:
            return all([m.is_freq_flat() for m in materials])