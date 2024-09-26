"""Module that writes out all outputs by default for APT SQL files.

Authors
-------
    - Mees Fix
"""

import warnings

import argparse
import pathlib

from jwst_rogue_path_tool.constants import PROJECT_DIRNAME
from jwst_rogue_path_tool.detect_claws import aptProgram


def process_programs(apt_programs, output_directory):
    """Process and write output from program."""
    for apt_program in apt_programs:
        print(
            f"Processing program file {apt_program} to output directory {output_directory}"
        )
        try:
            program = aptProgram(apt_program, angular_step=1.0)
            program.run()
            supported_observations = program.observations.supported_observations.keys()

            for obs_number in supported_observations:
                observation = program.observations.data[obs_number]
                program.plot_exposures(observation, output_directory)
                program.plot_observation(observation, output_directory)
                program.plot_v3pa_vs_flux(observation, output_directory)
                program.make_report(observation, output_directory)

            del program
        except Exception as e:
            warnings.warn(f"EXCEPTION FOR PROGRAM {apt_program}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, help="Filename or filename search pattern"
    )
    parser.add_argument("--output_dir", type=str, help="Directory to write output")
    args = parser.parse_args()

    if not args.output_dir:
        raise Exception("No output directory specified!")
    else:
        output_dir = pathlib.Path(args.output_dir)
        if not output_dir.is_dir():
            raise Exception(f"{args.output_dir} is not a directory!")

    files_path = pathlib.Path(args.filename)
    if files_path.is_file():
        process_programs([files_path], output_dir)
    else:
        file_path = files_path.parent
        file_pattern = files_path.name
        all_files = file_path.glob(file_pattern)
        print(all_files, file_pattern, file_path)
        process_programs(all_files, output_dir)
