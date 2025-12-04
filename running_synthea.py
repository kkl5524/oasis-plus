import subprocess
import os
import glob
import pandas as pd
import shutil

def run_synthea_jar(
    jar_path="synthea-with-dependencies.jar",
    main_output_dir="output/synthea_output",
    temp_output_dir="output/synthea_temp",
    state="Massachusetts",
    num_patients=1000,
    max_memory="4g"
):
    """
    Run Synthea multiple times and append CSVs to main output folder.

    Args:
        jar_path (str): Path to synthea-with-dependencies.jar
        main_output_dir (str): Folder where final CSVs will be stored
        temp_output_dir (str): Temporary folder for each Synthea run
        state (str): US state or module to simulate
        num_patients (int): Number of patients per batch
        max_memory (str): Java Xmx heap size
    """

    os.makedirs(main_output_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    for batch in range(20):  # Run 10 batches
        print(f"\n>>> Starting batch {batch + 1} of 10")

        cmd = [
            "java",
            f"-Xmx{max_memory}",
            "-jar", jar_path,
            "-p", str(num_patients),
            state,
            "-o", temp_output_dir,
            "--exporter.csv.export=true",
            f"--exporter.baseDirectory={temp_output_dir}",
            "--exporter.json.export=false",
            "--exporter.fhir.export=false"
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Batch {batch + 1} complete. Merging CSVs...")

            # Merge CSVs from temp_output_dir into main_output_dir
            for temp_file in glob.glob(os.path.join(temp_output_dir, "**", "*.csv"), recursive=True):
                filename = os.path.basename(temp_file)
                main_file = os.path.join(main_output_dir, filename)

                if os.path.exists(main_file):
                    df_old = pd.read_csv(main_file)
                    df_new = pd.read_csv(temp_file)
                    df_merged = pd.concat([df_old, df_new], ignore_index=True)
                    df_merged.to_csv(main_file, index=False)
                else:
                    shutil.move(temp_file, main_file)

            # Clear temp folder completely
            shutil.rmtree(temp_output_dir)
            os.makedirs(temp_output_dir, exist_ok=True)

        except subprocess.CalledProcessError as e:
            print(f"Error during batch {batch + 1}:", e)
            break

    print(f"\nAll batches complete. Final CSVs are in: {main_output_dir}")


if __name__ == "__main__":
    run_synthea_jar(
        jar_path="synthea/synthea-with-dependencies.jar",
        main_output_dir="output/synthea_output",
        temp_output_dir="output/synthea_temp",
        state="Massachusetts",
        num_patients=500,
        max_memory="8g"  # Increase memory for larger runs
    )
