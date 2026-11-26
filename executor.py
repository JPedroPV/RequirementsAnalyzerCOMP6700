import csv
import os
import zipfile
import subprocess
import json
import pandas as pd

# --------------------------------------------------------------------------------------------
# Read text file from comparator and turn into dictionary
def read_txt(comp_file):
    sections = {}
    current_section = None

    with open(comp_file, "r") as file:
        for line in file:
            line = line.strip()

            # Check if there are no differences
            if line == "NO DIFFERENCES IN REGARDS TO ELEMENT NAMES" or line == "NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS":
                #print(line)
                return sections

            # Identify a new section
            if line.startswith("Requirements only in"):
                current_section = line.replace("Requirements only in ", "").replace(":", "")
                sections[current_section] = []
            elif line.startswith("Element names only in"):
                current_section = line.replace("Requirements only in ", "").replace(":", "")
                sections[current_section] = []

            # Parse current section
            elif line.startswith("- ") and current_section is not None:
                requirement = line[2:].strip()
                sections[current_section].append(requirement)

    #print(sections)
    return sections

# --------------------------------------------------------------------------------------------
# Find differences between TEXT file and map rules
def find_diff(name_file, req_file):
    # Read names and requirements files and convert to dictionaries
    names = read_txt(name_file)
    requirements = read_txt(req_file)

    # Manually create Hadolint rules and map it based on that, rules is a dictionary where each item is:
    # "Rule ID: "Key Words that imply violation"
    # Based on the github for dockerfile specifc benchmarks for CIS, we get these general rules:
    # CIS 4.1 Create a user for the container
    # CIS 4.2 Use trusted base images for containers (user provide trusted base image list)
    # CIS 4.3 Do not install unnecessary packages in the container (user provide the disallowed package list)
    # CIS 4.6 Add HEALTHCHECK instruction to the container image
    # CIS 4.7 Do not use update instructions alone in the Dockerfile
    # CIS 4.9 Use COPY instead of ADD in Dockerfile
    # CIS 4.10 Do not store secrets in Dockerfiles (user provide the secret pattern, only checks contents in ENV and LABEL instructions)
    # Then we derive our rules from the Hadolint rules based on the above benchmarks
    rules = {
        # 4.1
        "DL3002": ["last", "root"],
        "DL3066": ["non-numeric", "user-id", "resolvable", "host"],
        # 4.2
        "DL3006": ["tag", "version", "image", "explicitly"],
        "DL3007": ["latest", "image", "update", "pin", "version", "explicitly", "release", "tag"],
        "DL3026": ["allowed", "registry", "FROM", "image"],
        # 4.3
        "DL3015": ["avoid", "additional", "packages", "--no-install-recommends"],
        # 4.6
        "DL3057": ["HEALTHCHECK", "missing"],
        "DL3012": ["multiple", "HEALTHCHECK"],
        # 4.7
        "DL3019": ["--no-cache", "switch", "--update", "/var/cache/apk/*", "installing", "package"],
        # 4.9
        "DL3020": ["COPY", "ADD", "files", "folders"],
        "DL3010": ["ADD", "extract", "archives", "image"],
        # 4.10
        "DL3064": ["sensitive", "data", "ARG", "ENV"],
        # Additional rules from manually looking
        "DL3061": ["instruction", "order", "FROM", "ARG", "comment"],
        "DL3047": ["wget", "flag", "--progress", "bloated", "build", "logs", "download", "large"],
        "DL3067": ["copy", "filesystem", "stage"]
    }

    # Write to text file the mapped Hadolint rules based on found differences
    with open("differences.txt", "w", newline="") as file:
        # Check if names and requirements are empty, if they are then there are no differences
        if len(names) == 0 and len(requirements) == 0:
            file.write("NO DIFFERENCES FOUND")
        else:
            mapped_rules = []
            # Check names for keyword matches first
            for file_n, name in names.items():
                for rule, keywords in rules.items():
                    matches = [keyword for keyword in keywords
                               if any(keyword.lower() in line.lower() for line in name)]
                    if matches and rule not in mapped_rules:
                        mapped_rules.append(rule)

            # Check requirements for keyword matches
            for file_n, requirement in requirements.items():
                for rule, keywords in rules.items():
                    matches = [keyword for keyword in keywords 
                               if any(keyword.lower() in line.lower() for line in requirement)]
                    if matches and rule not in mapped_rules:
                        mapped_rules.append(rule)

            # Check if no mappings were found
            if mapped_rules:
                file.write("\n".join(mapped_rules))
            else:
                file.write("NO MAPPINGS FOUND")
    return file

# --------------------------------------------------------------------------------------------
# Execute Hadolint tool based on TEXT file
def exec_hadolint(zipped_docker, differences_file):
    # Check if the zipped dockerfiles already exists, if not unzip them
    if not os.path.exists("extracted_dockerfiles"):
        with zipfile.ZipFile(zipped_docker, "r") as zip_ref:
            zip_ref.extractall("extracted_dockerfiles")

    # Open differences TEXT file and check for contents to determine what rules to run
    with open(differences_file, "r") as file:
        rules = []
        for line in file:
            line = line.strip()
            # Run with all rules
            if line == "NO DIFFERENCES FOUND" or line == "NO MAPPINGS FOUND":
                continue
            # Run with only rules in the text file
            else:
                rules.append(line)

    # The way I implement the rules checking is by executing all rules then filtering at the end
    # Since there are so many files to check I am saving the df and checking if it already exists
    if os.path.exists("hadolint_results.parquet"):
        df = pd.read_parquet("hadolint_results.parquet")

    else:
        # Get all the dockerfiles to run Hadolint on
        dockerfiles = []
        for root, dirs, files in os.walk("extracted_dockerfiles"):
            for file in files:
                # Ignoring macOS metadata files
                if file.startswith("._"):
                    continue
                filepath = os.path.join(root, file)
                dockerfiles.append(filepath)
        print(f"There are {len(dockerfiles)} dockerfiles to check.")

        # Run hadolint subprocesses with the dockerfiles
        results = []
        for filepath in dockerfiles:
            result = subprocess.run(
                ["hadolint", "--format", "json", filepath],
                capture_output = True,
                text = True
            )
            if result.stdout.strip(): #Check if anything is found
                findings = json.loads(result.stdout) #Convert to python obj
                for finding in findings: #Add filepath field to data info
                    finding["file"] = filepath
                results.extend(findings)

        # Create pandas dataframe from hadolint results
        df = pd.DataFrame(results)
        #print(df)

        # Saves the data frame so that in the future you don't need to run it mutiple times
        # since all I'm doing is filtering for rules after running all of them
        df.to_parquet("hadolint_results.parquet", index=False)

    # Filter for rules
    if rules:
        df = df[df["code"].isin(rules)]
    #print(df)

    return df

# --------------------------------------------------------------------------------------------
# Generate CSV file
def generate_CSV(df, output):
    # Group together by file, severity, and rule IDs then count the number of instances
    csv_df = (df.groupby(["file", "level", "code"]).size().reset_index(name="COUNT"))

    # Rename headers
    csv_df = csv_df.rename(columns={
        "file": "FilePath",
        "level": "DefaultSeverity",
        "code": "RULEID"
    })

    # Turn pandas df to CSV
    csv_df.to_csv(output, index=False)
    return csv_df


# --------------------------------------------------------------------------------------------
#find_diff("./Comparator_Outputs/test_v0_v1_name_differences.txt", "./Comparator_Outputs/test_no_req.txt")
#find_diff("./Comparator_Outputs/test_v0_v1_name_differences.txt", "./Comparator_Outputs/test_v0_v1_requirements_differences.txt")
# find_diff("./Comparator_Outputs/v0_v1_name_differences.txt", "./Comparator_Outputs/v0_v1_requirements_differences.txt")
# df = exec_hadolint("dockerfiles.zip", "differences.txt")
# #df.to_csv("test.csv", index=False)
# generate_CSV(df, "executor_data.csv")

def main():
    # Test against all files, generating CSVs everytime
    find_diff("./Comparator_Outputs/test_v0_v1_name_differences.txt", "./Comparator_Outputs/test_no_req.txt")
    df = exec_hadolint("dockerfiles.zip", "differences.txt")
    generate_CSV(df, "./Executor_Outputs/test_v0_v1_executor_data.csv")

    find_diff("./Comparator_Outputs/v0_v1_name_differences.txt", "./Comparator_Outputs/v0_v1_requirements_differences.txt")
    df = exec_hadolint("dockerfiles.zip", "differences.txt")
    generate_CSV(df, "./Executor_Outputs/v0_v1_executor_data.csv")

    find_diff("./Comparator_Outputs/v1_v2_name_differences.txt", "./Comparator_Outputs/v1_v2_requirements_differences.txt")
    df = exec_hadolint("dockerfiles.zip", "differences.txt")
    generate_CSV(df, "./Executor_Outputs/v1_v2_executor_data.csv")

main()