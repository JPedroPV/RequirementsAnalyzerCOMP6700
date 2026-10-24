from pathlib import Path

import sys

import yaml

def get_yamls(path1, path2):
	with Path(path1).open(encoding="utf-8") as yaml_file1, Path(path2).open(encoding="utf-8") as yaml_file2:
		data1 = yaml.safe_load(yaml_file1)
		data2 = yaml.safe_load(yaml_file2)
		entries1 = data1 if isinstance(data1, list) else [data1]
		entries2 = data2 if isinstance(data2, list) else [data2]
		return entries1, entries2

def get_element_names(entries):
	return {str(entry["name"]) for entry in entries if isinstance(entry, dict) and "name" in entry}

def get_element_requirements(entries):
	requirements = set()
	for entry in entries:
		if isinstance(entry, dict):
			reqs = entry.get("requirements")
			if isinstance(reqs, list):
				requirements.update("{},{}".format(entry["name"], req) for req in reqs)
	return requirements

def main(file_one, file_two, name_out, req_out):
	entries1, entries2 = get_yamls(file_one, file_two)
	first_names = get_element_names(entries1)
	second_names = get_element_names(entries2)
	first_requirements = get_element_requirements(entries1)
	second_requirements = get_element_requirements(entries2)
	only_first_names = sorted(first_names - second_names)
	only_second_names = sorted(second_names - first_names)
	only_first_reqs = sorted(first_requirements - second_requirements)
	only_second_reqs = sorted(second_requirements - first_requirements)

	if only_first_names or only_second_names:
		if only_first_names:
			lines_names = [f"Element names only in {file_one}:"]
			lines_names.extend("- {}".format(name) for name in only_first_names)
		else:
			lines_names = [f"No element names only in {file_one}."]
		if only_second_names:
			lines_names.append(f"\nElement names only in {file_two}:")
			lines_names.extend("- {}".format(name) for name in only_second_names)
		else:
			lines_names.append(f"No element names only in {file_two}.")
	else:
		lines_names = [f"NO DIFFERENCES IN REGARDS TO ELEMENT NAMES"]
	with Path(name_out).open("w", encoding="utf-8") as f:
		f.write("\n".join(lines_names) + "\n")
	if only_first_reqs:
		lines_reqs = [f"Requirements only in {file_one}:"]
		lines_reqs.extend("- {}".format(req) for req in only_first_reqs)
	else:
		lines_reqs = [f"No requirements only in {file_one}."]
	if only_second_reqs:
		lines_reqs.append(f"\nRequirements only in {file_two}:")
		lines_reqs.extend("- {}".format(req) for req in only_second_reqs)
	else:
		lines_reqs.append(f"No requirements only in {file_two}.")
	with Path(req_out).open("w", encoding="utf-8") as f:
		f.write("\n".join(lines_reqs) + "\n")


if __name__ == "__main__":
	if len(sys.argv) != 5:
		sys.exit("Usage: comparator.py <file1> <file2> <name_diff_txt> <req_diff_txt>")
	FILE_ONE, FILE_TWO, NAME_OUT, REQ_OUT = sys.argv[1:5]
	main(FILE_ONE, FILE_TWO, NAME_OUT, REQ_OUT)
