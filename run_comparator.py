"""Runs the comparator on specified YAML files and outputs the differences, alternatively you can run python comparator.py directly with the appropriate arguments."""
import comparator

comparator.main("./Extractor_Outputs/test_v0_v1_diff.yaml", "./Extractor_Outputs/test_v1_v2_diff.yaml", "./Comparator_Outputs/test_v0_v1_name_differences.txt", "./Comparator_Outputs/test_v0_v1_requirements_differences.txt")

comparator.main("./Extractor_Outputs/docker-cis-v0.yaml", "./Extractor_Outputs/docker-cis-v1.yaml", "./Comparator_Outputs/v0_v1_name_differences.txt", "./Comparator_Outputs/v0_v1_requirements_differences.txt")
comparator.main("./Extractor_Outputs/docker-cis-v1.yaml", "./Extractor_Outputs/docker-cis-v2.yaml", "./Comparator_Outputs/v1_v2_name_differences.txt", "./Comparator_Outputs/v1_v2_requirements_differences.txt")
