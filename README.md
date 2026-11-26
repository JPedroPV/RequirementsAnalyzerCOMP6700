# Team Members

| Name              | BannerID  | Email              |
| ----------------- | --------- | ------------------ |
| Joe Lin           | wha       | jzl0277@auburn.edu |
| Jeremy Wang       |           | jdw0134@auburn.edu |
| Pedro Vasconcelos | 904197253 | jpv0016@auburn.edu |

# Usage

## Pip

To install our packages with pip, we recommend installing and running from a python  virtual environment

To create the virtual environment do:

```Shell
python -m venv .venv
```

(this will be named .venv and create a directory of the same name)

To activate this venv do:

```Shell
source .venv/bin/activate
```

To install the packages needed to run our scripts do this while in a venv:

```Shell
pip install -r requirements.txt
```

And when you're done and wish the leave the venv run:

```Shell
deactivate
```

## Conda

Assuming you have conda installed to use conda with our project, create the environment with the packages needed with:

```Shell
conda env create --file environment.yml
```

And activate it with:

```Shell
conda activate requirements-analyzer
```

Deactivate by doing:

```Shell
conda deactivate
```

# Task-1: Extractor

*The LLM we are using for Task 1 is: [Joe please put the name of the llm here]*

Lorum ipsum

# Task-2: Comparator

## Usage

```Shell
python comparator.py <file1.yaml> <file2.yaml> <name_diff_txt> <req_diff_txt>
```

## Description

This takes in two yamls (outputs from Extractor), compares them finding what is the difference between the names of the key data elements between the two inputs given by <file1.yaml> and <file2.yaml>.

It starts by listing what names were in each file that wasn't in the other and putting the result into a filename that is passed in the arg <name_diff_txt>. If there are no element names unique to a file, the output txt will contain: `NO DIFFERENCES IN REGARDS TO ELEMENT NAMES`

Then, it does the same to the requirements attached to each key data element and then puts the unique ones into the file specified by <req_diff_txt>. If there are no requirements unique to a file, the output txt will contain: `NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS`

If you wish to set the args and run this task on a lot of inputs you can preset, you can call the main function and specify the args in `run_comparator.py` and just run that.

# Task-3: Executor

lolololol good luck

# Task-4: Repository Organization

Something something github actions

The binary that can be used to run the entire project is called by:

```Shell
./run_everything
```
