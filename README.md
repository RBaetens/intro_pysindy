# An introduction to pySINDy
An introductory notebook to system indentification with pySINDy, as well as some simulation tools to generate example problems.

## Get started

1) Clone the repository into the desired directory.

    - any terminal
        ```bash
        git clone https://github.com/RBaetens/intro_pysindy
        ```


2) Make sure you have a recent python version, and the following VS code extensions installed.

    - Python
    - Pylance
    - Python Environments
    - Jupyter


3) Set up your python environment in VS code.

    - bash/Linux terminal
        ```bash
        py -m venv ENV
        source ENV/Scripts/activate
        pip install -r requirements.txt
        ```

        Or for a specific python version:

        ```bash
        py -3.13 -m venv ENV
        source ENV/Scripts/activate
        pip install -r requirements.txt
        ```
        
    - cmd/Windows terminal
        ```cmd
        py -m venv ENV
        ENV/Scripts/activate.bat
        pip install -r requirements.txt
        ```
        Or for a specific python version:

        ```cmd
        py -3.13 -m venv ENV
        ENV/Scripts/activate.bat
        pip install -r requirements.txt
        ```


4) Select your python environment as the interpreter.

    Open the command palette, `ctrl + shift + P`, click 'Python: select interpreter' and choose your new environment.


5) Run a notebook.

    Select your new environment as the kernel (top right corner if you have a notebook open) for the notebook you want to run and start exploring!
