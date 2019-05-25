# Python-Project
This repo contains code to deploy pure python projects as PEX (refer to vision project). PEX can deploy the project with dependencies inside it.

<strong>movies:</strong> a bare template to convert a python project into executable zip. <br />
To convert into zip, run: `python -m zipapp movies` from root directory (current directory).

To make the zip executable, it's important to have `__main__.py` script in the root of the project. There are no dependencies installed in the zip, should run it inside a conda environment.

