# Natural Computing: Pacman Project

## Environment

We use Pipenv to manage our dependencies inside of a virtual environment. `Pipfile.lock` contains the exact versions and hashes of the dependencies which we used. To reproduce this environment, follow these steps:

* install Pipenv: `pip install --user pipenv`
* install depencies: `pipenv sync`
* enable virtual environment: `pipenv shell`

## Training Agents

`main.py` contains a Command Line Interface (CLI), which can be used to train and run the agents. It has a lot of parameters which can be used to tweak their behavior.

To see an overview over the available options, run `python main.py --help`.

For the simplest case, just run `python main.py`.
