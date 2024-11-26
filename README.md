# pymatlib


Pymatlib is a Python library designed to simulate and analyze material properties. 
It provides tools for modeling alloys, interpolating material properties, and more.


<!-- # TODO
![Build Status](https://img.shields.io/github/actions/workflow/status/rahil.doshi/pymatlib/build.yml)
-->
[![Pipeline Status](https://i10git.cs.fau.de/rahil.doshi/pymatlib/badges/master/pipeline.svg)](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/pipelines)
![License](https://img.shields.io/badge/license-GPLv3-blue)


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Classes](#classes)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## Installation
To install `pymatlib`, you can use pip. It is recommended to install it in a virtual environment:
```bash
pip install -e .
```
This command installs the package in editable mode, allowing you to make changes without needing to reinstall.

### Installation Requirements
- **Python version**: >= 3.10
- Required packages:
    - `numpy`
    - `sympy`
    - `pytest`
    - [`pystencils`](https://i10git.cs.fau.de/pycodegen/pystencils/-/tree/v2.0-dev?ref_type=heads)

### Installation Steps
1. Clone the repository:
```bash
git clone https://i10git.cs.fau.de/rahil.doshi/pymatlib.git
cd pymatlib
```
2. Install the package:
```bash
pip install -e .
```
This installs pymatlib in editable mode for development.


## Usage
Here are some examples of how to use the library:

### Example 1: Creating an Alloy
```bash
from pymatlib.core.alloy import Alloy

alloy = Alloy(
  elements=['Fe', 'Cr'], 
  composition=[0.7, 0.3], 
  temperature_solidus=1700.0, 
  temperature_liquidus=1800.0)
print(alloy)
# Output: Alloy(Fe: 70%, Cr: 30%, solidus: 1700.0 K, liquidus: 1800.0 K)
```

### Example 2: Interpolating Material Properties
```bash
from pymatlib.core.interpolators import interpolate_property

T = 1400.0
temp_array = [1300.0, 1400.0, 1500.0]
prop_array = [100.0, 200.0, 300.0]

result = interpolate_property(T, temp_array, prop_array)
print(result)
# Output: 200.0
```


## Features
- **Material Modeling**: Define alloys and their properties with ease. Supports various material properties including density, thermal conductivity, and heat capacity.
- **Interpolation**: Interpolate material properties across temperature ranges. Provides functions for interpolation of properties based on temperature.
- **Symbolic Calculations**: Utilize symbolic math with tools like sympy. Includes data structures for managing assignments and symbolic variables.
- **Extensibility**: Add new material properties and interpolators as needed.


## Classes

### Alloy
A dataclass for alloys, with properties like:
- ```elements```
- ```composition```
- ```temperature_solidus```
- ```temperature_liquidus```

### Assignment
A dataclass representing an assignment operation with:
- lhs: Left-hand side (symbolic variable).
- rhs: Right-hand side (expression or tuple).
- lhs_type: Type of the left-hand side.

### MaterialProperty
A dataclass representing a material property that can be evaluated as a function of a symbolic variable (e.g., temperature) and includes symbolic assignments.


## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch:
```bash
git checkout -b feature/your-feature
```
3. Make your changes and commit:
```bash
git commit -m "Add new feature"
```
4. Push to the branch:
```bash
git push origin feature/your-feature
```
5. Open a pull request.


## License
This project is licensed under the GNU General Public License v3 (GPLv3). See the [LICENSE](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/blob/master/LICENSE?ref_type=heads) file for details.


## Contact
For inquiries or issues:
- **Author**: Rahil Doshi 
- **Email**: rahil.doshi@fau.de
- **Project Homepage**: [pymatlib](https://i10git.cs.fau.de/rahil.doshi/pymatlib)
- **Bug Tracker**: [Issues](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/issues)


<!--
## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://i10git.cs.fau.de/rahil.doshi/pymatlib.git
git branch -M master
git push -uf origin master
```

## Integrate with your tools

- [ ] [Set up project integrations](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
-->