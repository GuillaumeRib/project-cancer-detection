# Data analysis
- Document here the project: project-cancer-detection
- Description:
    * Objective - Predict presence of metastatic tissue on lymph node scan images
    * Data - Separated images between Train , Validation and Test splits together with creation of random samples (can be found here: gs://cancer-detection-small-datasets)
    * Models - Built and trained several Deep Learning Convolutional Neural Network (CNN) models including transfer learning (ie: VGG16, ResNet,        MobileNetV2) and using MLFlow
    * Front-End - Streamlit web-page browsing images for model prediction results.
- Data Source: Kaggle dataset sourced from former competition: https://www.kaggle.com/competitions/histopathologic-cancer-detection
- Type of analysis: Image recognition / Binary classification


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for project-cancer-detection in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/project-cancer-detection`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "project-cancer-detection"
git remote add origin git@github.com:{group}/project-cancer-detection.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
project-cancer-detection-run
```

# Install

Go to `https://github.com/{group}/project-cancer-detection` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/project-cancer-detection.git
cd project-cancer-detection
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
project-cancer-detection-run
```
