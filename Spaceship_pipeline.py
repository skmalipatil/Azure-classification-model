# -----------------------------------------------------
# This is the Job Script/Run Configuration script for 
# building a pipeline and running it in an experiment
# -----------------------------------------------------

from azureml.core import Workspace

#access the workspace
ws = Workspace.from_config(path =  r'C:\Users\91888\OneDrive\Desktop\Azure\Azure_sdk')

# -----------------------------------------------------------------------------
# create custome environment
# -----------------------------------------------------------------------------
print('The environment creation has started')
from azureml.core import Environment
from azureml.core.environment import CondaDependencies

# create environment
Spaceship_env = Environment(name = 'Spaceship_Environment')

# create dependies object
# Create the Conda dependencies object
myenv_dep = CondaDependencies.create(
    conda_packages=['scikit-learn', 'pandas', 'pip'],  # Conda packages
    pip_packages=[
        'azureml-core',              # Core Azure ML functionality
        'azureml-pipeline-core',      # Core pipeline functionalities
        'azureml-pipeline-steps',     # Pipeline steps like PythonScriptStep
        'azureml-pipeline',           # General pipeline utilities
        'azureml-defaults',         
        'azureml-interpret'
    ]
)

Spaceship_env.python.conda_dependencies = myenv_dep

Spaceship_env.register(ws)
print('The environment creation has ended')

# ---------------------------------------------------
# Create a compute cluster for pipeline
# ---------------------------------------------------

print('Creating the compute cluster')
cluster_name = "Spaceship-pipeline-cluster"

from azureml.core.compute import AmlCompute

compute_config = AmlCompute.provisioning_configuration(
    vm_size = 'STANDARD_D2D_V5',
    max_nodes = 2
    )

from azureml.core.compute import ComputeTarget
compute_cluster = ComputeTarget.create(ws, cluster_name, compute_config)

compute_cluster.wait_for_completion()

print('Creating the compute cluster Completed')

# -----------------------------------------------------------------------------
# create run configuration for the script
# -----------------------------------------------------------------------------

from azureml.core.runconfig import RunConfiguration
run_config = RunConfiguration()

run_config.target = compute_cluster
run_config.environment =  Spaceship_env

# -----------------------------------------------------------------------------
# difining the pipeline
# -----------------------------------------------------------------------------
print('Defining the data pipeline')

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineData

input_ds = ws.datasets.get('Spaceship_Train')

datafolder = PipelineData('datafolder', datastore=ws.get_default_datastore())

# -----------------------------------------------------------------------------
# Step 01 - data preparation
# -----------------------------------------------------------------------------
print('Executing the data preparation pipeline started')

data_prep_step = PythonScriptStep(name = '01 Data Preparation',
                                  source_directory = '.',
                                  script_name = 'Spaceship_Dataprep.py',
                                  inputs = [input_ds.as_named_input('raw_data')],
                                  outputs = [datafolder],
                                  runconfig = run_config,
                                  arguments = ['--datafolder', datafolder]
    )

print('Executing the data preparation pipeline completed')
# -----------------------------------------------------------------------------
# Step 02
# -----------------------------------------------------------------------------

print('Executing the training pipeline started')

Training_step = PythonScriptStep(name='02 Training the model',
                                 source_directory='.',
                                 script_name='Spaceship_training.py',
                                 # inputs=[input_ds.as_named_input('raw_data')],
                                 inputs=[datafolder],
                                 runconfig=run_config,
                                 arguments=['--datafolder', datafolder])


print('Executing the training pipeline ended')

# -----------------------------------------------------------------------------
# configure and build the pipeline
# -----------------------------------------------------------------------------

steps = [data_prep_step, Training_step]

from azureml.pipeline.core import Pipeline

new_pipeline = Pipeline(workspace = ws, steps = steps)


# -----------------------------------------------------------------------------
# create the experiment and run the pipeline
# -----------------------------------------------------------------------------

from azureml.core import Experiment

new_experiment = Experiment(workspace = ws, name = 'Spaceship_PipelineExp01')
new_pipeline_run = new_experiment.submit(new_pipeline)

new_pipeline_run.wait_for_completion(show_output=True)


