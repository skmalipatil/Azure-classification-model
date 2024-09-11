# Import azureml classes

from azureml.core import Workspace

# access the workspace using config file
print('accessing the workspace for the job')
ws = Workspace.from_config(path =  r'C:\Users\91888\OneDrive\Desktop\Azure\Azure_sdk')

# -----------------------------------------------------------------------------
# create custom environment
# -----------------------------------------------------------------------------

from azureml.core import Environment
from azureml.core.environment import CondaDependencies

# Create environment
Spaceship_deploy_env = Environment(name = 'Spaceship_deploy_env')

# create dependency object
print('creating dependencies....')

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

Spaceship_deploy_env.python.conda_dependencies = myenv_dep

# register the environemt
print('Registering the environemnt')
Spaceship_deploy_env.register(ws)


# -----------------------------------------------------------------------------
# create azureml kubernatics cluster
# -----------------------------------------------------------------------------

from azureml.core.compute import AksCompute, ComputeTarget
cluster_name = 'space-cluster-1'

if cluster_name not in ws.compute_targets:
    print(cluster_name, 'does not exist in workspace creating new one')
    print('creating provisioning config for aks cluster...')
    
    aks_config = AksCompute.provisioning_configuration(location='southindia',
                                                       vm_size='STANDARD_D2D_V5',
                                                       agent_count = 1,
                                                       cluster_purpose='DevTest'
                                                       )
    
    print('creating the AKS cluster')
    production_cluster = ComputeTarget.create(ws, cluster_name, aks_config)
    
    production_cluster.wait_for_completion(show_output = True)
    
else:
    print(cluster_name, 'Exists, using it....')
    production_cluster = ws.compute_targets[cluster_name]
    
# -----------------------------------------------------------------------------
# create inference configuration
# -----------------------------------------------------------------------------

from azureml.core.model import InferenceConfig

print('creating inference configuration')
Inference_config = InferenceConfig(source_directory = './service_files',
                                   entry_script = 'scoring_script.py',
                                   environment = Spaceship_deploy_env)

# -----------------------------------------------------------------------------
# create service deployment configuration
# -----------------------------------------------------------------------------

from azureml.core.webservice import AksWebservice

print('create deployment configurationfor webservice')
deploy_config = AksWebservice.deploy_configuration(cpu_cores = 0.5,
                                                   memory_gb = 0.5)

# -----------------------------------------------------------------------------
# create and deploy webservice
# -----------------------------------------------------------------------------

from azureml.core.model import Model
model = ws.models['Spaceship_RandomForest']

print('deploying the webservice....')

service = Model.deploy(workspace = ws,
                       name = 'spaceship-service',
                       models = [model],
                       inference_config = Inference_config,
                       deployment_config = deploy_config,
                       deployment_target = production_cluster
                       )

service.wait_for_deployment(show_output = True)

# -----------------------------------------------------------------------------




#print(service.get_logs())





