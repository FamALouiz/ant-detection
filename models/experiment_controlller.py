import mlflow as mlf

def start_new_experiment(experiment_name: str) -> str:
    '''
    This function starts a new experiment in MLflow if it does not exist else return the id of the existing experiment.
    
    Parameters:
        - experiment_name (str): The name of the experiment.
    
    Returns:
        - str: The id of the experiment.
    '''

    try: 
        experiment_id = mlf.create_experiment(experiment_name)
    except mlf.exceptions.RestException:
        experiment_id = mlf.get_experiment_by_name(experiment_name).experiment_id
        
    return experiment_id

