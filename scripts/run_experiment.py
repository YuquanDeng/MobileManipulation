import sys
import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.append("../")
import infrastructure.pytorch_util as ptu
from infrastructure.utils import *
from infrastructure.logger import Logger
from agents.MLP_agent import MLPagent
from infrastructure.logger import Logger

class PolicyTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.device = ptu.init_gpu(use_gpu=not self.params['no_gpu'], gpu_id=self.params['which_gpu'])
        self.agent = MLPagent(self.params)

    def get_dataloader(self, preprocessed_features: bool, preprocessed_dataset: bool):
        """
        Load dataset.
        """
        dataset_dir = os.path.join(os.getcwd(), '../data/')
        if preprocessed_features:
            feature_dataset = load(filepath=dataset_dir + 'feature_dataset.pkl')
        else:
            # Load a1 dataset and extract features.
            if not preprocessed_dataset:
                a1_dataset = get_a1_dataset()
                save(a1_dataset, filepath=dataset_dir + 'a1_dataset.pkl')
            else:
                a1_dataset = load(filepath=dataset_dir + 'a1_dataset.pkl')
            
            resnet50 = pretrained_ResNet50(self.device)
            feature_dataset = {x: extract_batch_feature(self.device, resnet50, a1_dataset[x]) for x in ['part_1', 'part_2']}
            save(feature_dataset, filepath=dataset_dir + 'feature_dataset.pkl')

        train_loader, valid_loader = get_a1_dataloader(
            feature_dataset, 
            batch_size=self.params['batch_size'], 
            interval=self.params['interval']
            )
        return train_loader, valid_loader

    def run_training_loop(self):
        # Load dataset, Train agent, and log results.
        train_loader, valid_loader = self.get_dataloader(self.params['preprocessed_features'], self.params['preprocessed_dataset'])
        all_logs = self.agent.train(train_loader, valid_loader, self.device)

        if self.params['save_params']:
            print('\nSaving agent params.')
            self.agent.save_model_params('{}/MLP_policy.pt'.format(self.params['logdir']))

        self.perform_logging(all_logs, '{}/experiment_result.pkl'.format(self.params['logdir']))

        # perform the logging.
        # TODO: haven't intall some package so TS doesn't work so far. log
        # as dictionary for now.
        logger = Logger(self.params['logdir'])
        for i in range(self.params['epochs']):
            for key, value in all_logs.items():
                print(f"key: {key}, value[{i}]: {value[i]}")
                logger.log_scalar(value[i], key, i)
        print('Done logging...\n\n')
        logger.flush()
        
    def perform_logging(self, file, filepath) -> None:
        with open(filepath, 'wb') as fp:
            pickle.dump(file, fp)
        print(f"Successfully saved as {filepath}")
        print("-"*100)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('-'*100)

    # Run training
    trainer = PolicyTrainer(cfg)
    trainer.run_training_loop()

    # configs:
    # learning_rate: 0.001
    # epochs: 80

    # preprocessed_features: true
    # preprocessed_dataset: true
    # batch_size: 512
    # interval: 3

    # no_gpu: false
    # which_gpu: 0

    # save_params: true
    # logdir: &reference_dir '../conf_outputs/${now:%Y-%m-%d}/${now:%H-%M}'

if __name__ == "__main__":
    main()
