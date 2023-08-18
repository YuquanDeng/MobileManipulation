import sys
import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.append("../")
import utilities.pytorch_util as ptu
from utilities.utils import *
from agents.MLP_agent import MLPagent


class PolicyTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.device = ptu.init_gpu(use_gpu=not self.params['no_gpu'], gpu_id=self.params['which_gpu'])
        self.agent = MLPagent(self.params)

    def get_dataloader(self, preprocessed_features: bool, preprocessed_dataset: bool):
        """
        Load dataset.
        """
        dataset_dir = os.path.join(os.getcwd(), '../../data/siamese/')
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        if preprocessed_features:
            feature_dataset = load(filepath=dataset_dir + 'feature_dataset.pkl')
        else:
            # Load a1 dataset and extract features.
            if not preprocessed_dataset:
                a1_dataset = get_a1_dataset(data_dir=self.params['dataset_dir'])
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
            create_directory(self.params['logdir'])
            self.agent.save_model_params('{}/MLP_policy.pt'.format(self.params['logdir']))

        if self.params['save_log']:
            create_directory(self.params['logdir'])
            self.perform_logging(all_logs, '{}/experiment_result.pkl'.format(self.params['logdir']))
   
    def perform_logging(self, file, filepath) -> None:
        with open(filepath, 'wb') as fp:
            pickle.dump(file, fp)
        print(f"saved {filepath}")
        print("-"*100)

@hydra.main(version_base=None, config_path="../configs", config_name="train_config")
def main(cfg: DictConfig):
    # For enabling auto configs logging, go to train_config.yaml and comment by following 
    # the instruction there.
    print('-'*100)
    print(OmegaConf.to_yaml(cfg))
    print('-'*100)

    # Run training
    trainer = PolicyTrainer(cfg)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()
