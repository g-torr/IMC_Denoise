from IMC_Denoise.IMC_Denoise_main.multi_sample import Denoise_predict,Denoise_train
import argparse
import logging
import yaml
logger = logging.getLogger(__name__)

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    #-----IMC_Denoise---
    Denoise_train.main_train(config)
    Denoise_predict.main(config)
# Entry point for the script,  loads configuration
if __name__ == '__main__':
    main()
