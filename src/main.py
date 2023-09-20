
from core.facades.data_processing_facade import DataProcessingFacade
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Procesamiento de datos')
    parser.add_argument('--type_process', type=str, required=True, choices=['TRAIN_EVALUATE', 'TRAIN_AND_EVALUATE_INDIVIDUAL_SERIES', 'SELECT_MODEL', 'EXPLORATION_DATA_ANALYSIS'], help='Tipo de proceso')
    args = parser.parse_args()

    facade = DataProcessingFacade()
    facade.run(args.type_process)
