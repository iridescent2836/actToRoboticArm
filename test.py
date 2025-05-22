import argparse
import pickle

def convert_to_args(config):
    print("Entering convert_to_args")

    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Iterate over the config dictionary to generate command-line arguments
    for key, value in config.items():
        # Choose the appropriate argparse type based on the value's type
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', type=bool, default=value)
        elif isinstance(value, list):
            # For list types, allow the user to input a comma-separated string, then convert to list
            parser.add_argument(f'--{key}', type=str, default=','.join(map(str, value)))
        else:
            parser.add_argument(f'--{key}', type=type(value), default=value)

    # Static arguments
    parser.add_argument('--masks', type=bool, default=False)
    parser.add_argument('--dilation', type=bool, default=False)
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--pre_norm', type=bool, default=False)

    args = parser.parse_args()
    print("Exiting convert_to_args")

    return args

# Example of how the convert_to_args function is called
def main(config_args):
    with open('D:/PRML/act-plus-plus/temp/config.pkl', 'rb') as f:
        config = pickle.load(f)
        print(f'the whole config:{config}')
        
    policy_config = config['policy_config']

    # Call convert_to_args once
    policy_config = convert_to_args(policy_config)
    print(f'converted policy {policy_config}')

if __name__ == "__main__":
    with open('D:/PRML/act-plus-plus/temp/config.pkl', 'rb') as f:
        config_args = pickle.load(f)
    main(config_args)