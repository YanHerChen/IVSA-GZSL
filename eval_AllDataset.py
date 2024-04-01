import os
import yaml
start_dire = r"./eval.py"


settings = ['gzsl']
dataset = ['CUB', 'AWA2', 'SUN', 'FLO', 'APY']

for i in range(0, len(settings), 1):
    print('-'*30, settings[i], '-'*30)
    for j in range(0, len(dataset), 1):
        with open("config/Control.yaml") as f:
            y = yaml.safe_load(f)
            y['settings'] = settings[i]
            y['dataset'] = dataset[j]

        with open("config/Control.yaml", "w") as f:
            doc = yaml.dump(y, f)

        r = os.system("python %s" % start_dire)
