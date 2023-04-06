import wandb, random

PROJECT = "another_launch_project2"
# ENTITY = "wandb"
ENTITY = None

default_config = dict(lr=0.001,epochs=10)

def train(config):
    run = wandb.init(project=PROJECT, entity=ENTITY, config=config)
    run.log_code()
    config = wandb.config
    for epoch in range(config.epochs):
        loss = random.random()
        run.log({"loss": loss, "epoch": epoch})
    run.finish()

if __name__ == "__main__": 
    train(default_config)
    