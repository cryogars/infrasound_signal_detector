import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import optuna
from optuna.trial import TrialState

from train_cnn import create_data_loader
from utils import config as config
from utils.infrasounddataset import InfrasoundDataset
from utils.helpers import get_logger

logger = get_logger(__name__)

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = 1000
EPOCHS = 25
N_TRIALS = 100


class Net(nn.Module):

    def __init__(
        self, trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1
    ):

        super(Net, self).__init__()  # Initialize parent class
        in_size = 32  #
        kernel_size = 3
        self.conv = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=(3, 3))]
        )  # List with the Conv layers
        out_size = in_size - kernel_size + 1
        out_size = int(out_size / 2)
        for i in range(1, num_conv_layers):
            self.conv.append(
                nn.Conv2d(
                    in_channels=num_filters[i - 1],
                    out_channels=num_filters[i],
                    kernel_size=(3, 3),
                )
            )
            out_size = out_size - kernel_size + 1
            out_size = int(out_size / 2)
        self.conv2_drop = nn.Dropout2d(p=drop_conv2)
        self.out_feature = num_filters[num_conv_layers - 1] * out_size * out_size
        self.fc1 = nn.Linear(self.out_feature, num_neurons)
        self.fc2 = nn.Linear(num_neurons, out_features=config.CLASSES)
        self.p1 = drop_fc1

        # Initialize weights with the He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.conv[i].weight, nonlinearity="relu")
            if self.conv[i].bias is not None:
                nn.init.constant_(self.conv[i].bias, val=0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")

    def forward(self, x):
        for i, conv_i in enumerate(self.conv):
            if i == 2:
                x = F.relu(F.max_pool2d(self.conv2_drop(conv_i(x)), kernel_size=2))
            else:
                x = F.relu(F.max_pool2d(conv_i(x), kernel_size=2))
        x = x.view(-1, self.out_feature)  # Flatten tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.p1, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(network, optimizer):
    network.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = network(data.to(config.DEVICE))
        loss = F.cross_entropy(output, target.to(config.DEVICE))
        loss.backward()
        optimizer.step()


def test(network):
    network.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(config.DEVICE))  # Forward propagation
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(config.DEVICE).data.view_as(pred)).sum()
    accuracy_test = correct / len(test_loader.dataset)
    return accuracy_test


def objective(trial):
    num_conv_layers = trial.suggest_int("num_conv_layers", 2, 3)
    num_filters = [
        int(trial.suggest_float(name="num_filter_" + str(i), low=16, high=128, step=16))
        for i in range(num_conv_layers)
    ]
    num_neurons = trial.suggest_int(name="num_neurons", low=10, high=400, step=10)
    drop_conv2 = trial.suggest_float(name="drop_conv2", low=0.2, high=0.5)
    drop_fc1 = trial.suggest_float(name="drop_fc1", low=0.2, high=0.5)

    model = Net(
        trial, num_conv_layers, num_filters, num_neurons, drop_conv2, drop_fc1
    ).to(config.DEVICE)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", low=1e-5, high=1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
        train(model, optimizer)
        accuracy = test(model)

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy



if __name__ == "__main__":
    random_seed = 42
    torch.backends.cudnn.enabled = True
    torch.manual_seed(random_seed)

    train_feature_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_X_train_.parquet"
    train_label_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_y_train_.parquet"
    test_feature_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_X_test_.parquet"
    test_label_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_y_test_.parquet"
    save_path="optuna_result_sensor"

    # Train Loader
    isd = InfrasoundDataset(
        labels_file=train_label_file,
        waveform_file=train_feature_file,
        transformation=config.TRANSFORMATION,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
    )

    train_loader = create_data_loader(isd, BATCH_SIZE_TRAIN)

    # Validation Loader
    test_isd = InfrasoundDataset(
        labels_file=test_label_file,
        waveform_file=test_feature_file,
        transformation=config.TRANSFORMATION,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
    )

    test_loader = create_data_loader(test_isd, BATCH_SIZE_VALID)

    # Create an Optuna study to maximize test accuracy
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Display the study statistics
    logger.info(f"\nStudy statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")

    trial = study.best_trial
    logger.info(f"Best trial:")
    logger.info(f"  Value: {trial.value}")
    logger.info(f"  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key:<15}: {value}")

    # Save results to csv file
    df = study.trials_dataframe()
    df = df.loc[df["state"] == "COMPLETE"]
    df = df.drop("state", axis=1)
    df = df.sort_values("value")
    df.to_csv(f"{save_path}.csv", index=False)

    # Display results in a dataframe
    logger.info("\nOverall Results (ordered by accuracy):\n {}".format(df))

    most_important_parameters = optuna.importance.get_param_importances(
        study, target=None
    )

    logger.info("\nMost important hyperparameters:")
    for key, value in most_important_parameters.items():
        logger.info(f"  {key:<20}:{value * 100:.2f}%")



    # run_study(train_feature_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_X_train_.parquet",
    #           train_label_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_y_train_.parquet",
    #           test_feature_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_X_test_.parquet",
    #           test_label_file=f"{config.PROCESSED_DATA}/two_high_signal/high_signal_y_test_.parquet",
    #           save_path="optuna_result_sensor")
    #
    #
    # run_study(train_feature_file=f"{config.PROCESSED_DATA}/first_ten/x_train.parquet",
    #           train_label_file=f"{config.PROCESSED_DATA}/first_ten/y_train.parquet",
    #           test_feature_file=f"{config.PROCESSED_DATA}/first_ten/x_test.parquet",
    #           test_label_file=f"{config.PROCESSED_DATA}/first_ten/y_test.parquet",
    #           save_path="optuna_result_temporal")
    #
    # run_study(train_feature_file=f"{config.PROCESSED_DATA}/even_odd/x_train.parquet",
    #           train_label_file=f"{config.PROCESSED_DATA}/even_odd/y_train.parquet",
    #           test_feature_file=f"{config.PROCESSED_DATA}/even_odd/x_test.parquet",
    #           test_label_file=f"{config.PROCESSED_DATA}/even_odd/y_test.parquet",
    #           save_path="optuna_result_meteorological")




