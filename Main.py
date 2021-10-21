import numpy as np
import pickle
import time
import argparse
import sys

# keras/sklearn libraries
import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Input, Reshape, BatchNormalization
from keras.layers import (
    Conv1D,
    GlobalAveragePooling1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Reshape,
    AveragePooling1D,
    Flatten,
    Concatenate,
)
from keras import backend
from keras.callbacks import TensorBoard, LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


parser = argparse.ArgumentParser(description="ML framework")

parser.add_argument(
    "--multi_adsorbate",
    default=0,
    type=int,
    help="train for single adsorbate (0) or multiple (1) (default: 0)",
)
parser.add_argument(
    "--data_dir",
    default="CH_data",
    type=str,
    help="path to file containing DOS and targets (default: CH_data)",
)
parser.add_argument(
    "--run_mode",
    default=0,
    type=int,
    help="run regular (0) or 5-fold CV (1) (default: 0)",
)
parser.add_argument(
    "--split_ratio", default=0.2, type=float, help="train/test ratio (default:0.2)"
)
parser.add_argument(
    "--epochs", default=60, type=int, help="number of total epochs to run (default:60)"
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="batch size (default:32)"
)
parser.add_argument(
    "--channels", default=9, type=int, help="number of channels (default: 9)"
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="seed for data split(epochs), 0=random (default:0)",
)
parser.add_argument(
    "--save_model",
    default=0,
    type=int,
    help="path to file containing DOS and targets (default: 0)",
)
parser.add_argument(
    "--load_model",
    default=0,
    type=int,
    help="path to file containing DOS and targets (default: 0)",
)
args = parser.parse_args(sys.argv[1:])


def main():
    start_time = time.time()

    # load data (replace with your own depending on the data format)
    # Data format for x_surface_dos and x_adsorbate_dos is a numpy array with shape: (A, B, C) where A is number of samples, B is length of DOS file (2000), C is number of channels.
    # Number of channels here is 27 for x_surface_dos which contains 9 orbitals x up to 3 adsorbing surface atoms. E.g. a top site will have the first 9 channels filled and remaining as zeros.
    x_surface_dos, x_adsorbate_dos, y_targets = load_data(
        args.multi_adsorbate, args.data_dir
    )

    if args.seed == 0:
        args.seed = np.random.randint(1, 1e6)

    if args.run_mode == 0:
        run_training(args, x_surface_dos, x_adsorbate_dos, y_targets)
    elif args.run_mode == 1:
        run_kfold(args, x_surface_dos, x_adsorbate_dos, y_targets)

    print("--- %s seconds ---" % (time.time() - start_time))


def load_data(multi_adsorbate, data_dir):
    ###load data containing: (1) dos of surface, (2) adsorption energy(target), (3) dos of adsorbate in gas phase (for multi-adsorbate)
    if args.multi_adsorbate == 0:
        with open(args.data_dir, "rb") as f:
            surface_dos = pickle.load(f)
            targets = pickle.load(f)
        x_adsorbate_dos = []
    elif args.multi_adsorbate == 1:
        with open(args.data_dir, "rb") as f:
            surface_dos = pickle.load(f)
            targets = pickle.load(f)
            x_adsorbate_dos = pickle.load(f)
    ###Some data rearranging, depends on if atomic params are to be included as extra features in the DOS series or separately
    ###entries 1700-2200 of the data are set to zero, these are states far above fermi level which seem to cause additional errors, reason being some states are not physically reasonable

    ###First column is energy; not used in current implementation
    surface_dos = surface_dos[:, 0:2000, 1:28]
    ###States far above fermi level can be unphysical and set to zero
    surface_dos[:, 1800:2000, 0:27] = 0
    ###float32 is used for memory concerns
    surface_dos = surface_dos.astype(np.float32)

    if args.multi_adsorbate == 1:
        x_adsorbate_dos = x_adsorbate_dos[:, 0:2000, 1:10]
        x_adsorbate_dos = x_adsorbate_dos.astype(np.float32)

    return surface_dos, x_adsorbate_dos, targets


###Creates the ML model with keras
###This is the overall model where all 3 adsorption sites are fitted at the same time
def create_model(shared_conv, channels):

    ###Each input represents one out of three possible bonding atoms
    input1 = Input(shape=(2000, channels))
    input2 = Input(shape=(2000, channels))
    input3 = Input(shape=(2000, channels))

    conv1 = shared_conv(input1)
    conv2 = shared_conv(input2)
    conv3 = shared_conv(input3)

    convmerge = Concatenate(axis=-1)([conv1, conv2, conv3])
    convmerge = Flatten()(convmerge)
    convmerge = Dropout(0.2)(convmerge)
    convmerge = Dense(200, activation="linear")(convmerge)
    convmerge = Dense(1000, activation="relu")(convmerge)
    convmerge = Dense(1000, activation="relu")(convmerge)

    out = Dense(1, activation="linear")(convmerge)
    # shared_conv.summary()
    model = Model(input=[input1, input2, input3], output=out)
    return model


###This is the overall model where all 3 adsorption sites are fitted at the same time, and all adsorbates are fitted as well
def create_model_combined(shared_conv, channels):

    ###Each input represents one out of three possible bonding atoms
    input1 = Input(shape=(2000, channels))
    input2 = Input(shape=(2000, channels))
    input3 = Input(shape=(2000, channels))
    input4 = Input(shape=(2000, channels))

    conv1 = shared_conv(input1)
    conv2 = shared_conv(input2)
    conv3 = shared_conv(input3)

    adsorbate_conv = adsorbate_dos_featurizer(channels)
    conv4 = adsorbate_conv(input4)

    convmerge = Concatenate(axis=-1)([conv1, conv2, conv3, conv4])
    convmerge = Flatten()(convmerge)
    convmerge = Dropout(0.2)(convmerge)
    convmerge = Dense(200, activation="linear")(convmerge)
    convmerge = Dense(1000, activation="relu")(convmerge)
    convmerge = Dense(1000, activation="relu")(convmerge)

    out = Dense(1, activation="linear")(convmerge)

    model = Model(input=[input1, input2, input3, input4], output=out)
    return model


###This sub-model is the convolutional network for the DOS
###Uses the same model for each atom input channel
###Input is a 2000 length DOS data series
def dos_featurizer(channels):
    input_dos = Input(shape=(2000, channels))
    x1 = AveragePooling1D(pool_size=4, strides=4, padding="same")(input_dos)
    x2 = AveragePooling1D(pool_size=25, strides=4, padding="same")(input_dos)
    x3 = AveragePooling1D(pool_size=200, strides=4, padding="same")(input_dos)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Conv1D(50, 20, activation="relu", padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(75, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(100, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(125, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(150, 3, activation="relu", padding="same", strides=1)(x)
    shared_model = Model(input_dos, x)
    return shared_model


###Uses the same model for adsorbate but w/ separate weights
def adsorbate_dos_featurizer(channels):
    input_dos = Input(shape=(2000, channels))
    x1 = AveragePooling1D(pool_size=4, strides=4, padding="same")(input_dos)
    x2 = AveragePooling1D(pool_size=25, strides=4, padding="same")(input_dos)
    x3 = AveragePooling1D(pool_size=200, strides=4, padding="same")(input_dos)
    x = Concatenate(axis=-1)([x1, x2, x3])
    x = Conv1D(50, 20, activation="relu", padding="same", strides=2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(75, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(100, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(125, 3, activation="relu", padding="same", strides=2)(x)
    x = AveragePooling1D(pool_size=3, strides=2, padding="same")(x)
    x = Conv1D(150, 3, activation="relu", padding="same", strides=1)(x)
    shared_model = Model(input_dos, x)
    return shared_model


###Simple learning rate scheduler
def decay_schedule(epoch, lr):
    if epoch == 0:
        lr = 0.001
    elif epoch == 15:
        lr = 0.0005
    elif epoch == 35:
        lr = 0.0001
    elif epoch == 45:
        lr = 0.00005
    elif epoch == 55:
        lr = 0.00001
    return lr


# regular training
def run_training(args, x_surface_dos, x_adsorbate_dos, y_targets):

    ###Split data into train and test
    if args.multi_adsorbate == 0:
        x_train, x_test, y_train, y_test = train_test_split(
            x_surface_dos, y_targets, test_size=args.split_ratio, random_state=88
        )
    elif args.multi_adsorbate == 1:
        x_train, x_test, y_train, y_test, ads_train, ads_test = train_test_split(
            x_surface_dos,
            y_targets,
            x_adsorbate_dos,
            test_size=args.split_ratio,
            random_state=88,
        )
    ###Scaling data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[2])).reshape(
        x_train.shape
    )
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[2])).reshape(x_test.shape)

    if args.multi_adsorbate == 1:
        ads_train = scaler.fit_transform(
            ads_train.reshape(-1, ads_train.shape[2])
        ).reshape(ads_train.shape)
        ads_test = scaler.transform(ads_test.reshape(-1, ads_test.shape[2])).reshape(
            ads_test.shape
        )

    ###call and fit model
    shared_conv = dos_featurizer(args.channels)
    lr_scheduler = LearningRateScheduler(decay_schedule, verbose=0)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), histogram_freq=1)

    ###FOr testing purposes, a model where 3 adsorption sites fitted simultaneously and 3 separately are done by comparison
    if args.multi_adsorbate == 0:
        if args.load_model == 0:
            model = create_model(shared_conv, args.channels)
            model.compile(
                loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"]
            )
        elif args.load_model == 1:
            print("Loading model...")
            model = load_model("DOSnet_saved.h5", compile=False)
            model.compile(
                loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"]
            )
        model.summary()
        model.fit(
            [x_train[:, :, 0:9], x_train[:, :, 9:18], x_train[:, :, 18:27]],
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(
                [x_test[:, :, 0:9], x_test[:, :, 9:18], x_test[:, :, 18:27]],
                y_test,
            ),
            callbacks=[tensorboard, lr_scheduler],
        )
        train_out = model.predict(
            [x_train[:, :, 0:9], x_train[:, :, 9:18], x_train[:, :, 18:27]]
        )
        train_out = train_out.reshape(len(train_out))
        test_out = model.predict(
            [x_test[:, :, 0:9], x_test[:, :, 9:18], x_test[:, :, 18:27]]
        )
        test_out = test_out.reshape(len(test_out))

    elif args.multi_adsorbate == 1:
        model = create_model_combined(shared_conv, args.channels)
        model.compile(
            loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"]
        )
        model.summary()
        model.fit(
            [x_train[:, :, 0:9], x_train[:, :, 9:18], x_train[:, :, 18:27], ads_train],
            y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(
                [x_test[:, :, 0:9], x_test[:, :, 9:18], x_test[:, :, 18:27], ads_test],
                y_test,
            ),
            callbacks=[tensorboard, lr_scheduler],
        )
        train_out = model.predict(
            [x_train[:, :, 0:9], x_train[:, :, 9:18], x_train[:, :, 18:27], ads_train]
        )
        train_out = train_out.reshape(len(train_out))
        test_out = model.predict(
            [x_test[:, :, 0:9], x_test[:, :, 9:18], x_test[:, :, 18:27], ads_test]
        )
        test_out = test_out.reshape(len(test_out))

    ###this is just to write the results to a file
    print("train MAE: ", mean_absolute_error(y_train, train_out))
    print("train RMSE: ", mean_squared_error(y_train, train_out) ** (0.5))
    print("test MAE: ", mean_absolute_error(y_test, test_out))
    print("test RMSE: ", mean_squared_error(y_test, test_out) ** (0.5))

    with open("predict_train.txt", "w") as f:
        np.savetxt(f, np.stack((y_train, train_out), axis=-1))
    with open("predict_test.txt", "w") as f:
        np.savetxt(f, np.stack((y_test, test_out), axis=-1))

    if args.save_model == 1:
        print("Saving model...")
        model.save("DOSnet_saved.h5")


# kfold
def run_kfold(args, x_surface_dos, x_adsorbate_dos, y_targets):
    cvscores = []
    count = 0
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)

    for train, test in kfold.split(x_surface_dos, y_targets):

        scaler_CV = StandardScaler()
        x_surface_dos[train, :, :] = scaler_CV.fit_transform(
            x_surface_dos[train, :, :].reshape(-1, x_surface_dos[train, :, :].shape[-1])
        ).reshape(x_surface_dos[train, :, :].shape)
        x_surface_dos[test, :, :] = scaler_CV.transform(
            x_surface_dos[test, :, :].reshape(-1, x_surface_dos[test, :, :].shape[-1])
        ).reshape(x_surface_dos[test, :, :].shape)
        if args.multi_adsorbate == 1:
            x_adsorbate_dos[train, :, :] = scaler.fit_transform(
                x_adsorbate_dos[train, :, :].reshape(
                    -1, x_adsorbate_dos[train, :, :].shape[-1]
                )
            ).reshape(x_adsorbate_dos[train, :, :].shape)
            x_adsorbate_dos[test, :, :] = scaler.transform(
                x_adsorbate_dos[test, :, :].reshape(
                    -1, x_adsorbate_dos[test, :, :].shape[-1]
                )
            ).reshape(x_adsorbate_dos[test, :, :].shape)

        keras.backend.clear_session()
        shared_conv = dos_featurizer(args.channels)
        lr_scheduler = LearningRateScheduler(decay_schedule, verbose=0)
        if args.multi_adsorbate == 0:
            model_CV = create_model(shared_conv, args.channels)
            model_CV.compile(
                loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"]
            )
            model_CV.fit(
                [
                    x_surface_dos[train, :, 0:9],
                    x_surface_dos[train, :, 9:18],
                    x_surface_dos[train, :, 18:27],
                ],
                y_targets[train],
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=0,
                callbacks=[lr_scheduler],
            )
            scores = model_CV.evaluate(
                [
                    x_surface_dos[test, :, 0:9],
                    x_surface_dos[test, :, 9:18],
                    x_surface_dos[test, :, 18:27],
                ],
                y_targets[test],
                verbose=0,
            )
            train_out_CV_temp = model_CV.predict(
                [
                    x_surface_dos[test, :, 0:9],
                    x_surface_dos[test, :, 9:18],
                    x_surface_dos[test, :, 18:27],
                ]
            )
            train_out_CV_temp = train_out_CV_temp.reshape(len(train_out_CV_temp))
        elif args.multi_adsorbate == 1:
            model_CV = create_model_combined(shared_conv, args.channels)
            model_CV.compile(
                loss="logcosh", optimizer=Adam(0.001), metrics=["mean_absolute_error"]
            )
            model_CV.fit(
                [
                    x_surface_dos[train, :, 0:9],
                    x_surface_dos[train, :, 9:18],
                    x_surface_dos[train, :, 18:27],
                    x_adsorbate_dos[train, :, :],
                ],
                y_targets[train],
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=0,
                callbacks=[lr_scheduler],
            )
            scores = model_CV.evaluate(
                [
                    x_surface_dos[test, :, 0:9],
                    x_surface_dos[test, :, 9:18],
                    x_surface_dos[test, :, 18:27],
                    x_adsorbate_dos[test, :, :],
                ],
                y_targets[test],
                verbose=0,
            )
            train_out_CV_temp = model_CV.predict(
                [
                    x_surface_dos[test, :, 0:9],
                    x_surface_dos[test, :, 9:18],
                    x_surface_dos[test, :, 18:27],
                    x_adsorbate_dos[test, :, :],
                ]
            )
            train_out_CV_temp = train_out_CV_temp.reshape(len(train_out_CV_temp))
        print((model_CV.metrics_names[1], scores[1]))
        cvscores.append(scores[1])
        if count == 0:
            train_out_CV = train_out_CV_temp
            test_y_CV = y_targets[test]
            test_index = test
        elif count > 0:
            train_out_CV = np.append(train_out_CV, train_out_CV_temp)
            test_y_CV = np.append(test_y_CV, y_targets[test])
            test_index = np.append(test_index, test)
        count = count + 1
    print((np.mean(cvscores), np.std(cvscores)))
    print(len(test_y_CV))
    print(len(train_out_CV))
    with open("CV_predict.txt", "w") as f:
        np.savetxt(f, np.stack((test_y_CV, train_out_CV), axis=-1))


if __name__ == "__main__":
    main()
