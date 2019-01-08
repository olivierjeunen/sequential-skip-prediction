import argparse
import datetime
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from glob import glob
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

def evaluation(sizes, label_matrix, prediction_matrix):
    '''
    Return MAP and first-prediction-accuracy
    :param sizes:    A list of session sizes
    :param labels:      The correct labels
    :param predictions: The predicted labels
    :returns:           Mean Average Precision and First Prediction Accuracy
    '''
    # Set variables
    score = 0.0
    first_acc = 0.0
    # For every session
    for size, label_row, prediction_row in zip(sizes, label_matrix, prediction_matrix):
        # Set variables
        n_correct    = 0
        session      = 0.0
        # For the first 'size' predictions
        for i in range(size):        
            # If the prediction is correct:
            if label_row[i] == prediction_row[i]:
                # Increase counter of correct predictions
                n_correct += 1
                session += n_correct / (i + 1)
                # If first prediction
                if i == 0:
                    first_acc += 1
        # Save session score
        score += session / size
    return score/sizes.shape[0], first_acc/sizes.shape[0]

def generate_model(history, future):
    h_input  = keras.layers.Input(shape = (history.shape[1], history.shape[2]))
    h_subnet = keras.layers.Dropout(.35)(h_input)
    h_subnet = keras.layers.CuDNNLSTM(256, return_sequences = True)(h_subnet)
    h_subnet = keras.layers.CuDNNLSTM(256)(h_subnet)

    f_input  = keras.layers.Input(shape = (future.shape[1], future.shape[2]))
    f_subnet = keras.layers.Dropout(.35)(f_input)
    f_subnet = keras.layers.CuDNNLSTM(128, return_sequences = True)(f_subnet)
    f_subnet = keras.layers.CuDNNLSTM(128)(f_subnet)

    td_subnet = keras.layers.TimeDistributed(keras.layers.Dense(32))(f_input)
    td_subnet = keras.layers.Flatten()(td_subnet)

    concat = keras.layers.concatenate([h_subnet, f_subnet, td_subnet])
    concat = keras.layers.Dropout(.25)(concat)

    dense1  = keras.layers.Dense(512, activation = 'elu')(concat)
    dense1  = keras.layers.BatchNormalization()(dense1)
    dense2  = keras.layers.Dense(512, activation = 'elu')(dense1)

    outputs = [keras.layers.Dense(1, activation = 'sigmoid')(dense2) for _ in range(10)]

    model = keras.Model(inputs = [h_input, f_input], outputs = outputs)
    
    return model

def process_sessions_train(subset, features, track_feature_list, label_index):
    session_histories = []
    session_futures   = []
    session_labels    = []
    session_sizes     = []
    for session_id, subgroup in subset[['session_id'] + features + track_feature_list].groupby('session_id', sort = False):
        # Split session in two
        half = int(np.floor(subgroup.shape[0] / 2.0))
        first = subgroup[features + track_feature_list].head(half).values
        second = subgroup[features + track_feature_list].tail(subgroup.shape[0] - half).values
        
        # Generate padding for session history
        padding = np.zeros((int(max_seq_len / 2.0) - first.shape[0], len(features) + len(track_feature_list)))
        session_history = np.vstack([padding,first])
        
        # Generate padding for session future
        padding = np.zeros((int(max_seq_len / 2.0) - second.shape[0], len(track_feature_list)))
        session_future = np.vstack([second[:,-len(track_feature_list):],padding])
        
        # Generate labels
        current_labels = second[:,label_index].astype(int)
        
        # Save results
        session_histories.append(session_history)
        session_futures.append(session_future)
        session_labels.append(np.hstack([current_labels,np.zeros(int(max_seq_len / 2.0) - current_labels.shape[0])]))
        session_sizes.append(current_labels.shape[0])

    return session_histories, session_futures, session_labels, session_sizes 

def process_sessions_test(subset, features, track_feature_list):
    session_histories = []
    session_futures   = []
    session_sizes     = []
    for session_id, subgroup in subset[['session_id'] + features + track_feature_list].groupby('session_id', sort = False):
        # Split session in two
        half = int(np.floor(subgroup.shape[0] / 2.0))
        first = subgroup[features + track_feature_list].head(half).values
        second = subgroup[features + track_feature_list].tail(subgroup.shape[0] - half).values

        # Generate padding for session history
        padding = np.zeros((int(max_seq_len / 2.0) - first.shape[0], len(features) + len(track_feature_list)))
        session_history = np.vstack([padding,first])

        # Generate padding for session future
        padding = np.zeros((int(max_seq_len / 2.0) - second.shape[0], len(track_feature_list)))
        session_future = np.vstack([second[:,-len(track_feature_list):],padding])

        session_histories.append(session_history)
        session_futures.append(session_future)
        session_sizes.append(second.shape[0])

    return session_histories, session_futures, session_sizes 

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type = str, help = 'Path to the folder containing all data files')
    parser.add_argument('start_day', type = int, help = 'Integer between 0-65, to indicate the day to start on (inclusive)')
    parser.add_argument('end_day', type = int, help = 'Integer between 1-66, to indicate the day to end on (exclusive)')
    parser.add_argument('sub_ID', type = str, help = 'ID for the submission - to make sure different runs do not overwrite files')
    args = parser.parse_args()

    # Set path to data location
    path = '/scratch/antwerpen/204/vsc20402/spotify-data/'

    # Get list of logs
    input_logs = sorted(glob(path + 'training_set/log_0*.gz'))
    
    # Get list of track features
    track_features =  pd.concat([pd.read_csv(f) for f in sorted(glob(path + 'track_features/*'))]).rename(columns = {'track_id': 'track_id_clean'})
    track_features = pd.get_dummies(track_features, columns = ['key'], prefix = ['key'])
    track_features = pd.get_dummies(track_features, columns = ['mode'], prefix = ['mode'])

    # Set number of logs to train on - between 1 and 10
    n_logs = 5

    # Set data types for fields to consider
    dtypes = {
        'session_id': str,
        'session_position': np.uint8,
        'session_length': np.uint8,
        'track_id_clean': str,
        'skip_1': np.bool_,
        'skip_2': np.bool_,
        'skip_3': np.bool_,
        'not_skipped': np.bool_,
        'context_switch': np.bool_,
        'no_pause_before_play': np.bool_,
        'short_pause_before_play': np.bool_,
        'long_pause_before_play': np.bool_,
        'hist_user_behavior_n_seekfwd': np.uint8,
        'hist_user_behavior_n_seekback': np.uint8,
        'hist_user_behavior_is_shuffle': np.bool_,
        'hour_of_day': np.uint8,
        'premium': np.bool_,
        'context_type': str,
        'hist_user_behavior_reason_start': str,
        'hist_user_behavior_reason_end': str
    }

    test_dtypes = {
        'session_id': str,
        'track_id_clean': str,
        'session_position': np.uint8,
        'session_length': np.uint8,
    }

    # For every day
    for log in input_logs[args.start_day:args.end_day]:
        # Get start timestamp
        day_start = datetime.datetime.now()

        # Extract date from log-file
        date = log[len(path)+19:len(path)+27]
            
        # Read in the files associated with that day
        print('{0}\t\tParsing training data for date {1}...'.format(datetime.datetime.now(),date))
        data = pd.concat([pd.read_csv(log.replace('log_0','log_'+str(i),1), usecols = dtypes.keys(), dtype = dtypes) for i in range(n_logs)])
        data['train_flag'] = True

        print('{0}\t\tParsing test data for date {1}...'.format(datetime.datetime.now(),date))
        test_data_hist  = pd.read_csv(path + 'test_set/log_prehistory_{0}_000000000000.csv.gz'.format(date), usecols = dtypes.keys(), dtype = dtypes)
        test_data_input = pd.read_csv(path + 'test_set/log_input_{0}_000000000000.csv.gz'.format(date), usecols = test_dtypes.keys(), dtype = test_dtypes)
        test_data_hist['train_flag']  = False
        test_data_input['train_flag'] = False

        print('{0}\t\tMerging data...'.format(datetime.datetime.now()))
        data = pd.concat([data,test_data_hist])

        # Generate dummy columns for categorical variables
        print('{0}\t\tOne-hot-encoding categorical variables...'.format(datetime.datetime.now()))
        data = pd.get_dummies(data, columns = ['context_type'], prefix = ['context_type'])
        data = pd.get_dummies(data, columns = ['hist_user_behavior_reason_start'], prefix = ['hist_user_behavior_reason_start'])
        data = pd.get_dummies(data, columns = ['hist_user_behavior_reason_end'], prefix = ['hist_user_behavior_reason_end'])
        
        # Features to use
        print('{0}\t\tNormalising context features...'.format(datetime.datetime.now()))
        features = list(set(data.columns) - set(['session_id','track_id_clean','date', 'skip_2', 'session_position', 'train_flag']))
        data[features] = StandardScaler().fit_transform(data[features])
        features.append('skip_2')
        features.append('session_position')
        
        # Add second halves of test sessions as well, doing this earlier skews distribution of features
        data = pd.concat([data,test_data_input])
        data.fillna(.0,inplace=True)

        # Merge data with track features
        print('{0}\t\tNormalising track features...'.format(datetime.datetime.now()))
        data = data.merge(track_features, on = 'track_id_clean', how = 'left')
        data['rel_pos'] = data['session_position'] / data['session_length']
        session_track_count = data.groupby(['session_id','track_id_clean'], as_index = False).size().reset_index().rename(columns = {0 : 'track_session_count'})
        data = data.merge(session_track_count, on = ['session_id', 'track_id_clean'], how = 'left')
        track2count = data['track_id_clean'].value_counts().reset_index().rename(columns = {'index': 'track_id_clean', 'track_id_clean': 'track_count'})
        data = data.merge(track2count, on = 'track_id_clean', how = 'left')
        data['track_count'] = np.log(np.log(data['track_count']) + 1)


        # Count number of unique songs in a session
        session_unique_count = data[['session_id','track_id_clean']].drop_duplicates()['session_id'].value_counts().reset_index().rename(columns = {'index': 'session_id', 'session_id': 'n_unique_tracks'})
        data = data.merge(session_unique_count, on = ['session_id'], how = 'left')
        data['r_unique_tracks'] = data['n_unique_tracks'] / data['session_length']

        # List of track- and content-features
        track_feature_list = ['rel_pos',
                              'track_session_count',
                              'track_count',
                              'mode_minor',
                              'mode_major',
                              'release_year',
                              'duration',
                              'us_popularity_estimate',
                              'n_unique_tracks',
                              'r_unique_tracks',
                              'acousticness',
                              'beat_strength',
                              'bounciness',
                              'danceability',
                              'dyn_range_mean',
                              'energy',
                              'flatness',
                              'instrumentalness',
                              'liveness',
                              'loudness',
                              'mechanism',
                              'organism',
                              'speechiness',
                              'tempo',
                              'time_signature',
                              'valence',
                              'acoustic_vector_0',
                              'acoustic_vector_1',
                              'acoustic_vector_2',
                              'acoustic_vector_3',
                              'acoustic_vector_4',
                              'acoustic_vector_5',
                              'acoustic_vector_6',
                              'acoustic_vector_7']
        data[track_feature_list] = StandardScaler().fit_transform(data[track_feature_list])

        label_index = features.index('skip_2')
        position_index = features.index('session_position')
        
        # Max sequence length
        max_seq_len = 20

        # Make sure data is ordered correctly
        data.sort_values(['train_flag','session_id','session_position'], inplace = True)

        # If data already exists, don't do it again
        if glob(path + 'preprocessed/{0}_{1}_train.npz'.format(date,n_logs)):
            print('{0}\t\tSkipping training data for date {1}...'.format(datetime.datetime.now(),date))
            # Read in preprocessed matrices
            start = datetime.datetime.now()
            print('{0}\t\tLoading preprocessed data...'.format(start))       
            preprocessed = np.load(path + 'preprocessed/{0}_{1}_train.npz'.format(date,n_logs))
            
            # Extract
            train_history = preprocessed['train_history']
            train_future  = preprocessed['train_future']
            train_labels  = preprocessed['train_labels']
            train_sizes   = preprocessed['train_sizes']
            
            del preprocessed
            
            end = datetime.datetime.now()
            print('{0}\t\tLoaded preprocessed data in {1}...'.format(end, end - start)) 
        else:
            ###
            # PREPROCESS TRAINING SET
            ###
            start = datetime.datetime.now()
            print('{0}\t\tPreprocessing {1} training sessions in parallel...'.format(start, data.loc[data.train_flag == True]['session_id'].nunique()))       

            # Placeholders
            train_history  = []
            train_future   = []
            train_labels   = []
            train_sizes    = []

            # Process all sessions
            n_workers = 25
            train_sessions = data.loc[data.train_flag == True]['session_id'].unique()
            results = Parallel(n_jobs = n_workers, verbose = 10)(delayed(process_sessions_train)(data.merge(pd.DataFrame(chunk, columns = ['session_id']), on = 'session_id', how = 'right'), features, track_feature_list, label_index) for chunk in np.array_split(train_sessions, n_workers))
            print('{0}\t\tCollecting results from workers...'.format(datetime.datetime.now())) 
            for t_h, t_f, t_l, t_s in tqdm(results):
                train_history.extend(t_h)
                train_future.extend(t_f)
                train_labels.extend(t_l)
                train_sizes.extend(t_s)
            
            end = datetime.datetime.now()
            print('{0}\t\tPreprocessed training data in {1}...'.format(end, end - start))
    
            # Encode sessions as numpy arrays
            train_history  = np.asarray(train_history)
            train_future   = np.asarray(train_future)
            train_labels   = np.asarray(train_labels)
            train_sizes    = np.asarray(train_sizes)
            
            # Write out preprocessed matrices
            start = datetime.datetime.now()
            print('{0}\t\tPersisting preprocessed data...'.format(start))       

            # Dump to file
            np.savez_compressed(path + 'preprocessed/{0}_{1}_train.npz'.format(date,n_logs),
                train_history  = train_history,
                train_future   = train_future,
                train_labels   = train_labels,
                train_sizes    = train_sizes
            )
            
            end = datetime.datetime.now()
            print('{0}\t\tPersisted preprocessed data in {1}...'.format(end, end - start))       
        
        # If data already exists, don't do it again
        if glob(path + 'preprocessed/{0}_test.npz'.format(date)):
            print('{0}\t\tSkipping test data preprocessing for date {1}...'.format(datetime.datetime.now(),date))
            # Read in preprocessed matrices
            start = datetime.datetime.now()
            print('{0}\t\tLoading preprocessed data...'.format(start))       
            preprocessed = np.load(path + 'preprocessed/{0}_test.npz'.format(date))
            
            # Extract
            test_history  = preprocessed['test_history']
            test_future   = preprocessed['test_future']
            test_sizes    = preprocessed['test_sizes']
            
            del preprocessed
            
            end = datetime.datetime.now()
            print('{0}\t\tLoaded preprocessed data in {1}...'.format(end, end - start))       
        else:
            ###
            # PREPROCESS TEST SET
            ###
            start = datetime.datetime.now()
            print('{0}\t\tPreprocessing {1} test sessions...'.format(start, data.loc[data.train_flag == False]['session_id'].nunique()))       
            
            # Placeholders
            test_history  = []
            test_future   = []
            test_sizes    = []

            # Process all sessions
            n_workers = 25
            test_sessions = data.loc[data.train_flag == False]['session_id'].unique()
            results = Parallel(n_jobs = n_workers, verbose = 10)(delayed(process_sessions_test)(data.merge(pd.DataFrame(chunk, columns = ['session_id']), on = 'session_id', how = 'right'), features, track_feature_list) for chunk in np.array_split(test_sessions, n_workers))
            print('{0}\t\tCollecting results from workers...'.format(datetime.datetime.now())) 
            for t_h, t_f, t_s in tqdm(results):
                test_history.extend(t_h)
                test_future.extend(t_f)
                test_sizes.extend(t_s)
            end = datetime.datetime.now()
            print('{0}\t\tPreprocessed test data in {1}...'.format(end, end - start))       

            # Encode as numpy arrays
            test_history  = np.asarray(test_history)
            test_future   = np.asarray(test_future)
            test_sizes    = np.asarray(test_sizes)
            
            # Write out preprocessed matrices
            start = datetime.datetime.now()
            print('{0}\t\tPersisting preprocessed data...'.format(start))       
            
            # Dump to file
            np.savez_compressed(path + 'preprocessed/{0}_test.npz'.format(date),
                test_history = test_history,
                test_future  = test_future,
                test_sizes   = test_sizes
            )
            end = datetime.datetime.now()
            print('{0}\t\tPersisted preprocessed data in {1}...'.format(end, end - start))       

        # Free up memory
        del data
        #del track_features

        # k-fold Cross-validation grouped on sessions
        k = 5
        n_epochs = 50
        test_predictions = []
        all_maps = []
        for fold_id, (train_idx, valid_idx) in enumerate(KFold(n_splits = k).split(train_history)):
            print('{0}\t----- FOLD {1} -----'.format(datetime.datetime.now(),fold_id))
            # Filter out training and testing data
            h_train = train_history[train_idx]
            h_valid = train_history[valid_idx]
            f_train = train_future[train_idx]
            f_valid = train_future[valid_idx]
            l_train = train_labels[train_idx]
            l_valid = train_labels[valid_idx]
            s_train = train_sizes[train_idx]
            s_valid = train_sizes[valid_idx]

            # Loss weights
            weights = np.asarray([(1 / val + sum((1 / (2*n)) for n in range(val + 1,11))) for val in range(1,11)])
            weights /= weights.max()
    
            # Generate model
            with tf.device('/cpu:0'):
                model = generate_model(h_train, f_train)
            parallel_model = keras.utils.multi_gpu_model(model, gpus = 2)
            parallel_model.compile(loss='binary_crossentropy',
                                   optimizer = keras.optimizers.Adam(lr = 0.002, amsgrad = True),
                                   metrics=['binary_accuracy'],
                                   loss_weights = weights.tolist())

            # Early stopping
            best_map = .0
            best_weights = None
            best_epoch = 0

            # For every epoch
            for epoch_id in range(n_epochs):
                parallel_model.fit([h_train, f_train],
                                    [l_train[:,i] for i in range(10)],
                                    validation_data = ([h_valid, f_valid], [l_valid[:,i] for i in range(10)]),
                                    batch_size = 2048,
                                    epochs = 1,
                                    verbose = 0)
                p_valid = parallel_model.predict([h_valid, f_valid], batch_size = 4096)
                MAP, FPA = evaluation(s_valid, l_valid, np.swapaxes(np.round(p_valid),0,1))
                print('{0}\t\tValid\tMAP:\t{1}\tFPA:\t{2}'.format(datetime.datetime.now(),MAP, FPA))
                if MAP > best_map:
                    best_map = MAP
                    best_epoch = epoch_id
                    best_weights = parallel_model.get_weights()
                elif epoch_id - best_epoch >= 5:
                    break

            print('=========================================================')
            print('{0}\t\tStopping at epoch {1}, best epoch was {2} with MAP {3}'.format(datetime.datetime.now(),epoch_id, best_epoch, best_map))
            print('=========================================================')
            all_maps.append(best_map)

            print('{0}\t\tPredicting for test set...'.format(datetime.datetime.now()))
            # Reload best weights
            parallel_model.set_weights(best_weights)

            # Predict for test set
            p_test = parallel_model.predict([test_history, test_future], batch_size = 4096)
            test_predictions.append(np.swapaxes(p_test,0,1))

        print('=========================================================')
        print('{0}\t\tAverage best MAP over all folds:\t\t{1}...'.format(datetime.datetime.now(), np.mean(all_maps)))
        print('=========================================================')
        print('{0}\t\tGenerating submission...'.format(datetime.datetime.now()))
        # Geometric mean of predictions over folds
        p_test = np.prod(test_predictions, axis = 0) ** (1.0 / len(test_predictions))
        
        # Open submission file
        with open(path + 'predictions/p_{0}_{1}.csv'.format(args.sub_ID, date), 'w') as predictions:
            predictions.write('prediction\n')
            with open(path + 'submissions/sub_{0}_{1}.txt'.format(args.sub_ID, date),'w') as submission:
                # For every prediction row
                for prediction_row, size in zip(p_test, test_sizes):
                    # Placeholder for output
                    out = ''
                    # For every prediction
                    for i in range(size):
                        # Append to output
                        out += str(int(round(prediction_row[i][0])))
                        predictions.write(str(prediction_row[i][0]) + '\n')
                    # Write output
                    submission.write(out + '\n')

        # Print out how long it took
        day_end = datetime.datetime.now()
        print('{0}\t\tFinished date {1} in {2}...'.format(day_end, date, day_end - day_start))
