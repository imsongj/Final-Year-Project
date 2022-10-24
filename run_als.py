#! /usr/bin/env python3

import os
import sys
import numpy as np
from random import randint
from datetime import datetime
from argparse import ArgumentParser
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

SCRIPTS_PATH = '/data/share/users/zyma/h-axe/scripts'
JOB_JSON_PATH = SCRIPTS_PATH + '/job_confs/als.json'
MOVIE_FEATURE_OUTPUT_PATH = SCRIPTS_PATH + '/movie-feature-output'
USER_FEATURE_OUTPUT_PATH = SCRIPTS_PATH + '/user-feature-output'
IDMAP_OUTPUT_PATH = SCRIPTS_PATH + '/idmap-output'
NETFLIX_DATA_PATH = SCRIPTS_PATH + '/netflix_data'
FINAL_OUTPUT_PATH = SCRIPTS_PATH + '/final-output'
NUM_WORKER = 10
DATA_FILES = [
    '/combined_data_1_modified.txt',
    '/combined_data_2_modified.txt',
    '/combined_data_3_modified.txt',
    '/combined_data_4_modified.txt',
]

def make_probe_file(data_file): #create probe file with 1/100 size of data file TODO:: make new combined data and with deleted these values
    with open(NETFLIX_DATA_PATH + data_file) as file1, open(NETFLIX_DATA_PATH + '/probe_test.txt', 'a') as file2:
        movie_id = 0
        for line in file1:
            #print('line:' , line)
            if line[-2] == ':':
                movie_id = int(line[:-2])
                if movie_id % 10 == 0:
                    file2.write(line)
            else:
                if movie_id % 10 == 0:
                    rnd = np.random.uniform(0,1) 
                    if rnd < 0.05:
                        file2.write(line)

def make_spark_file(data_file, new_name): #create probe file with 1/100 size of data file TODO:: make new combined data and with deleted these values
    with open(NETFLIX_DATA_PATH + data_file) as file1, \
        open(NETFLIX_DATA_PATH + new_name, 'w') as file2:
        movie_id = 0
        for line in file1:
            #print('line:' , line)
            if line[-2] == ':':
                movie_id = int(line[:-2])
            else:
                x = line.split(',')
                file2.write('{},{},{}\n'.format(x[0], movie_id, x[1]))
                        
def make_qualifying_file(data_file, new_name):
    with open(NETFLIX_DATA_PATH + data_file) as file1, \
        open(NETFLIX_DATA_PATH + '/qualifying_test_unvarified.txt', 'a') as file2, \
        open(NETFLIX_DATA_PATH + new_name, 'w') as file3:
        movie_id = 0
        data_count = 0
        for line in file1:
            if line[-2] == ':':
                movie_id = int(line[:-2])
                if movie_id % 10 == 0:
                    file2.write(line)
                file3.write(line)
            else:
                if movie_id % 10 == 0:
                    rnd = np.random.uniform(0,1) 
                    if rnd < 0.1:
                        file2.write(line)
                        data_count = data_count + 1
                    else:
                        file3.write(line)
                else:
                        file3.write(line)

        print('extracted {} data from {} to {}\n'.format(data_count, data_file, '/qualifying_test.txt'))

def check_qualifying_file(id_map): #check and remove users that doesn't exist on data file anymore;
    with open(NETFLIX_DATA_PATH + '/qualifying_test_unvarified.txt') as file, \
        open(NETFLIX_DATA_PATH + '/qualifying_test.txt', 'w') as file2:
        count = 0
        for line in file:
            if line[-2] == ':':
                movie_id = int(line[:-2])
                file2.write(line)
            else:
                x = line.split(',')
                user_index = id_map.get(int(x[0]))
                if user_index != None:
                    file2.write(line)
                else:
                    count = count + 1
        print('removed {} from qualifying file\n'.format(count))




def get_ini(path):
    ini = {}
    with open(path) as file:
        for line in file:
            splits = line.split('=')
            ini[splits[0].strip()] = splits[1].strip()
    return ini

def read_feature_matrix(target_path, num_workers):
    feature_matrix = np.loadtxt(target_path + '/part0')
    for i in range(num_workers - 1):
        feature_matrix = np.append(feature_matrix, np.loadtxt(target_path + '/part' + str(i + 1)), axis = 0)
    feature_matrix = feature_matrix[feature_matrix[:,0].argsort()]
    feature_matrix = np.delete(feature_matrix, 0, 1)
    #for i in range(10):
    #    print(feature_matrix[i])
    print('shape of feature matrix: {}'.format(feature_matrix.shape))
    return feature_matrix

def read_idmap():
    id_map = {}
    with open(IDMAP_OUTPUT_PATH + '/part0') as file:
        for line in file:
            splits = line.split(' ')
            key = int(splits[0])
            id_map[key] = int(splits[1])
    return id_map

def read_probe_file(probe_file):
    with open(NETFLIX_DATA_PATH + '/' + probe_file) as file:
        num_lines = sum(1 for line in file if line[-2] != ':')
        movie_id = -1
        count = 0
        prediction_arr = np.zeros(shape =(num_lines, 4))
        file.seek(0, 0)
        for line in file:
            #print('line:' , line)
            if line[-2] == ':':
                movie_id = int(line[:-2])
            else:
                x = line.split(',')
                prediction_arr[count][0] = movie_id
                prediction_arr[count][1] = int(x[0]) #user id
                prediction_arr[count][2] = -1 #prediction rate
                prediction_arr[count][3] = float(x[1])    #actual rate
                count = count + 1
        return prediction_arr

def read_test_file(test_file): #read test file without actual ratings
    with open(NETFLIX_DATA_PATH + '/' + test_file) as file:
        num_lines = sum(1 for line in file if line[-2] != ':')
        movie_id = -1
        count = 0
        prediction_arr = np.zeros(shape =(num_lines, 4))
        for line in file:
            if line[-2] == ':':
                movie_id = int(line[:-2])
            else:
                x = line.split(',')
                prediction_arr[count][0] = movie_id
                prediction_arr[count][1] = int(x[0]) #user id
                prediction_arr[count][2] = -1 #prediction rate
                prediction_arr[count][3] = -1    #actual rate
                count = count + 1
        return prediction_arr

def read_movie_title(): 
    title_map = {}
    with open(NETFLIX_DATA_PATH + '/movie_titles.csv') as file:
        for line in file:
            splits = line.split(',')
            key = int(splits[0])
            title_map[key] = splits[2]
    return title_map


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calculate_rmse(prediction_arr):
    prediction_vector = [row[2] for row in prediction_arr]
    target_vector = [row[3] for row in prediction_arr]
    random_vector = [randint(0, 5) for row in prediction_arr]
    r = rmse(np.array(prediction_vector), np.array(target_vector))
    print('number of predictions: {}'.format(prediction_arr.shape[0]))
    print('rmse = {}\n'.format(r))
    return r

def find_watched_movie(user_id):
    print('Scanning for watched movies...')
    watched_set = set()
    user_id_str = str(user_id)
    for data_file in DATA_FILES:
        with open(NETFLIX_DATA_PATH + data_file) as file:
            movie_id = 0
            for line in file:
                #print('line:' , line)
                if line[-2] == ':':
                    movie_id = int(line[:-2])
                else:
                    if line[:len(user_id_str)] == user_id_str:
                        watched_set.add(movie_id)
    return watched_set

def print_watched_movies(watched_set, title_map):
    for movie_id in watched_set:
        print('{}, {}'.format(movie_id, title_map[movie_id]))

def plot_graph(rmse_arr):
    x = range(1, rmse_arr.shape[0] + 1)
    plt.plot(x, rmse_arr, marker='o', markersize=3)
    plt.ylabel("RMSE")
    plt.xlabel("number of iterations")
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.savefig(FINAL_OUTPUT_PATH + '/RMSE_plot.png')

def create_test_output(test, prediction_arr, ini, iteration, num_movies, num_users, timestamp):
    #config: num machines, parallelism, nf, lambda, iterations, seed, input output files
    #rmse, size of data, time spent, time stamp
    #prediction data file
    with open(FINAL_OUTPUT_PATH + '/statistics.txt', 'a') as file:
        # datetime object containing current date and time
        if test == 'true':
            rm = ''
            file.write('\n-------test run: {}\n'.format(timestamp))
        else:
            rm = calculate_rmse(prediction_arr)
            file.write('\n-------run: {}\n'.format(timestamp))
        
        file.write('number of workers: {}, parallelism: {}\n\n'.format(ini['n_job_processes'], ini['parallelism']))
        file.writelines('data: {}\nnumber of movies: {}, number of users: {}\n\n' \
            .format(ini['data'], num_movies, num_users))
        file.writelines('prediction: {}\nnumber of predictions: {}\n\n'.format(ini['test'], prediction_arr.shape[0]))
        file.writelines('nf: {}\nlambda: {}\niterations: {}\nseed: {}\n\n' \
            .format(ini['nf'], ini['lambda'], iteration, ini['seed']))
        file.write('rmse: {} \n'.format(rm))
    with open(FINAL_OUTPUT_PATH + '/prediction.txt', 'w') as file:
        current_movie_id = 0
        for item in prediction_arr:
            if item[0] != current_movie_id:
                current_movie_id = int(item[0])
                file.write('{}:\n'.format(current_movie_id))
            file.write('{}, {}\n'.format(int(item[1]), item[2]))
    return rm


def make_prediction(target_file, id_map, iteration, num_workers):
    movie_feature_matrix = read_feature_matrix(MOVIE_FEATURE_OUTPUT_PATH + '/itr' + str(iteration), num_workers)
    user_feature_matrix = read_feature_matrix(USER_FEATURE_OUTPUT_PATH + '/itr' + str(iteration), num_workers)
    prediction_arr = read_probe_file(target_file)
    num_movies, _ = movie_feature_matrix.shape
    num_users, _ = user_feature_matrix.shape
    for item in prediction_arr:
        movie_id = int(item[0])
        user_index = id_map.get(item[1])
        if user_index == None:
            item[2] = -1
        else:
            movie_feature_row = movie_feature_matrix[movie_id - 1] # 1 * nf
            T_user_feature_row = np.transpose(user_feature_matrix[int(user_index)]) # 1 * nf
            item[2] = np.matmul(movie_feature_row, T_user_feature_row)
    return prediction_arr, num_movies, num_users


def make_recommendation(user_id, user_index, iteration, num_workers):
    title_map = read_movie_title()
    if user_index == None:
        print('User id: {} is a new user.\nRecommendation of 7 random movies to begin with: \n'.format(user_id))
        for i in range(7):
            print('    {}. {}\n'.format(i + 1, title_map[randint(0, len(title_map) - 1)]))
    else:
        watched_set = find_watched_movie(user_id)
        movie_feature_matrix = read_feature_matrix(MOVIE_FEATURE_OUTPUT_PATH + '/itr' + str(iteration), num_workers)
        user_feature_matrix = read_feature_matrix(USER_FEATURE_OUTPUT_PATH + '/itr' + str(iteration), num_workers)
        
        num_movies, _ = movie_feature_matrix.shape
        recommendation_arr = np.zeros(shape =(num_movies, 3))
        for i in range(num_movies):
            recommendation_arr[i][0] = i + 1
            recommendation_arr[i][1] = user_id #user id
            movie_feature_row = movie_feature_matrix[i] # 1 * nf
            T_user_feature_row = np.transpose(user_feature_matrix[int(user_index)]) # 1 * nf
            recommendation_arr[i][2] = np.matmul(movie_feature_row, T_user_feature_row)
        with open(FINAL_OUTPUT_PATH + '/recommendation.txt', 'w') as file:
            for item in recommendation_arr:
                file.write('{}:\n'.format(item[0]))
                file.write('{}, {}\n'.format(int(item[1]), item[2]))
        recommendation_arr = recommendation_arr[recommendation_arr[:,2].argsort()]
        print_watched_movies()
        print('Recommendation of 7 movies based on prediction: \n')
        count = 0
        for i in range(num_movies):
            if recommendation_arr[-(i+1)][0] not in watched_set:
                #print(recommendation_arr[-(i+1)])
                print('    {}. {}\n'.format(count + 1, title_map[recommendation_arr[-(i+1)][0]]))
                count = count + 1
                if count >= 7:
                    break
        print('full predicted ratings in {}'.format(FINAL_OUTPUT_PATH + '/recommendation.txt'))

def submit_ursa_job():
    os.system('./submit-job.py --config {} -t'.format(JOB_JSON_PATH ))

if __name__ == '__main__':
    parser = ArgumentParser(description= 'Recommendation System on Netflix Data. -h to see more usage')
    parser.add_argument('-ALS', '--als', action='store_true', help='Optimize feature matrices using ALS program on Ursa Cluster')
    parser.add_argument('-P', '--prev', action='store_true', help='Record statistic for previous iterations')
    parser.add_argument('-T', '--test', action='store_true', help='Test with test data without truth rating value(no RMSE calculation)')
    parser.add_argument('-R', '--reco', action='store_true', help='Make recommendation for a specific user')
    
    args = parser.parse_args()

    if args.als == True:
        print('Submiting ALS job')
        submit_ursa_job()
        print('ALS job finished. Optimizing done')
    
    id_map = read_idmap()
    ini = get_ini(SCRIPTS_PATH + '/job_confs/als.ini')
    iterations = int(ini['iterations'])
    num_workers = int(ini['n_job_processes'])
    if args.reco == True:
        user_id = input("Enter user id for recommendation: ")
        user_index = id_map.get(user_id)
        make_recommendation(user_id, user_index, iterations, num_workers)
    else:
        
        if args.test == True:
            test_path = 'test.txt'
        else:
            test_path = 'qualifying_test.txt'
        ini['test'] = test_path
        if args.prev == True:
            rmse_arr = np.zeros((iterations))
            for itr in range(iterations):
                print('Processing for iteration: {}'.format(itr + 1))
                prediction_arr, num_movies, num_users = make_prediction(test_path, id_map, itr + 1, num_workers)
                for element in prediction_arr:
                    if element[2] == -1:
                        print('error: non-existing user id: ' + str(element[1]))
                        sys.exit()
                now = datetime.now()
                rmse_arr[itr] = create_test_output(args.test, prediction_arr,  ini, itr + 1, num_movies, num_users, now)
            plot_graph(rmse_arr)

        else:        
            prediction_arr, num_movies, num_users = make_prediction(test_path, id_map, iterations, num_workers)
            for element in prediction_arr:
                if element[2] == -1:
                    print('error: non-existing user id: ' + str(element[1]))
                    sys.exit()
            now = datetime.now()
            create_test_output(args.test, prediction_arr,  ini, iterations, num_movies, num_users, now)
    

