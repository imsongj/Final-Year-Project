// Copyright 2020 HDL
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <memory>
#include <numeric>
#include <algorithm>
#include <queue>
#include <random>
#include <vector>
#include <cmath>
#include <math.h>

#include "glog/logging.h"
#include "base/tokenizer.h"
#include "common/engine.h"

#include "../../tools/lib/eigen-3.4.0/Eigen/Dense"

using ListPtr = std::shared_ptr<std::vector<std::pair<int,double>>>; //
using MatrixRow = std::vector<std::pair<int, double>>; //int = row index, double = value
using PredictionItem = std::pair<std::pair<int, int>, std::pair<int, double>>; //user id, user index order, movie id, rating

class Item { //change to user and Item strut
 public:
  Item() : rate_list_(std::make_shared<std::vector<std::pair<int,double>>>()) {}
  Item(int id) : item_id_(id), rate_list_(std::make_shared<std::vector<std::pair<int,double>>>()) {}
  Item(int id, const ListPtr& rate_list) : item_id_(id), rate_list_(rate_list) {}

  int GetId() const { return item_id_; }
  const ListPtr& GetRateList() const { return rate_list_; }

  double GetMemory() const {
    double ret = sizeof(int) + rate_list_->size() * sizeof(std::pair<int,double>);
    return ret;
  }
  bool operator<(const Item& other) const { return item_id_ < other.item_id_; }
  bool operator==(const Item& other) const { return item_id_ == other.item_id_; }

  friend void operator<<(axe::base::BinStream& bin_stream, const Item& i) { bin_stream << i.item_id_ << *(i.rate_list_); }
  friend void operator>>(axe::base::BinStream& bin_stream, Item& i) { bin_stream >> i.item_id_ >> *(i.rate_list_); }

 private:
  int item_id_;
  ListPtr rate_list_;
};

int ReadInt(const std::string& line, size_t& ptr) {
  int ret = 0;
  while (ptr < line.size() && !isdigit(line.at(ptr)))
    ++ptr;
  CHECK(ptr < line.size()) << "Invalid Input";
  while (ptr < line.size() && isdigit(line.at(ptr))) {
    ret = ret * 10 + line.at(ptr) - '0';
    ++ptr;
  }
  return ret;
}

std::vector<std::vector<double>> CreateIdentityMatrix(int dim, double initial_value = 1.0){
  std::vector<std::vector<double>> i_matrix; //std::vector<std::pair<int, double>>;
  for(int i = 0; i < dim; i++){
    std::vector<double> row;
    for(int j = 0; j < dim; j++){
      double element;
      if(i == j){
        element = initial_value;
      }else{
        element = 0;
      }
      row.push_back(element);
    }
    i_matrix.push_back(row);
  }
  return i_matrix;
}
double Product(double x, double* y) {return x * (*y);}

std::vector<std::vector<double>> LocalMatrixMul(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B){
    std::vector<std::vector<double>> result;

    int rowA = static_cast<int>(A.size());
    int colA = static_cast<int>(A[0].size());
    int rowB = static_cast<int>(B.size());
    int colB = static_cast<int>(B[0].size());
    
    if(colA != rowB)return result;
    result.resize(rowA, std::vector<double>(colB));
    std::vector<double*> colPtr;
    for(int i = 0; i < rowB; i++) colPtr.push_back(&B[i][0]);
    for(int c = 0; c < colB; c++){
      for(int r = 0; r < rowA; r++){
          result[r][c] = std::inner_product(A[r].begin(),A[r].end(),colPtr.begin(), 0.0f, std::plus<double>(), Product);
      }
      std::transform(colPtr.begin(),colPtr.end(),colPtr.begin(),[](double* x){return ++x;});
    }
    return result;
}
std::vector<std::vector<double>> LocalTranspose(std::vector<std::vector<double>> &source){
  std::vector<std::vector<double>> result;
  int row = static_cast<int>(source.size());
  int col = static_cast<int>(source[0].size());
  result.resize(col, std::vector<double>(row));
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      result[j][i] = source[i][j];
    }
  }
  return result;
}
std::vector<std::vector<double>> LocalGetCofactor(std::vector<std::vector<double>> A, int p, int q, int n, int dim){
    int i = 0, j = 0;
    std::vector<std::vector<double>> temp;
    temp.resize(dim, std::vector<double>(dim));
    for (int row = 0; row < n; row++){
        for (int col = 0; col < n; col++){
            if (row != p && col != q){
                temp[i][j++] = A[row][col];
                if (j == n - 1){
                    j = 0;
                    i++;
                }
            }
        }
    }
    return temp;
}
double LocalDeterminant(std::vector<std::vector<double>> &matrix, int N) {
    //int N = static_cast<int>(matrix.size());
    double det = 1;

    for (int i = 0; i < N; ++i) {

        double pivotElement = matrix[i][i];
        int pivotRow = i;
        for (int row = i + 1; row < N; ++row) {
            if (std::abs(matrix[row][i]) > std::abs(pivotElement)) {
                pivotElement = matrix[row][i];
                pivotRow = row;
            }
        }
        if (pivotElement == 0.0f) {
            return 0.0f;
        }
        if (pivotRow != i) {
            matrix[i].swap(matrix[pivotRow]);
            det *= -1.0f;
        }
        det *= pivotElement;

        for (int row = i + 1; row < N; ++row) {
            for (int col = i + 1; col < N; ++col) {
                matrix[row][col] -= matrix[row][i] * matrix[i][col] / pivotElement;
            }
        }
    }

    return det;
}

std::vector<std::vector<double>> LocalAdjoint(std::vector<std::vector<double>> A, int dim){
    std::vector<std::vector<double>> adj;
    adj.resize(dim, std::vector<double>(dim));
    
    if (dim == 1){
        adj[0][0] = 1;
        return adj;
    }
 
    int sign = 1;
    std::vector<std::vector<double>> temp; // To store cofactors
 
    for (int i=0; i<dim; i++){
        for (int j=0; j<dim; j++){
            temp = LocalGetCofactor(A, i, j, dim, dim);
            sign = ((i+j)%2==0)? 1: -1;
 
            adj[j][i] = (sign)*(LocalDeterminant(temp, dim-1));
        }
    }
    return adj;
}
 
std::vector<std::vector<double>> LocalInverse(std::vector<std::vector<double>> A, int dim){   
    std::vector<std::vector<double>> inverse;
    inverse.resize(dim, std::vector<double>(dim));
    std::vector<std::vector<double>> copy = A;
    double det = LocalDeterminant(copy, dim);
    if (det == 0){
        LOG(WARNING) << "Singular matrix, can't find its inverse";
        return inverse;
    }
    
    std::vector<std::vector<double>> adj;
    adj = LocalAdjoint(A, dim);
    
    
    for (int i=0; i<dim; i++)
        for (int j=0; j<dim; j++)
            inverse[i][j] = adj[i][j]/double(det);
 
    return inverse;
}
std::vector<std::vector<double>> LocalScalarMul(double scalar, std::vector<std::vector<double>>& matrix){
  int row = static_cast<int>(matrix.size());
  int col = static_cast<int>(matrix.at(0).size());
  std::vector<std::vector<double>> result;
  result.resize(row, std::vector<double>(col));
  
  for (int i=0; i<row; i++){
    for (int j=0; j<col; j++){
      result[i][j] = scalar * matrix[i][j];
    }
  }
  return result;
}
std::vector<std::vector<double>> LocalMatrixSum(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B){
  int row = static_cast<int>(A.size());
  int col = static_cast<int>(B.at(0).size());
  std::vector<std::vector<double>> result;
  result.resize(row, std::vector<double>(col));
    
  for (int i=0; i<row; i++){
    for (int j=0; j<col; j++){
      result[i][j] = A[i][j] + B[i][j];
    }
  }
  return result;
}
std::vector<std::vector<double>> ItemToMarix(Item item){ //returns num_element * 1 matrix
  std::vector<std::vector<double>> matrix; //std::vector<std::pair<int, double>>;
  ListPtr rate_list = item.GetRateList();
  std::vector<double> row;
  for(const auto& pair : *rate_list){
    double element = pair.second;;
    row.push_back(element);
  }
  matrix.push_back(row);
  return matrix;
}
void DisplayMarix(std::vector<std::vector<double>> A){
  int n = static_cast<int>(A.size());
  int m = static_cast<int>(A.at(0).size());
  for (int i=0; i<n; i++){
    std::string output;
    for (int j=0; j<m; j++){
      output = output + std::to_string(A[i][j]) + ", ";
    }
    LOG(INFO) << output;
  }
  LOG(INFO) << " ";
}

Eigen::MatrixXd ConvertMatrix(std::vector<std::vector<double>> target){
  int num_row = target.size();
  int num_col = target[0].size();
  Eigen::MatrixXd m(num_row, num_col);
  for(int i = 0; i < num_row; i++){
    for(int j = 0; j < num_col; j++){
      m(i, j) = target[i][j];
    }
  }
  return m;
}

DatasetPartition<Item> ParseLine(const std::string& line, std::shared_ptr<std::vector<int>> num_movie_vec, int i) {
  size_t ptr = 0;
  DatasetPartition<Item> ret;
  int movie_id;
  
  if((line[line.length() - 2] == ':') || (line[line.length() - 1] == ':')){ //get Item id; returns empty Item obj
    num_movie_vec->at(i) = ReadInt(line, ptr);
    movie_id = num_movie_vec->at(i);
  }else{
    movie_id = num_movie_vec->at(i);
    auto movie_rate_list = std::make_shared<std::vector<std::pair<int,double>>>();
    int user_id;
    int rate;
    axe::base::StrtokTokenizer tokenizer(line, ","); 
    char* tok;
    tok = tokenizer.next();
    user_id = std::atoi(tok);
    tok = tokenizer.next();
    rate = std::atoi(tok);
    movie_rate_list->push_back(std::make_pair(movie_id - 1, (double)rate));
    ret.push_back(Item(user_id, movie_rate_list));
  }
  return ret;
}

void PrintItem(const Item& item){
    std::string output = "Item ID = " + std::to_string(item.GetId()) + ": ";
    ListPtr tmpList = item.GetRateList();
    for(const auto& kv : *tmpList){
        std::string tmp = std::to_string(kv.first) + " " + std::to_string(kv.second) + ", ";
        output += tmp;
    }
    LOG(INFO) << output;
}

class ALS : public Job {
 public:
  void Run(TaskGraph* tg, const std::shared_ptr<Properties>& config) const override {
    auto input_dirs_str = config->GetOrSet("data", "");
    int n_partitions = std::stoi(config->GetOrSet("parallelism", "20"));
    int nf = std::stoi(config->GetOrSet("nf", "8"));
    double lambda = std::stod(config->GetOrSet("lambda", "0.05"));
    int n_iterations = std::stoi(config->GetOrSet("iterations", "5"));
    int seed = std::stoi(config->GetOrSet("seed", "100"));

    std::shared_ptr<std::vector<int>> num_movie_vec = std::make_shared<std::vector<int>>(); 
    std::shared_ptr<int> test_movie_vec = std::make_shared<int>(0); 
    std::shared_ptr<int> counter = std::make_shared<int>(0); 

    axe::base::StrtokTokenizer tokenizer(input_dirs_str, ","); //separate input file dirs
    char* tok;
    std::vector<std::string> input_dirs_vec = std::vector<std::string>();
    tok = tokenizer.next();
    int num_input_files = 0;
    while(tok != NULL){ //get input dir
      input_dirs_vec.push_back(tok);
      //LOG(INFO) << tok;
      tok = tokenizer.next();
      num_input_files +=1;
    }
    std::shared_ptr<std::vector<int>> user_id_list = std::make_shared<std::vector<int>>(); //delete                       
    // Load data
    for(int i = 0; i < num_input_files; i++){
      int tmp = 0;
      num_movie_vec->push_back(tmp);
    }
    auto input_data = TextSourceDataset(input_dirs_vec[0], tg, 1) //load first input file
                      .FlatMap([num_movie_vec](const std::string& line) { return ParseLine(line, num_movie_vec, 0); },
                      [](const std::vector<double>& input) {
                        double ret = 0;
                        for (double x : input) {
                          ret += x;
                        }
                        return ret * 2;
                      })
                      .PartitionBy([](const Item& i) { return i.GetId(); }, n_partitions);
    for(int i = 1; i < num_input_files; i++){
      auto additional_input_data = TextSourceDataset(input_dirs_vec[i], tg, 1) //load additional input files
                      .FlatMap([num_movie_vec, i](const std::string& line) { return ParseLine(line, num_movie_vec, i); },
                      [](const std::vector<double>& input) {
                        double ret = 0;
                        for (double x : input) {
                          ret += x;
                        }
                        return ret * 2;
                      })
                      .PartitionBy([](const Item& i) { return i.GetId(); }, n_partitions);

      input_data.UpdatePartitionWith(&additional_input_data, 
                  [](DatasetPartition<Item>& input_data, DatasetPartition<Item> additional_input_data){
                    for(Item& item : additional_input_data){
                      input_data.push_back(item);
                    }
                  });
    }  
    
    auto R_users = std::make_shared<axe::common::Dataset<Item>>(input_data.ReduceByWithoutLocalAgg([](const Item& ele) { return ele.GetId(); }, //get rate matrix(user oriented)
                [](Item& agg, const Item& ele) {
                  ListPtr aggList = agg.GetRateList();
                  ListPtr eleList = ele.GetRateList(); 
                  for(const auto& pair : *eleList){
                    aggList->push_back(pair);
                  }
                },
                n_partitions));     
    auto user_id_map = std::make_shared<axe::common::Dataset<std::pair<int, int>>>(R_users->MapPartition([](const DatasetPartition<Item>& users){ //map pairs for user id and row id auto user_id_map = 
                            DatasetPartition<std::pair<int, int>> id_list;
                            for(const Item& item : users){
                              id_list.push_back(std::make_pair(item.GetId(), 0));
                            }
                            return id_list;
                          }));     

    auto local_user_id_map = user_id_map->Broadcast([](auto ele) { return ele.first; }, n_partitions); //sorted user by id
    local_user_id_map.UpdatePartition([](auto& id_list) {
                            int count = 0;
                            for(auto& id : id_list){
                              id.second = count;
                              count++;
                            }
                          });
    R_users = std::make_shared<axe::common::Dataset<Item>>(R_users->SharedDataMapPartitionWith(&local_user_id_map, //update user id to index 0-num_users
                  [](const DatasetPartition<Item>& users, const DatasetPartition<std::pair<int, int>>& id_map){
                   DatasetPartition<Item> new_users;
                    int offset = 0;
                    int num_users = id_map.size();
                    //LOG(INFO) << "num_users " << num_users;
                    for(const Item& user : users){
                      int user_id = user.GetId();
                      int user_index = -1;
                      for(int i = offset; i < num_users; i++){
                        //LOG(INFO) << "id map " << id_map.at(i).first;
                        if(id_map.at(i).first == user_id){
                          user_index = i;
                          offset = i;
                          break;
                        }
                      }
                      const ListPtr rate_list = user.GetRateList();
                      new_users.push_back(Item(user_index, rate_list));
                    }
                    return new_users;
                  })); 
    auto ParseMovie = [](const DatasetPartition<Item>& input_data){
      DatasetPartition<Item> movies_map;
      for(const Item& item : input_data){
        ListPtr rate_list = item.GetRateList();
        int user_id, movie_id;
        double rate;
        user_id = item.GetId();
        for(const auto& pair : *rate_list){
          auto user_rate_list = std::make_shared<std::vector<std::pair<int, double>>>();
          movie_id = pair.first;
          rate = pair.second;
          user_rate_list->push_back(std::make_pair(user_id, rate));
          movies_map.push_back(Item(movie_id, user_rate_list));
          //LOG(INFO) << user_id << " " <<movie_id << " " << rate;
        }
      }
      return movies_map;     
    };
    auto movies = R_users->MapPartition(ParseMovie) //get rate matrix(movie oriented)
                .ReduceBy([](const Item& ele) { return ele.GetId(); },
                [](Item& agg, const Item& ele) {
                  ListPtr aggList = agg.GetRateList();
                  ListPtr eleList = ele.GetRateList(); 
                  for(const auto& pair : *eleList){
                    aggList->push_back(pair);
                  }
                },
                n_partitions);
    movies.UpdatePartition([](DatasetPartition<Item>& movies){
                            for(const Item& item : movies){
                              ListPtr rateList = item.GetRateList();
                              std::sort((*rateList).begin(), (*rateList).end());
                            }
                          }); //update partition
                          
    
    auto movie_avg_rating = movies.MapPartition([](const DatasetPartition<Item>& movies){ //calculate movie avg rating
                                                   DatasetPartition<std::pair<int, double>> movie_ratings;
                                                    for(const Item& item : movies){
                                                      int movie_id = item.GetId();
                                                      ListPtr rate_list = item.GetRateList();
                                                      int len = rate_list->size();
                                                      double sum = 0;
                                                      for(const auto pair : *rate_list){
                                                        sum += pair.second;
                                                      }
                                                      movie_ratings.push_back(std::make_pair(movie_id, sum / len)); 
                                                    }
                                                    return movie_ratings;//maybe do this in mapping movies
                                                }); //not in order?
    auto movie_feature_matrix = movies.MapPartitionWith(&movie_avg_rating, //MatrixRow structure needed to broadcast and order them in correct order
                                [nf, seed](const DatasetPartition<Item>& movies, const DatasetPartition<std::pair<int, double>>& avg_ratings){
                                  DatasetPartition<MatrixRow> feature_matrix;
                                  int i = 0;
                                  for(const Item& item : movies){
                                    double avg = avg_ratings.at(i).second;
                                    int movie_id = item.GetId();
                                    //LOG(INFO) << "feature matrix movie id: " << movie_id;
                                    MatrixRow row; 
                                    if(avg_ratings.at(i).first != item.GetId()){
                                      LOG(INFO) << "````````````different id ";
                                      break;
                                    }
                                    std::mt19937 mt(seed);
                                    for(int j = 0; j < nf; j++){
                                      if(j == 0){
                                        row.push_back(std::make_pair(movie_id, avg));
                                      }else{
                                        //std::random_device rd;
                                        //std::mt19937 mt(rd());
                                        std::uniform_real_distribution<double> dist(0., 1.);
                                        double rnd = dist(mt) / 100;
                                        row.push_back(std::make_pair(movie_id, rnd));
                                      }
                                    }
                                    feature_matrix.push_back(row);
                                    i++;  
                                  }
                                  return feature_matrix;
                                });
                                
    auto local_movie_feature_matrix = movie_feature_matrix.Broadcast([](auto ele) { return ele.at(0).first; }, n_partitions)
                      .MapPartition([](const DatasetPartition<MatrixRow>& movie_feature_matrix){ //convert distributed matrix to local matrix 
                        DatasetPartition<std::vector<double>> new_feature_matrix;
                        for(const MatrixRow& row : movie_feature_matrix){
                          std::vector<double> new_row;
                          for(const std::pair<int, double>& pair : row){
                            new_row.push_back(pair.second);
                          }
                          new_feature_matrix.push_back(new_row);
                        }
                        return new_feature_matrix;
                      }); //num_movies * nf
    auto user_feature_matrix = R_users->MapPartition([](const DatasetPartition<Item>& users){
                                DatasetPartition<MatrixRow> user_feature_matrix;
                                for(const Item& item : users){
                                  MatrixRow dummy;
                                  dummy.push_back(std::make_pair(item.GetId(), 3.0f));
                                  user_feature_matrix.push_back(dummy);
                                }
                                return user_feature_matrix;
                              });
                               //create empty feature matrix
    auto local_user_feature_matrix = user_feature_matrix.MapPartition([](const DatasetPartition<MatrixRow>& user_feature_matrix){ //convert distributed matrix to local matrix //TODO::later use prev name
                                      DatasetPartition<std::vector<double>> new_feature_matrix;
                                      for(const MatrixRow& row : user_feature_matrix){
                                          std::vector<double> dummy;
                                          for(const std::pair<int, double>& pair : row){
                                            dummy.push_back(pair.second);
                                          }
                                          new_feature_matrix.push_back(dummy);
                                        }
                                      return new_feature_matrix;
                                    }); //create local empty feature matrix
    
    for(int itr = 0; itr < n_iterations; itr++){ 
      //update U
      user_feature_matrix = R_users->SharedDataMapPartitionWith(&local_movie_feature_matrix, 
                            [nf, lambda](const DatasetPartition<Item>& users, const DatasetPartition<std::vector<double>>& local_movie_feature_matrix){
                              DatasetPartition<MatrixRow> new_feature_matrix;
                              int num_movies = local_movie_feature_matrix.size();
                              std::vector<std::vector<double>> lambda_identity = CreateIdentityMatrix(nf, lambda); //nf * nf
                              Eigen::MatrixXd e_identity = ConvertMatrix(lambda_identity);
                              //DisplayMarix(lambda_identity);
                              for(const Item& user : users){
                                int user_id = user.GetId();
                                std::vector<std::vector<double>> sub_feature_matrix; //num_movies rated by this user * nf
                                ListPtr rate_list = user.GetRateList();
                                for(const auto& rated_movie : *rate_list){ //add movies rated by user
                                  int movie_id = rated_movie.first;
                                  if(movie_id < 0){
                                    LOG(WARNING) << "#####error feature: get movie id " << movie_id;
                                  }else{
                                    std::vector<double> feature_row = local_movie_feature_matrix[movie_id];
                                    sub_feature_matrix.push_back(feature_row);
                                  }
                                }
                                std::vector<std::vector<double>> user_vector = ItemToMarix(user);
                                Eigen::MatrixXd e_user_v = ConvertMatrix(user_vector);
                                Eigen::MatrixXd e_sub_fm = ConvertMatrix(sub_feature_matrix);
                                Eigen::MatrixXd r_v = e_sub_fm.transpose()*e_user_v.transpose();
                                Eigen::MatrixXd r_m_1 = e_sub_fm.transpose()*e_sub_fm;
                                Eigen::MatrixXd r_m_2 = sub_feature_matrix.size() * e_identity;
                                Eigen::MatrixXd r_m = r_m_1 + r_m_2;
                                Eigen::MatrixXd inv_r_m = r_m.inverse();
                                Eigen::MatrixXd new_m = inv_r_m * r_v;
                                MatrixRow new_row;  
                                for(int i = 0; i < new_m.rows(); i++){ //convert to MatrixRow
                                  new_row.push_back(std::make_pair(user_id, new_m(i, 0)));
                                }
                                new_feature_matrix.push_back(new_row);
                              }
                              return new_feature_matrix;
                            });
                            
      local_user_feature_matrix = user_feature_matrix.Broadcast([](auto ele) { return ele.at(0).first; }, n_partitions)
                                  .MapPartition([](const DatasetPartition<MatrixRow>& user_feature_matrix){ //convert distributed matrix to local matrix //TODO::later use prev name
                                        DatasetPartition<std::vector<double>> new_feature_matrix;
                                        for(const MatrixRow& row : user_feature_matrix){
                                          std::vector<double> new_row;
                                          for(const std::pair<int, double>& pair : row){
                                            new_row.push_back(pair.second);
                                          }
                                          new_feature_matrix.push_back(new_row);
                                        }
                                        return new_feature_matrix;
                                      }); //num_movies * nf
      movie_feature_matrix = movies.SharedDataMapPartitionWith(&local_user_feature_matrix, 
                            [nf, lambda](const DatasetPartition<Item>& movies, const DatasetPartition<std::vector<double>>& local_user_feature_matrix){
                              DatasetPartition<MatrixRow> new_feature_matrix;
                              int num_users = local_user_feature_matrix.size();
                              std::vector<std::vector<double>> lambda_identity = CreateIdentityMatrix(nf, lambda); //nf * nf
                              Eigen::MatrixXd e_identity = ConvertMatrix(lambda_identity);
                              //DisplayMarix(lambda_identity);
                              for(const Item& movie : movies){
                                int movie_id = movie.GetId();
                                std::vector<std::vector<double>> sub_feature_matrix; //num_movies rated by this user * nf
                                ListPtr rate_list = movie.GetRateList();
                                for(const auto& rated_user : *rate_list){ //add movies rated by user
                                  int user_id = rated_user.first;
                                  if(user_id < 0){
                                    LOG(WARNING) << "#####error feature: get user id " << user_id;
                                  }else{
                                    std::vector<double> feature_row = local_user_feature_matrix[user_id];
                                    sub_feature_matrix.push_back(feature_row);
                                  }
                                }//change row index to 0-num_rows
                        
                                std::vector<std::vector<double>> movie_vector = ItemToMarix(movie);
                                Eigen::MatrixXd e_movie_v = ConvertMatrix(movie_vector);
                                Eigen::MatrixXd e_sub_fm = ConvertMatrix(sub_feature_matrix);
                                Eigen::MatrixXd r_v = e_sub_fm.transpose()*e_movie_v.transpose();
                                Eigen::MatrixXd r_m_1 = e_sub_fm.transpose()*e_sub_fm;
                                Eigen::MatrixXd r_m_2 = sub_feature_matrix.size() * e_identity;
                                Eigen::MatrixXd r_m = r_m_1 + r_m_2;
                                Eigen::MatrixXd inv_r_m = r_m.inverse();
                                Eigen::MatrixXd new_m = inv_r_m * r_v;
                                MatrixRow new_row;  
                                for(int i = 0; i < new_m.rows(); i++){ //convert to MatrixRow
                                  new_row.push_back(std::make_pair(movie_id, new_m(i, 0)));
                                }
                                new_feature_matrix.push_back(new_row);
                              }
                              return new_feature_matrix;
                            });
                            
      local_movie_feature_matrix = movie_feature_matrix.Broadcast([](auto ele) { return ele.at(0).first; }, n_partitions)
                  .MapPartition([](const DatasetPartition<MatrixRow>& movie_feature_matrix){ //convert distributed matrix to local matrix //TODO::later use prev name
                  DatasetPartition<std::vector<double>> new_feature_matrix;
                  for(const MatrixRow& row : movie_feature_matrix){
                    std::vector<double> new_row;
                    for(const std::pair<int, double>& pair : row){
                      new_row.push_back(pair.second);
                    }
                    new_feature_matrix.push_back(new_row);
                  }
                  return new_feature_matrix;
                }); //num_movies * nf
      //update M
      movie_feature_matrix.SaveAs("/data/share/users/zyma/h-axe/scripts/movie-feature-output/itr" + std::to_string(itr + 1), 
      [itr](DatasetPartition<MatrixRow>* feature_matrix, std::ofstream& ofs) {
      for (const MatrixRow& row : *feature_matrix) {
        int row_num = row[0].first;
        ofs << row_num;
        for(const std::pair<int, double>& pair : row){
          ofs << " " << pair.second;
        }
        ofs << std::endl;
      }
      LOG(INFO) << "movie feature saved for itr " + std::to_string(itr);
      });
      
      user_feature_matrix.SaveAs("/data/share/users/zyma/h-axe/scripts/user-feature-output/itr" + std::to_string(itr + 1), 
        [itr](DatasetPartition<MatrixRow>* feature_matrix, std::ofstream& ofs) {
        for (const MatrixRow& row : *feature_matrix) {
          int row_num = row[0].first;
          ofs << row_num;
          for(const std::pair<int, double>& pair : row){
            ofs << " " << pair.second;
          }
          ofs << std::endl;
        }
        LOG(INFO) << "user feature saved for itr " + std::to_string(itr);
      });
    }
    
    local_user_id_map.SaveAs("/data/share/users/zyma/h-axe/scripts/idmap-output", [](DatasetPartition<std::pair<int, int>>* map, std::ofstream& ofs) {
      for (const std::pair<int, int>& pair : *map) {
        ofs << pair.first << " " << pair.second;
        ofs << std::endl;
      }
      LOG(INFO) << "id map saved";
    });
    
    axe::common::JobDriver::ReversePrintTaskGraph(*tg);              
  }
};

int main(int argc, char** argv) {
  axe::common::JobDriver::Run(argc, argv, ALS());
  return 0;
}
