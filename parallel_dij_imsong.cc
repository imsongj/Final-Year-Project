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

#include <string.h>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "glog/logging.h"

#include "common/engine.h"

#define MAX_INT 2147000000

class Node{
  public:
    Node() : neighbor_list_(std::make_shared<std::vector<std::pair<int,int>>>()) {}
    Node(int node_id, int min_distance, int parent_node_id) : 
      node_id_(node_id), min_distance_(min_distance), parent_node_id_(parent_node_id), neighbor_list_(std::make_shared<std::vector<std::pair<int,int>>>()) {}
    Node(int node_id, int min_distance, int parent_node_id, const std::shared_ptr<std::vector<std::pair<int,int>>>& neighbor_list) : 
      node_id_(node_id), min_distance_(min_distance), parent_node_id_(parent_node_id), neighbor_list_(neighbor_list) {}
    
    int GetNodeID() const { return node_id_; }
    int GetMinDistance() const { return min_distance_; }
    int GetParentNodeID() const { return parent_node_id_; }
    const std::shared_ptr<std::vector<std::pair<int,int>>>& GetNeighborList() const { return neighbor_list_; }

    double GetMemory() const {
      double ret = sizeof(int) * 3 + neighbor_list_->size() * sizeof(std::pair<int,int>);
      return ret;
    }
    bool operator<(const Node& other) const { return node_id_ < other.node_id_; }
    bool operator==(const Node& other) const { return node_id_ == other.node_id_; }

    friend void operator<<(axe::base::BinStream& bin_stream, const Node& n) { bin_stream << n.node_id_ << n.min_distance_ << n.parent_node_id_ << *(n.neighbor_list_); }
    friend void operator>>(axe::base::BinStream& bin_stream, Node& n) { bin_stream >> n.node_id_ >> n.min_distance_ >> n.parent_node_id_ >> *(n.neighbor_list_); }

  private:
    int node_id_;
    int min_distance_;
    int parent_node_id_;
    std::shared_ptr<std::vector<std::pair<int,int>>> neighbor_list_;
};

DatasetPartition<Node> ParseLine(const std::string& line, const int start_node_id) {
  char* pos;
  std::unique_ptr<char[]> record_ptr(new char[line.size() + 1]);
  strncpy(record_ptr.get(), line.data(), line.size());
  record_ptr.get()[line.size()] = '\0';
  char* tok = strtok_r(record_ptr.get(), " ", &pos);

  int i = 0;
  int node_id;
  int neighbor_node_id;
  int neighbor_distance;
  auto neighbor_list = std::make_shared<std::vector<std::pair<int,int>>>();
  std::pair<int,int> tmp;
  while (tok != NULL) {
    if (i == 0) {
      node_id = std::atoi(tok);
      i = 1;
    } else if (i == 1) {
      neighbor_node_id = std::atoi(tok);
      i = 2;
    } else {
      neighbor_distance = std::atoi(tok);
      i = 3;
    }
    tok = strtok_r(NULL, " ", &pos);
  }
  neighbor_list->push_back(std::make_pair(neighbor_node_id, neighbor_distance));
  DatasetPartition<Node> ret;
  if(node_id == start_node_id){
    ret.push_back(Node(node_id, 0, -1, neighbor_list));
  }else{
    ret.push_back(Node(node_id, MAX_INT, -1, neighbor_list));
  }
  
  return ret;
}

void PrintNode(const Node& node, int iter){
    std::string output = "Node ID = " + std::to_string(node.GetNodeID()) 
    + " minDist = " + std::to_string(node.GetMinDistance())
    + " parent_node_id = " + std::to_string(node.GetParentNodeID()) + " neighbors = ";
    std::shared_ptr<std::vector<std::pair<int,int>>> tmpList = node.GetNeighborList();
    for(int i = 0; i < tmpList->size(); i++){
        std::string tmp = std::to_string(tmpList->at(i).first) + " " + std::to_string(tmpList->at(i).second) + " ";
        output += tmp;
    }
    LOG(INFO) << output;
}

class ParallelDijkstra : public Job {
 public:
  void Run(TaskGraph* tg, const std::shared_ptr<Properties>& config) const override {
    auto input = config->GetOrSet("data", "");
    int n_partitions = std::stoi(config->GetOrSet("parallelism", "20"));
    int n_iters = std::stoi(config->GetOrSet("n_iters", "3"));
    int start_ID = std::stoi(config->GetOrSet("start_node", "1"));
    // Load data
    auto graph = TextSourceDataset(input, tg, n_partitions)
                    .FlatMap([start_ID](const std::string& line) { return ParseLine(line, start_ID); })
                    .ReduceBy([](const Node& ele) { return ele.GetNodeID(); },
                      [](Node& agg, const Node& ele) {
                        std::shared_ptr<std::vector<std::pair<int,int>>> aggList = agg.GetNeighborList();
                        std::shared_ptr<std::vector<std::pair<int,int>>> eleList = ele.GetNeighborList();
                        aggList->push_back(eleList->front());
                      },
                      n_partitions);
    
    graph.UpdatePartition([](DatasetPartition<Node>& graph) {
      std::sort(graph.begin(), graph.end(), [](const Node& a, const Node& b) { return a.GetNodeID() < b.GetNodeID(); });
    });
    /*
    graph.ApplyRead([](const auto& partition) {
                      for (auto& par : partition) {
                        PrintNode(par, 0);
                      }
                      google::FlushLogFiles(google::INFO);
                    });*/

    auto get_distances = [](const DatasetPartition<Node>& graph){
      DatasetPartition<std::pair<int, std::pair<int, int>>> distances;
      for(const Node& n : graph){
        distances.push_back(std::make_pair(n.GetNodeID(), std::make_pair(n.GetMinDistance(), n.GetParentNodeID())));
        for(std::pair<int,int> neighbor : *n.GetNeighborList()){
            int new_distance = MAX_INT;
            if(n.GetMinDistance() < MAX_INT)
              new_distance = neighbor.second + n.GetMinDistance();
          distances.push_back(std::make_pair(neighbor.first, std::make_pair(new_distance, n.GetNodeID())));
        }
      }
      return distances;
    };

    

    auto apply_updates = [](const DatasetPartition<Node>& graph, const DatasetPartition<std::pair<int, std::pair<int, int>>>& min_distances) {
      DatasetPartition<Node> updated_graph;
      /*for(const std::pair<int, std::pair<int, int>>& p : min_distances){
        int node_id = p.first;
        bool updated = false;
        int curr_idx = node_id;
        while(curr_idx < graph.size() && graph.at(curr_idx).GetNodeID() <= node_id){
          if(graph.at(curr_idx).GetNodeID() == node_id){
            updated_graph.push_back(Node(node_id, p.second.first, p.second.second, graph.at(curr_idx).GetNeighborList()));
            updated = true;
            break;
          }
          ++curr_idx;
        }
        if(!updated){
          updated_graph.push_back(Node(node_id, p.second.first, p.second.second));
        }
      }*/
      for(const std::pair<int, std::pair<int, int>>& p : min_distances){ //better implementation to find matching node in graph(use node class to utilize reduceby)
        int node_id = p.first;
        bool updated = false;
        for(const Node& n : graph){ //sort
          if(n.GetNodeID() == node_id){
            updated_graph.push_back(Node(node_id, p.second.first, p.second.second, n.GetNeighborList()));
            updated = true;
            break;
          }
        }
        if(!updated){
          updated_graph.push_back(Node(node_id, p.second.first, p.second.second));
        }
      }
      return updated_graph;
    };
    auto updated_graph = std::make_shared<axe::common::Dataset<Node>>(graph);
    std::shared_ptr<axe::common::Dataset<std::pair<int, std::pair<int, int>>>> min_distances;
    
    for(int i = 0; i < n_iters; ++i){
      min_distances = std::make_shared<axe::common::Dataset<std::pair<int, std::pair<int, int>>>>(updated_graph->MapPartition(get_distances)
                          .ReduceBy([](const std::pair<int, std::pair<int, int>>& ele) { return ele.first; },
                                    [](std::pair<int, std::pair<int, int>>& agg, const std::pair<int, std::pair<int, int>>& ele) {
                                      if(agg.second.first > ele.second.first){
                                        agg.second.first = ele.second.first; //update min distance
                                        agg.second.second = ele.second.second; //update parent ID
                                      }
                                    },
                                    n_partitions));

      updated_graph = std::make_shared<axe::common::Dataset<Node>>(
          updated_graph->SharedDataMapPartitionWith(min_distances.get(), apply_updates));

      /*updated_graph->ApplyRead( [i](const auto& partition) {
                                for (auto& par : partition) {
                                  PrintNode(par, i + 1);
                                }
                                google::FlushLogFiles(google::INFO);
                              });*/
    }

    auto get_output = [](const DatasetPartition<Node>& graph){
      DatasetPartition<std::pair<int, std::pair<int, int>>> output;
      for(const Node& n : graph){
        if(n.GetMinDistance() < MAX_INT){
          output.push_back(std::make_pair(n.GetNodeID(), std::make_pair(n.GetMinDistance(), n.GetParentNodeID())));
        }
      }
      return output;
    };

    updated_graph->MapPartition(get_output)
                  .PartitionBy([](const std::pair<int, std::pair<int, int>>&) { return 0; }, 1)
                  .ApplyRead( [](const auto& partition) {
                                for (auto& par : partition) {
                                  LOG(INFO) << "NodeID = " << par.first << " Distance =  " << par.second.first << " Parent NodeID =  " << par.second.second;
                                }
                                google::FlushLogFiles(google::INFO);
                              });
    axe::common::JobDriver::ReversePrintTaskGraph(*tg);               
  }
};

int main(int argc, char** argv) {
  axe::common::JobDriver::Run(argc, argv, ParallelDijkstra());
  return 0;
}
