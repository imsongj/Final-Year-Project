// Copyright 2018 H-AXE
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

#include <cctype>
#include <memory>
#include <string>
#include <utility>

#include "glog/logging.h"

#include "base/tokenizer.h"
#include "common/engine.h"

using axe::base::Tokenizer;

void ParseLine(DatasetPartition<std::pair<int, int>>& collection, const std::string& line) {
  if (line.empty()) {
    return;
  }
  axe::base::WhiteSpaceTokenizer tokenizer(line);
  std::string tok;
  std::unordered_map<int, int> length_count;
  while (tokenizer.next(tok)) {
    length_count[tok.length()] += 1;
  }
  for (auto& pair : length_count) {
    collection.push_back(pair);
  }
}

class WordCountJob : public Job {
 public:
  void Run(TaskGraph* tg, const std::shared_ptr<Properties>& config) const override {
    TextSourceDataset(config->Get("input"), tg, std::stoi(config->Get("parallelism")))
        .FlatMap([](const std::string& line) {
          DatasetPartition<std::pair<int, int>> ret;
          ParseLine(ret, line);
          return ret;
        })
        .ReduceBy([](const std::pair<int, int>& ele) { return ele.first; },
                  [](std::pair<int, int>& agg, const std::pair<int, int>& update) { agg.second += update.second; }, 1)
        .ApplyRead([](auto data) {
          int count;
          count = data.size();
          for(int i = 0; i < count; i++){
            std::pair<int, int> cur_data = data.at(i);
            LOG(INFO) << "output: " << cur_data.first << ": " << cur_data.second;
          }
          google::FlushLogFiles(google::INFO);
        });
  }
};

int main(int argc, char** argv) {
  axe::common::JobDriver::Run(argc, argv, WordCountJob());
  return 0;
}
