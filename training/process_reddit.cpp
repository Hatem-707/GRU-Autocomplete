#include <algorithm>
#include <cctype>
#include <fstream>
#include <glaze/glaze.hpp>
#include <ios>
#include <iostream>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::string> splitSentence(std::string sentence) {
  std::vector<std::string> words;
  std::string word;
  for (char &letter : sentence) {
    if (!letter || letter == ' ') {
      if (!word.empty()) {
        std::transform(word.begin(), word.end(), word.begin(),
                       [](char c) { return std::tolower(c); });
        words.push_back(word);
      }
      word.clear();
    } else if (std::isalpha(letter) || letter == '\'') {
      word.push_back(letter);
    }
  }
  return words;
}

int main() {
  std::ifstream cleanSentences;
  cleanSentences.open("clean_big.json", std::ios::binary);
  std::unordered_map<std::string, long long> frequency;
  std::string line;
  cleanSentences.seekg(0, std::ios_base::end);
  long long dataSize = cleanSentences.tellg();
  std::string jsonData;
  jsonData.resize(dataSize);
  cleanSentences.seekg(0, std::ios_base::beg);
  cleanSentences.read(&jsonData[0], dataSize);

  std::unordered_map<std::string, std::string> dataset;

  // 3. Parse using Glaze
  // Glaze detects that it is a map and parses all key-value pairs
  // automatically.
  auto result = glz::read_json(dataset, jsonData);

  if (result) {
    std::cerr << "Error parsing JSON: " << result.ec << "\n";
    return 1;
  }

  // 4. Access the data
  std::cout << "Loaded " << dataset.size() << " sentences.\n";

  std::ofstream processedSentences;
  processedSentences.open("processed_reddit.txt", std::ios::trunc);
  cleanSentences.close();
  for (auto &[id, sentence] : dataset) {
    auto words = splitSentence(sentence);
    for (auto &word : words) {
      frequency[word]++;
    }
    std::transform(sentence.begin(), sentence.end(), sentence.begin(),
                   [](char c) { return std::tolower(c); });
    auto processed = sentence | std::views::filter([](char c) {
                       return std::isalpha(c) || c == '\'' || c == ' ';
                     });
    for (const char &letter : processed) {
      processedSentences << letter;
    }
    processedSentences << "\n";
  }
  processedSentences.close();
  std::vector<std::pair<std::string, long long>> counts;
  for (auto &[key, value] : frequency) {
    counts.emplace_back(key, value);
  }
  std::sort(counts.begin(), counts.end(),
            [](auto a, auto b) { return a.second > b.second; });
  std::cout << counts.size() << std::endl;
  std::ofstream output_vocab;
  output_vocab.open("reddit_top.txt", std::ios::trunc);
  for (int i = 0; i < std::min(counts.size(), 10000ul); i++) {
    output_vocab << counts[i].first << "\n";
  }
  output_vocab.close();
}