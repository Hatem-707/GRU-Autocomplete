
#include <algorithm>
#include <cctype>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

std::vector<std::string> splitSentence(std::string &sentence) {
  std::vector<std::string> words;
  std::string temp;
  for (const char &letter : sentence) {
    if (!letter || letter == ' ') {
      if (!temp.empty()) {
        std::transform(temp.begin(), temp.end(), temp.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        while (std::ispunct(temp.back())) {
          temp.pop_back();
        }
        if (!temp.empty())
          words.push_back(temp);
        temp.clear();
      }
    } else {
      temp.push_back(letter);
    }
  }
  return words;
}

std::vector<std::vector<std::string>> extractSentenece(std::string &line) {
  std::vector<std::vector<std::string>> sentences;
  std::vector<std::string> sentence;
  std::string word;
  for (const char &letter : line) {
    if (!letter || letter == ' ') {
      if (word.size() > 1 || word == "I" || word == "i" || word == "a") {
        if (word == "eou") {
          sentences.push_back(sentence);
          sentence.clear();
          word.clear();
        } else if (word == "’") {
          continue;
        } else if (word != ".." && word != "...") {
          std::transform(word.begin(), word.end(), word.begin(),
                         [](const char &c) { return std::tolower(c); });
          sentence.push_back(word);
          word.clear();
        } else {
          word.clear();
        }
      } else {
        word.clear();
      }
    } else {
      if (std::isalpha(letter) || letter == '\'')
        word.push_back(letter);
    }
  }
  if (!sentence.empty()) {
    sentences.push_back(sentence);
  }
  return sentences;
}

int main() {
  std::ifstream daily_dialogue;
  daily_dialogue.open("daily_dialog/train/dialogues_train.txt");
  std::ofstream cleaned_daily;
  cleaned_daily.open("clean_daily.txt", std::ios::trunc);
  std::string line;
  std::unordered_map<std::string, int> freq;
  while (std::getline(daily_dialogue, line)) {
    auto sentences = extractSentenece(line);
    for (const auto &sentence : sentences) {
      for (const auto &word : sentence) {
        cleaned_daily << word + " ";
        freq[word]++;
      }
      cleaned_daily << std::endl;
    }
  }
  daily_dialogue.close();
  std::vector<std::pair<std::string, int>> entries;
  for (const auto entry : freq) {
    entries.push_back({entry});
  }
  std::sort(entries.begin(), entries.end(),
            [](std::pair<std::string, int> a, std::pair<std::string, int> b) {
              return a.second > b.second;
            });
  std::cout << entries.size() << std::endl;
  std::ofstream topWords;
  topWords.open("daily_dialogue_top20k.txt", std::ios::trunc);
  for (int i = 0; i < std::min(10000ul, entries.size()); i++) {
    topWords << entries[i].first << std::endl;
  }
  topWords.close();
}