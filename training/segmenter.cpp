#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>

// Include Glaze (ensure glaze/glaze.hpp is in your include path)
#include <glaze/glaze.hpp>

// Include ICU
#include <unicode/brkiter.h>
#include <unicode/unistr.h>
#include <unicode/ustream.h>

// Function to segment text into sentences using ICU
std::vector<std::string> segment_text_icu(const std::string& text) {
    std::vector<std::string> sentences;
    
    UErrorCode status = U_ZERO_ERROR;
    icu::Locale locale = icu::Locale::getUS(); // Equivalent to language="en"
    
    // Create the BreakIterator for sentences
    std::unique_ptr<icu::BreakIterator> bi(
        icu::BreakIterator::createSentenceInstance(locale, status)
    );

    if (U_FAILURE(status)) {
        std::cerr << "Error creating ICU BreakIterator: " << u_errorName(status) << std::endl;
        return sentences;
    }

    // ICU works with UnicodeString
    icu::UnicodeString uText = icu::UnicodeString::fromUTF8(text);
    bi->setText(uText);

    int32_t start = bi->first();
    int32_t end = bi->next();

    while (end != icu::BreakIterator::DONE) {
        // Extract substring
        icu::UnicodeString sentence_uni = uText.tempSubStringBetween(start, end);
        
        // Convert back to UTF-8 std::string
        std::string sentence_str;
        sentence_uni.toUTF8String(sentence_str);

        // Optional: specific cleaning to match pysbd's clean=True
        // Remove excessive newlines or trim
        sentence_str.erase(std::remove(sentence_str.begin(), sentence_str.end(), '\n'), sentence_str.end());
        sentence_str.erase(std::remove(sentence_str.begin(), sentence_str.end(), '\r'), sentence_str.end());

        if (!sentence_str.empty()) {
            sentences.push_back(sentence_str);
        }

        start = end;
        end = bi->next();
    }
    
    return sentences;
}

int main() {
    // 1. Read input data
    // Assuming the python script dumped the raw content to a file to avoid Parquet/Arrow C++ overhead
    std::string inputFilename = "raw_corpus.txt"; 
    std::ifstream inFile(inputFilename, std::ios::binary);
    
    if (!inFile) {
        std::cerr << "Could not open " << inputFilename << "\n";
        return 1;
    }

    std::cout << "Reading file..." << std::endl;
    // Read entire file into string
    std::string content((std::istreambuf_iterator<char>(inFile)),
                         std::istreambuf_iterator<char>());
    inFile.close();

    // 2. Perform Segmentation
    std::cout << "Segmenting sentences..." << std::endl;
    std::vector<std::string> sentences = segment_text_icu(content);
    std::cout << "Found " << sentences.size() << " sentences." << std::endl;

    // 3. Format as Map {"0": "sent", "1": "sent"} to match Python output
    std::unordered_map<std::string, std::string> dict_sent;
    dict_sent.reserve(sentences.size());

    for (size_t i = 0; i < sentences.size(); ++i) {
        dict_sent[std::to_string(i)] = sentences[i];
    }

    // 4. Write to JSON using Glaze
    std::cout << "Writing JSON..." << std::endl;
    std::string json_output;
    auto write_err = glz::write_json(dict_sent, json_output); // defaults to minified (like orjson)
    
    if (write_err) {
        std::cerr << "Error writing JSON" << std::endl;
        return 1;
    }

    std::ofstream outFile("clean_big.json", std::ios::binary);
    outFile.write(json_output.data(), json_output.size());
    outFile.close();

    std::cout << "Done. Saved to clean_big.json" << std::endl;
    return 0;
}
