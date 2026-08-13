"""
Builds overlapped corpora for the Lexical Overlap Experiment.

Reads the generated English corpus and counts the frequency of tokens, delivering a sorted frequency list.
Selects anchors based on the specified target percentage and frequency strategy.
Maps these anchors to the corresponding artificial language corpora using their respective dictionary files.

Usage:
    python create_overlapped_corpora.py \
        --eng_corpus_path generated_sentences.txt \
        --languages cjk hiragana
"""

from collections import Counter
import json
import random

def filter_anchors_by_strategy(all_tokens_sorted, strategy):
    """
    Filters the sorted frequency list into designated frequency pools based on the specified strategy: 
    - 'high': indices 0-9
    - 'mid': indices 10-49
    - 'low': indices 50-199
    """
    if strategy == "high":
        return all_tokens_sorted[:10]
    elif strategy == "mid":
        return all_tokens_sorted[10:50]
    elif strategy == "low":
        return all_tokens_sorted[50:200]

def counter(file_name):
    """
    Counts the frequency of tokens in the parallel (English) corpus and selects lexical overlaps based on the specified target percentage and frequency strategy.
    """
    corpus_frequency = Counter()
    number_of_tokens = 0
    
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line_parts = line.strip().split(' ')
            # Filter out empty strings
            tokens = [x for x in line_parts if x]
            # Update the total token count
            number_of_tokens += len(tokens)
            # Update the frequency counter with the tokens from this line
            corpus_frequency.update(tokens)
    
    # Sort all tokens by frequency in descending order
    all_tokens_sorted = corpus_frequency.most_common()

    return all_tokens_sorted, number_of_tokens

def select_anchors(file_name, target_percentage, strategy):
    """
    Selects anchors from the given file based on the specified target percentage and frequency strategy.
    """
    # Handle the case where the target percentage is 0.0
    if target_percentage == 0.0:
        print(f"\n[STRATEGY: {strategy.upper()}] Selected unique words (anchors): 0")
        print(f"Total tokens covered: 0 out of {number_of_tokens} (0.00%)")
        return []
    
    # Count the frequency of tokens in the file and get the total number of tokens
    all_tokens_sorted, number_of_tokens = counter(file_name)
    
    # Filter the words based on the specified strategy
    filtered_pool = filter_anchors_by_strategy(all_tokens_sorted, strategy)
    
    # Shuffle the filtered pool to ensure randomness in selection
    randomized_pool = list(filtered_pool)
    random.shuffle(randomized_pool)

    # Calculate the target number of tokens to cover based on the specified percentage
    target_tokens_count = (target_percentage / 100.0) * number_of_tokens
    
    anchors = []
    accumulated_count = 0
    
    # Select anchors from the randomized pool until the accumulated count reaches the target token count
    for token, count in randomized_pool:
        # Check if adding this token's count would exceed the target token count
        if (accumulated_count + count) > target_tokens_count:
            continue
        # If not, add the token and its count to the anchors list and update the accumulated count
        anchors.append((token, count))
        accumulated_count += count

        # If the accumulated count has reached or exceeded the target, break out of the loop
        if accumulated_count >= target_tokens_count:
            break
    
    # Actual percentage of tokens covered by the selected anchors (not always exactly equal to the target percentage due to selection pipeline)
    actual_percentage = (accumulated_count / number_of_tokens) * 100
    
    print(f"\n[STRATEGY: {strategy.upper()}] Selected unique words (anchors): {len(anchors)}")
    print(f"Total tokens covered: {accumulated_count} out of {number_of_tokens} ({actual_percentage:.2f}%)")
    
    # Print the first 10 selected anchors for reference
    for i, (token, count) in enumerate(anchors[:10], start=1):
        print(f"{i}. '{token}' -> {count} ({count / number_of_tokens * 100:.2f}%)")
    if len(anchors) > 10:
        print("... (and remaining words within the target percentage threshold)")
        
    print(f"Total number of tokens in corpus: {number_of_tokens}\n")
    
    return anchors

def get_replacement_mapping(dictionary_path, anchors):
    """
    Creates a replacement mapping for a specific language based on the selected anchors and the corresponding dictionary file.
    """
    with open(dictionary_path, "r", encoding="utf-8") as f:
        synset_data = json.load(f)
    
    replacement_map = {}
    
    for anchor, _ in anchors:
        # If the term exists in the dictionary, change the artificial token to the corresponding (English) anchor token
        if anchor in synset_data:
            artificial = synset_data[anchor].get("artificial", "")
            if artificial:
                replacement_map[artificial] = anchor

    print(f"--- REPLACEMENT MAPPING FOR {dictionary_path} ---")
    print(f"Number of mapped replacements: {len(replacement_map)}")
    return replacement_map

def overwrite(corpus_path, replacement_mapping, lang, target_percentage, strategy):
    """
    Overwrites the artificial corpus with the selected tokens replaced by anchors.
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    modified_lines = []
    for line in lines:
        tokens = line.strip().split(' ')
        # If a token exists in the replacement mapping, replace it
        new_tokens = [replacement_mapping.get(token, token) for token in tokens if token]
        # Save the modified line to the list of modified lines
        modified_lines.append(' '.join(new_tokens) + '\n')
    
    # Create the output corpus path, e.g., "corpus_cjk_P5_high.txt"
    output_corpus_path = f"corpus_{lang}_P{int(target_percentage)}_{strategy}.txt"
    
    # Write the modified lines to the new corpus file
    with open(output_corpus_path, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)
    print(f"Saved corpus file: {output_corpus_path}")
        
def create_corpora(eng_corpus_path, languages):
    """
    Creates overlapped corpora for the specified languages based on the English corpus, target coverage percentages, and frequency strategies.
    """
    # Defined target percentages for lexical overlap
    percentage_values = [0.0, 2.5, 5.0, 7.5, 10.0]
    # Defined frequency-based selection strategies for anchors
    strategies = ["high", "mid", "low"]
    
    for target_percentage in percentage_values:
        for strategy in strategies:
            print(f"=========================================")
            print(f"GENERATING FIXED ANCHORS FOR OVERLAP = {target_percentage}%, STRATEGY = {strategy.upper()}")
            print(f"=========================================")
            
            # Generate anchors ONCE per combination of percentage and strategy
            fixed_anchors = counter(eng_corpus_path, target_percentage, strategy)
            
            # Distribute the exact same anchors to BOTH target languages
            for lang in languages:
                print(f"Processing language: {lang.upper()}")
                dictionary_path = f"synset_pos_artificial_{lang}.json"
                artificial_corpus_path = f"corpus_{lang}_synset.txt"
                
                # Map the exact same anchors to this language's dictionary
                mapping = get_replacement_mapping(dictionary_path, fixed_anchors)
                overwrite(artificial_corpus_path, mapping, lang, target_percentage, strategy)


def main():
    languages_list = ["cjk", "hiragana"]
    create_corpora("generated_sentences.txt", languages_list)

if __name__ == "__main__":
    main()
    
