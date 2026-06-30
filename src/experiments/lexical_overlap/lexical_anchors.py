from collections import Counter
import json
import random

def filter_anchors_by_strategy(all_tokens_sorted, strategy):
    """
    Filters the sorted frequency list strictly within the top 100 most frequent tokens.
    - 'high': Top 10 words (indices 0-9) -> Grammatical framework
    - 'mid': From 11th to 50th place (indices 10-49) -> Relational/Taxonomic framework
    - 'low': From 51st to 100th place (indices 50-99) -> Descriptive/Conceptual tokens
    """
    if strategy == "high":
        return all_tokens_sorted[:10]
    elif strategy == "mid":
        return all_tokens_sorted[10:50]
    elif strategy == "low":
        return all_tokens_sorted[50:200]

def counter(file_name, target_percentage, strategy="high"):
    corpus_frequency = Counter()
    number_of_tokens = 0
    
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line_parts = line.strip().split(' ')
            tokens = [x for x in line_parts if x]
            number_of_tokens += len(tokens)
            corpus_frequency.update(tokens)
    
    # Sort all tokens by frequency (most common first)
    all_tokens_sorted = corpus_frequency.most_common()
    
    # Filter the word pool based on the frequency strategy (high, mid, low)
    filtered_pool = filter_anchors_by_strategy(all_tokens_sorted, strategy)
    
    # Create a copy of the pool and shuffle it randomly
    randomized_pool = list(filtered_pool)
    random.shuffle(randomized_pool)

    # Calculate how many tokens we need to accumulate to reach the target corpus coverage
    target_tokens_count = (target_percentage / 100.0) * number_of_tokens
    
    anchors = []
    accumulated_count = 0
    
    # Dynamically accumulate unique words until the targeted coverage threshold is met
    for token, count in randomized_pool:
        if (accumulated_count + count) > target_tokens_count:
            continue
        anchors.append((token, count))
        accumulated_count += count

        if accumulated_count >= target_tokens_count:
            break

    actual_percentage = (accumulated_count / number_of_tokens) * 100
    
    print(f"\n[STRATEGY: {strategy.upper()}] Selected unique words (anchors): {len(anchors)}")
    print(f"Total tokens covered: {accumulated_count} out of {number_of_tokens} ({actual_percentage:.2f}%)")
    
    # Print the first 10 selected anchors for tracking/verification
    for i, (token, count) in enumerate(anchors[:10], start=1):
        print(f"{i}. '{token}' -> {count} ({count / number_of_tokens * 100:.2f}%)")
    if len(anchors) > 10:
        print("... (and remaining words within the target percentage threshold)")
        
    print(f"Total number of tokens in corpus: {number_of_tokens}\n")
    
    return anchors

def get_replacement_mapping(dictionary_path, anchors):
    """
    Creates a replacement mapping for a specific language using a pre-selected, fixed list of anchors.
    """
    with open(dictionary_path, "r", encoding="utf-8") as f:
        synset_data = json.load(f)
        
    replacement_map = {}
    
    for anchor, _ in anchors:
        if anchor in synset_data:
            artificial = synset_data[anchor].get("artificial", "")
            if artificial:
                replacement_map[artificial] = anchor

    print(f"--- REPLACEMENT MAPPING FOR {dictionary_path} ---")
    print(f"Number of mapped replacements: {len(replacement_map)}")
    return replacement_map

def overwrite(corpus_path, replacement_mapping, lang, target_percentage, strategy="high"):
    with open(corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    modified_lines = []
    for line in lines:
        tokens = line.strip().split(' ')
        new_tokens = [replacement_mapping.get(token, token) for token in tokens if token]
        modified_lines.append(' '.join(new_tokens) + '\n')
    
    # The output file name reflects the configuration (e.g., corpus_cjk_P10_mid.txt)
    output_corpus_path = f"corpus_{lang}_P{int(target_percentage)}_{strategy}.txt"
        
    with open(output_corpus_path, "w", encoding="utf-8") as f:
        f.writelines(modified_lines)
    print(f"Saved corpus file: {output_corpus_path}")
        
def create_corpora(eng_corpus_path, languages):
    # We define the target coverage percentages (e.g., 5% and 10%)
    percentage_values = [2.5, 5.0, 7.5, 10.0]
    # The three frequency strategies to evaluate the Anchor-Token Hypothesis
    strategies = ["high", "mid", "low"]
    
    for target_percentage in percentage_values:
        for strategy in strategies:
            print(f"=========================================")
            print(f"GENERATING FIXED ANCHORS FOR OVERLAP = {target_percentage}%, STRATEGY = {strategy.upper()}")
            print(f"=========================================")
            
            # 1. Generate anchors ONCE per combination of percentage and strategy
            fixed_anchors = counter(eng_corpus_path, target_percentage, strategy)
            
            # 2. Distribute the exact same anchors to all target languages
            for lang in languages:
                print(f"Processing language: {lang.upper()}")
                dictionary_path = f"synset_pos_artificial_{lang}.json"
                artificial_corpus_path = f"corpus_{lang}_synset.txt"
                
                # Map the exact same anchors to this language's dictionary
                mapping = get_replacement_mapping(dictionary_path, fixed_anchors)
                overwrite(artificial_corpus_path, mapping, lang, target_percentage, strategy)

# Execute the pipeline
languages_list = ["cjk", "hiragana"]
create_corpora("eng_sentences_with_adj.txt", languages_list)
