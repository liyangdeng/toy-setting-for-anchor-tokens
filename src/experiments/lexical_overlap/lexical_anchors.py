from collections import Counter
import json

def filter_anchors_by_strategy(all_tokens_sorted, strategy):
    """
    Helper function that filters the sorted frequency list based on the chosen strategy.
    - 'high': Keeps the list from the top (most frequent structural/function words).
    - 'mid': Skips the top 15 most frequent tokens and takes content words from the middle.
    - 'low': Takes lower-frequency words from the bottom part of the distribution.
    """
    if strategy == "high":
        return all_tokens_sorted
    elif strategy == "mid":
        cutoff_top = 15  # Skips top words like 'and', 'is', 'remains'...
        return all_tokens_sorted[cutoff_top:]
    elif strategy == "low":
        cutoff_mid = int(len(all_tokens_sorted) * 0.3)  # Skips the top 30% of the entire vocabulary
        return all_tokens_sorted[cutoff_mid:]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

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
    
    # Calculate how many tokens we need to accumulate to reach the target corpus coverage
    target_tokens_count = (target_percentage / 100.0) * number_of_tokens
    
    anchors = []
    accumulated_count = 0
    
    # Dynamically accumulate unique words until the targeted coverage threshold is met
    for token, count in filtered_pool:
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

def replacement_mapping(dictionary, corpus, target_percentage, strategy="high"):
    anchors = counter(corpus, target_percentage, strategy)
    
    with open(dictionary, "r", encoding="utf-8") as f:
        synset_data = json.load(f)
        
    replacement_mapping = {}
    
    for anchor, _ in anchors:
        if anchor in synset_data:
            artificial = synset_data[anchor].get("artificial", "")
            if artificial:
                replacement_mapping[artificial] = anchor

    print("--- REPLACEMENT MAPPING ---")
    print(f"Number of mapped replacements: {len(replacement_mapping)}")
    return replacement_mapping

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
            for lang in languages:
                print(f"=========================================")
                print(f"RUNNING EXPERIMENT FOR LANG = {lang.upper()}, OVERLAP = {target_percentage}%, STRATEGY = {strategy.upper()}")
                print(f"=========================================")
                
                dictionary_path = f"synset_pos_artificial_{lang}.json"
                artificial_corpus_path = f"corpus_{lang}_synset.txt"
                
                mapping = replacement_mapping(dictionary_path, eng_corpus_path, target_percentage, strategy)
                overwrite(artificial_corpus_path, mapping, lang, target_percentage, strategy)

# Execute the pipeline
languages_list = ["cjk", "hiragana"]
create_corpora("eng_sentences_with_adj.txt", languages_list)
