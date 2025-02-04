

# Displaying the dataset as a table
dataset = [
    # (No., Age, Student, Income, Credit, Buys)
    ["1", "Young", "Yes", "High", "Fair", "No"],
    ["2", "Senior", "No", "High", "Excellent", "Yes"],
    ["3", "Middle", "Yes", "Medium", "Fair", "Yes"],
    ["4", "Young", "Yes", "Low", "Fair", "No"],
    ["5", "Middle", "Yes", "Low", "Excellent", "Yes"],
    ["6", "Senior", "No", "Medium", "Excellent", "No"],
    ["7", "Young", "No", "Medium", "Excellent", "Yes"],
    ["8", "Young", "Yes", "Medium", "Fair", "Yes"],
    ["9", "Middle", "Yes", "High", "Excellent", "Yes"],
    ["10", "Senior", "No", "Low", "Fair", "No"],
]

# Print table
print(f"{'No.':<5}{'Age':<10}{'Student':<10}{'Income':<10}{'Credit':<10}{'Buys':<5}")
print("=" * 50)
for row in dataset:
    print(f"{row[0]:<5}{row[1]:<10}{row[2]:<10}{row[3]:<10}{row[4]:<10}{row[5]:<5}")

# Preprocessing: Tokenize and build vocabulary
vocab = set()
word_counts = {"Yes": {}, "No": {}}
class_counts = {"Yes": 0, "No": 0}

def tokenize(text):
    return text.lower().split()

# Convert dataset to a format suitable for Naïve Bayes
training_data = [(" ".join(row[1:5]), row[5]) for row in dataset]

# Count word occurrences in each class
for entry, label in training_data:
    words = tokenize(entry)
    class_counts[label] += 1
    for word in words:
        vocab.add(word)
        if word not in word_counts[label]:
            word_counts[label][word] = 0
        word_counts[label][word] += 1

# Compute probabilities (Naïve Bayes formula)
num_words = len(vocab)

def compute_word_probs(word_counts, class_counts, class_label):
    total_words = sum(word_counts[class_label].values()) + num_words  # Add num_words for Laplace smoothing
    probs = {word: (word_counts[class_label].get(word, 0) + 1) / total_words for word in vocab}
    return probs

yes_probs = compute_word_probs(word_counts, class_counts, "Yes")
no_probs = compute_word_probs(word_counts, class_counts, "No")

# Predict function using Bayes' Theorem
def predict(entry):
    words = tokenize(entry)
    yes_prob = class_counts["Yes"] / sum(class_counts.values())
    no_prob = class_counts["No"] / sum(class_counts.values())

    for word in words:
        if word in vocab:
            yes_prob *= yes_probs.get(word, 1 / (sum(word_counts["Yes"].values()) + num_words))
            no_prob *= no_probs.get(word, 1 / (sum(word_counts["No"].values()) + num_words))

    return "Yes" if yes_prob > no_prob else "No"

# Test the classifier
test_entry = "Young Yes Medium Fair"
result = predict(test_entry)
print(f"\nPrediction for '{test_entry}': {result}")

# Visualization
labels = list(vocab)
yes_values = [yes_probs[word] for word in vocab]
no_values = [no_probs[word] for word in vocab]

x = np.arange(len(labels))
width = 0.4

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width / 2, yes_values, width, label="Yes")
ax.bar(x + width / 2, no_values, width, label="No")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.set_ylabel("Probability")
ax.set_title("Word Probabilities in Naïve Bayes Classification")
ax.legend()

plt.show()
