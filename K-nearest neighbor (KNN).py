import math

class KNearestNeighbor:
    def __init__(self, k=1):
        self.k = k
        self.data = []

    def fit(self, dataset):
        self.data = dataset

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def predict(self, guess):
        distances = []
        for data in self.data:
            dist = self.euclidean_distance(guess[0], guess[1], data[0], data[1])
            distances.append((dist, data[2]))
        
        distances.sort(key=lambda x: x[0])  # Sort by distance
        k_nearest = distances[:self.k]  # Get k nearest neighbors
        
        # Majority voting
        class_votes = {}
        for _, label in k_nearest:
            class_votes[label] = class_votes.get(label, 0) + 1
        
        guess[2] = max(class_votes, key=class_votes.get)
        print(f"Predicted Category: {guess[2]}")
        return guess[2]

# Example dataset
data = [
    [30, 15, "Apple"],
    [55, 60, "Banana"],
    [65, 95, "Banana"],
    [15, 30, "Apple"],
    [75, 75, "Banana"],
    [65, 15, "Apple"],
    [35, 85, "Banana"]
]

guess = [25, 40, ""]

knn = KNearestNeighbor(k=1)
knn.fit(data)
knn.predict(guess)