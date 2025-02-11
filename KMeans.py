class KMeans:
    def __init__(self, k=4, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = []
        self.clusters = {}

    def calculate_distance(self, dx, dy, cx, cy):
        first = (dx - cx) ** 2 + (dy - cy) ** 2
        if first == 0:
            return 0
        last_guess = first / 2.0
        while True:
            guess = (last_guess + first / last_guess) / 2
            if abs(guess - last_guess) < .00000000001:
                return guess
            last_guess = guess

    def assign_clusters(self, datapoints):
        self.clusters = {key: [] for key in range(self.k)}
        for point in datapoints:
            x, y = point
            closest = min(range(self.k), key=lambda i: self.calculate_distance(x, y, self.centroids[i][0], self.centroids[i][1]))
            self.clusters[closest].append(point)

    def calculate_new_centroids(self):
        new_centroids = []
        for cluster in self.clusters.values():
            if cluster:
                x_avg = sum(p[0] for p in cluster) / len(cluster)
                y_avg = sum(p[1] for p in cluster) / len(cluster)
                new_centroids.append([x_avg, y_avg])
            else:
                new_centroids.append(self.centroids[len(new_centroids)])
        return new_centroids

    def fit(self, datapoints):
        self.centroids = [datapoints[i] for i in range(self.k)]
        for _ in range(self.max_iterations):
            self.assign_clusters(datapoints)
            new_centroids = self.calculate_new_centroids()
            if new_centroids == self.centroids:
                break
            self.centroids = new_centroids
        return self.centroids, self.clusters

# Example dataset
datapoints = [
    [3, 12], [2, 6], [9, 5], [6, 9], 
    [8, 6], [7, 5], [2, 3], [5, 11],
    [4, 7], [10, 8]
]

kmeans = KMeans(k=4)
final_centroids, final_clusters = kmeans.fit(datapoints)

# Output results
print("Final Centroids:", final_centroids)
for cluster_id, points in final_clusters.items():
    print(f"Cluster {cluster_id + 1}: {points}")
