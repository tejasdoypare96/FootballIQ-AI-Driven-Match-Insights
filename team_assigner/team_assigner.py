# Import necessary library for clustering
from sklearn.cluster import KMeans

# Define a class to assign players to teams based on their uniform colors
class TeamAssigner:
    def __init__(self):
        """
        Initialize the TeamAssigner class.
        `team_colors`: Dictionary to store team colors.
        `player_team_dict`: Dictionary to map player IDs to their respective teams.
        """
        self.team_colors = {}  # Store the RGB color of each team
        self.player_team_dict = {}  # Store which team each player belongs to

    def get_clustering_model(self, image):
        """
        Create a K-means clustering model to identify dominant colors in the image.
        :param image: The input image to be clustered (usually the player's top half).
        :return: Fitted K-means model.
        """
        # Reshape the image into a 2D array where each row is a pixel with 3 color channels (RGB)
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters (since there are two teams)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        # Return the trained K-means model
        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant color of the player's uniform from the given bounding box (bbox).
        :param frame: The entire video frame/image where the player is detected.
        :param bbox: The bounding box around the player (top-left and bottom-right coordinates).
        :return: RGB color of the player's uniform.
        """
        # Crop the image to the bounding box of the player
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Focus only on the top half of the player's bounding box (usually where the jersey is most visible)
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get the clustering model for the top half of the player image
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel in the image
        labels = kmeans.labels_

        # Reshape the labels to match the 2D shape of the image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Check the corner pixels to determine which cluster likely represents the background (non-player)
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)  # Background cluster
        player_cluster = 1 - non_player_cluster  # The other cluster is the player's uniform

        # Get the dominant color of the player's uniform based on the player cluster
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assign teams to players by clustering the colors of their uniforms.
        :param frame: The entire video frame/image containing the players.
        :param player_detections: Dictionary of detected players, each with a bounding box (bbox).
        """
        player_colors = []  # List to store the colors of all detected players
        
        # Loop through each player detection and extract their uniform color
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]  # Get the bounding box of the player
            player_color = self.get_player_color(frame, bbox)  # Get the player's uniform color
            player_colors.append(player_color)  # Add the player's color to the list

        # Cluster the extracted player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans  # Store the K-means model for future use

        # Assign the cluster centers as the two team colors
        self.team_colors[1] = kmeans.cluster_centers_[0]  # First team color
        self.team_colors[2] = kmeans.cluster_centers_[1]  # Second team color

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predict the team of a player based on their uniform color.
        :param frame: The entire video frame/image containing the player.
        :param player_bbox: The bounding box of the player.
        :param player_id: The unique ID of the player.
        :return: The predicted team ID (1 or 2).
        """
        # If the player's team has already been determined, return the cached team ID
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player's uniform color from the current frame
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team ID based on the player's uniform color using the stored K-means model
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Adjust from 0/1 to 1/2

        # Hardcoded rule: Assign player ID 91 to team 1 (could be a special case or error correction)
        if player_id == 91:
            team_id = 1

        # Store the player's team assignment for future reference
        self.player_team_dict[player_id] = team_id

        return team_id
