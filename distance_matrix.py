import csv
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import unicodedata  # Added import
import random  # Added import
from itertools import combinations  # Added import


def load_feature_system_csv(file_path):
    """
    Loads the feature system from a CSV file.
    """
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        sounds = []
        feature_matrix = []
        features = header[2:]  # Features start from the third column
        descriptions = []

        for row in reader:
            sounds.append(row[0])  # First column is the sound
            descriptions.append(row[1])  # Second column is the description
            try:  # Added error handling for parsing
                feature_matrix.append([float(value) for value in row[2:]])
            except ValueError as e:
                print(f"Error parsing row for sound '{row[0]}': {e}")
                # Handle error, e.g., skip row or fill with default values
                # For now, let's skip the row or raise an error
                raise ValueError(
                    f"Invalid numeric value in feature data for sound '{row[0]}'"
                ) from e

    return sounds, features, np.array(feature_matrix), descriptions


def get_feature_ranges(feature_matrix):
    """
    Determine the range of each feature in the data.
    """
    feature_ranges = []
    for i in range(feature_matrix.shape[1]):
        col = feature_matrix[:, i]
        non_zero = col[col != 0]

        if len(non_zero) > 0:
            min_val, max_val = np.min(non_zero), np.max(non_zero)
            feature_ranges.append((min_val, max_val))
        else:
            feature_ranges.append((0, 0))

    return feature_ranges


def normalize_features(feature_matrix, feature_ranges=None):
    """
    Normalize features based on their range in the data, not fixed ranges.
    """
    normalized_matrix = np.zeros_like(feature_matrix, dtype=float)

    if feature_ranges is None:
        feature_ranges = get_feature_ranges(feature_matrix)

    for i in range(feature_matrix.shape[1]):
        col = feature_matrix[:, i]
        min_val, max_val = feature_ranges[i]

        if min_val != max_val:
            for j in range(len(col)):
                if col[j] != 0:
                    normalized_matrix[j, i] = (col[j] - min_val) / (max_val - min_val)
        else:
            # Handle constant features (only 0 or one value)
            normalized_matrix[:, i] = (col != 0).astype(
                float
            )  # Binary 0/1 if range is zero

    return normalized_matrix


def compute_feature_difference_vector(
    sound1_idx, sound2_idx, feature_matrix, feature_ranges
):
    """
    Compute a vector representing the feature differences between two sounds.
    Properly handles different feature ranges and ignores cases where both have zero.
    """
    # Extract feature vectors for the two sounds
    vec1 = feature_matrix[sound1_idx]
    vec2 = feature_matrix[sound2_idx]

    # Create masks for various conditions
    both_nonzero_mask = (vec1 != 0) & (vec2 != 0)
    either_nonzero_mask = (vec1 != 0) | (vec2 != 0)
    # only1_nonzero_mask = (vec1 != 0) & (vec2 == 0) # Not explicitly used in final vector below
    # only2_nonzero_mask = (vec1 == 0) & (vec2 != 0) # Not explicitly used in final vector below

    # Calculate normalized absolute difference for features where both are non-zero
    abs_diff = np.zeros_like(vec1, dtype=float)
    for i in range(len(vec1)):
        if both_nonzero_mask[i]:
            # Normalize by feature range
            min_val, max_val = feature_ranges[i]
            if min_val != max_val:
                range_size = max_val - min_val
                # Avoid division by zero if range_size is somehow zero despite min_val != max_val
                if range_size > 1e-9:
                    abs_diff[i] = abs(vec1[i] - vec2[i]) / range_size
                else:
                    abs_diff[i] = float(
                        vec1[i] != vec2[i]
                    )  # Treat as binary if range is tiny
            else:
                # If min_val == max_val, difference is 0 unless values differ (shouldn't happen)
                abs_diff[i] = float(vec1[i] != vec2[i])

    # Create metrics for each feature type
    missing_features = np.zeros(
        1
    )  # Ratio of features where one has a value but other doesn't
    num_either_nonzero = np.sum(either_nonzero_mask)
    if num_either_nonzero > 0:
        missing_ratio = 1.0 - (np.sum(both_nonzero_mask) / num_either_nonzero)
        missing_features[0] = missing_ratio
    else:
        missing_features[0] = 0.0  # No features active for either sound

    # Create synthetic features for each non-zero difference
    # This helps the model learn which specific feature differences matter
    feature_vectors = []
    for i in range(len(vec1)):
        if both_nonzero_mask[i]:
            # Use the calculated normalized difference
            feature_vectors.append(abs_diff[i])
        elif either_nonzero_mask[i]:
            # When only one sound has this feature, assign max difference (1.0)
            feature_vectors.append(1.0)
        else:
            # Both sounds lack this feature (value is 0) - treat as NA, assign 0 difference
            feature_vectors.append(0.0)

    # Combine into a single feature vector
    # The vector now contains:
    # - Per-feature difference (0 if NA, 1 if only one has it, normalized diff if both have it)
    # - Overall ratio of non-shared features among active features
    return np.array(feature_vectors + missing_features.tolist())


def expert_defined_phoneme_pairs():
    """
    Define phoneme pairs with expert-assigned distance values.
    These will be used as ground truth for the ML model.
    """
    pairs = [
        # Vowel pairs (small distances between similar vowels)
        ("a", "ɑ", 0.05),  # Very similar open vowels
        ("a", "æ", 0.15),  # Both front, different height
        ("i", "ɪ", 0.05),  # Almost allophonic high front vowels
        ("u", "ʊ", 0.05),  # Almost allophonic high back vowels
        ("e", "ɛ", 0.1),  # Mid front vowels
        ("o", "ɔ", 0.1),  # Mid back vowels
        ("a", "i", 0.4),  # Low vs high vowel (distant)
        ("a", "u", 0.45),  # Low front vs high back (distant)
        ("i", "u", 0.3),  # Front vs back high vowels
        # New vowel pairs
        ("a", "ɐ", 0.1),  # Open front vs near-open central vowel
        ("e", "ɪ", 0.2),  # Mid front vs high front vowel
        ("o", "ʊ", 0.2),  # Mid back vs high back vowel
        ("æ", "ɛ", 0.1),  # Near-open front vs open-mid front vowel
        # Added vowel length pairs
        ("a", "aː", 0.05),  # Short vs long vowel (minimal difference)
        ("a", "aːː", 0.1),  # Short vs extra-long vowel
        ("aː", "aːː", 0.05),  # Long vs extra-long vowel
        # Added vowel-consonant pairs
        ("a", "bː", 0.8),  # Vowel vs long voiced bilabial stop
        ("a", "bːʰ", 0.85),  # Vowel vs long aspirated bilabial stop
        # Consonant pairs (similar manner, different place)
        ("p", "t", 0.25),  # Voiceless stops, different place
        ("p", "k", 0.3),  # Voiceless stops, different place
        ("t", "k", 0.25),  # Voiceless stops, different place
        ("b", "d", 0.25),  # Voiced stops, different place
        ("b", "g", 0.3),  # Voiced stops, different place
        ("d", "g", 0.25),  # Voiced stops, different place
        ("f", "s", 0.25),  # Voiceless fricatives, different place
        ("f", "x", 0.35),  # Voiceless fricatives, different place
        ("s", "x", 0.3),  # Voiceless fricatives, different place
        ("m", "n", 0.2),  # Nasals, different place
        ("m", "ŋ", 0.3),  # Nasals, different place
        ("n", "ŋ", 0.25),  # Nasals, different place
        # New consonant pairs
        ("p", "b", 0.1),  # Voiceless vs voiced bilabial stop
        ("t", "d", 0.1),  # Voiceless vs voiced alveolar stop
        ("k", "g", 0.1),  # Voiceless vs voiced velar stop
        ("s", "z", 0.1),  # Voiceless vs voiced alveolar fricative
        ("f", "v", 0.1),  # Voiceless vs voiced labiodental fricative
        ("ʃ", "ʒ", 0.1),  # Voiceless vs voiced postalveolar fricative
        # Sonority differences (increasing distance with sonority difference)
        ("p", "m", 0.35),  # Stop vs nasal (same place)
        ("t", "n", 0.35),  # Stop vs nasal (same place)
        ("p", "l", 0.45),  # Stop vs liquid
        ("p", "w", 0.5),  # Stop vs glide
        ("m", "l", 0.25),  # Nasal vs liquid
        ("m", "w", 0.3),  # Nasal vs glide
        ("l", "w", 0.2),  # Liquid vs glide
        # Vowel-consonant pairs (large distances)
        ("a", "p", 0.85),  # Vowel vs stop
        ("i", "t", 0.85),  # Vowel vs stop
        ("u", "k", 0.85),  # Vowel vs stop
        ("a", "m", 0.75),  # Vowel vs nasal
        ("i", "n", 0.75),  # Vowel vs nasal
        ("u", "ŋ", 0.75),  # Vowel vs nasal
        ("a", "l", 0.7),  # Vowel vs liquid
        ("i", "l", 0.7),  # Vowel vs liquid
        ("a", "w", 0.65),  # Vowel vs glide
        # Special cases - smaller differences
        ("i", "j", 0.2),  # High front vowel vs palatal glide
        ("u", "w", 0.2),  # High back vowel vs labio-velar glide
        # Consonants with secondary articulation
        ("p", "pʷ", 0.1),  # Plain vs labialized
        ("t", "tʷ", 0.15),  # Plain vs labialized
        ("p", "pʲ", 0.15),  # Plain vs palatalized
        ("t", "tʲ", 0.1),  # Plain vs palatalized
        # Pre-nasalized consonants
        ("t", "ⁿt", 0.2),  # Plain vs pre-nasalized
        ("d", "ⁿd", 0.2),  # Plain vs pre-nasalized
        # Ejectives
        ("p", "pʼ", 0.15),  # Plain vs ejective
        ("t", "tʼ", 0.15),  # Plain vs ejective
        ("k", "kʼ", 0.15),  # Plain vs ejective
        # Tones
        ("²", "³", 0.1),  # Low-mid vs mid tone
        ("³", "¹", 0.2),  # Mid vs low tone
        ("²", "¹", 0.1),  # Low-mid vs low tone
        ("²³", "²¹", 0.3),  # Rising vs falling from low-mid
        ("²⁵", "²¹", 0.5),  # High rise vs fall from low-mid
        ("²⁵", "¹⁵", 0.25),  # High rise from low-mid vs from low
        ("²⁵²", "²⁵", 0.2),  # Complex vs simple contour
        # Rhotacized vowels
        ("a˞", "a", 0.1),  # Rhotacized vs plain vowel
        ("ə˞", "ə", 0.1),  # Rhotacized vs plain schwa
        # Spread/aspirated features
        ("bʰ", "b", 0.1),  # Aspirated vs plain voiced stop
        ("bʰː", "bː", 0.1),  # Long aspirated vs long plain
        # Retracted sounds
        ("wˠ", "w", 0.2),  # Retracted vs plain approximant
        # Trilled consonants
        ("r", "ɾ", 0.15),  # Trilled vs tap rhotic
        ("rː", "r", 0.1),  # Long vs short trill
        ("rʲ", "r", 0.15),  # Palatalized vs plain trill
    ]

    # Remove potential duplicates (like the i-j pair) keeping the last occurrence
    unique_pairs_dict = {(s1, s2): dist for s1, s2, dist in pairs}
    unique_pairs = [(s1, s2, dist) for (s1, s2), dist in unique_pairs_dict.items()]

    return unique_pairs


def find_sound_index(sound, sounds):
    """
    Find the index of a sound in the sounds list, with fallback options.
    """
    try:
        return sounds.index(sound)
    except ValueError:
        pass  # Try normalization

    # Try Unicode normalized forms
    try:
        normalized = unicodedata.normalize("NFD", sound)
        return sounds.index(normalized)
    except ValueError:
        # Fallback: Check if sound without diacritics exists (simple version)
        base_sound = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        try:
            return sounds.index(base_sound)
        except ValueError:
            return None  # Not found


def get_available_expert_pairs(sounds, expert_pairs):
    """
    Filter expert-defined pairs to only include sounds found in our dataset.
    """
    available_pairs = []
    not_found_sounds = set()

    for sound1, sound2, distance in expert_pairs:
        idx1 = find_sound_index(sound1, sounds)
        idx2 = find_sound_index(sound2, sounds)

        if idx1 is not None and idx2 is not None:
            available_pairs.append((idx1, idx2, distance))
        else:
            if idx1 is None:
                not_found_sounds.add(sound1)
            if idx2 is None:
                not_found_sounds.add(sound2)

    if not_found_sounds:
        print(
            f"Warning: The following sounds from expert_pairs were not found in feature_system.csv: {', '.join(sorted(list(not_found_sounds)))}"
        )

    return available_pairs


def generate_training_data(
    sounds, feature_matrix, feature_ranges, expert_pairs, n_additional_samples=500
):
    """
    Generate training data combining expert-defined pairs and additional random pairs.
    """
    # Get expert-defined pairs that are available in our sound inventory
    available_expert_pairs = get_available_expert_pairs(sounds, expert_pairs)
    print(
        f"Using {len(available_expert_pairs)} expert-defined phoneme pairs for training."
    )

    # Create feature vectors and target distances from expert pairs
    X = []
    y = []

    for idx1, idx2, distance in available_expert_pairs:
        feature_vector = compute_feature_difference_vector(
            idx1, idx2, feature_matrix, feature_ranges
        )
        X.append(feature_vector)
        y.append(distance)

    # Generate additional random pairs
    num_sounds = len(sounds)
    # Create a set of pairs already used (consider both orders)
    used_pairs = set()
    for idx1, idx2, _ in available_expert_pairs:
        used_pairs.add(tuple(sorted((idx1, idx2))))

    all_potential_pairs = list(combinations(range(num_sounds), 2))
    random.shuffle(all_potential_pairs)

    additional_count = 0
    for idx1, idx2 in all_potential_pairs:
        # Check if this pair (in any order) has already been used
        if tuple(sorted((idx1, idx2))) not in used_pairs:
            # Create a feature difference vector
            feature_vector = compute_feature_difference_vector(
                idx1, idx2, feature_matrix, feature_ranges
            )

            # Synthesize a plausible distance based on feature differences
            # This is a heuristic approach that will be refined by the ML model
            vec1 = feature_matrix[idx1]
            vec2 = feature_matrix[idx2]

            both_nonzero = (vec1 != 0) & (vec2 != 0)
            either_nonzero = (vec1 != 0) | (vec2 != 0)
            num_both_nonzero = np.sum(both_nonzero)
            num_either_nonzero = np.sum(either_nonzero)

            # Calculate feature difference for shared features
            shared_diff_sum = 0.0
            if num_both_nonzero > 0:
                for i in range(len(vec1)):
                    if both_nonzero[i]:
                        min_val, max_val = feature_ranges[i]
                        range_size = max_val - min_val
                        if range_size > 1e-9:
                            shared_diff_sum += abs(vec1[i] - vec2[i]) / range_size
                        # else: # If range is zero, difference is 0 if values are same

            # Calculate ratio of shared to either features
            feature_sharing_ratio = (
                num_both_nonzero / num_either_nonzero if num_either_nonzero > 0 else 1.0
            )

            # Synthetic distance calculation (heuristic)
            # Base distance related to how many features are NOT shared
            synthetic_distance = (
                1.0 - feature_sharing_ratio
            ) * 0.6  # Weight non-shared features more

            # Add contribution from differences in shared features
            if num_both_nonzero > 0:
                # Average difference across shared features, weighted
                synthetic_distance += (shared_diff_sum / num_both_nonzero) * 0.4

            # Add a small random variation to diversify synthetic data
            synthetic_distance += (random.random() - 0.5) * 0.1  # Smaller variation
            synthetic_distance = max(
                0.0, min(1.0, synthetic_distance)
            )  # Clip to [0, 1]

            X.append(feature_vector)
            y.append(synthetic_distance)

            used_pairs.add(tuple(sorted((idx1, idx2))))
            additional_count += 1

            if additional_count >= n_additional_samples:
                break

    print(f"Added {additional_count} synthetic training pairs.")
    return np.array(X), np.array(y)


def compute_ml_distance_matrix(sounds, feature_matrix, features, descriptions=None):
    """
    Compute distance matrix using a machine learning model trained on expert-defined pairs.
    """
    num_sounds = len(sounds)

    # Get feature ranges
    feature_ranges = get_feature_ranges(feature_matrix)

    # Normalize features
    normalized_matrix = normalize_features(feature_matrix, feature_ranges)

    # Get expert-defined phoneme pairs
    expert_pairs = expert_defined_phoneme_pairs()

    # --- Feature Coverage Check ---
    print("\n--- Checking Feature Coverage in Expert Pairs ---")
    feature_value_map = defaultdict(list)
    for sound_idx, sound_features in enumerate(
        feature_matrix
    ):  # Use original matrix for mapping
        for feature_idx, value in enumerate(sound_features):
            if value != 0:  # Exclude zero values
                feature_value_map[(feature_idx, value)].append(sound_idx)

    expert_sound_indices = set()
    available_expert_pairs_indices = get_available_expert_pairs(
        sounds, expert_pairs
    )  # Get indices directly
    for idx1, idx2, _ in available_expert_pairs_indices:
        expert_sound_indices.add(idx1)
        expert_sound_indices.add(idx2)

    missing_feature_values = []
    for (feature_idx, value), sound_indices_with_feature in feature_value_map.items():
        is_represented = False
        for sound_idx in sound_indices_with_feature:
            if sound_idx in expert_sound_indices:
                is_represented = True
                break
        if not is_represented:
            missing_feature_values.append(
                ((feature_idx, value), sound_indices_with_feature)
            )

    if not missing_feature_values:
        print(
            "Coverage Check: All non-zero feature/value combinations are represented by at least one sound in the expert pairs list."
        )
    else:
        print(
            f"WARNING: Found {len(missing_feature_values)} non-zero feature/value combinations NOT represented in the expert pairs list."
        )
        print(
            "Consider adding pairs involving these features/values to improve model training:"
        )
        count = 0
        for (feature_idx, value), sound_indices_with_feature in sorted(
            missing_feature_values, key=lambda item: features[item[0][0]]
        ):  # Sort by feature name
            feature_name = features[feature_idx]
            example_sounds = [
                sounds[idx] for idx in sound_indices_with_feature[:3]
            ]  # Get up to 3 examples
            print(f"  - Feature '{feature_name}' = {value}:")
            # print(f"    (Not represented by any sound in expert pairs)") # Redundant
            print(f"    Example sounds: {', '.join(example_sounds)}")
            count += 1
            if count >= 15:  # Limit warnings printed
                print("  ... (additional missing features not shown)")
                break
    print("--- End Feature Coverage Check ---\n")
    # --- End Feature Coverage Check ---

    # Generate training data
    print("Generating training data...")
    X_train_full, y_train_full = generate_training_data(
        sounds,
        normalized_matrix,
        feature_ranges,
        expert_pairs,
        n_additional_samples=2000,
    )  # Increased synthetic samples

    if len(X_train_full) == 0:
        raise ValueError(
            "No training data generated. Check expert pairs and sound list."
        )

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # Train a Random Forest model
    print("Training model...")
    # Optimized hyperparameters to better handle multicollinearity and prevent overfitting
    model = RandomForestRegressor(
        n_estimators=300,  # More trees for robustness
        max_depth=20,  # Limiting depth to prevent overfitting
        min_samples_split=6,  # Increased to reduce overfitting
        min_samples_leaf=4,  # Increased to reduce overfitting
        max_features="sqrt",  # Use sqrt of features for each split - helps with multicollinearity
        bootstrap=True,  # Use bootstrapping for robustness
        oob_score=True,  # Out-of-bag scoring to monitor overfitting
        n_jobs=-1,  # Use all available cores
        random_state=42,
    )

    # Balance weight between expert pairs and synthetic pairs
    sample_weights = np.ones(len(X_train))
    # Calculate indices for expert pairs in the training set (first len(available_expert_pairs_indices) * 0.8 samples)
    num_expert_pairs = int(
        0.8 * len(available_expert_pairs_indices)
    )  # 80% of expert pairs are in training
    # Weight expert pairs higher
    sample_weights[:num_expert_pairs] = 2.0

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Evaluate model performance
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    print(f"Validation MSE: {val_mse:.4f}")
    if hasattr(model, "oob_score_"):
        print(f"Out-of-bag score: {model.oob_score_:.4f}")

    # Generate all pair combinations for prediction
    print("Predicting distances for all pairs...")
    distance_matrix = np.zeros((num_sounds, num_sounds))

    # Create a list of all phoneme pairs for batch prediction
    all_pairs_features = []
    all_pair_indices = []

    for i in range(num_sounds):
        for j in range(i + 1, num_sounds):  # Only compute upper triangle
            feature_vector = compute_feature_difference_vector(
                i, j, normalized_matrix, feature_ranges
            )
            all_pairs_features.append(feature_vector)
            all_pair_indices.append((i, j))

    # Batch predict all distances
    if all_pairs_features:  # Ensure there are pairs to predict
        all_distances = model.predict(np.array(all_pairs_features))
    else:
        all_distances = []

    # Fill in the distance matrix
    for (i, j), distance in zip(all_pair_indices, all_distances):
        # Ensure distance is within [0, 1] range after prediction
        clipped_distance = max(0.0, min(1.0, distance))
        distance_matrix[i, j] = distance_matrix[j, i] = clipped_distance

    # --- Post-prediction Analysis ---
    # Feature Importance
    feature_importances = model.feature_importances_
    # Map importance back to original features (handling the extra 'missing_ratio' feature)
    num_orig_features = len(features)
    importance_dict = {}
    for i in range(num_orig_features):
        importance_dict[features[i]] = feature_importances[i]
    if len(feature_importances) > num_orig_features:
        importance_dict["_missing_feature_ratio_"] = feature_importances[
            num_orig_features
        ]

    sorted_importances = sorted(
        importance_dict.items(), key=lambda item: item[1], reverse=True
    )

    print("\nTop 15 Feature Importances:")
    for feature_name, importance in sorted_importances[:15]:
        print(f"  {feature_name}: {importance:.4f}")

    # Sample Predictions vs Expert
    if available_expert_pairs_indices:
        print("\nSample of predicted vs expert distances (from available pairs):")
        sample_indices = random.sample(
            available_expert_pairs_indices, min(10, len(available_expert_pairs_indices))
        )

        # Calculate average difference for all expert pairs
        all_diffs = []
        for idx1, idx2, expert_distance in available_expert_pairs_indices:
            predicted_distance = distance_matrix[idx1, idx2]
            diff = abs(predicted_distance - expert_distance)
            all_diffs.append(diff)

        avg_diff = sum(all_diffs) / len(all_diffs) if all_diffs else 0
        print(
            f"Average difference between predicted and expert distances: {avg_diff:.4f}"
        )

        # Show sample pairs
        for idx1, idx2, expert_distance in sample_indices:
            predicted_distance = distance_matrix[idx1, idx2]
            sound1 = sounds[idx1]
            sound2 = sounds[idx2]
            diff = abs(predicted_distance - expert_distance)
            print(
                f"  {sound1:<4} - {sound2:<4}: pred={predicted_distance:.3f}, expert={expert_distance:.3f}, diff={diff:.3f}"
            )
    # --- End Post-prediction Analysis ---

    return distance_matrix


def save_distance_matrix(file_path, sounds, distance_matrix):
    """
    Saves the distance matrix to a text file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("\t" + "\t".join(sounds) + "\n")
        # Write rows
        for i, sound in enumerate(sounds):
            f.write(
                sound
                + "\t"
                + "\t".join(f"{distance:.4f}" for distance in distance_matrix[i])
                + "\n"
            )


def derive_soundclass(sounds, feature_matrix, features):
    """
    Derive Dolgopolsky sound classes dynamically based on features.
    """
    sound_classes = defaultdict(list)

    for sound_idx, sound_features in enumerate(feature_matrix):
        sound = sounds[sound_idx]

        # Classify tones (keep existing tone management)
        if "tone" in features:
            tone_indices = [i for i, f in enumerate(features) if "tone" in f]
            tone_values = [sound_features[i] for i in tone_indices]
            if any(tone_values):
                if tone_values == [1, 1]:  # Example: low even tones
                    sound_classes["0"].append(sound)
                elif tone_values == [1, 3]:  # Example: rising tones
                    sound_classes["1"].append(sound)
                elif tone_values == [5, 1]:  # Example: falling tones
                    sound_classes["2"].append(sound)
                elif tone_values == [3, 3]:  # Example: mid even tones
                    sound_classes["3"].append(sound)
                elif tone_values == [4, 4]:  # Example: high even tones
                    sound_classes["4"].append(sound)
                elif tone_values == [2]:  # Example: short tones
                    sound_classes["5"].append(sound)
                elif tone_values == [2, 1, 4]:  # Example: complex tones
                    sound_classes["6"].append(sound)
                continue

        # Classify sounds based on Dolgopolsky system
        if "labial" in features and sound_features[features.index("labial")] == 1:
            if sound_features[features.index("continuant")] == -1:
                sound_classes["P"].append(sound)  # Labial obstruents
            elif sound_features[features.index("nasal")] == 1:
                sound_classes["M"].append(sound)  # Labial nasal
            elif sound_features[features.index("continuant")] == 1:
                sound_classes["W"].append(sound)  # Voiced labial fricative
            continue

        if "dental" in features or "alveolar" in features:
            if sound_features[features.index("continuant")] == -1:
                sound_classes["T"].append(sound)  # Dental obstruents
            elif sound_features[features.index("sibilant")] == 1:
                sound_classes["S"].append(sound)  # Alveolar/postalveolar fricatives
            elif sound_features[features.index("nasal")] == 1:
                sound_classes["N"].append(sound)  # Remaining nasals
            elif sound_features[features.index("lateral")] == 1 or sound_features[features.index("trill")] == 1 or sound_features[features.index("tap")] == 1:
                sound_classes["R"].append(sound)  # Trills, taps, flaps, lateral approximants
            continue

        if "velar" in features or "uvular" in features:
            if sound_features[features.index("continuant")] == -1:
                sound_classes["K"].append(sound)  # Velar/postvelar obstruents and affricates
            elif sound_features[features.index("nasal")] == 1:
                sound_classes["ø"].append(sound)  # Initial velar nasal
            continue

        if "palatal" in features and sound_features[features.index("approximant")] == 1:
            sound_classes["J"].append(sound)  # Palatal approximant
            continue

        if "laryngeal" in features and sound_features[features.index("laryngeal")] == 1:
            sound_classes["ø"].append(sound)  # Laryngeals
            continue

        # Default to "other" if no specific classification is found
        sound_classes["other"].append(sound)

    return sound_classes


def adjust_distances_for_soundclass(
    sound1_idx, sound2_idx, feature_matrix, feature_ranges, sound_classes, sounds
):
    """
    Adjust distances based on Dolgopolsky sound classes.
    """
    sound1 = sounds[sound1_idx]
    sound2 = sounds[sound2_idx]

    # Determine the classes of the two sounds
    class1 = next((cls for cls, members in sound_classes.items() if sound1 in members), None)
    class2 = next((cls for cls, members in sound_classes.items() if sound2 in members), None)

    # If one sound is a tone and the other is not, set maximum distance
    if (class1 in {"0", "1", "2", "3", "4", "5", "6"} and class2 not in {"0", "1", "2", "3", "4", "5", "6"}) or \
       (class2 in {"0", "1", "2", "3", "4", "5", "6"} and class1 not in {"0", "1", "2", "3", "4", "5", "6"}):
        return 1.0

    # If sounds are in different classes, increase the distance proportionally
    if class1 != class2:
        base_distance = compute_feature_difference_vector(
            sound1_idx, sound2_idx, feature_matrix, feature_ranges
        ).mean()  # Use mean feature difference as base
        return min(1.0, base_distance + 0.2)  # Add a proportional penalty

    # If sounds are in the same class, use the base distance
    return compute_feature_difference_vector(
        sound1_idx, sound2_idx, feature_matrix, feature_ranges
    ).mean()


def compute_ml_distance_matrix_with_soundclass(sounds, feature_matrix, features, descriptions=None):
    """
    Compute distance matrix using a machine learning model, incorporating Dolgopolsky sound classes.
    """
    num_sounds = len(sounds)

    # Get feature ranges
    feature_ranges = get_feature_ranges(feature_matrix)

    # Normalize features
    normalized_matrix = normalize_features(feature_matrix, feature_ranges)

    # Derive Dolgopolsky sound classes
    sound_classes = derive_soundclass(sounds, feature_matrix, features)
    print("\nDerived Dolgopolsky Sound Classes:")
    for cls, members in sound_classes.items():
        print(f"  {cls}: {', '.join(members)}")

    # Compute distance matrix
    distance_matrix = np.zeros((num_sounds, num_sounds))

    for i in range(num_sounds):
        for j in range(i + 1, num_sounds):  # Only compute upper triangle
            distance = adjust_distances_for_soundclass(
                i, j, normalized_matrix, feature_ranges, sound_classes, sounds
            )
            distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


if __name__ == "__main__":
    # Load the feature system
    try:
        sounds, features, feature_matrix, descriptions = load_feature_system_csv(
            "feature_system.csv"
        )
    except FileNotFoundError:
        print("Error: feature_system.csv not found in the current directory.")
        exit()
    except ValueError as e:
        print(f"Error processing feature_system.csv: {e}")
        exit()

    # Compute ML-based distance matrix
    distance_matrix = compute_ml_distance_matrix(
        sounds, feature_matrix, features, descriptions
    )

    # Save distance matrix to a file
    output_filename = "ml_distance_matrix.txt"
    save_distance_matrix(output_filename, sounds, distance_matrix)
    print(f"\nDistance matrix saved to {output_filename}")

    # --- Final Analysis Output ---
    num_sounds = len(sounds)
    if num_sounds > 1:
        # Flatten upper triangle for analysis
        flat_distances = distance_matrix[np.triu_indices(num_sounds, k=1)]

        # Print overall stats
        print(f"\nDistance Matrix Stats:")
        print(f"  Min distance: {np.min(flat_distances):.4f}")
        print(f"  Max distance: {np.max(flat_distances):.4f}")
        print(f"  Mean distance: {np.mean(flat_distances):.4f}")
        print(f"  Median distance: {np.median(flat_distances):.4f}")

        # Combine sounds and distances for sorting
        dist_list = [
            (sounds[i], sounds[j], distance_matrix[i, j])
            for i in range(num_sounds)
            for j in range(i + 1, num_sounds)
        ]

        # Print the top 10 largest distances
        dist_list.sort(key=lambda x: x[2], reverse=True)
        print("\nTop 10 Largest Distances:")
        for sound1, sound2, distance in dist_list[:10]:
            print(f"  {sound1:<4} - {sound2:<4}: {distance:.4f}")

        # Print the top 10 smallest distances
        dist_list.sort(key=lambda x: x[2])
        print("\nTop 10 Smallest Distances (excluding self):")
        for sound1, sound2, distance in dist_list[:10]:
            print(f"  {sound1:<4} - {sound2:<4}: {distance:.4f}")
    else:
        print("\nNot enough sounds to calculate distances.")
    # --- End Final Analysis Output ---
