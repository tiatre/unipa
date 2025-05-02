import csv

import distfeat
from collections import Counter
import unicodedata

CLTS_GRAPHEMES_FILE = "graphemes.tsv"


def read_clts():
    """
    Builds the sound catalogue by reading the CLTS graphemes file and filtering out unwanted sounds.
    """
    sounds = {}
    exclusion_keywords = {
        "unspecified-manner",
        "unspecified-place",
        "unspecified-voice",
        "weak",
        "with-falling_tone",
        "with-high_tone",
        "with-low_tone",
        "with-mid_tone",
        "with-rising_tone",
    }

    with open(CLTS_GRAPHEMES_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["DATASET"] != "bipa":
                continue

            grapheme = row["GRAPHEME"]
            name = sorted(row["NAME"].split())

            # Skip sounds containing any exclusion keywords
            if any(keyword in name for keyword in exclusion_keywords):
                continue

            sounds[grapheme] = name

    return sounds


sounds = read_clts()

# Convert descriptors to features and print them
feature_collection = []
my_sounds = {}
for sound, descriptors in sorted(sounds.items()):
    features = distfeat.descriptors_to_feature_bundle(descriptors)
    feature_collection.append(tuple(sorted(features.items())))
    my_sounds[sound] = tuple(sorted(features.items()))

# Define a list of feature sets to exclude from duplicate verification
excluded_feature_sets = [
    (
        ("alveolar", 1),
        ("bilabial", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("alveolar", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("syllabic", -1),
        ("velar", 1),
        ("voice", 1),
    ),
    (
        ("alveolar", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("syllabic", -1),
        ("velar", 1),
        ("voice", -1),
    ),
    (
        ("alveolar", 1),
        ("bilabial", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("strident", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", 1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("consonantal", 1),
        ("continuant", 1),
        ("dorsal", 1),
        ("sonorant", -1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", 1),
        ("coronal", 1),
        ("distributed", 1),
        ("sonorant", -1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("consonantal", 1),
        ("continuant", -1),
        ("dorsal", 1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("consonantal", 1),
        ("continuant", -1),
        ("dorsal", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("consonantal", 1),
        ("continuant", -1),
        ("dorsal", 1),
        ("pharyngeal", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("labial", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("labial", 1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("consonantal", 1),
        ("continuant", 1),
        ("labial", 1),
        ("sonorant", -1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("labial", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("labial", 1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("dorsal", 1),
        ("syllabic", -1),
        ("voice", 1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("dorsal", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("anterior", 1),
        ("consonantal", 1),
        ("continuant", -1),
        ("coronal", 1),
        ("distributed", -1),
        ("nasal", 1),
        ("syllabic", -1),
        ("voice", -1),
    ),
    (
        ("consonantal", 1),
        ("continuant", 1),
        ("dorsal", 1),
        ("pharyngeal", 1),
        ("sonorant", -1),
        ("syllabic", -1),
        ("voice", -1),
    ),
]


def find_repeated_features(feature_collection, excluded_feature_sets, sounds):
    """
    Identifies repeated features in the feature collection, excluding specified feature sets.

    Args:
        feature_collection (list): A list of feature bundles.
        excluded_feature_sets (list): A list of feature sets to exclude from duplicate verification.
        sounds (dict): A dictionary of graphemes and their descriptors.

    Returns:
        dict: A dictionary of repeated features and their counts.
    """
    feature_counter = Counter(feature_collection)
    repeated_features = {}

    for feature, count in feature_counter.items():
        if count > 1:
            # Skip features in the exclusion list
            if feature in excluded_feature_sets:
                continue

            # Collect graphemes with this feature
            graphemes_with_feature = [
                (sound, descriptors)
                for sound, descriptors in sounds.items()
                if tuple(
                    sorted(distfeat.descriptors_to_feature_bundle(descriptors).items())
                )
                == feature
            ]
            # Check if all graphemes have the same descriptors
            descriptor_sets = {
                tuple(descriptors) for _, descriptors in graphemes_with_feature
            }
            if len(descriptor_sets) > 1:  # Only consider if descriptors differ
                # Apply filtering iteratively to handle multiple sounds
                filtered_descriptor_sets = descriptor_sets
                for exclusion_set in [
                    {"voiceless", "voiced", "devoiced"},
                    {"voiced", "voiceless", "revoiced"},
                    {"unreleased"},
                ]:
                    filtered_descriptor_sets = {
                        tuple(set(descriptor_set) - exclusion_set)
                        for descriptor_set in filtered_descriptor_sets
                    }
                # If differences remain after filtering, count as repeated
                if len(filtered_descriptor_sets) > 1:
                    repeated_features[feature] = count

    return repeated_features


# Use the function to find repeated features
repeated_features = find_repeated_features(
    feature_collection, excluded_feature_sets, sounds
)

if repeated_features:
    print("\nRepeated features found:")
    for idx, (feature, count) in enumerate(repeated_features.items(), start=1):
        print(f"{idx}. Feature: {feature}")
        print(f"   Count: {count}")

        # Collect graphemes with this feature and group by unique descriptor bundles
        grapheme_groups = {}
        for sound, descriptors in sorted(sounds.items()):
            features = tuple(
                sorted(distfeat.descriptors_to_feature_bundle(descriptors).items())
            )
            if features == feature:
                descriptor_bundle = tuple(descriptors)
                if descriptor_bundle not in grapheme_groups:
                    grapheme_groups[descriptor_bundle] = sound

        print(f"   Unique descriptor bundles: {len(grapheme_groups)}")
        print("   Graphemes with this feature (one per unique bundle):")
        for descriptors, sound in grapheme_groups.items():
            print(f"     - Grapheme: {sound}")
            print(f"       Descriptors: {', '.join(descriptors)}")
        print("-" * 50)  # Separator for better readability
else:
    print("\nNo repeated features found.")


def generate_feature_system_csv(sounds, output_file="feature_system.csv"):
    """
    Generates a CSV file with the complete feature system, where each feature is a separate column,
    and includes a "description" column with textual descriptors.

    Args:
        sounds (dict): A dictionary of graphemes and their feature-value pairs.
        output_file (str): The name of the output CSV file.
    """
    # Collect all unique features across all sounds
    all_features = sorted(
        {feature for descriptors in sounds.values() for feature, _ in descriptors}
    )

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row: "sound", "description", followed by all features
        writer.writerow(["sound", "description"] + all_features)

        # Write each sound and its feature values
        for sound, descriptors in sorted(
            sounds.items(), key=lambda item: unicodedata.normalize("NFD", item[0])
        ):
            # Normalize grapheme to NFD Unicode format
            normalized_sound = unicodedata.normalize("NFD", sound)
            # Create the textual description by joining feature-value pairs
            description = " ".join(
                f"{feature}={value}" for feature, value in descriptors
            )
            # Create a row with the actual feature values (-1, 0, or +1)
            feature_values = [
                next((value for feature, value in descriptors if feature == f), 0)
                for f in all_features
            ]
            writer.writerow([normalized_sound, description] + feature_values)


# Generate the feature system CSV
generate_feature_system_csv(my_sounds)
