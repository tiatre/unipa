# Unified mapping: descriptors (as tuples) to distinctive features
# The longer the descriptor tuple, the more specific it is
# The function will match the longest applicable tuple from a descriptor set

descriptor_to_features = {
    # Multi-descriptor mappings
    ("alveolar-and-bilabial",): {
        "coronal": +1,
        "anterior": +1,
        "distributed": -1,
        "labial": +1,
    },
    ("alveolar-and-velar",): {
        "coronal": +1,
        "anterior": +1,
        "distributed": -1,
        "dorsal": +1,
    },
    ("bilabial-and-alveolar",): {
        "labial": +1,
        "anterior": +1,
        "coronal": +1,
        "distributed": -1,
    },
    ("bilabial-and-velar",): {"labial": +1, "anterior": +1, "dorsal": +1},
    ("devoiced", "voiced", "implosive"): {"voice": -1, "constricted": +1},
    ("devoiced", "voiced"): {"voice": -1},
    ("labialized", "bilabial"): {"labial": +1, "round": +1},
    ("labialized", "labio-dental"): {"labial": +1, "round": +1},
    ("labialized", "velar"): {"labial": +1, "dorsal": +1, "distributed": -1},
    ("non-syllabic", "vowel"): {"syllabic": -1, "consonantal": -1},
    ("palatalized", "alveolar"): {"dorsal": +1, "atr": +1, "distributed": -1},
    ("palatalized", "labio-dental"): {"dorsal": +1, "distributed": +1, "atr": +1},
    ("palatalized", "post-alveolar"): {"dorsal": +1, "atr": +1, "distributed": +1},
    ("palatalized", "retroflex"): {"coronal": +1, "distributed": -1, "dorsal": +1, "atr": +1},
    ("pre-glottalized-and-nasalized",): {"preglottalized": +1, "nasal": +1},
    ("revoiced", "voiceless"): {"voice": +1},
    ("unspecified-manner",): {"continuant": 0},
    ("unspecified-place",): {"place": 0},
    ("unspecified-voice",): {"voice": 0},
    ("velar-and-alveolar",): {
        "coronal": +1,
        "anterior": +1,
        "distributed": -1,
        "dorsal": +1,
    },
    ("velar-and-bilabial",): {"dorsal": +1, "labial": +1, "constricted": +1},
    ("velarized", "labio-velar"): {"retracted": +1},
    ("with-falling_tone",): {"tone_contour": -1},
    ("with-high_tone",): {"tone_level": +1},
    ("with-low_tone",): {"tone_level": -1},
    ("with-nasal-release",): {"nasal": +1},
    ("with-rising_tone",): {"tone_contour": +1},
    # Single descriptors
    ("advanced",): {"atr": +1},
    ("affricate",): {"continuant": -1, "strident": +1},
    ("alveolar",): {"coronal": +1, "anterior": +1, "distributed": -1},
    ("alveolo-palatal",): {"coronal": +1, "dorsal": +1},
    ("apical",): {"apical": +1},
    ("approximant",): {"approximant": +1, "sonorant": +1, "consonantal": +1},
    ("aspirated",): {"spread": +1},
    ("back",): {"back": +1},
    ("bilabial",): {"labial": +1, "anterior": +1},
    ("breathy",): {"breathy": +1},
    ("central",): {"centralized": +1},
    ("centralized",): {"centralized": +1},
    ("click",): {"click": +1, "continuant": -1, "consonantal": +1, "sonorant": -1},
    ("close-mid",): {"high": +1, "low": -1},
    ("close",): {"high": +1, "tense": +1},
    ("consonant",): {"consonantal": +1, "syllabic": -1},
    ("creaky",): {"creaky": +1},
    ("dental",): {"coronal": +1, "anterior": +1, "distributed": +1},
    ("devoiced",): {"voice": -1},
    ("ejective",): {"constricted": +1},
    ("epiglottal",): {"pharyngeal": +1, "constricted": +1},
    ("falling",): {"tone_contour": -1},
    ("flat",): {"tone_contour": 0},
    ("fricative",): {"continuant": +1, "sonorant": -1},
    ("from-high",): {"tone_start": 1},
    ("from-low",): {"tone_start": -1},
    ("from-mid-high",): {"tone_start": 0.5},
    ("from-mid-low",): {"tone_start": -0.5},
    ("from-mid",): {"tone_start": 0},
    ("front",): {"back": -1},
    ("glottal",): {"glottal": +1},
    ("glottalized",): {"constricted": +1},
    ("high",): {"high": +1},
    ("implosive",): {"voice": +1, "constricted": +1},
    ("labialized",): {"round": +1},
    ("labio-dental",): {"labial": +1},
    ("labio-palatal",): {"labial": +1, "coronal": +1},
    ("labio-velar",): {"labial": +1, "dorsal": +1},
    ("laminal",): {"laminal": +1},
    ("lateral",): {"lateral": +1},
    ("linguolabial",): {"labial": +1, "coronal": +1},
    ("long",): {"length": +1},
    ("low",): {"low": +1},
    ("mid-long",): {"length": 0},
    ("mid",): {"high": 0, "low": 0},
    ("nasal-click",): {"nasal": +1, "click": +1},
    ("nasal",): {"nasal": +1},
    ("nasalized",): {"nasal": +1},
    ("near-back",): {"back": +1},
    ("near-close",): {"high": +1, "tense": -1},
    ("near-front",): {"back": -1},
    ("near-open",): {"low": +1, "tense": -1},
    ("non-syllabic",): {"syllabic": -1},
    ("open-mid",): {"low": +1, "high": -1},
    ("open",): {"low": +1, "tense": +1},
    ("palatal-velar",): {"dorsal": +1, "coronal": +1, "distributed": +1},
    ("palatal",): {"coronal": +1},
    ("palatalized",): {"dorsal": +1, "atr": +1},
    ("pharyngeal",): {"pharyngeal": +1},
    ("pharyngealized",): {"pharyngeal": +1, "atr": -1},
    ("post-alveolar",): {"coronal": +1, "distributed": +1},
    ("pre-aspirated",): {"preaspirated": +1},
    ("pre-glottalized",): {"preglottalized": +1},
    ("pre-nasalized",): {"prenasal": +1},
    ("retracted-tongue-root",): {"atr": -1},
    ("retracted",): {"atr": -1},
    ("retroflex",): {"coronal": +1, "distributed": -1},
    ("revoiced",): {"voice": +1},
    ("rhotacized",): {"rhotacized": +1},
    ("rising",): {"tone_contour": 1},
    ("rounded",): {"round": +1},
    ("short",): {"length": -1},
    ("sibilant",): {"sibilant": +1},
    ("stop",): {"continuant": -1},
    ("syllabic",): {"syllabic": +1},
    ("tap",): {"vibrant": +1},
    ("tense",): {"tense": +1},
    ("to-high",): {"tone_end": 1},
    ("to-low",): {"tone_end": -1},
    ("to-mid-high",): {"tone_end": 0.5},
    ("to-mid-low",): {"tone_end": -0.5},
    ("to-mid",): {"tone_end": 0},
    ("tone",): {"tone_level": 0},
    ("trill",): {"vibrant": +1, "trilled": +1},
    ("ultra-long",): {"length": +2},
    ("ultra-short",): {"length": -2},
    ("unrounded",): {"round": -1},
    ("uvular",): {"dorsal": +1, "pharyngeal": +1},
    ("velar",): {"dorsal": +1},
    ("velarized",): {"dorsal": +1, "atr": -1},
    ("via-high",): {"tone_peak": 1},
    ("via-low",): {"tone_peak": -1},
    ("via-mid-high",): {"tone_peak": 0.5},
    ("via-mid-low",): {"tone_peak": -0.5},
    ("via-mid",): {"tone_peak": 0},
    ("voiced",): {"voice": +1},
    ("voiceless",): {"voice": -1},
    ("vowel",): {"syllabic": +1, "consonantal": -1},
    ("whistled-sibilant",): {"sibilant": +1, "strident": +1, "atr": -1},
    ("with-friction",): {"strident": +1},
}


def descriptors_to_feature_bundle(descriptor_list):
    descriptors = set(descriptor_list)
    bundle = {}

    while descriptors:
        matched = False
        # Try to match longest descriptor sets first
        for key in sorted(descriptor_to_features.keys(), key=lambda k: -len(k)):
            if set(key).issubset(descriptors):
                bundle.update(descriptor_to_features[key])
                descriptors.difference_update(key)
                matched = True
                break
        if not matched:
            desc = descriptors.pop()  # remove one to prevent infinite loop
            # optionally log or handle unmatched descriptors

    return bundle


if __name__ == "__main__":
    example = ["voiceless", "alveolar", "stop", "unrounded"]
    example = ["consonant", "dental", "devoiced", "implosive", "voiced"]
    print(descriptors_to_feature_bundle(example))
