def generate_pinyin_vocab() -> list[str]:
    """
    Generate a complete list of valid Mandarin Pinyin syllables, with:
    - Tones 1–5
    - Toneless form (轻音)
    - Basic formatting chars like space, newline
    """

    initials = [
        "", "b", "p", "m", "f", "d", "t", "n", "l",
        "g", "k", "h", "j", "q", "x", "zh", "ch", "sh",
        "r", "z", "c", "s", "y", "w"
    ]
    finals = [
        "a", "ai", "an", "ang", "ao",
        "e", "ei", "en", "eng", "er",
        "i", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu",
        "o", "ong", "ou",
        "u", "ua", "uai", "uan", "uang", "ue", "ui", "un", "uo",
        "v", "ve"
    ]

    base_syllables = set()
    for ini in initials:
        for fin in finals:
            if ini == "y":
                syllable = fin if fin.startswith("i") else "y" + fin
                base_syllables.add(syllable)
            elif ini == "w":
                syllable = fin if fin.startswith("u") else "w" + fin
                base_syllables.add(syllable)
            
            syllable = ini + fin
            base_syllables.add(syllable)

    # Add all syllables with tones 1–5
    vocab = [f"{syl}{tone}" for syl in base_syllables for tone in range(1, 6)]

    # Add toneless (neutral tone, e.g., "de", "ma")
    vocab += list(base_syllables)

    # Add common formatting and special tokens
    vocab += ["<PAD>", "<UNK>", " ", "\n", "\t", ".", ",", "!", "?", "…", "：", "；", "-", "(", ")"]

    return sorted(set(vocab))
