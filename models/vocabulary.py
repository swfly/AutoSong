def generate_pinyin_vocab() -> list[str]:
    """
    Generate a complete list of valid Mandarin Pinyin syllables with tones (1–5).

    Returns
    -------
    vocab : list[str]
        A list of pinyin syllables like ['wo3', 'ai4', 'ni3', ..., 'zhi1', 'zhuang4']
    """
    initials = [
        "", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
        "zh", "ch", "sh", "r", "z", "c", "s", "y", "w"
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
            elif ini == "w":
                syllable = fin if fin.startswith("u") else "w" + fin
            else:
                syllable = ini + fin
            base_syllables.add(syllable)

    vocab = [f"{syl}{tone}" for syl in base_syllables for tone in range(1, 6)]
    return sorted(vocab)
