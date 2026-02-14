#!/usr/bin/env python3
"""
Generate additional training samples from David Hamner's YouTube transcripts.

All samples are based on actual words from the transcripts - no hallucinated text.
Extracts, rearranges, and reformats real transcript content into various sample types:
  - Longer coherent passages (500-800 words)
  - Paragraph-level chunks with proper sentence boundary detection
  - Monologue-style samples (teaching/explaining framing)
  - Short quips and opinions
  - Technical explanations

Outputs to data/personal/training_samples_extended.jsonl
Then merges with existing data/personal/training_samples.jsonl
"""

import json
import re
import os
import hashlib
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRANSCRIPTS_PATH = "data/personal/all_transcripts.txt"
EXISTING_SAMPLES_PATH = "data/personal/training_samples.jsonl"
EXTENDED_SAMPLES_PATH = "data/personal/training_samples_extended.jsonl"

# Titles that are music/songs - we'll extract from these differently
MUSIC_TITLES = {
    "IT Guy's Brain #music #geekpop",
    "Your Phone is Terrible | Music Video",
    "Maemo Leste SystemV Hackjob #music",
    "The Best Voice AI, Just Not locally.  #music #ai #voiceassistant #sesame #csm #localai #opensource",
    "Gemma 3 27B Tested #music",
    "NN trained on Zelda music (Overfit example)",
}

# Titles with garbled/non-English transcripts to skip
SKIP_TITLES = {
    "Librem 5 | Software Improvements",  # garbled non-English transcript
}


# ---------------------------------------------------------------------------
# Parse transcripts
# ---------------------------------------------------------------------------

def parse_transcripts(path):
    """Parse all_transcripts.txt into a list of (title, text) tuples."""
    transcripts = []
    current_title = None
    current_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            # Detect title lines: === Title ===
            m = re.match(r"^=== (.+?) ===$", line)
            if m:
                if current_title and current_lines:
                    text = "\n".join(current_lines).strip()
                    if text:
                        transcripts.append((current_title, text))
                current_title = m.group(1)
                current_lines = []
            else:
                if current_title is not None:
                    current_lines.append(line)

        # Don't forget the last one
        if current_title and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                transcripts.append((current_title, text))

    return transcripts


def make_title_slug(title):
    """Create a slug from a title for source field."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    return slug[:60]


# ---------------------------------------------------------------------------
# Sentence boundary detection
# ---------------------------------------------------------------------------

# These transcripts are mostly unpunctuated speech. We need heuristics.

# Common sentence-starting phrases in David's speech
SENTENCE_STARTERS = [
    "so ", "now ", "but ", "and ", "also ", "if ", "when ", "while ",
    "let's ", "let me ", "here ", "here's ", "there ", "there's ",
    "this ", "that ", "these ", "those ", "the ", "a ",
    "I ", "I'm ", "I've ", "I'll ", "I was ", "I am ", "I think ",
    "we ", "we're ", "we've ", "we can ", "we need ",
    "you ", "you'll ", "you can ", "you're ", "you need ",
    "it ", "it's ", "it'll ", "its ",
    "one ", "first ", "second ", "next ", "finally ",
    "okay ", "ok ", "alright ", "all right ", "cool ",
    "hello ", "hey ", "hi ",
    "for ", "from ", "with ", "by ", "in ", "on ", "at ",
    "once ", "after ", "before ", "during ",
    "however ", "unfortunately ", "fortunately ", "basically ",
    "another ", "most ", "some ", "any ", "each ", "every ",
    "what ", "how ", "why ", "where ", "who ",
    "notice ", "keep ", "do ", "don't ", "just ", "simply ",
    "to ", "going ", "step ",
]

# Phrases that strongly suggest a sentence boundary before them
BOUNDARY_PHRASES = [
    "so what ", "so the ", "so if ", "so now ", "so let", "so from ",
    "now let", "now we", "now that", "now the", "now if",
    "but the ", "but if ", "but this ", "but that ", "but here",
    "okay so", "ok so", "alright so", "all right so",
    "let's go", "let's take", "let's see", "let's do", "let's try",
    "the next ", "the first ", "the biggest ",
    "another ", "one of the ",
    "I'm gonna", "I'm going to", "I'm not ", "I was ", "I think ",
    "you can ", "you'll ", "you're gonna", "you need ",
    "once you ", "once that", "once this",
    "after that", "after you", "after the",
    "if you ", "if we ", "if the ", "if this ",
    "for example", "for those",
    "here's what", "here's the", "here is ",
    "this is ", "this means", "this will",
    "thanks for watching",
    "anyway ", "moving on",
    "pretty cool", "pretty good", "pretty nice",
    "do note ", "keep in mind",
    "the problem ", "the issue ", "the fix ",
    "unfortunately ", "fortunately ",
    "however ", "although ",
]


def find_approximate_sentences(text, min_words=5, max_words=80):
    """
    Split transcript text into approximate sentences.
    Since transcripts lack punctuation, we use heuristic boundary detection.
    """
    # First, handle any actual punctuation that exists
    # Replace period/question/exclamation followed by space+capital with a split marker
    text = re.sub(r'([.!?])\s+([A-Z])', r'\1|||SPLIT|||\2', text)

    # Split on actual sentence endings
    chunks = text.split("|||SPLIT|||")

    sentences = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_words:
            if len(words) >= min_words:
                sentences.append(chunk.strip())
            continue

        # For longer chunks, try to find natural boundaries
        current = []
        for i, word in enumerate(words):
            current.append(word)

            # Check if current position looks like a sentence boundary
            if len(current) >= min_words:
                remaining_text = " ".join(words[i+1:i+4]).lower() if i+1 < len(words) else ""

                is_boundary = False

                # Check for boundary phrases
                for bp in BOUNDARY_PHRASES:
                    if remaining_text.startswith(bp.lower()):
                        is_boundary = True
                        break

                # Check for sentence starters (weaker signal, needs more words)
                if not is_boundary and len(current) >= 12:
                    for ss in SENTENCE_STARTERS:
                        if remaining_text.startswith(ss.lower()):
                            is_boundary = True
                            break

                # Force break at max_words
                if len(current) >= max_words:
                    is_boundary = True

                if is_boundary:
                    sent = " ".join(current).strip()
                    if len(sent.split()) >= min_words:
                        sentences.append(sent)
                    current = []

        # Leftover
        if current:
            sent = " ".join(current).strip()
            if len(sent.split()) >= min_words:
                sentences.append(sent)

    return sentences


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------

def generate_long_passages(transcripts, min_words=500, max_words=800):
    """
    Extract longer coherent passages (500-800 words) that capture natural flow.
    These come from single videos where David speaks at length.
    """
    samples = []
    for title, text in transcripts:
        if title in SKIP_TITLES or title in MUSIC_TITLES:
            continue

        words = text.split()
        if len(words) < min_words:
            continue

        # Slide a window through the text, trying to break at sentence-ish boundaries
        sentences = find_approximate_sentences(text)
        if not sentences:
            continue

        current_passage = []
        current_word_count = 0

        for sent in sentences:
            sent_words = len(sent.split())

            if current_word_count + sent_words > max_words and current_word_count >= min_words:
                # Emit this passage
                passage_text = " ".join(current_passage)
                samples.append({
                    "text": passage_text,
                    "source": f"youtube_long_passage",
                    "title": title,
                })
                current_passage = []
                current_word_count = 0

            current_passage.append(sent)
            current_word_count += sent_words

        # Emit remainder if long enough
        if current_word_count >= min_words:
            passage_text = " ".join(current_passage)
            samples.append({
                "text": passage_text,
                "source": f"youtube_long_passage",
                "title": title,
            })

    return samples


def generate_paragraph_chunks(transcripts, min_words=80, max_words=250):
    """
    Create paragraph-level chunks with better sentence boundary detection.
    These are medium-length coherent chunks.
    """
    samples = []
    for title, text in transcripts:
        if title in SKIP_TITLES or title in MUSIC_TITLES:
            continue

        words = text.split()
        if len(words) < min_words:
            continue

        sentences = find_approximate_sentences(text, min_words=8, max_words=60)
        if not sentences:
            continue

        current_chunk = []
        current_word_count = 0

        for sent in sentences:
            sent_words = len(sent.split())

            if current_word_count + sent_words > max_words and current_word_count >= min_words:
                chunk_text = " ".join(current_chunk)
                samples.append({
                    "text": chunk_text,
                    "source": "youtube_paragraph",
                    "title": title,
                })
                current_chunk = []
                current_word_count = 0

            current_chunk.append(sent)
            current_word_count += sent_words

        if current_word_count >= min_words:
            chunk_text = " ".join(current_chunk)
            samples.append({
                "text": chunk_text,
                "source": "youtube_paragraph",
                "title": title,
            })

    return samples


def generate_monologue_samples(transcripts, min_words=150, max_words=400):
    """
    Create monologue-style samples framed as David explaining/teaching something.
    These add a topic framing around actual transcript content.
    """
    samples = []

    # Teaching/explaining indicators in David's speech
    teaching_indicators = [
        "let me show you", "I'm gonna show you", "I'm going to show you",
        "here's how", "here's what", "let's go over", "let's take a look",
        "let's dive in", "let's get started", "let's check out",
        "the way", "the idea is", "the goal", "the concept",
        "what exactly is", "what is the difference", "so what is",
        "you can", "you'll want to", "you need to", "you should",
        "step 1", "step one", "first off", "first things first",
        "the most important", "the biggest", "keep in mind",
        "it turns out", "it's actually", "it's basically",
        "this is how", "this is what", "this is where",
        "in order to", "the way I went about",
    ]

    for title, text in transcripts:
        if title in SKIP_TITLES or title in MUSIC_TITLES:
            continue

        words = text.split()
        if len(words) < min_words:
            continue

        text_lower = text.lower()

        # Find segments that start with teaching indicators
        for indicator in teaching_indicators:
            idx = text_lower.find(indicator)
            while idx != -1:
                # Extract a chunk starting from this indicator
                remaining = text[idx:]
                remaining_words = remaining.split()

                if len(remaining_words) >= min_words:
                    # Take up to max_words, trying to end at a sentence boundary
                    chunk_words = remaining_words[:max_words]
                    chunk_text = " ".join(chunk_words)

                    # Try to find a good ending point
                    sentences = find_approximate_sentences(chunk_text, min_words=8)
                    if sentences and len(sentences) > 1:
                        # Use all but maybe the last partial sentence
                        good_text = " ".join(sentences[:-1])
                        if len(good_text.split()) >= min_words:
                            chunk_text = good_text

                    # Frame as a monologue with topic context
                    framed = f"Topic: {title}\n\n{chunk_text}"
                    samples.append({
                        "text": framed,
                        "source": "youtube_monologue",
                        "title": title,
                    })

                # Look for next occurrence
                idx = text_lower.find(indicator, idx + len(indicator) + 50)

    return samples


def generate_short_quips(transcripts, min_words=10, max_words=50):
    """
    Extract short punchy statements - opinions, reactions, conclusions.
    """
    samples = []

    # Patterns that indicate opinion/quip statements
    quip_starters = [
        "I think ", "I bet ", "I expect ", "I would ", "I can't wait",
        "I'm sure ", "I'm hoping ", "I'm guessing ", "I'm impressed",
        "I would love ", "I'm amazed ", "I was ", "I have to say",
        "I strongly recommend", "I encourage", "I predict",
        "honestly ", "personally ", "arguably ",
        "this is amazing", "this is really", "this is cool",
        "that's cool", "that's pretty", "that's actually",
        "pretty cool", "pretty good", "pretty impressive",
        "not bad", "not gonna lie",
        "the biggest ", "the most ", "the best ", "the worst ",
        "it's not ", "it's actually ", "it's a ", "it's the ",
        "you gotta ", "you should ", "you can't ",
        "welcome to ", "hello and welcome",
        "thanks for watching",
        "do note ", "keep in mind", "important to note",
        "the downside ", "the upside ", "the catch ",
        "at the end of the day",
        "overall ", "all in all ", "in my opinion",
        "the important takeaway",
        "anyway ", "so yeah ",
    ]

    for title, text in transcripts:
        if title in SKIP_TITLES:
            continue

        sentences = find_approximate_sentences(text, min_words=5, max_words=60)

        for sent in sentences:
            sent_lower = sent.lower().strip()
            word_count = len(sent.split())

            if word_count < min_words or word_count > max_words:
                continue

            # Check if this sentence starts with a quip indicator
            is_quip = False
            for qs in quip_starters:
                if sent_lower.startswith(qs.lower()):
                    is_quip = True
                    break

            if is_quip:
                samples.append({
                    "text": sent.strip(),
                    "source": "youtube_quip",
                    "title": title,
                })

    return samples


def generate_technical_explanations(transcripts, min_words=100, max_words=350):
    """
    Identify and extract segments where David is explaining technical concepts.
    """
    samples = []

    # Technical topic indicators
    tech_indicators = [
        "kernel", "bootloader", "grub", "filesystem", "root filesystem",
        "partition", "mount", "chroot", "driver", "module",
        "gpu", "cpu", "ram", "vram", "processor",
        "repository", "package", "dependencies", "install",
        "docker", "container", "vm", "virtual",
        "network", "ip address", "mac address", "tcp", "dns",
        "x11", "wayland", "compositor", "display",
        "python", "script", "command", "terminal", "bash", "shell",
        "neural net", "ai ", "model", "training",
        "usb", "hdmi", "serial", "gpio",
        "efi", "uefi", "bios",
        "standard out", "standard error", "pipe",
        "floppy", "boot", "reboot",
        "api", "protocol", "server",
        "debian", "opensuse", "suse", "ubuntu", "arch",
        "rpm", "deb ", "flat pack", "flatpak",
        "convolutional", "neural", "diffusion",
    ]

    for title, text in transcripts:
        if title in SKIP_TITLES or title in MUSIC_TITLES:
            continue

        words = text.split()
        if len(words) < min_words:
            continue

        text_lower = text.lower()

        # Count technical terms to see if this is a technical video
        tech_count = sum(1 for t in tech_indicators if t in text_lower)
        if tech_count < 3:
            continue

        # Extract chunks that are rich in technical content
        sentences = find_approximate_sentences(text, min_words=8, max_words=60)
        if not sentences:
            continue

        current_chunk = []
        current_word_count = 0
        current_tech_density = 0

        for sent in sentences:
            sent_lower = sent.lower()
            sent_words = len(sent.split())
            sent_tech = sum(1 for t in tech_indicators if t in sent_lower)

            current_chunk.append(sent)
            current_word_count += sent_words
            current_tech_density += sent_tech

            if current_word_count >= max_words or (
                current_word_count >= min_words and
                current_tech_density >= 2
            ):
                if current_word_count >= min_words and current_tech_density >= 2:
                    chunk_text = " ".join(current_chunk)
                    samples.append({
                        "text": f"Topic: {title}\n\n{chunk_text}",
                        "source": "youtube_technical",
                        "title": title,
                    })
                current_chunk = []
                current_word_count = 0
                current_tech_density = 0

        # Remainder
        if current_word_count >= min_words and current_tech_density >= 2:
            chunk_text = " ".join(current_chunk)
            samples.append({
                "text": f"Topic: {title}\n\n{chunk_text}",
                "source": "youtube_technical",
                "title": title,
            })

    return samples


def generate_intro_outros(transcripts):
    """
    Extract David's characteristic intro and outro patterns.
    These are short but very voice-distinctive.
    """
    samples = []

    for title, text in transcripts:
        if title in SKIP_TITLES:
            continue

        text_lower = text.lower()

        # Extract intros (first ~50 words that contain greeting patterns)
        words = text.split()
        intro_patterns = [
            "hello and welcome", "hello and welcome back",
            "hello internet", "hello everybody",
            "hi everyone",
        ]

        for ip in intro_patterns:
            if text_lower.startswith(ip):
                # Take the intro segment - up to first major topic shift
                intro_words = words[:60]
                intro_text = " ".join(intro_words)
                # Try to end at a sentence boundary
                sents = find_approximate_sentences(intro_text, min_words=5)
                if sents:
                    intro_text = " ".join(sents[:2]) if len(sents) > 1 else sents[0]
                if len(intro_text.split()) >= 10:
                    samples.append({
                        "text": intro_text,
                        "source": "youtube_intro",
                        "title": title,
                    })
                break

        # Extract outros (last ~40 words containing "thanks for watching")
        if "thanks for watching" in text_lower:
            idx = text_lower.rfind("thanks for watching")
            outro = text[idx:]
            outro_words = outro.split()
            if 3 <= len(outro_words) <= 50:
                samples.append({
                    "text": outro.strip(),
                    "source": "youtube_outro",
                    "title": title,
                })

    return samples


def generate_song_lyrics(transcripts):
    """
    Extract song/music content as its own sample type.
    David writes songs about tech topics - these are distinctive.
    """
    samples = []

    for title, text in transcripts:
        if title not in MUSIC_TITLES:
            continue
        if title in SKIP_TITLES:
            continue

        words = text.split()
        if len(words) < 15:
            continue

        samples.append({
            "text": f"Song: {title}\n\n{text}",
            "source": "youtube_song",
            "title": title,
        })

    return samples


def generate_conversational_snippets(transcripts, min_words=30, max_words=150):
    """
    Extract segments with David's conversational style - asides, reactions,
    humor, and informal commentary.
    """
    samples = []

    casual_markers = [
        "okey dokey", "okie dokie", "alright so",
        "pretty cool", "pretty schwifty", "bitchin",
        "oh my goodness", "oh wow", "oh boy",
        "hooray", "huzzah", "tada",
        "that's cool", "that's awesome", "that's pretty",
        "yep", "nope", "anyway",
        "I'm gonna", "let's go ahead",
        "so yeah", "well it's", "I mean",
        "I don't know", "I'm not sure",
        "fun though right", "let's be honest",
        "geez", "oh man", "oh no",
        "good game", "well played",
        "but wait", "here's the thing",
        "I should also mention", "I also want",
        "I have to give them credit",
    ]

    for title, text in transcripts:
        if title in SKIP_TITLES or title in MUSIC_TITLES:
            continue

        text_lower = text.lower()
        words = text.split()

        if len(words) < min_words:
            continue

        for marker in casual_markers:
            idx = text_lower.find(marker)
            while idx != -1:
                # Extract a snippet around this casual marker
                # Find word boundaries
                before_text = text[:idx]
                before_words = before_text.split()

                # Go back ~15 words for context
                start_word_idx = max(0, len(before_words) - 15)
                snippet_start = len(" ".join(before_words[:start_word_idx])) + (1 if start_word_idx > 0 else 0)

                remaining = text[snippet_start:]
                remaining_words = remaining.split()

                if len(remaining_words) >= min_words:
                    snippet_words = remaining_words[:max_words]
                    snippet = " ".join(snippet_words)

                    # Try to end cleanly
                    sents = find_approximate_sentences(snippet, min_words=5)
                    if sents and len(sents) > 1:
                        good_text = " ".join(sents[:-1])
                        if len(good_text.split()) >= min_words:
                            snippet = good_text

                    if min_words <= len(snippet.split()) <= max_words:
                        samples.append({
                            "text": snippet.strip(),
                            "source": "youtube_conversational",
                            "title": title,
                        })

                idx = text_lower.find(marker, idx + len(marker) + 100)

    return samples


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def text_hash(text):
    """Create a hash for deduplication."""
    # Normalize whitespace and case for comparison
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate_samples(samples, existing_hashes=None):
    """Remove duplicate samples based on text content."""
    if existing_hashes is None:
        existing_hashes = set()

    unique = []
    seen = set(existing_hashes)

    for sample in samples:
        h = text_hash(sample["text"])

        # Also check if this is a substantial substring of something we've seen
        # by checking a shorter hash of the first 200 chars
        short_text = " ".join(sample["text"].lower().split()[:40])
        short_h = hashlib.md5(short_text.encode()).hexdigest()

        if h not in seen and short_h not in seen:
            seen.add(h)
            seen.add(short_h)
            unique.append(sample)

    return unique


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("GENERATING EXTENDED VOICE SAMPLES")
    print("=" * 70)

    # Parse transcripts
    transcripts = parse_transcripts(TRANSCRIPTS_PATH)
    print(f"Parsed {len(transcripts)} transcripts")

    # Filter out music and skip titles for stats
    content_transcripts = [
        (t, txt) for t, txt in transcripts
        if t not in SKIP_TITLES and t not in MUSIC_TITLES
    ]
    total_words = sum(len(txt.split()) for _, txt in content_transcripts)
    print(f"Content transcripts: {len(content_transcripts)} ({total_words:,} words)")

    # Load existing samples for dedup
    existing_hashes = set()
    existing_samples = []
    if os.path.exists(EXISTING_SAMPLES_PATH):
        with open(EXISTING_SAMPLES_PATH) as f:
            for line in f:
                sample = json.loads(line)
                existing_samples.append(sample)
                existing_hashes.add(text_hash(sample["text"]))
        print(f"Loaded {len(existing_samples)} existing samples")

    # Generate all sample types
    print("\nGenerating samples...")

    long_passages = generate_long_passages(transcripts)
    print(f"  Long passages (500-800 words): {len(long_passages)}")

    paragraph_chunks = generate_paragraph_chunks(transcripts)
    print(f"  Paragraph chunks (80-250 words): {len(paragraph_chunks)}")

    monologues = generate_monologue_samples(transcripts)
    print(f"  Monologue samples (150-400 words): {len(monologues)}")

    quips = generate_short_quips(transcripts)
    print(f"  Short quips (10-50 words): {len(quips)}")

    technical = generate_technical_explanations(transcripts)
    print(f"  Technical explanations (100-350 words): {len(technical)}")

    intros_outros = generate_intro_outros(transcripts)
    print(f"  Intros/Outros: {len(intros_outros)}")

    songs = generate_song_lyrics(transcripts)
    print(f"  Song lyrics: {len(songs)}")

    conversational = generate_conversational_snippets(transcripts)
    print(f"  Conversational snippets: {len(conversational)}")

    # Combine all new samples
    all_new = (
        long_passages +
        paragraph_chunks +
        monologues +
        quips +
        technical +
        intros_outros +
        songs +
        conversational
    )
    print(f"\nTotal raw new samples: {len(all_new)}")

    # Deduplicate against existing and within new
    unique_new = deduplicate_samples(all_new, existing_hashes)
    print(f"After deduplication: {len(unique_new)}")

    # Write extended samples
    with open(EXTENDED_SAMPLES_PATH, "w") as f:
        for sample in unique_new:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(unique_new)} new samples to {EXTENDED_SAMPLES_PATH}")

    # Write combined samples (original + new)
    combined = existing_samples + unique_new
    with open(EXISTING_SAMPLES_PATH, "w") as f:
        for sample in combined:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Wrote {len(combined)} total samples to {EXISTING_SAMPLES_PATH}")

    # Stats summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original samples:  {len(existing_samples)}")
    print(f"New samples added: {len(unique_new)}")
    print(f"Total samples:     {len(combined)}")

    # Source breakdown
    sources = {}
    for s in combined:
        src = s.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print("\nSamples by source:")
    for src in sorted(sources):
        print(f"  {src}: {sources[src]}")

    # Word count stats
    word_counts = [len(s["text"].split()) for s in unique_new]
    if word_counts:
        print(f"\nNew sample word counts:")
        print(f"  Min: {min(word_counts)}")
        print(f"  Max: {max(word_counts)}")
        print(f"  Avg: {sum(word_counts) / len(word_counts):.0f}")
        print(f"  Total words: {sum(word_counts):,}")


if __name__ == "__main__":
    main()
