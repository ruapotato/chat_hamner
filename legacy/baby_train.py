"""
Hamner Baby-Step Training
==========================
Teaches the model one fact at a time, like teaching a baby.

Approach:
  1. Start with "1 + 1 = 2"
  2. Train until the model can output it correctly N times in a row
  3. Add the next fact (e.g., "2 + 2 = 4")
  4. Train on BOTH facts until mastery of the new one (while checking old ones don't regress)
  5. Continue fact by fact, always mixing in all previously mastered facts

Facts progress from simple arithmetic to word problems to multi-step reasoning.

Usage:
    python baby_train.py                          # start fresh from curriculum checkpoint
    python baby_train.py --resume                 # resume baby training
    python baby_train.py --checkpoint path/to/ckpt
"""

import os
import sys
import json
import time
import math
import random
import signal
import datetime
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from model import HamnerModel, HamnerConfig
from variants import emotional_param_groups


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = "checkpoints/baby"
LOG_FILE = "logs/baby.log"
METRICS_FILE = "logs/baby_metrics.csv"

BASE_LR = 5e-5          # gentle LR for fine-grained drilling
BATCH_SIZE = 32          # all 32 = same concept, different words
SEQ_LEN = 128            # facts are short, no need for 1024
EMOTIONAL_LAYERS = 6
EMOTIONAL_LR_SCALE = 0.2

# Mastery criteria
MASTERY_CHECKS = 10      # test 10 random problems per check (less noisy than 5)
CHECK_EVERY = 100        # test mastery every N training steps
REGRESSION_CHECKS = 3    # spot-check this many old facts each mastery test
MAX_STEPS_PER_LEVEL = 20_000  # hard ceiling per level
PLATEAU_PATIENCE = 30    # 30 checks × 100 steps = 3000 steps of no improvement before moving on

# Repetition mixing
OLD_FACT_RATIO = 0.4     # 40% of each batch is old facts (spaced repetition)


# ---------------------------------------------------------------------------
# Procedural Fact Generator
# ---------------------------------------------------------------------------
# Instead of a fixed list, we generate thousands of variations per level
# so the model learns the CONCEPT, not memorized strings.

OBJECTS = [
    ("apple", "apples"), ("cat", "cats"), ("dog", "dogs"), ("bird", "birds"),
    ("fish", "fish"), ("ball", "balls"), ("cookie", "cookies"), ("rock", "rocks"),
    ("egg", "eggs"), ("cup", "cups"), ("shoe", "shoes"), ("book", "books"),
    ("star", "stars"), ("car", "cars"), ("hat", "hats"), ("toy", "toys"),
    ("coin", "coins"), ("tree", "trees"), ("flower", "flowers"), ("pen", "pens"),
    ("bug", "bugs"), ("frog", "frogs"), ("duck", "ducks"), ("bear", "bears"),
    ("cake", "cakes"), ("box", "boxes"), ("key", "keys"), ("leaf", "leaves"),
    ("sock", "socks"), ("ring", "rings"),
]

# Templates for variety — each returns (full_text, prompt, expected)
# The generator picks templates randomly so the model sees many phrasings.


class BabyFactGenerator:
    """Procedurally generates arithmetic facts at progressive difficulty levels.

    Levels:
      1: a + b = c  (a,b in 1..5)
      2: a + b = c  (a,b in 1..10)
      3: a - b = c  (result >= 0, a,b in 1..10)
      4: Word addition  ("3 apples plus 2 apples is 5 apples")
      5: Word subtraction  ("5 cats take away 2 cats is 3 cats")
      6: Two-step word  ("3 apples plus 2 apples is 5 apples. Take 1 away and you have 4 apples.")
      7: a + b = c  (a,b in 1..50)
      8: a - b = c  (a,b in 1..50)
      9: "If you have X and get/lose Y" phrasing
     10: Multi-step word problems with varied phrasing
    """

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.level = 1

    def set_level(self, level: int):
        self.level = level

    def _pick_obj(self):
        singular, plural = self.rng.choice(OBJECTS)
        return singular, plural

    def _pluralize(self, n, singular, plural):
        return singular if n == 1 else plural

    def generate(self) -> tuple:
        """Returns (full_text, prompt, expected_completion)."""
        level = self.level
        # Pre-arithmetic: teach number output
        if level == 101:
            return self._counting_sequence(1, 20)
        elif level == 102:
            return self._number_after(1, 30)
        elif level == 103:
            return self._number_copy(1, 50)
        # Arithmetic
        elif level == 1:
            return self._addition_symbolic(1, 5)
        elif level == 2:
            return self._addition_symbolic(1, 10)
        elif level == 3:
            return self._subtraction_symbolic(1, 10)
        elif level == 4:
            return self._addition_word(1, 10)
        elif level == 5:
            return self._subtraction_word(1, 10)
        elif level == 6:
            return self._two_step_word(1, 10)
        elif level == 7:
            return self._addition_symbolic(1, 50)
        elif level == 8:
            return self._subtraction_symbolic(1, 50)
        elif level == 9:
            return self._if_you_have(1, 20)
        elif level == 10:
            return self._multi_step_varied(1, 20)
        return self._addition_symbolic(1, 5)

    # --- Pre-arithmetic: counting sequences ---
    def _counting_sequence(self, lo, hi):
        """Teach: 1, 2, 3, ... 10, 11, 12 (multi-digit output)"""
        start = self.rng.randint(lo, hi - 3)
        length = self.rng.randint(3, min(8, hi - start))
        nums = list(range(start, start + length + 1))
        seq = ", ".join(str(n) for n in nums)
        prompt_nums = nums[:-1]
        answer = str(nums[-1])
        prompt_seq = ", ".join(str(n) for n in prompt_nums) + ","
        template = self.rng.choice([
            lambda: (f"Count: {seq}", f"Count: {prompt_seq}", f" {answer}"),
            lambda: (f"{seq}", f"{prompt_seq}", f" {answer}"),
            lambda: (f"Next: {seq}", f"Next: {prompt_seq}", f" {answer}"),
            lambda: (f"Counting: {seq}.", f"Counting: {prompt_seq}", f" {answer}."),
        ])
        return template()

    # --- Pre-arithmetic: what number comes after N ---
    def _number_after(self, lo, hi):
        """Teach: the number after 9 is 10, after 14 is 15, etc."""
        n = self.rng.randint(lo, hi - 1)
        nxt = n + 1
        template = self.rng.choice([
            lambda: (f"The number after {n} is {nxt}.",
                     f"The number after {n} is", f" {nxt}."),
            lambda: (f"After {n} comes {nxt}.",
                     f"After {n} comes", f" {nxt}."),
            lambda: (f"{n} + 1 = {nxt}",
                     f"{n} + 1 =", f" {nxt}"),
            lambda: (f"What comes after {n}? {nxt}.",
                     f"What comes after {n}?", f" {nxt}."),
            lambda: (f"Count: {n}, {nxt}.",
                     f"Count: {n},", f" {nxt}."),
        ])
        return template()

    # --- Pre-arithmetic: number echo/copy ---
    def _number_copy(self, lo, hi):
        """Teach: outputting specific multi-digit numbers."""
        n = self.rng.randint(lo, hi)
        template = self.rng.choice([
            lambda: (f"The number is {n}.",
                     f"The number is", f" {n}."),
            lambda: (f"Say {n}. {n}.",
                     f"Say {n}.", f" {n}."),
            lambda: (f"Number: {n}",
                     f"Number:", f" {n}"),
            lambda: (f"Write the number {n}: {n}",
                     f"Write the number {n}:", f" {n}"),
        ])
        return template()

    def generate_for_level(self, level: int) -> tuple:
        """Generate a fact at a specific level (for mixing old levels)."""
        old_level = self.level
        self.level = level
        result = self.generate()
        self.level = old_level
        return result

    def generate_concept_batch(self, count: int) -> list:
        """Generate `count` different phrasings of ONE random problem.

        Picks one set of numbers, then expresses the same fact many ways.
        Returns list of (full_text, prompt, expected) tuples.
        """
        level = self.level
        # Pre-arithmetic levels: just generate count independent samples
        # (these already have variety built in)
        if level in (101, 102, 103):
            return [self.generate() for _ in range(count)]
        elif level <= 2:
            return self._addition_all_phrasings(
                1, 5 if level == 1 else 10, count)
        elif level in (3, 8):
            return self._subtraction_all_phrasings(
                1, 10 if level == 3 else 50, count)
        elif level == 4:
            return self._addition_word_all_phrasings(1, 10, count)
        elif level == 5:
            return self._subtraction_word_all_phrasings(1, 10, count)
        elif level == 6:
            return self._two_step_all_phrasings(1, 10, count)
        elif level == 7:
            return self._addition_all_phrasings(1, 50, count)
        elif level == 9:
            return self._if_you_have_all_phrasings(1, 20, count)
        elif level == 10:
            return self._multi_step_all_phrasings(1, 20, count)
        return self._addition_all_phrasings(1, 5, count)

    def generate_concept_batch_for_level(self, level: int, count: int) -> list:
        old_level = self.level
        self.level = level
        result = self.generate_concept_batch(count)
        self.level = old_level
        return result

    # --- ALL phrasings for addition (one set of numbers) ---
    def _addition_all_phrasings(self, lo, hi, count):
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, hi)
        c = a + b
        objs = [self._pick_obj() for _ in range(6)]
        templates = [
            (f"{a} + {b} = {c}", f"{a} + {b} =", f" {c}"),
            (f"{a} plus {b} equals {c}", f"{a} plus {b} equals", f" {c}"),
            (f"{b} + {a} = {c}", f"{b} + {a} =", f" {c}"),
            (f"{b} plus {a} equals {c}", f"{b} plus {a} equals", f" {c}"),
            (f"What is {a} + {b}? The answer is {c}.", f"What is {a} + {b}? The answer is", f" {c}."),
            (f"What is {b} + {a}? The answer is {c}.", f"What is {b} + {a}? The answer is", f" {c}."),
            (f"Add {a} and {b} to get {c}.", f"Add {a} and {b} to get", f" {c}."),
            (f"Add {b} and {a} to get {c}.", f"Add {b} and {a} to get", f" {c}."),
            (f"{a} and {b} is {c}", f"{a} and {b} is", f" {c}"),
            (f"The sum of {a} and {b} is {c}.", f"The sum of {a} and {b} is", f" {c}."),
            (f"If you add {a} and {b} you get {c}.", f"If you add {a} and {b} you get", f" {c}."),
            (f"{a} + {b} = {c}. {b} + {a} = {c}.", f"{a} + {b} = {c}. {b} + {a} =", f" {c}."),
        ]
        # Add word-problem phrasings with different objects
        for s, p in objs:
            oa = self._pluralize(a, s, p)
            ob = self._pluralize(b, s, p)
            oc = self._pluralize(c, s, p)
            templates.extend([
                (f"{a} {oa} plus {b} {ob} is {c} {oc}.",
                 f"{a} {oa} plus {b} {ob} is", f" {c} {oc}."),
                (f"{a} {oa} and {b} {ob} makes {c} {oc}.",
                 f"{a} {oa} and {b} {ob} makes", f" {c} {oc}."),
                (f"You have {a} {oa}. You get {b} more. Now you have {c} {oc}.",
                 f"You have {a} {oa}. You get {b} more. Now you have", f" {c} {oc}."),
                (f"There are {a} {oa} and {b} {ob}. That is {c} {oc} total.",
                 f"There are {a} {oa} and {b} {ob}. That is", f" {c} {oc} total."),
            ])
        # Sample with replacement to fill count
        result = []
        for _ in range(count):
            result.append(self.rng.choice(templates))
        return result

    # --- ALL phrasings for subtraction ---
    def _subtraction_all_phrasings(self, lo, hi, count):
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, a)
        c = a - b
        objs = [self._pick_obj() for _ in range(6)]
        templates = [
            (f"{a} - {b} = {c}", f"{a} - {b} =", f" {c}"),
            (f"{a} minus {b} equals {c}", f"{a} minus {b} equals", f" {c}"),
            (f"What is {a} - {b}? The answer is {c}.", f"What is {a} - {b}? The answer is", f" {c}."),
            (f"Subtract {b} from {a} to get {c}.", f"Subtract {b} from {a} to get", f" {c}."),
            (f"Take {b} from {a} and you get {c}.", f"Take {b} from {a} and you get", f" {c}."),
            (f"{a} take away {b} is {c}.", f"{a} take away {b} is", f" {c}."),
            (f"If you subtract {b} from {a} you get {c}.", f"If you subtract {b} from {a} you get", f" {c}."),
            (f"The difference of {a} and {b} is {c}.", f"The difference of {a} and {b} is", f" {c}."),
        ]
        for s, p in objs:
            oa = self._pluralize(a, s, p)
            oc = self._pluralize(c, s, p)
            templates.extend([
                (f"{a} {oa} take away {b} is {c} {oc}.",
                 f"{a} {oa} take away {b} is", f" {c} {oc}."),
                (f"You have {a} {oa}. You lose {b}. Now you have {c} {oc}.",
                 f"You have {a} {oa}. You lose {b}. Now you have", f" {c} {oc}."),
                (f"Start with {a} {oa}. Remove {b}. That leaves {c} {oc}.",
                 f"Start with {a} {oa}. Remove {b}. That leaves", f" {c} {oc}."),
                (f"There are {a} {oa}. {b} go away. Now there are {c} {oc}.",
                 f"There are {a} {oa}. {b} go away. Now there are", f" {c} {oc}."),
            ])
        result = []
        for _ in range(count):
            result.append(self.rng.choice(templates))
        return result

    # --- ALL phrasings for word addition ---
    def _addition_word_all_phrasings(self, lo, hi, count):
        return self._addition_all_phrasings(lo, hi, count)

    # --- ALL phrasings for word subtraction ---
    def _subtraction_word_all_phrasings(self, lo, hi, count):
        return self._subtraction_all_phrasings(lo, hi, count)

    # --- ALL phrasings for two-step ---
    def _two_step_all_phrasings(self, lo, hi, count):
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, hi)
        mid = a + b
        take = self.rng.randint(1, mid)
        final = mid - take
        objs = [self._pick_obj() for _ in range(6)]
        templates = []
        for s, p in objs:
            oa = self._pluralize(a, s, p)
            ob = self._pluralize(b, s, p)
            om = self._pluralize(mid, s, p)
            of = self._pluralize(final, s, p)
            templates.extend([
                (f"{a} {oa} plus {b} {ob} is {mid} {om}. Take {take} away and you have {final} {of}.",
                 f"{a} {oa} plus {b} {ob} is {mid} {om}. Take {take} away and you have", f" {final} {of}."),
                (f"You start with {a} {oa} and add {b} more for {mid} {om}. Then you remove {take}, leaving {final} {of}.",
                 f"You start with {a} {oa} and add {b} more for {mid} {om}. Then you remove {take}, leaving", f" {final} {of}."),
                (f"Start with {a}. Add {b} to get {mid}. Subtract {take} to get {final}.",
                 f"Start with {a}. Add {b} to get {mid}. Subtract {take} to get", f" {final}."),
                (f"{a} + {b} = {mid}. {mid} - {take} = {final}.",
                 f"{a} + {b} = {mid}. {mid} - {take} =", f" {final}."),
            ])
        result = []
        for _ in range(count):
            result.append(self.rng.choice(templates))
        return result

    # --- ALL phrasings for if-you-have ---
    def _if_you_have_all_phrasings(self, lo, hi, count):
        if self.rng.random() < 0.5:
            return self._addition_all_phrasings(lo, hi, count)
        else:
            return self._subtraction_all_phrasings(lo, hi, count)

    # --- ALL phrasings for multi-step ---
    def _multi_step_all_phrasings(self, lo, hi, count):
        return self._two_step_all_phrasings(lo, hi, count)

    # --- Single-phrasing generators (for mastery testing) ---

    # --- Level 1-2: Symbolic addition ---
    def _addition_symbolic(self, lo, hi):
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, hi)
        c = a + b
        template = self.rng.choice([
            lambda: (f"{a} + {b} = {c}", f"{a} + {b} =", f" {c}"),
            lambda: (f"{a} plus {b} equals {c}", f"{a} plus {b} equals", f" {c}"),
            lambda: (f"What is {a} + {b}? The answer is {c}.",
                     f"What is {a} + {b}? The answer is", f" {c}."),
            lambda: (f"Add {a} and {b} to get {c}.", f"Add {a} and {b} to get", f" {c}."),
            lambda: (f"{a} + {b} = {c}", f"{a} + {b} =", f" {c}"),
        ])
        return template()

    # --- Level 3,8: Symbolic subtraction ---
    def _subtraction_symbolic(self, lo, hi):
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, a)  # ensure result >= 0
        c = a - b
        template = self.rng.choice([
            lambda: (f"{a} - {b} = {c}", f"{a} - {b} =", f" {c}"),
            lambda: (f"{a} minus {b} equals {c}", f"{a} minus {b} equals", f" {c}"),
            lambda: (f"What is {a} - {b}? The answer is {c}.",
                     f"What is {a} - {b}? The answer is", f" {c}."),
            lambda: (f"Subtract {b} from {a} to get {c}.",
                     f"Subtract {b} from {a} to get", f" {c}."),
        ])
        return template()

    # --- Level 4: Word addition ---
    def _addition_word(self, lo, hi):
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, hi)
        c = a + b
        sing, plur = self._pick_obj()
        oa = self._pluralize(a, sing, plur)
        ob = self._pluralize(b, sing, plur)
        oc = self._pluralize(c, sing, plur)
        template = self.rng.choice([
            lambda: (f"{a} {oa} plus {b} {ob} is {c} {oc}.",
                     f"{a} {oa} plus {b} {ob} is",
                     f" {c} {oc}."),
            lambda: (f"{a} {oa} and {b} {ob} makes {c} {oc}.",
                     f"{a} {oa} and {b} {ob} makes",
                     f" {c} {oc}."),
            lambda: (f"You have {a} {oa}. You get {b} more. Now you have {c} {oc}.",
                     f"You have {a} {oa}. You get {b} more. Now you have",
                     f" {c} {oc}."),
            lambda: (f"There are {a} {oa} and {b} {ob}. That is {c} {oc} total.",
                     f"There are {a} {oa} and {b} {ob}. That is",
                     f" {c} {oc} total."),
        ])
        return template()

    # --- Level 5: Word subtraction ---
    def _subtraction_word(self, lo, hi):
        a = self.rng.randint(lo + 1, hi)
        b = self.rng.randint(lo, a)
        c = a - b
        sing, plur = self._pick_obj()
        oa = self._pluralize(a, sing, plur)
        oc = self._pluralize(c, sing, plur)
        template = self.rng.choice([
            lambda: (f"{a} {oa} take away {b} is {c} {oc}.",
                     f"{a} {oa} take away {b} is",
                     f" {c} {oc}."),
            lambda: (f"You have {a} {oa}. You lose {b}. Now you have {c} {oc}.",
                     f"You have {a} {oa}. You lose {b}. Now you have",
                     f" {c} {oc}."),
            lambda: (f"Start with {a} {oa}. Remove {b}. That leaves {c} {oc}.",
                     f"Start with {a} {oa}. Remove {b}. That leaves",
                     f" {c} {oc}."),
            lambda: (f"There are {a} {oa}. {b} go away. Now there are {c} {oc}.",
                     f"There are {a} {oa}. {b} go away. Now there are",
                     f" {c} {oc}."),
        ])
        return template()

    # --- Level 6: Two-step word ---
    def _two_step_word(self, lo, hi):
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, hi)
        mid = a + b
        take = self.rng.randint(1, mid)
        final = mid - take
        sing, plur = self._pick_obj()
        om = self._pluralize(mid, sing, plur)
        of = self._pluralize(final, sing, plur)
        oa = self._pluralize(a, sing, plur)
        ob = self._pluralize(b, sing, plur)
        template = self.rng.choice([
            lambda: (f"{a} {oa} plus {b} {ob} is {mid} {om}. Take {take} away and you have {final} {of}.",
                     f"{a} {oa} plus {b} {ob} is {mid} {om}. Take {take} away and you have",
                     f" {final} {of}."),
            lambda: (f"You start with {a} {oa} and add {b} more for {mid} {om}. Then you remove {take}, leaving {final} {of}.",
                     f"You start with {a} {oa} and add {b} more for {mid} {om}. Then you remove {take}, leaving",
                     f" {final} {of}."),
        ])
        return template()

    # --- Level 9: "If you have" phrasing ---
    def _if_you_have(self, lo, hi):
        sing, plur = self._pick_obj()
        if self.rng.random() < 0.5:
            # addition
            a = self.rng.randint(lo, hi)
            b = self.rng.randint(lo, hi)
            c = a + b
            oa = self._pluralize(a, sing, plur)
            oc = self._pluralize(c, sing, plur)
            verb = self.rng.choice(["find", "get", "receive", "pick up", "are given"])
            template = self.rng.choice([
                lambda: (f"If you have {a} {oa} and {verb} {b} more, you have {c} {oc}.",
                         f"If you have {a} {oa} and {verb} {b} more, you have",
                         f" {c} {oc}."),
                lambda: (f"A child has {a} {oa}. They {verb} {b} more. Now they have {c} {oc}.",
                         f"A child has {a} {oa}. They {verb} {b} more. Now they have",
                         f" {c} {oc}."),
            ])
        else:
            # subtraction
            a = self.rng.randint(lo + 1, hi)
            b = self.rng.randint(lo, a)
            c = a - b
            oa = self._pluralize(a, sing, plur)
            oc = self._pluralize(c, sing, plur)
            verb = self.rng.choice(["eat", "lose", "drop", "give away", "break"])
            template = self.rng.choice([
                lambda: (f"If you have {a} {oa} and {verb} {b}, you have {c} {oc} left.",
                         f"If you have {a} {oa} and {verb} {b}, you have",
                         f" {c} {oc} left."),
                lambda: (f"A child has {a} {oa}. They {verb} {b}. Now they have {c} {oc}.",
                         f"A child has {a} {oa}. They {verb} {b}. Now they have",
                         f" {c} {oc}."),
            ])
        return template()

    # --- Level 10: Multi-step varied ---
    def _multi_step_varied(self, lo, hi):
        sing, plur = self._pick_obj()
        a = self.rng.randint(lo, hi)
        b = self.rng.randint(lo, hi)
        mid = a + b
        c = self.rng.randint(1, mid)
        final = mid - c
        om = self._pluralize(mid, sing, plur)
        of = self._pluralize(final, sing, plur)
        oa = self._pluralize(a, sing, plur)

        get_verb = self.rng.choice(["finds", "gets", "picks up", "is given"])
        lose_verb = self.rng.choice(["gives away", "loses", "drops", "eats"])

        template = self.rng.choice([
            lambda: (f"A boy has {a} {oa}. He {get_verb} {b} more, so he has {mid} {om}. He {lose_verb} {c}, so he has {final} {of}.",
                     f"A boy has {a} {oa}. He {get_verb} {b} more, so he has {mid} {om}. He {lose_verb} {c}, so he has",
                     f" {final} {of}."),
            lambda: (f"Start with {a} {oa}. Add {b} to get {mid} {om}. Subtract {c} to get {final} {of}.",
                     f"Start with {a} {oa}. Add {b} to get {mid} {om}. Subtract {c} to get",
                     f" {final} {of}."),
            lambda: (f"There are {a} {oa} on a table. Someone puts {b} more. Now there are {mid} {om}. Then {c} fall off. Now there are {final} {of}.",
                     f"There are {a} {oa} on a table. Someone puts {b} more. Now there are {mid} {om}. Then {c} fall off. Now there are",
                     f" {final} {of}."),
        ])
        return template()


# Levels define which generator level to use and what to test mastery on.
# Each level has a number range / complexity, and mastery is tested with
# fresh random problems the model hasn't seen during training.

@dataclass
class Level:
    name: str
    gen_level: int       # BabyFactGenerator level
    description: str

LEVELS = [
    # Pre-arithmetic: teach the model to output multi-digit numbers
    Level("counting",           101, "1, 2, 3, ... 10, 11, 12 sequences"),
    Level("number-after",       102, "the number after N is N+1"),
    Level("number-copy",        103, "echo/copy numbers up to 50"),
    # Arithmetic
    Level("tiny addition",       1,  "a + b (1-5)"),
    Level("addition to 10",      2,  "a + b (1-10)"),
    Level("subtraction to 10",   3,  "a - b (1-10)"),
    Level("word addition",       4,  "N objects plus M objects"),
    Level("word subtraction",    5,  "N objects take away M"),
    Level("two-step word",       6,  "add then subtract, word form"),
    Level("bigger addition",     7,  "a + b (1-50)"),
    Level("bigger subtraction",  8,  "a - b (1-50)"),
    Level("if-you-have",         9,  "natural language add/sub"),
    Level("multi-step stories", 10,  "multi-step with varied phrasing"),
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_metrics(step, loss, level_idx, level_name, mastered, accuracy):
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    write_header = not os.path.exists(METRICS_FILE)
    with open(METRICS_FILE, "a") as f:
        if write_header:
            f.write("timestamp,step,loss,level_idx,level_name,mastered,accuracy\n")
        ts = datetime.datetime.now().isoformat()
        f.write(f'{ts},{step},{loss:.6f},{level_idx},"{level_name}",{mastered},{accuracy:.3f}\n')


# ---------------------------------------------------------------------------
# Mastery testing
# ---------------------------------------------------------------------------

@torch.no_grad()
def test_fact(model, tokenizer, prompt, expected, device="cuda"):
    """Feed prompt, generate, check if output starts with expected completion."""
    model.eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate just enough tokens to cover the expected answer
    max_tokens = len(tokenizer.encode(expected, add_special_tokens=False)) + 5

    output = model.generate(
        input_ids, max_new_tokens=max_tokens,
        temperature=0.01, top_k=1, top_p=1.0,  # near-greedy
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id or 0,
    )

    generated_ids = output[0][len(tokens):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    model.train()

    # Extract expected number and its digit count
    import re
    expected_num = re.search(r'\d+', expected.strip())
    expected_num = expected_num.group(0) if expected_num else expected.strip()
    num_digits = len(expected_num)

    # Read exactly that many leading digit tokens as the answer
    # (each digit is its own token in this tokenizer)
    answer_digits = []
    for tid in generated_ids:
        tok_str = tokenizer.decode([tid]).strip()
        if tok_str.isdigit() and len(answer_digits) < num_digits:
            answer_digits.append(tok_str)
        else:
            break
    generated_answer = "".join(answer_digits) if answer_digits else generated_text[:10]

    passed = generated_answer == expected_num
    return passed, f"{generated_answer} [{generated_text[:30]}]"


def run_mastery_test(model, tokenizer, gen, level_idx, device="cuda"):
    """Test mastery on FRESH random problems the model hasn't seen.

    Generates MASTERY_CHECKS new random problems at the current level.
    Also spot-checks old levels for regression.
    Returns (mastered, accuracy, results_list).
    """
    # Use a separate RNG so we don't affect training generation
    test_gen = BabyFactGenerator(seed=int(time.time() * 1000) % 2**31)
    test_gen.set_level(LEVELS[level_idx].gen_level)

    # Test current level with fresh problems
    results = []
    for _ in range(MASTERY_CHECKS):
        full, prompt, expected = test_gen.generate()
        passed, generated = test_fact(model, tokenizer, prompt, expected, device)
        results.append((passed, generated, prompt, expected))

    correct = sum(1 for p, _, _, _ in results if p)
    accuracy = correct / MASTERY_CHECKS
    all_passed = correct == MASTERY_CHECKS

    # Spot-check old levels for regression
    regression = False
    if level_idx > 0 and all_passed:
        old_levels = list(range(level_idx))
        random.shuffle(old_levels)
        for old_lvl in old_levels[:min(REGRESSION_CHECKS, len(old_levels))]:
            test_gen.set_level(LEVELS[old_lvl].gen_level)
            full, prompt, expected = test_gen.generate()
            old_passed, old_gen = test_fact(model, tokenizer, prompt, expected, device)
            if not old_passed:
                log(f"  REGRESSION on level {old_lvl} ({LEVELS[old_lvl].name}): "
                    f"'{prompt}' -> '{old_gen}' (expected '{expected}')")
                regression = True

    return all_passed and not regression, accuracy, results


# ---------------------------------------------------------------------------
# Training batch construction
# ---------------------------------------------------------------------------

def make_training_batch(tokenizer, gen, current_level_idx, batch_size, seq_len):
    """Build a batch of diverse problems, each expressed in varied phrasings.

    Each batch item is a DIFFERENT random problem at the current level.
    Each sequence packs multiple varied phrasings of its problem, so the model
    sees "3+4=7. 3 plus 4 equals 7. Add 3 and 4 to get 7." in one row.
    ~40% of items review old levels for spaced repetition.
    """
    input_ids_list = []
    labels_list = []

    for _ in range(batch_size):
        # Pick level: current or review old?
        if current_level_idx > 0 and random.random() < OLD_FACT_RATIO:
            lvl = random.randint(0, current_level_idx - 1)
        else:
            lvl = current_level_idx

        # Generate many phrasings of ONE problem for this sequence
        phrasings = gen.generate_concept_batch_for_level(
            LEVELS[lvl].gen_level, 16
        )
        # Pack different phrasings separated by newlines (clear boundary between facts)
        texts = [full for full, _, _ in phrasings]
        random.shuffle(texts)
        combined = "\n".join(texts) + "\n"
        tokens = tokenizer.encode(combined, add_special_tokens=False)[:seq_len]

        while len(tokens) < seq_len:
            tokens.append(tokenizer.pad_token_id or 0)

        # NO pre-shifting — model.forward() handles shift internally
        input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
        labels_list.append(torch.tensor(tokens, dtype=torch.long))

    return torch.stack(input_ids_list), torch.stack(labels_list)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, config, step, loss,
                    current_level_idx, mastered_levels):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    raw_state = model.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}

    ckpt_data = {
        "model_state_dict": clean_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config.__dict__,
        "step": step,
        "avg_loss": loss,
        "current_level_idx": current_level_idx,
        "mastered_levels": mastered_levels,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"step_{step:07d}.pt")
    torch.save(ckpt_data, ckpt_path)

    latest_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(ckpt_data, latest_path)

    log(f"  Checkpoint saved: {ckpt_path}")

    # Keep last 5 + milestones
    all_ckpts = sorted(Path(CHECKPOINT_DIR).glob("step_*.pt"))
    to_keep = set(all_ckpts[-5:])
    for c in all_ckpts:
        if c not in to_keep:
            c.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def baby_train(checkpoint_path=None, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log("=" * 70)
    log("HAMNER BABY-STEP TRAINING")
    log("=" * 70)
    log(f"Teaching {len(LEVELS)} levels, each with infinite procedural variations")
    log(f"Mastery = {MASTERY_CHECKS}/{MASTERY_CHECKS} correct on FRESH random problems")
    log(f"Checked every {CHECK_EVERY} steps, max {MAX_STEPS_PER_LEVEL} steps per level")
    log(f"Move on only after {PLATEAU_PATIENCE} checks with no accuracy improvement")

    for i, lvl in enumerate(LEVELS):
        log(f"  Level {i}: {lvl.name} - {lvl.description}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Find checkpoint
    if checkpoint_path is None:
        baby_latest = Path(CHECKPOINT_DIR) / "latest.pt"
        if resume and baby_latest.exists():
            checkpoint_path = str(baby_latest)
        else:
            for p in ["checkpoints/curriculum/latest.pt", "checkpoints/training/latest.pt"]:
                if Path(p).exists():
                    checkpoint_path = p
                    break

    if checkpoint_path is None:
        log("ERROR: No checkpoint found.")
        sys.exit(1)

    log(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Build model
    config_dict = ckpt["config"]
    config = HamnerConfig(**config_dict)
    config.emotional_layers = EMOTIONAL_LAYERS
    config.emotional_lr_scale = EMOTIONAL_LR_SCALE
    config.gradient_checkpointing = False
    config.vocab_size = tokenizer.vocab_size

    model = HamnerModel(config).to(device)
    state_dict = ckpt["model_state_dict"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)

    total_p, _ = model.count_parameters()
    log(f"Model: {total_p:,} params | {config.hidden_size}h x {config.num_layers}L")

    # Resume state
    start_level = 0
    mastered_levels = []
    global_step = 0
    if resume and "current_level_idx" in ckpt:
        start_level = ckpt["current_level_idx"]
        mastered_levels = ckpt.get("mastered_levels", [])
        global_step = ckpt.get("step", 0)
        log(f"Resuming from level {start_level}, step {global_step}")
        log(f"Previously mastered: {len(mastered_levels)} levels")

    # Optimizer
    param_groups = emotional_param_groups(model, BASE_LR)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.01)

    if resume and "current_level_idx" in ckpt and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            log("Restored optimizer state")
        except Exception as e:
            log(f"Fresh optimizer (could not restore: {e})")

    scaler = torch.amp.GradScaler("cuda")
    if resume and "scaler_state_dict" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass

    # Compile
    if hasattr(torch, "compile"):
        log("Compiling model with torch.compile...")
        model = torch.compile(model)

    model.train()

    # Fact generator for training
    gen = BabyFactGenerator(seed=42)

    # Graceful shutdown
    shutdown_requested = False
    def handle_signal(signum, frame):
        nonlocal shutdown_requested
        log("Shutdown signal received, saving checkpoint...")
        shutdown_requested = True
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # --- Pre-test: sample each level ---
    log("\n--- PRE-TEST: Testing 3 random problems per level ---")
    model.eval()
    test_gen = BabyFactGenerator(seed=9999)
    for i, lvl in enumerate(LEVELS):
        test_gen.set_level(lvl.gen_level)
        passes = 0
        for _ in range(3):
            full, prompt, expected = test_gen.generate()
            passed, generated = test_fact(model, tokenizer, prompt, expected, device)
            if passed:
                passes += 1
        log(f"  Level {i} ({lvl.name}): {passes}/3 correct")
    model.train()

    # --- Main training loop: one level at a time ---
    start_time = time.time()
    losses = []

    for level_idx in range(start_level, len(LEVELS)):
        if shutdown_requested:
            break

        lvl = LEVELS[level_idx]
        log(f"\n{'='*70}")
        log(f"LEVEL {level_idx + 1}/{len(LEVELS)}: {lvl.name}")
        log(f"Description: {lvl.description}")
        log(f"Generator level: {lvl.gen_level}")
        log(f"Mastered so far: {len(mastered_levels)}/{len(LEVELS)}")
        log(f"{'='*70}")

        # Show some example problems
        example_gen = BabyFactGenerator(seed=level_idx * 1000)
        example_gen.set_level(lvl.gen_level)
        log("  Example problems:")
        for _ in range(5):
            full, prompt, expected = example_gen.generate()
            log(f"    '{full}'")

        level_step = 0
        mastered = False
        best_accuracy = 0.0
        no_improve_count = 0  # consecutive mastery checks with no accuracy gain

        while level_step < MAX_STEPS_PER_LEVEL and not shutdown_requested:
            # Build batch with fresh procedural problems
            input_ids, labels_ = make_training_batch(
                tokenizer, gen, level_idx, BATCH_SIZE, SEQ_LEN
            )
            input_ids = input_ids.to(device)
            labels_ = labels_.to(device)

            # Forward + backward
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids, labels=labels_)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            losses.append(loss_val)
            global_step += 1
            level_step += 1

            # Log progress
            if level_step % 10 == 0:
                avg_loss = sum(losses[-50:]) / len(losses[-50:])
                elapsed = time.time() - start_time
                log(f"  step {global_step} (lvl_step {level_step}) | loss {avg_loss:.4f} | "
                    f"level {level_idx+1}/{len(LEVELS)} ({lvl.name}) | "
                    f"best acc {best_accuracy:.0%} | {elapsed/60:.1f}m")
                log_metrics(global_step, avg_loss, level_idx, lvl.name,
                            len(mastered_levels), best_accuracy)

            # Mastery check on FRESH random problems
            if level_step % CHECK_EVERY == 0:
                log(f"\n  --- MASTERY TEST: Level {level_idx} ({lvl.name}) ---")
                is_mastered, accuracy, results = run_mastery_test(
                    model, tokenizer, gen, level_idx, device
                )

                for passed, generated, prompt, expected in results:
                    status = "PASS" if passed else "FAIL"
                    log(f"    [{status}] '{prompt}' -> '{generated}' (expected '{expected}')")

                log(f"  Accuracy: {accuracy:.0%} ({sum(1 for p,_,_,_ in results if p)}/{MASTERY_CHECKS})")

                if is_mastered:
                    log(f"  ** MASTERED level {level_idx} ({lvl.name}) in {level_step} steps! **")
                    mastered = True
                    mastered_levels.append(level_idx)

                    avg_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
                    save_checkpoint(model, optimizer, scaler, config,
                                    global_step, avg_loss, level_idx + 1, mastered_levels)
                    break
                else:
                    # Track if accuracy is still improving
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        no_improve_count = 0
                        log(f"  New best accuracy: {best_accuracy:.0%} (step {level_step})")
                    else:
                        no_improve_count += 1
                        log(f"  No improvement ({no_improve_count}/{PLATEAU_PATIENCE}) | "
                            f"best: {best_accuracy:.0%} | step {level_step}/{MAX_STEPS_PER_LEVEL}")

                    # Only give up if accuracy has truly plateaued
                    if no_improve_count >= PLATEAU_PATIENCE:
                        log(f"  Accuracy plateaued at {best_accuracy:.0%} after {PLATEAU_PATIENCE} "
                            f"checks with no improvement, moving on")
                        break

        if not mastered and not shutdown_requested:
            log(f"  Could not master level {level_idx} ({lvl.name}) | "
                f"best accuracy: {best_accuracy:.0%} | steps: {level_step}")
            avg_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
            save_checkpoint(model, optimizer, scaler, config,
                            global_step, avg_loss, level_idx + 1, mastered_levels)

    # Final summary
    elapsed = time.time() - start_time
    log("\n" + "=" * 70)
    log(f"BABY TRAINING {'STOPPED' if shutdown_requested else 'COMPLETE'}")
    log(f"Total steps: {global_step}")
    log(f"Levels mastered: {len(mastered_levels)}/{len(LEVELS)}")
    log(f"Time: {elapsed/60:.1f} minutes")

    if mastered_levels:
        log("\nMastered levels:")
        for idx in mastered_levels:
            log(f"  [{idx}] {LEVELS[idx].name}")

    # Final comprehensive test
    log("\n--- FINAL TEST (10 fresh problems per level) ---")
    model.eval()
    final_gen = BabyFactGenerator(seed=77777)
    for i, lvl in enumerate(LEVELS):
        final_gen.set_level(lvl.gen_level)
        correct = 0
        for _ in range(10):
            full, prompt, expected = final_gen.generate()
            passed, generated = test_fact(model, tokenizer, prompt, expected, device)
            if passed:
                correct += 1
        status = "MASTERED" if correct >= 9 else "PARTIAL" if correct >= 5 else "FAILED"
        log(f"  [{status}] Level {i} ({lvl.name}): {correct}/10")
    log("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hamner Baby-Step Training")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    baby_train(checkpoint_path=args.checkpoint, resume=args.resume)
