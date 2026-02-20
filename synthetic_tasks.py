"""
Synthetic Task Generators for Curriculum Training
==================================================
Generates structured training examples on-the-fly for embedding reorganization.
All tasks return plain text strings ready for tokenization.

9 task types, each with difficulty scaling (1=easy, 5=hard):
  1. Arithmetic      - "Calculate: 34 + 17 = 51"
  2. Counting        - "Count the items: apple, banana, cherry. Answer: 3"
  3. Bracket matching - "Brackets: ( [ { } ] ) Valid: yes"
  4. ListOps         - "[MAX 3 [MIN 7 2] 5] = 5"
  5. Sorting         - "Sort: 7 2 9 1 -> 1 2 7 9"
  6. Copy/Repeat     - "Repeat 3 times: hello -> hello hello hello"
  7. Context Recall  - "Alice likes pizza. Bob likes sushi. What does Alice like? pizza"
  8. Logic           - "If it rains then the ground is wet. It rains. Therefore the ground is wet."
  9. Comparison      - "Which is bigger: 15 or 23? 23"
"""

import random
from typing import List


COMMON_WORDS = [
    "apple", "banana", "cherry", "dog", "cat", "house", "tree", "river",
    "mountain", "book", "phone", "computer", "water", "fire", "sun", "moon",
    "star", "cloud", "rain", "snow", "bird", "fish", "flower", "grass",
    "stone", "bridge", "road", "car", "train", "ship", "plane", "hat",
    "shoe", "chair", "table", "door", "window", "light", "clock", "bell",
    "key", "cup", "bowl", "fork", "knife", "pen", "paper", "bag",
    "box", "ball", "ring", "coin", "drum", "flag", "lamp", "map",
    "cake", "milk", "bread", "cheese", "egg", "salt", "rice", "soup",
    "garden", "forest", "ocean", "island", "village", "castle", "tower", "farm",
    "robot", "dragon", "wizard", "knight", "queen", "king", "prince", "giant",
]

NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Ruby", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
]

PROPERTIES = [
    "likes pizza", "likes sushi", "has a dog", "has a cat", "plays guitar",
    "plays piano", "lives in Paris", "lives in Tokyo", "drives a truck",
    "rides a bike", "reads novels", "reads comics", "wears glasses",
    "wears a hat", "speaks French", "speaks Spanish", "likes hiking",
    "likes swimming", "collects stamps", "collects coins", "drinks tea",
    "drinks coffee", "studies math", "studies art", "is tall", "is short",
]

LOGIC_CONDITIONS = [
    ("it rains", "the ground is wet"),
    ("it snows", "the roads are slippery"),
    ("the sun shines", "the flowers bloom"),
    ("the alarm rings", "everyone wakes up"),
    ("the dog barks", "the cat hides"),
    ("the wind blows", "the leaves fall"),
    ("the bell rings", "class begins"),
    ("the light turns green", "cars start moving"),
    ("the temperature drops", "water freezes"),
    ("someone knocks", "the door opens"),
    ("the music plays", "people dance"),
    ("the phone rings", "someone answers"),
]


class SyntheticTaskGenerator:
    """Generates structured training examples with adjustable difficulty."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.difficulty = 1

    def set_difficulty(self, level: int):
        self.difficulty = max(1, min(5, level))

    # ------------------------------------------------------------------
    # 1. Arithmetic
    # ------------------------------------------------------------------
    def generate_arithmetic(self) -> str:
        d = self.difficulty
        if d <= 2:
            a = self.rng.randint(1, 10 * d)
            b = self.rng.randint(1, 10 * d)
            op = self.rng.choice(["+", "-"])
            result = a + b if op == "+" else a - b
            return f"Calculate: {a} {op} {b} = {result}"
        elif d == 3:
            a = self.rng.randint(1, 100)
            b = self.rng.randint(1, 100)
            op = self.rng.choice(["+", "-", "*"])
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            else:
                result = a * b
            return f"Calculate: {a} {op} {b} = {result}"
        elif d == 4:
            a = self.rng.randint(10, 500)
            b = self.rng.randint(10, 500)
            op = self.rng.choice(["+", "-", "*"])
            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            else:
                result = a * b
            return f"Calculate: {a} {op} {b} = {result}"
        else:
            a = self.rng.randint(10, 500)
            b = self.rng.randint(10, 500)
            c = self.rng.randint(1, 200)
            op1 = self.rng.choice(["+", "-"])
            op2 = self.rng.choice(["+", "-"])
            result = a
            result = result + b if op1 == "+" else result - b
            result = result + c if op2 == "+" else result - c
            return f"Calculate: {a} {op1} {b} {op2} {c} = {result}"

    # ------------------------------------------------------------------
    # 2. Counting
    # ------------------------------------------------------------------
    def generate_counting(self) -> str:
        d = self.difficulty
        n = self.rng.randint(2, 2 + d * 2)
        words = [self.rng.choice(COMMON_WORDS) for _ in range(n)]
        if d >= 3 and self.rng.random() < 0.5:
            # Count occurrences of a specific word
            target = self.rng.choice(words)
            extra = [self.rng.choice(COMMON_WORDS) for _ in range(self.rng.randint(1, d))]
            all_words = words + extra
            self.rng.shuffle(all_words)
            count = sum(1 for w in all_words if w == target)
            return f"Count '{target}' in: {', '.join(all_words)}. Answer: {count}"
        else:
            return f"Count the items: {', '.join(words)}. Answer: {len(words)}"

    # ------------------------------------------------------------------
    # 3. Bracket Matching
    # ------------------------------------------------------------------
    def _check_brackets(self, tokens: List[str]) -> bool:
        stack = []
        matching = {")": "(", "]": "[", "}": "{"}
        openers = set("([{")
        closers = set(")]}")
        for t in tokens:
            if t in openers:
                stack.append(t)
            elif t in closers:
                if not stack or stack[-1] != matching[t]:
                    return False
                stack.pop()
        return len(stack) == 0

    def generate_brackets(self) -> str:
        d = self.difficulty
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]
        depth = self.rng.randint(1, min(d + 1, 6))

        # Build valid nested brackets
        stack = []
        result = []
        for _ in range(depth):
            pair = self.rng.choice(pairs[:min(d, 3)])
            stack.append(pair[1])
            result.append(pair[0])
        # Optionally add content word in the middle
        if self.rng.random() < 0.3:
            result.append(self.rng.choice(COMMON_WORDS))
        while stack:
            result.append(stack.pop())

        # Optionally corrupt
        valid = True
        if self.rng.random() < 0.4:
            idx = self.rng.randint(0, len(result) - 1)
            all_brackets = list("()[]{}")
            result[idx] = self.rng.choice(all_brackets)
            valid = self._check_brackets([t for t in result if t in "()[]{}"])

        return f"Brackets: {' '.join(result)} Valid: {'yes' if valid else 'no'}"

    # ------------------------------------------------------------------
    # 4. ListOps
    # ------------------------------------------------------------------
    def _eval_listop(self, op: str, values: list) -> int:
        if op == "MAX":
            return max(values)
        elif op == "MIN":
            return min(values)
        elif op == "SUM":
            return sum(values)
        return 0

    def _gen_listop_expr(self, max_depth: int) -> tuple:
        """Returns (expression_string, value)."""
        ops = ["MAX", "MIN", "SUM"]
        op = self.rng.choice(ops)
        n_args = self.rng.randint(2, 4)
        parts = []
        values = []
        for _ in range(n_args):
            if max_depth > 0 and self.rng.random() < 0.3:
                expr, val = self._gen_listop_expr(max_depth - 1)
                parts.append(expr)
                values.append(val)
            else:
                num = self.rng.randint(0, 9)
                parts.append(str(num))
                values.append(num)
        result = self._eval_listop(op, values)
        expr = f"[{op} {' '.join(parts)}]"
        return expr, result

    def generate_listops(self) -> str:
        d = self.difficulty
        if d <= 2:
            # Flat: [MAX 3 7 2]
            max_depth = 0
        elif d <= 4:
            max_depth = 1
        else:
            max_depth = 2
        expr, result = self._gen_listop_expr(max_depth)
        return f"{expr} = {result}"

    # ------------------------------------------------------------------
    # 5. Sorting
    # ------------------------------------------------------------------
    def generate_sorting(self) -> str:
        d = self.difficulty
        n = self.rng.randint(3, 3 + d)
        max_val = 10 * d
        nums = [self.rng.randint(0, max_val) for _ in range(n)]
        sorted_nums = sorted(nums)
        if d >= 4 and self.rng.random() < 0.4:
            # Reverse sort
            sorted_nums = sorted(nums, reverse=True)
            return f"Sort descending: {' '.join(map(str, nums))} -> {' '.join(map(str, sorted_nums))}"
        return f"Sort: {' '.join(map(str, nums))} -> {' '.join(map(str, sorted_nums))}"

    # ------------------------------------------------------------------
    # 6. Copy / Repeat
    # ------------------------------------------------------------------
    def generate_copy(self) -> str:
        d = self.difficulty
        n = self.rng.randint(2, 2 + d)
        if d >= 3 and self.rng.random() < 0.4:
            # Multi-word repeat
            words = [self.rng.choice(COMMON_WORDS) for _ in range(self.rng.randint(2, 3))]
            phrase = " ".join(words)
            repeated = " | ".join([phrase] * n)
            return f"Repeat {n} times: {phrase} -> {repeated}"
        else:
            word = self.rng.choice(COMMON_WORDS)
            repeated = " ".join([word] * n)
            return f"Repeat {n} times: {word} -> {repeated}"

    # ------------------------------------------------------------------
    # 7. Context Recall
    # ------------------------------------------------------------------
    def generate_context_recall(self) -> str:
        d = self.difficulty
        # Number of facts scales with difficulty: 2-3 at d1, up to 8-12 at d5
        n_facts = self.rng.randint(2, 2 + d * 2)
        names = self.rng.sample(NAMES, min(n_facts, len(NAMES)))
        props = self.rng.sample(PROPERTIES, min(n_facts, len(PROPERTIES)))

        facts = []
        for i in range(n_facts):
            facts.append(f"{names[i]} {props[i]}")

        # Pick one to ask about
        ask_idx = self.rng.randint(0, n_facts - 1)
        target_name = names[ask_idx]
        target_prop = props[ask_idx]

        fact_text = ". ".join(facts) + "."

        # Higher difficulty: rephrase the question
        if d >= 3 and self.rng.random() < 0.5:
            templates = [
                f"What do we know about {target_name}?",
                f"Tell me about {target_name}.",
                f"What is true of {target_name}?",
            ]
            question = self.rng.choice(templates)
        else:
            question = f"What does {target_name} do?" if "likes" not in target_prop and "has" not in target_prop else f"What about {target_name}?"

        return f"{fact_text} {question} {target_name} {target_prop}."

    # ------------------------------------------------------------------
    # 8. Logic
    # ------------------------------------------------------------------
    def generate_logic(self) -> str:
        d = self.difficulty

        if d <= 2:
            # Simple modus ponens: If A then B. A. Therefore B.
            cond, result = self.rng.choice(LOGIC_CONDITIONS)
            return f"If {cond} then {result}. {cond.capitalize()}. Therefore {result}."

        elif d == 3:
            # Two rules, pick the matching one
            pairs = self.rng.sample(LOGIC_CONDITIONS, 2)
            trigger_idx = self.rng.randint(0, 1)
            cond, result = pairs[trigger_idx]
            rules = " ".join(f"If {c} then {r}." for c, r in pairs)
            return f"{rules} {cond.capitalize()}. Therefore {result}."

        elif d == 4:
            # Chained inference: A→B, B→C. A. Therefore C.
            pairs = self.rng.sample(LOGIC_CONDITIONS, 2)
            a, b = pairs[0]
            _, c = pairs[1]
            return (f"If {a} then {b}. If {b} then {c}. "
                    f"{a.capitalize()}. Therefore {c}.")

        else:
            # Chain + distractor
            pairs = self.rng.sample(LOGIC_CONDITIONS, 3)
            a, b = pairs[0]
            _, c = pairs[1]
            distractor_cond, distractor_result = pairs[2]
            return (f"If {a} then {b}. If {b} then {c}. "
                    f"If {distractor_cond} then {distractor_result}. "
                    f"{a.capitalize()}. Therefore {c}.")

    # ------------------------------------------------------------------
    # 9. Comparison
    # ------------------------------------------------------------------
    def generate_comparison(self) -> str:
        d = self.difficulty

        if d <= 2:
            # Two numbers, which is bigger
            a = self.rng.randint(1, 100)
            b = self.rng.randint(1, 100)
            while b == a:
                b = self.rng.randint(1, 100)
            bigger = max(a, b)
            return f"Which is bigger: {a} or {b}? {bigger}"

        elif d == 3:
            # Bigger OR smaller
            a = self.rng.randint(1, 500)
            b = self.rng.randint(1, 500)
            while b == a:
                b = self.rng.randint(1, 500)
            if self.rng.random() < 0.5:
                return f"Which is bigger: {a} or {b}? {max(a, b)}"
            else:
                return f"Which is smaller: {a} or {b}? {min(a, b)}"

        elif d == 4:
            # Max/min among 3-4 numbers
            n = self.rng.randint(3, 4)
            nums = [self.rng.randint(1, 1000) for _ in range(n)]
            while len(set(nums)) != len(nums):
                nums = [self.rng.randint(1, 1000) for _ in range(n)]
            nums_str = ", ".join(map(str, nums))
            if self.rng.random() < 0.5:
                return f"Find the largest: {nums_str}. Answer: {max(nums)}"
            else:
                return f"Find the smallest: {nums_str}. Answer: {min(nums)}"

        else:
            # Compare arithmetic expressions
            a1 = self.rng.randint(1, 50)
            a2 = self.rng.randint(1, 50)
            b1 = self.rng.randint(1, 20)
            b2 = self.rng.randint(1, 20)
            val_a = a1 + a2
            val_b = b1 * b2
            expr_a = f"{a1}+{a2}"
            expr_b = f"{b1}*{b2}"
            if val_a > val_b:
                answer = expr_a
            elif val_b > val_a:
                answer = expr_b
            else:
                answer = "equal"
            return f"Which is bigger: {expr_a} or {expr_b}? {answer}"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def get_random_task(self) -> str:
        generators = [
            self.generate_arithmetic,
            self.generate_counting,
            self.generate_brackets,
            self.generate_listops,
            self.generate_sorting,
            self.generate_copy,
            self.generate_context_recall,
            self.generate_logic,
            self.generate_comparison,
        ]
        return self.rng.choice(generators)()


if __name__ == "__main__":
    g = SyntheticTaskGenerator(seed=0)
    for d in range(1, 6):
        g.set_difficulty(d)
        print(f"\n--- Difficulty {d} ---")
        # Show one of each type, then 3 random
        generators = [
            ("Arithmetic", g.generate_arithmetic),
            ("Counting", g.generate_counting),
            ("Brackets", g.generate_brackets),
            ("ListOps", g.generate_listops),
            ("Sorting", g.generate_sorting),
            ("Copy", g.generate_copy),
            ("Context", g.generate_context_recall),
            ("Logic", g.generate_logic),
            ("Comparison", g.generate_comparison),
        ]
        for name, gen in generators:
            print(f"  [{name:>10s}] {gen()}")
