"""
Domain 4: Finite State Machines
"""
from __future__ import annotations
import random
from collections import deque
from VTASK.base import TaskEntry, TaskGenerator


class FSMGenerator(TaskGenerator):
    domain = "fsm"
    difficulty_range = (1, 5)

    def _build_fsm(self, rng: random.Random, n_states: int, alphabet: list) -> dict:
        states = [f"q{i}" for i in range(n_states)]
        start_state = "q0"
        # At least one accept state, at most half
        n_accept = rng.randint(1, max(1, n_states // 2))
        accept_states = set(rng.sample(states, n_accept))
        transitions = {}
        for s in states:
            transitions[s] = {}
            for sym in alphabet:
                transitions[s][sym] = rng.choice(states)
        return {
            "states": states,
            "alphabet": alphabet,
            "transitions": transitions,
            "start_state": start_state,
            "accept_states": list(accept_states),
        }

    def _build_nfa_with_epsilon(self, rng: random.Random, n_states: int, alphabet: list) -> dict:
        """Build an NFA with epsilon transitions."""
        states = [f"q{i}" for i in range(n_states)]
        start_state = "q0"
        n_accept = rng.randint(1, max(1, n_states // 2))
        accept_states = set(rng.sample(states, n_accept))
        transitions = {}
        for s in states:
            transitions[s] = {}
            for sym in alphabet:
                # NFA: 0-2 transitions per symbol
                n_trans = rng.randint(0, 2)
                transitions[s][sym] = rng.sample(states, min(n_trans, len(states)))
            # Epsilon transitions: randomly add 0 or 1
            if rng.random() < 0.4:
                transitions[s]["ε"] = [rng.choice(states)]
            else:
                transitions[s]["ε"] = []
        return {
            "states": states,
            "alphabet": alphabet,
            "transitions": transitions,
            "start_state": start_state,
            "accept_states": list(accept_states),
            "is_nfa": True,
        }

    def _accepts(self, fsm: dict, string: str) -> bool:
        if fsm.get("is_nfa"):
            return self._nfa_accepts(fsm, string)
        state = fsm["start_state"]
        for sym in string:
            state = fsm["transitions"][state].get(sym, None)
            if state is None:
                return False
        return state in fsm["accept_states"]

    def _epsilon_closure(self, fsm: dict, states: set) -> set:
        closure = set(states)
        stack = list(states)
        while stack:
            s = stack.pop()
            for t in fsm["transitions"].get(s, {}).get("ε", []):
                if t not in closure:
                    closure.add(t)
                    stack.append(t)
        return closure

    def _nfa_accepts(self, fsm: dict, string: str) -> bool:
        current = self._epsilon_closure(fsm, {fsm["start_state"]})
        for sym in string:
            nxt = set()
            for s in current:
                for t in fsm["transitions"].get(s, {}).get(sym, []):
                    nxt.add(t)
            current = self._epsilon_closure(fsm, nxt)
        return bool(current & set(fsm["accept_states"]))

    def _bfs_accepting_string(self, fsm: dict, max_len: int = 10) -> str | None:
        """BFS to find shortest string accepted by FSM."""
        if fsm.get("is_nfa"):
            return self._bfs_accepting_nfa(fsm, max_len)
        start = fsm["start_state"]
        alphabet = fsm["alphabet"]
        accept_states = set(fsm["accept_states"])
        queue = deque([(start, "")])
        visited = {start}
        while queue:
            state, path = queue.popleft()
            if state in accept_states:
                return path
            if len(path) >= max_len:
                continue
            for sym in alphabet:
                nxt = fsm["transitions"][state].get(sym)
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + sym))
        # Check empty string
        if start in accept_states:
            return ""
        return None

    def _bfs_accepting_nfa(self, fsm: dict, max_len: int = 10) -> str | None:
        start_closure = frozenset(self._epsilon_closure(fsm, {fsm["start_state"]}))
        accept_states = set(fsm["accept_states"])
        if start_closure & accept_states:
            return ""
        queue = deque([(start_closure, "")])
        visited = {start_closure}
        while queue:
            states, path = queue.popleft()
            if len(path) >= max_len:
                continue
            for sym in fsm["alphabet"]:
                nxt = set()
                for s in states:
                    for t in fsm["transitions"].get(s, {}).get(sym, []):
                        nxt.add(t)
                nxt = frozenset(self._epsilon_closure(fsm, nxt))
                if nxt & accept_states:
                    return path + sym
                if nxt and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + sym))
        return None

    def _fsm_to_description(self, fsm: dict) -> str:
        lines = []
        lines.append(f"States: {', '.join(fsm['states'])}")
        lines.append(f"Alphabet: {{{', '.join(fsm['alphabet'])}}}")
        lines.append(f"Start state: {fsm['start_state']}")
        lines.append(f"Accept states: {{{', '.join(sorted(fsm['accept_states']))}}}")
        lines.append("Transitions:")
        for s in fsm["states"]:
            for sym, dst in fsm["transitions"][s].items():
                if isinstance(dst, list):
                    if dst:
                        lines.append(f"  δ({s}, {sym}) = {{{', '.join(dst)}}}")
                else:
                    lines.append(f"  δ({s}, {sym}) = {dst}")
        return "\n".join(lines)

    def generate(self, seed: int, difficulty: int = 1) -> TaskEntry:
        rng = random.Random(seed)

        if difficulty == 1:
            alphabet = ["0", "1"]
            fsm = self._build_fsm(rng, n_states=3, alphabet=alphabet)
            test_str = "".join(rng.choices(alphabet, k=rng.randint(2, 5)))
            accepted = self._accepts(fsm, test_str)
            answer = "yes" if accepted else "no"
            desc = self._fsm_to_description(fsm)
            question = (
                f"Consider the following finite state machine (DFA):\n\n{desc}\n\n"
                f"Does the machine accept the string \"{test_str}\"? "
                f"Answer with just \"yes\" or \"no\"."
            )
            distractors = ["no" if accepted else "yes"]
            metadata = {"fsm": fsm, "test_str": test_str, "correct_answer": answer, "mode": "accept"}

        elif difficulty == 2:
            alphabet = ["a", "b", "c"]
            fsm = self._build_fsm(rng, n_states=4, alphabet=alphabet)
            test_str = "".join(rng.choices(alphabet, k=rng.randint(3, 6)))
            # Compute final state
            state = fsm["start_state"]
            for sym in test_str:
                state = fsm["transitions"][state][sym]
            answer = state
            desc = self._fsm_to_description(fsm)
            question = (
                f"Consider the following finite state machine (DFA):\n\n{desc}\n\n"
                f"What state does the machine end in after processing the string \"{test_str}\"? "
                f"Answer with just the state name."
            )
            other_states = [s for s in fsm["states"] if s != answer]
            distractors = rng.sample(other_states, min(3, len(other_states)))
            metadata = {"fsm": fsm, "test_str": test_str, "correct_answer": answer, "mode": "final_state"}

        elif difficulty == 3:
            alphabet = ["a", "b"]
            fsm = self._build_fsm(rng, n_states=5, alphabet=alphabet)
            # Find shortest accepting string, then pad to length N
            target_len = rng.randint(3, 6)
            shortest = self._bfs_accepting_string(fsm, max_len=10)
            if shortest is None:
                # No accepting string — make a trivially accepting fsm
                fsm["accept_states"] = [fsm["start_state"]]
                shortest = ""
            if len(shortest) <= target_len:
                # Pad by repeating a symbol that stays in current state or loops
                test_str = shortest
                state = fsm["start_state"]
                for sym in test_str:
                    state = fsm["transitions"][state][sym]
                while len(test_str) < target_len:
                    sym = rng.choice(alphabet)
                    test_str += sym
                    state = fsm["transitions"][state][sym]
                # Verify it's still accepted
                if not self._accepts(fsm, test_str):
                    # Fall back: BFS for a string of exactly target_len
                    test_str = self._bfs_exact_length(fsm, target_len, alphabet)
                    if test_str is None:
                        target_len = len(shortest)
                        test_str = shortest
            else:
                target_len = len(shortest)
                test_str = shortest

            answer = "yes" if self._accepts(fsm, test_str) else "no"
            # For difficulty 3 we ask them to produce an accepting string of length N
            desc = self._fsm_to_description(fsm)
            question = (
                f"Consider the following finite state machine (DFA):\n\n{desc}\n\n"
                f"Give a string of exactly length {target_len} over the alphabet "
                f"{{{', '.join(alphabet)}}} that is accepted by this machine. "
                f"If no such string exists, write \"NONE\"."
            )
            answer = test_str if self._accepts(fsm, test_str) else "NONE"
            distractors = []
            metadata = {
                "fsm": fsm,
                "target_len": target_len,
                "correct_answer": answer,
                "mode": "generate_string",
            }

        elif difficulty == 4:
            alphabet = ["0", "1"]
            fsm = self._build_nfa_with_epsilon(rng, n_states=6, alphabet=alphabet)
            test_str = "".join(rng.choices(alphabet, k=rng.randint(3, 7)))
            accepted = self._accepts(fsm, test_str)
            answer = "yes" if accepted else "no"
            desc = self._fsm_to_description(fsm)
            question = (
                f"Consider the following NFA with epsilon (ε) transitions:\n\n{desc}\n\n"
                f"Does the NFA accept the string \"{test_str}\"? "
                f"Answer with just \"yes\" or \"no\"."
            )
            distractors = ["no" if accepted else "yes"]
            metadata = {"fsm": fsm, "test_str": test_str, "correct_answer": answer, "mode": "accept_nfa"}

        else:  # difficulty 5
            alphabet = ["a", "b"]
            fsm_a = self._build_fsm(rng, n_states=4, alphabet=alphabet)
            fsm_b = self._build_fsm(rng, n_states=4, alphabet=alphabet)
            # Find a string accepted by A but rejected by B (or vice versa), length <= 6
            sym_diff_str = self._find_symmetric_difference(fsm_a, fsm_b, alphabet, max_len=6)
            if sym_diff_str is not None:
                answer = sym_diff_str
            else:
                answer = "NONE"
            desc_a = self._fsm_to_description(fsm_a)
            desc_b = self._fsm_to_description(fsm_b)
            question = (
                f"Consider these two finite state machines (DFAs):\n\n"
                f"Machine A:\n{desc_a}\n\n"
                f"Machine B:\n{desc_b}\n\n"
                f"Give a string of length at most 6 over {{{', '.join(alphabet)}}} "
                f"that is accepted by exactly one of the two machines. "
                f"If no such string exists (both machines accept and reject the same strings of length ≤ 6), "
                f"write \"NONE\"."
            )
            distractors = []
            metadata = {
                "fsm_a": fsm_a,
                "fsm_b": fsm_b,
                "correct_answer": answer,
                "mode": "symmetric_difference",
            }

        return TaskEntry(
            question=question,
            answer=answer,
            distractors=distractors,
            difficulty=difficulty,
            domain=self.domain,
            metadata=metadata,
            task_id="",
        )

    def _bfs_exact_length(self, fsm: dict, target_len: int, alphabet: list) -> str | None:
        """BFS to find an accepting string of exactly target_len."""
        accept_states = set(fsm["accept_states"])
        queue = deque([(fsm["start_state"], "")])
        while queue:
            state, path = queue.popleft()
            if len(path) == target_len:
                if state in accept_states:
                    return path
                continue
            for sym in alphabet:
                nxt = fsm["transitions"][state][sym]
                queue.append((nxt, path + sym))
        return None

    def _find_symmetric_difference(self, fsm_a: dict, fsm_b: dict, alphabet: list, max_len: int = 6) -> str | None:
        """BFS over product automaton to find a string in A⊕B of length ≤ max_len."""
        start = (fsm_a["start_state"], fsm_b["start_state"])
        queue = deque([(start, "")])
        visited = {start}
        accept_a = set(fsm_a["accept_states"])
        accept_b = set(fsm_b["accept_states"])

        def in_sym_diff(sa, sb):
            return (sa in accept_a) != (sb in accept_b)

        # Check empty string
        if in_sym_diff(fsm_a["start_state"], fsm_b["start_state"]):
            return ""

        while queue:
            (sa, sb), path = queue.popleft()
            if len(path) >= max_len:
                continue
            for sym in alphabet:
                nsa = fsm_a["transitions"][sa][sym]
                nsb = fsm_b["transitions"][sb][sym]
                new_path = path + sym
                if in_sym_diff(nsa, nsb):
                    return new_path
                state = (nsa, nsb)
                if state not in visited:
                    visited.add(state)
                    queue.append((state, new_path))
        return None

    def score_answer(self, answer: str, entry: TaskEntry) -> float:
        mode = entry.metadata.get("mode", "accept")
        answer = answer.strip()

        if mode in ("accept", "accept_nfa"):
            norm = answer.lower()
            correct = entry.metadata["correct_answer"].lower()
            if norm in ("true", "yes"):
                norm = "yes"
            elif norm in ("false", "no"):
                norm = "no"
            return 1.0 if norm == correct else 0.0

        elif mode == "final_state":
            return 1.0 if answer.lower() == entry.metadata["correct_answer"].lower() else 0.0

        elif mode == "generate_string":
            if entry.metadata["correct_answer"] == "NONE":
                return 1.0 if answer.upper() == "NONE" else 0.0
            if answer.upper() == "NONE":
                return 0.0
            fsm = entry.metadata["fsm"]
            target_len = entry.metadata["target_len"]
            if len(answer) != target_len:
                return 0.0
            valid_chars = set(fsm["alphabet"])
            if not all(c in valid_chars for c in answer):
                return 0.0
            return 1.0 if self._accepts(fsm, answer) else 0.0

        elif mode == "symmetric_difference":
            if entry.metadata["correct_answer"] == "NONE":
                return 1.0 if answer.upper() == "NONE" else 0.0
            if answer.upper() == "NONE":
                # Check if the correct answer is actually NONE
                return 1.0 if entry.metadata["correct_answer"] == "NONE" else 0.0
            fsm_a = entry.metadata["fsm_a"]
            fsm_b = entry.metadata["fsm_b"]
            if len(answer) > 6:
                return 0.0
            valid_chars = set(fsm_a["alphabet"])
            if not all(c in valid_chars for c in answer):
                return 0.0
            acc_a = self._accepts(fsm_a, answer)
            acc_b = self._accepts(fsm_b, answer)
            return 1.0 if (acc_a != acc_b) else 0.0

        return 0.0
