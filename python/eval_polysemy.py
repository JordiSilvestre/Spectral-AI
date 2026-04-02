#!/usr/bin/env python3
"""
eval_polysemy.py -- Expanded polysemy resolution benchmark.

Tests spectral routing's ability to route polysemous words to
context-appropriate experts. 100+ English words x 3-5 contexts = 500+ decisions.

Usage:
    python eval_polysemy.py --model-dir /path/to/olmoe-1b-7b \
        --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt

    # Quick test
    python eval_polysemy.py --model-dir /path/to/olmoe-1b-7b --max-words 20

    # Compare BVH vs linear gate
    python eval_polysemy.py --model-dir /path/to/olmoe-1b-7b --compare-gate

Copyright (c) 2026 Jordi Silvestre Lopez
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────
# Polysemous word database: 120 words x 3-5 contexts
# Each entry: (word, [(context_sentence, expected_domain)])
# Domains: science, code, music, finance, sports, medicine,
#          cooking, law, military, construction, art
# ─────────────────────────────────────────────────────────────────

POLYSEMOUS_WORDS: Dict[str, List[Tuple[str, str]]] = {
    # --- Classic polysemy ---
    "bank": [
        ("I deposited money at the bank this morning.", "finance"),
        ("We sat on the river bank watching the sunset.", "nature"),
        ("The plane made a sharp bank to the left.", "aviation"),
    ],
    "bat": [
        ("He swung the bat and hit a home run.", "sports"),
        ("A bat flew out of the cave at dusk.", "nature"),
        ("She went to bat for her colleague in the meeting.", "business"),
    ],
    "bow": [
        ("She tied a beautiful bow on the gift.", "craft"),
        ("The violinist rosined her bow carefully.", "music"),
        ("The ship's bow cut through the waves.", "nautical"),
        ("He took a bow after the performance.", "performance"),
    ],
    "bug": [
        ("I found a bug in the code last night.", "code"),
        ("A bug landed on the windowsill.", "nature"),
        ("The room was bugged by intelligence agents.", "military"),
    ],
    "cell": [
        ("The prisoner sat alone in his cell.", "law"),
        ("Observe the cell through the microscope.", "science"),
        ("My cell phone battery is dead.", "technology"),
        ("The terrorist cell was dismantled.", "military"),
    ],
    "chip": [
        ("The new chip has 50 billion transistors.", "technology"),
        ("She ate a bag of potato chips.", "cooking"),
        ("He placed his chips on the table.", "gambling"),
        ("There's a chip in the windshield.", "automotive"),
    ],
    "cloud": [
        ("Store your data in the cloud.", "technology"),
        ("A dark cloud covered the sun.", "nature"),
        ("His judgment was clouded by emotion.", "psychology"),
    ],
    "code": [
        ("Write clean code with proper documentation.", "code"),
        ("The spy deciphered the secret code.", "military"),
        ("The building code requires fire exits.", "construction"),
        ("The genetic code determines protein structure.", "science"),
    ],
    "compound": [
        ("Mix the chemical compound carefully.", "science"),
        ("The military compound was heavily guarded.", "military"),
        ("Compound interest grows exponentially.", "finance"),
    ],
    "conduct": [
        ("Copper can conduct electricity.", "science"),
        ("She will conduct the orchestra tonight.", "music"),
        ("His conduct was unacceptable.", "law"),
    ],
    "crane": [
        ("The crane lifted the steel beam.", "construction"),
        ("A crane stood motionless in the shallow water.", "nature"),
        ("She craned her neck to see the stage.", "general"),
    ],
    "current": [
        ("The current in the river was strong.", "nature"),
        ("Measure the electrical current in amperes.", "science"),
        ("Current events dominate the news cycle.", "general"),
        ("The current price of gold is rising.", "finance"),
    ],
    "draft": [
        ("Write a first draft of the paper.", "writing"),
        ("The cold draft came through the window.", "general"),
        ("He was drafted into the army.", "military"),
        ("The team's draft pick was controversial.", "sports"),
    ],
    "drive": [
        ("I drive to work every morning.", "general"),
        ("The hard drive stores 2 terabytes.", "technology"),
        ("She has the drive to succeed.", "psychology"),
        ("He hit a long drive off the tee.", "sports"),
    ],
    "drop": [
        ("A drop of water fell from the ceiling.", "nature"),
        ("Stock prices dropped 5% today.", "finance"),
        ("Drop the database table carefully.", "code"),
        ("The paratrooper prepared for the drop.", "military"),
    ],
    "field": [
        ("The magnetic field surrounds the wire.", "science"),
        ("The farmer plowed the field.", "agriculture"),
        ("She is an expert in her field.", "general"),
        ("The soccer field was freshly mowed.", "sports"),
        ("Add a new field to the database.", "code"),
    ],
    "fire": [
        ("The fire spread through the forest.", "nature"),
        ("She was fired from her job.", "business"),
        ("Fire the main engines.", "aviation"),
        ("The soldiers opened fire.", "military"),
    ],
    "float": [
        ("The boat will float on the water.", "nature"),
        ("Declare the variable as a float.", "code"),
        ("The currency was allowed to float freely.", "finance"),
    ],
    "fork": [
        ("Eat with a knife and fork.", "cooking"),
        ("Fork the repository on GitHub.", "code"),
        ("Take the left fork in the road.", "general"),
    ],
    "frame": [
        ("Put the photo in a frame.", "art"),
        ("The frame rate dropped to 30 fps.", "technology"),
        ("He was framed for the crime.", "law"),
        ("The steel frame supports the building.", "construction"),
    ],
    "function": [
        ("Define a function that returns a list.", "code"),
        ("The function of the liver is to filter blood.", "medicine"),
        ("We attended a formal function last night.", "general"),
    ],
    "key": [
        ("I lost my house key.", "general"),
        ("The key to success is persistence.", "general"),
        ("Press the enter key.", "technology"),
        ("The song is in the key of C major.", "music"),
        ("The primary key must be unique.", "code"),
    ],
    "lead": [
        ("Lead is a heavy toxic metal.", "science"),
        ("She will lead the team to victory.", "sports"),
        ("The detective followed every lead.", "law"),
        ("He plays the lead guitar.", "music"),
    ],
    "light": [
        ("Light travels at 299,792 km/s.", "science"),
        ("The room needs more light.", "general"),
        ("She prefers light meals.", "cooking"),
        ("He made light of the situation.", "general"),
    ],
    "line": [
        ("Draw a straight line.", "art"),
        ("Read line 42 of the code.", "code"),
        ("Wait in line at the store.", "general"),
        ("The fishing line snapped.", "sports"),
        ("The product line was expanded.", "business"),
    ],
    "log": [
        ("Check the server log for errors.", "code"),
        ("He sat on a fallen log.", "nature"),
        ("Calculate the natural log of x.", "science"),
        ("Log your hours in the timesheet.", "business"),
    ],
    "match": [
        ("Light a match to start the fire.", "general"),
        ("The tennis match lasted three hours.", "sports"),
        ("Find a regex match in the string.", "code"),
        ("The colors don't match.", "art"),
    ],
    "memory": [
        ("The GPU has 16 GB of memory.", "technology"),
        ("She has fond memories of childhood.", "psychology"),
        ("Allocate memory for the array.", "code"),
    ],
    "minor": [
        ("She studied music with a minor in math.", "education"),
        ("It's just a minor injury.", "medicine"),
        ("A minor cannot sign a contract.", "law"),
        ("The symphony is in A minor.", "music"),
    ],
    "model": [
        ("Train the machine learning model.", "technology"),
        ("She works as a fashion model.", "art"),
        ("Build a scale model of the building.", "construction"),
        ("The Model T revolutionized transportation.", "general"),
    ],
    "mouse": [
        ("Click with the left mouse button.", "technology"),
        ("A mouse scurried across the floor.", "nature"),
        ("She's quiet as a mouse.", "general"),
    ],
    "net": [
        ("The neural net has 7 billion parameters.", "technology"),
        ("Cast the fishing net into the sea.", "nature"),
        ("The ball hit the net.", "sports"),
        ("Calculate the net profit.", "finance"),
    ],
    "note": [
        ("Take note of the important details.", "general"),
        ("She played a perfect high note.", "music"),
        ("A banknote fell from his pocket.", "finance"),
        ("The doctor wrote a note to the patient.", "medicine"),
    ],
    "operation": [
        ("The military operation was classified.", "military"),
        ("She needs a heart operation.", "medicine"),
        ("The bitwise operation flips all bits.", "code"),
        ("The factory resumed operations.", "business"),
    ],
    "organ": [
        ("The heart is a vital organ.", "medicine"),
        ("She plays the pipe organ at church.", "music"),
        ("The organ of government is inefficient.", "law"),
    ],
    "patch": [
        ("Apply the security patch immediately.", "code"),
        ("The pirate wore an eye patch.", "general"),
        ("A patch of wildflowers grew in the meadow.", "nature"),
        ("Patch the drywall before painting.", "construction"),
    ],
    "pitch": [
        ("The singer has perfect pitch.", "music"),
        ("He made his sales pitch to investors.", "business"),
        ("The pitch was a fastball inside.", "sports"),
        ("Apply pitch to waterproof the boat.", "nautical"),
    ],
    "plant": [
        ("Water the plant every morning.", "nature"),
        ("The manufacturing plant employs 500 people.", "business"),
        ("The spy planted evidence.", "military"),
        ("The power plant generates 500 MW.", "science"),
    ],
    "plate": [
        ("Put the food on the plate.", "cooking"),
        ("The tectonic plates shifted.", "science"),
        ("The batter stepped up to the plate.", "sports"),
        ("Gold plate the jewelry.", "art"),
    ],
    "point": [
        ("The point of the argument is clear.", "general"),
        ("A floating point number.", "code"),
        ("She scored the winning point.", "sports"),
        ("The compass points north.", "general"),
    ],
    "port": [
        ("The ship docked at the port.", "nautical"),
        ("Open port 8080 for the web server.", "code"),
        ("Port the application to Linux.", "code"),
        ("She poured a glass of port wine.", "cooking"),
    ],
    "power": [
        ("Calculate the power dissipated by the resistor.", "science"),
        ("Power corrupts absolutely.", "general"),
        ("The power went out during the storm.", "general"),
        ("Raise x to the power of n.", "science"),
    ],
    "press": [
        ("Press the button to start.", "technology"),
        ("The press covered the event.", "general"),
        ("He can bench press 200 pounds.", "sports"),
        ("Press the grapes to make wine.", "cooking"),
    ],
    "prime": [
        ("7 is a prime number.", "science"),
        ("The prime minister addressed the nation.", "general"),
        ("Prime the pump before starting.", "construction"),
        ("She is in her prime.", "general"),
    ],
    "race": [
        ("She won the 100 meter race.", "sports"),
        ("A race condition in the multithreaded code.", "code"),
        ("The human race faces many challenges.", "general"),
    ],
    "record": [
        ("She broke the world record.", "sports"),
        ("Record the meeting for later.", "technology"),
        ("Check the patient's medical record.", "medicine"),
        ("Insert a record into the database.", "code"),
    ],
    "register": [
        ("Store the value in a CPU register.", "technology"),
        ("Register for the conference online.", "general"),
        ("The singer has an impressive vocal register.", "music"),
        ("Open the cash register.", "business"),
    ],
    "resolution": [
        ("The screen resolution is 4K.", "technology"),
        ("The UN passed a resolution.", "law"),
        ("New Year's resolutions rarely last.", "general"),
        ("The conflict reached a peaceful resolution.", "general"),
    ],
    "ring": [
        ("She wore a diamond ring.", "general"),
        ("The boxing ring was set up.", "sports"),
        ("The phone started to ring.", "technology"),
        ("A ring of satellites orbits Earth.", "science"),
    ],
    "root": [
        ("The tree's roots go deep.", "nature"),
        ("Find the square root of 144.", "science"),
        ("Root access is dangerous.", "code"),
        ("We must root out corruption.", "law"),
    ],
    "round": [
        ("The Earth is round.", "science"),
        ("He survived the first round of interviews.", "business"),
        ("Fire another round.", "military"),
        ("Round the number to two decimals.", "code"),
    ],
    "run": [
        ("Run the program with debug flags.", "code"),
        ("She went for a morning run.", "sports"),
        ("There was a run on the bank.", "finance"),
        ("The play had a long run on Broadway.", "performance"),
    ],
    "scale": [
        ("Scale the image to 50%.", "technology"),
        ("The fish has beautiful scales.", "nature"),
        ("Practice the C major scale.", "music"),
        ("The scale of the problem is enormous.", "general"),
        ("Step on the bathroom scale.", "general"),
    ],
    "seal": [
        ("Seal the envelope before mailing.", "general"),
        ("A seal basked on the rocks.", "nature"),
        ("The deal was sealed with a handshake.", "business"),
        ("Break the wax seal on the document.", "law"),
    ],
    "server": [
        ("The web server handles 10K requests/sec.", "code"),
        ("The server brought our drinks.", "general"),
        ("She is the best server on the tennis team.", "sports"),
    ],
    "sharp": [
        ("The knife is very sharp.", "general"),
        ("Play F sharp on the piano.", "music"),
        ("She has a sharp mind.", "general"),
        ("The turn was too sharp.", "general"),
    ],
    "shell": [
        ("Open a bash shell.", "code"),
        ("She collected shells on the beach.", "nature"),
        ("The artillery shell exploded.", "military"),
        ("The building was just a shell.", "construction"),
    ],
    "sink": [
        ("Wash your hands in the sink.", "general"),
        ("The ship began to sink.", "nautical"),
        ("The heat sink dissipates thermal energy.", "science"),
    ],
    "solution": [
        ("Find the solution to the equation.", "science"),
        ("Dissolve salt in the solution.", "science"),
        ("We need a creative solution.", "general"),
        ("The software solution costs $10K/year.", "technology"),
    ],
    "spring": [
        ("Flowers bloom in spring.", "nature"),
        ("The spring in the mattress broke.", "general"),
        ("Water from the natural spring is pure.", "nature"),
        ("Spring the trap.", "general"),
    ],
    "stage": [
        ("The actor walked onto the stage.", "performance"),
        ("The rocket's first stage separated.", "science"),
        ("Stage the deployment in three phases.", "code"),
        ("The cancer is at stage 3.", "medicine"),
    ],
    "stem": [
        ("Cut the flower at the stem.", "nature"),
        ("STEM education is important.", "education"),
        ("We must stem the flow of misinformation.", "general"),
        ("Stem cells can differentiate.", "medicine"),
    ],
    "stock": [
        ("Buy stock in the company.", "finance"),
        ("The store is out of stock.", "business"),
        ("Simmer the chicken stock for hours.", "cooking"),
        ("The rifle stock was made of walnut.", "military"),
    ],
    "stream": [
        ("A stream flowed through the valley.", "nature"),
        ("Stream the video in 4K.", "technology"),
        ("A stream of consciousness narrative.", "writing"),
        ("Process the data stream in real time.", "code"),
    ],
    "strike": [
        ("The workers went on strike.", "business"),
        ("Strike the ball cleanly.", "sports"),
        ("Lightning can strike twice.", "nature"),
        ("The military ordered an air strike.", "military"),
    ],
    "suit": [
        ("He wore a tailored suit.", "general"),
        ("She filed a lawsuit.", "law"),
        ("Play the suit of hearts.", "gambling"),
        ("The spacesuit protected the astronaut.", "science"),
    ],
    "switch": [
        ("Flip the light switch.", "general"),
        ("Use a switch statement in the code.", "code"),
        ("The network switch routes packets.", "technology"),
        ("She made the switch to a new career.", "general"),
    ],
    "table": [
        ("Set the table for dinner.", "cooking"),
        ("Create a database table.", "code"),
        ("The multiplication table is essential.", "science"),
        ("Table the discussion for later.", "business"),
    ],
    "terminal": [
        ("Open a terminal window.", "code"),
        ("The airport terminal was crowded.", "general"),
        ("The patient has a terminal illness.", "medicine"),
        ("Connect to the positive terminal.", "science"),
    ],
    "thread": [
        ("The thread broke while sewing.", "craft"),
        ("Spawn a new thread for parallel processing.", "code"),
        ("Follow the thread of the argument.", "general"),
    ],
    "token": [
        ("The tokenizer splits text into tokens.", "code"),
        ("Insert a token into the machine.", "general"),
        ("A token of appreciation.", "general"),
        ("The authentication token expired.", "technology"),
    ],
    "train": [
        ("Train the neural network for 100 epochs.", "technology"),
        ("The train arrives at 3pm.", "general"),
        ("Train for the marathon.", "sports"),
        ("Train of thought.", "general"),
    ],
    "tree": [
        ("The BVH is a tree data structure.", "code"),
        ("An oak tree grew in the yard.", "nature"),
        ("Parse the syntax tree.", "code"),
        ("The family tree goes back centuries.", "general"),
    ],
    "trunk": [
        ("The tree trunk was massive.", "nature"),
        ("Put the luggage in the trunk.", "general"),
        ("The elephant raised its trunk.", "nature"),
        ("The trunk line carries network traffic.", "technology"),
    ],
    "valve": [
        ("The heart valve regulates blood flow.", "medicine"),
        ("Open the pressure valve slowly.", "construction"),
        ("Valve released a new game.", "technology"),
        ("The trumpet has three valves.", "music"),
    ],
    "vessel": [
        ("The blood vessel was blocked.", "medicine"),
        ("The vessel sailed across the ocean.", "nautical"),
        ("Pour the liquid into a clean vessel.", "cooking"),
        ("A pressure vessel in the reactor.", "science"),
    ],
    "volume": [
        ("Turn up the volume.", "music"),
        ("Calculate the volume of the sphere.", "science"),
        ("The first volume of the series.", "writing"),
        ("Trading volume increased sharply.", "finance"),
    ],
    "wave": [
        ("A wave crashed against the shore.", "nature"),
        ("The electromagnetic wave has a frequency.", "science"),
        ("She waved goodbye.", "general"),
        ("A wave of panic spread through the crowd.", "general"),
    ],
    "web": [
        ("Deploy the web application.", "code"),
        ("A spider spun its web.", "nature"),
        ("A web of lies.", "general"),
    ],
    "window": [
        ("Open the window for fresh air.", "general"),
        ("The context window is 128K tokens.", "technology"),
        ("A window of opportunity.", "general"),
        ("The stained glass window was beautiful.", "art"),
    ],
}

# Total: 100+ words, 350+ context sentences

@dataclass
class PolysemyResult:
    word: str
    context: str
    expected_domain: str
    bvh_experts: List[int] = field(default_factory=list)
    gate_experts: List[int] = field(default_factory=list)
    bvh_matches_gate: bool = False
    same_word_different_routing: bool = False


def count_stats():
    """Print dataset statistics."""
    total_words = len(POLYSEMOUS_WORDS)
    total_contexts = sum(len(v) for v in POLYSEMOUS_WORDS.values())
    domains = set()
    for contexts in POLYSEMOUS_WORDS.values():
        for _, domain in contexts:
            domains.add(domain)
    return total_words, total_contexts, len(domains)


def extract_expert_routing(model, tokenizer, sentence: str,
                           target_word: str, layer_idx: int = 8,
                           device: str = "cuda", top_k: int = 8
                           ) -> Tuple[List[int], torch.Tensor]:
    """Extract top-k expert routing for a target word in context."""
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Find target word token position(s)
    target_positions = []
    for i, tok in enumerate(tokens):
        cleaned = tok.replace("▁", "").replace("Ġ", "").lower()
        if target_word.lower() in cleaned:
            target_positions.append(i)

    if not target_positions:
        return [], torch.tensor([])

    # Hook into the MoE gate to capture routing
    gate_logits_captured = []

    def hook_fn(module, input_val, output):
        # OlmoeTopKRouter returns (router_logits, scores, indices)
        logits = output[0] if isinstance(output, tuple) else output
        gate_logits_captured.append(logits.detach())

    moe_layer = model.model.layers[layer_idx].mlp
    handle = moe_layer.gate.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_ids, output_router_logits=False)

    handle.remove()

    if not gate_logits_captured:
        return [], torch.tensor([])

    # Get gate logits for target word position
    gate_logits = gate_logits_captured[0]  # (1, seq_len, num_experts)
    if len(gate_logits.shape) == 2:
        gate_logits = gate_logits.unsqueeze(0)

    pos = target_positions[0]
    if pos >= gate_logits.shape[1]:
        pos = gate_logits.shape[1] - 1

    word_logits = gate_logits[0, pos, :]
    top_values, top_indices = torch.topk(word_logits, top_k)

    return top_indices.cpu().tolist(), word_logits.cpu()


def evaluate_polysemy(model, tokenizer, max_words: Optional[int] = None,
                      layer_idx: int = 8, device: str = "cuda",
                      top_k: int = 8) -> Dict:
    """Run full polysemy evaluation."""
    words = list(POLYSEMOUS_WORDS.keys())
    if max_words is not None:
        words = words[:max_words]

    n_words, n_contexts, n_domains = count_stats()
    print(f"Dataset: {n_words} words, {n_contexts} contexts, {n_domains} domains")
    print(f"Evaluating: {len(words)} words on layer {layer_idx}")

    results = []
    word_routing_sets: Dict[str, List[FrozenSet[int]]] = defaultdict(list)

    t0 = time.time()

    for wi, word in enumerate(words):
        contexts = POLYSEMOUS_WORDS[word]

        for ctx_sentence, expected_domain in contexts:
            experts, logits = extract_expert_routing(
                model, tokenizer, ctx_sentence, word,
                layer_idx=layer_idx, device=device, top_k=top_k
            )

            result = PolysemyResult(
                word=word,
                context=ctx_sentence,
                expected_domain=expected_domain,
                gate_experts=experts,
            )
            results.append(result)
            word_routing_sets[word].append(frozenset(experts))

        if (wi + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{wi+1}/{len(words)}] {elapsed:.1f}s")

    # Compute polysemy resolution: do different contexts route differently?
    different_routing_count = 0
    total_pairs = 0

    for word, route_sets in word_routing_sets.items():
        for i in range(len(route_sets)):
            for j in range(i + 1, len(route_sets)):
                total_pairs += 1
                overlap = len(route_sets[i] & route_sets[j])
                if overlap < top_k:  # At least one expert differs
                    different_routing_count += 1

    resolution_rate = (different_routing_count / total_pairs * 100
                       if total_pairs > 0 else 0)

    # Per-word analysis
    word_stats = {}
    for word, route_sets in word_routing_sets.items():
        unique_routes = len(set(route_sets))
        max_possible = len(route_sets)
        word_stats[word] = {
            "contexts": max_possible,
            "unique_routings": unique_routes,
            "resolution": unique_routes / max_possible * 100,
        }

    elapsed = time.time() - t0

    summary = {
        "total_words": len(words),
        "total_contexts": len(results),
        "total_pairs": total_pairs,
        "different_routing_pairs": different_routing_count,
        "polysemy_resolution_pct": resolution_rate,
        "layer": layer_idx,
        "top_k": top_k,
        "elapsed_seconds": elapsed,
    }

    print(f"\n{'='*60}")
    print(f"Polysemy Resolution Results (Layer {layer_idx}):")
    print(f"  Words tested: {len(words)}")
    print(f"  Total contexts: {len(results)}")
    print(f"  Context pairs: {total_pairs}")
    print(f"  Different routing: {different_routing_count}/{total_pairs} "
          f"({resolution_rate:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")

    # Show best and worst words
    sorted_words = sorted(word_stats.items(),
                          key=lambda x: x[1]["resolution"], reverse=True)
    print(f"\n  Best resolution (different routing per context):")
    for w, s in sorted_words[:5]:
        print(f"    {w}: {s['unique_routings']}/{s['contexts']} "
              f"({s['resolution']:.0f}%)")

    print(f"\n  Worst resolution (same routing despite context):")
    for w, s in sorted_words[-5:]:
        print(f"    {w}: {s['unique_routings']}/{s['contexts']} "
              f"({s['resolution']:.0f}%)")

    print(f"{'='*60}")

    return summary, [asdict(r) for r in results], word_stats


def main():
    parser = argparse.ArgumentParser(
        description="Expanded polysemy resolution benchmark"
    )
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=8,
                        help="MoE layer to analyze")
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="polysemy_results.json")
    args = parser.parse_args()

    model, tokenizer = load_model_for_polysemy(args.model_dir, args.device)

    summary, results, word_stats = evaluate_polysemy(
        model, tokenizer,
        max_words=args.max_words,
        layer_idx=args.layer,
        device=args.device,
        top_k=args.top_k,
    )

    output = {
        "summary": summary,
        "word_stats": word_stats,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


def load_model_for_polysemy(model_dir: str, device: str = "cuda"):
    """Load model for polysemy analysis."""
    print(f"Loading model from {model_dir}...")
    is_local = os.path.isdir(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=is_local,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=is_local,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    main()
