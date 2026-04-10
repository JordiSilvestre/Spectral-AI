#!/usr/bin/env python3
"""
expert_analysis_common.py — Shared constants and utilities for expert specialization analysis.

Extracted from analyze_experts.py so that both the original script and the
multi-model analysis script (analyze_experts_multi.py) use identical inputs.
"""

# ── Text categories with diverse examples (30 categories, 240 texts) ──
CATEGORIES = {
    # ── STEM ──
    "algebra": [
        "Solve the equation 3x + 7 = 22. First subtract 7 from both sides",
        "The quadratic formula x = (-b +/- sqrt(b^2-4ac))/2a solves any quadratic",
        "Factoring x^2 - 5x + 6 = (x-2)(x-3) gives roots at x=2 and x=3",
        "Systems of linear equations can be solved using Gaussian elimination",
        "The determinant of a 2x2 matrix [[a,b],[c,d]] is ad - bc",
        "Polynomial long division of x^3 + 2x^2 by x + 1 yields",
        "The binomial theorem states (a+b)^n = sum of C(n,k) a^(n-k) b^k",
        "Simplify the expression (3x^2y)(4xy^3) = 12x^3y^4",
    ],
    "calculus": [
        "The derivative of x squared is 2x. Integration by parts yields",
        "The integral of sin(x)dx = -cos(x) + C by antidifferentiation",
        "L'Hopital's rule: lim f/g = lim f'/g' when both approach 0 or infinity",
        "The chain rule states d/dx f(g(x)) = f'(g(x)) * g'(x)",
        "Taylor series expansion of e^x = 1 + x + x^2/2! + x^3/3! + ...",
        "The fundamental theorem of calculus connects differentiation and integration",
        "A Riemann sum approximates the area under a curve using rectangles",
        "Partial derivatives measure the rate of change with respect to one variable",
    ],
    "statistics": [
        "The probability of rolling two sixes is 1/36. Expected value is",
        "The normal distribution has mean mu and standard deviation sigma",
        "Bayes theorem: P(A|B) = P(B|A) * P(A) / P(B) for conditional probability",
        "The correlation coefficient r ranges from -1 to 1 measuring linear association",
        "A 95% confidence interval means we are 95% confident the true value lies within",
        "The p-value represents the probability of observing results as extreme under H0",
        "Standard deviation measures the spread of data around the mean value",
        "Chi-squared test determines if observed frequencies differ from expected ones",
    ],
    "linear_algebra": [
        "The matrix multiplication AB where A is 3x3 and B is 3x2 gives",
        "The eigenvalues of the covariance matrix determine the principal components",
        "Singular value decomposition factorizes M = U Sigma V^T for any matrix",
        "An orthogonal matrix satisfies Q^T Q = I, preserving vector lengths",
        "The rank of a matrix equals the number of linearly independent rows",
        "Gram-Schmidt process converts a set of vectors into an orthonormal basis",
        "The trace of a matrix equals the sum of its diagonal elements",
        "A positive definite matrix has all positive eigenvalues and x^T A x > 0",
    ],
    "physics": [
        "The speed of light in vacuum is approximately 299,792,458 meters per second",
        "Newton's second law F=ma relates force to mass and acceleration",
        "The Schrodinger equation describes how quantum states evolve over time",
        "Entropy always increases in an isolated system according to the second law",
        "Maxwell's equations unify electricity and magnetism into electromagnetism",
        "Special relativity shows that E = mc^2 for mass-energy equivalence",
        "The Heisenberg uncertainty principle limits precision of position and momentum",
        "Conservation of angular momentum explains why spinning objects maintain rotation",
    ],
    "chemistry": [
        "The periodic table organizes elements by atomic number and electron configuration",
        "Covalent bonds form when atoms share electron pairs to achieve stability",
        "The pH scale measures acidity: pH 7 is neutral, below is acidic, above is basic",
        "Oxidation-reduction reactions involve the transfer of electrons between species",
        "Le Chatelier's principle predicts how equilibrium shifts when conditions change",
        "Avogadro's number 6.022e23 defines the number of particles in one mole",
        "Organic chemistry studies carbon-based compounds including hydrocarbons and polymers",
        "Electronegativity measures an atom's tendency to attract shared electrons",
    ],
    "biology": [
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
        "The double helix structure of DNA was discovered by Watson and Crick in 1953",
        "The mitochondria is the powerhouse of the cell, producing ATP through",
        "Natural selection drives evolution by favoring traits that improve survival",
        "Mitosis produces two identical daughter cells for growth and repair",
        "The central dogma: DNA is transcribed to RNA which is translated to protein",
        "CRISPR-Cas9 allows precise editing of DNA sequences in living organisms",
        "The human genome contains approximately 3 billion base pairs across 23 chromosomes",
    ],
    # ── COMPUTER SCIENCE ──
    "python_code": [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "import torch; model = nn.Linear(256, 64); optimizer = Adam(model.parameters())",
        "with open('data.json', 'r') as f: data = json.load(f)",
        "class DataProcessor: def __init__(self, config): self.config = config",
        "list_comprehension = [x**2 for x in range(10) if x % 2 == 0]",
        "async def fetch_data(url): async with aiohttp.ClientSession() as session:",
        "df = pd.DataFrame({'name': names, 'score': scores}).groupby('name').mean()",
        "try: result = api_call() except ConnectionError as e: logger.error(f'Failed: {e}')",
    ],
    "systems_code": [
        "int main(int argc, char* argv[]) { printf(\"Hello World\\n\"); return 0; }",
        "struct Node { int data; struct Node* next; }; // linked list in C",
        "void* malloc(size_t size); // allocates size bytes of uninitialized memory",
        "#include <pthread.h> pthread_mutex_lock(&mutex); critical_section(); pthread_mutex_unlock(&mutex);",
        "template<typename T> class Vector { T* data; size_t size; size_t capacity; };",
        "fn main() { let v: Vec<i32> = vec![1, 2, 3]; println!(\"{:?}\", v); }",
        "mov eax, [ebp+8] ; load first argument from stack frame into register",
        "kernel<<<blocks, threads>>>(d_input, d_output, N); cudaDeviceSynchronize();",
    ],
    "web_code": [
        "const app = express(); app.get('/api/users', async (req, res) => {",
        "SELECT users.name, COUNT(orders.id) FROM users LEFT JOIN orders ON",
        "<div className={styles.container}><h1>{title}</h1><p>{description}</p></div>",
        "fetch('/api/data').then(res => res.json()).then(data => setState(data))",
        "CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(255) NOT NULL);",
        "router.post('/login', passport.authenticate('local'), (req, res) => {",
        "@app.route('/api/items', methods=['GET']) def get_items(): return jsonify(items)",
        "const [count, setCount] = useState(0); useEffect(() => { fetchData(); }, []);",
    ],
    "devops": [
        "git commit -m 'fix: resolve null pointer in auth middleware'",
        "docker build -t myapp:latest . && docker run -p 8080:80 myapp:latest",
        "kubectl apply -f deployment.yaml && kubectl rollout status deployment/app",
        "name: CI Pipeline\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest",
        "terraform plan -var-file=prod.tfvars && terraform apply -auto-approve",
        "nginx -t && systemctl reload nginx # test config and reload",
        "ssh -i key.pem ec2-user@10.0.1.5 'sudo systemctl restart application'",
        "aws s3 sync ./dist s3://my-bucket --delete --cache-control max-age=86400",
    ],
    "algorithms": [
        "Binary search runs in O(log n) by halving the search space each step",
        "Quicksort uses divide and conquer with average complexity O(n log n)",
        "Dijkstra's algorithm finds shortest paths from a source to all vertices",
        "Dynamic programming breaks problems into overlapping subproblems stored in a table",
        "A hash table provides O(1) average lookup using a hash function",
        "BFS explores all neighbors at the current depth before moving deeper in a graph",
        "The traveling salesman problem is NP-hard: no known polynomial time solution",
        "A balanced binary search tree maintains O(log n) height for efficient operations",
    ],
    # ── LANGUAGE & WRITING ──
    "narrative": [
        "The beautiful sunset painted the sky in shades of orange and purple",
        "She walked through the ancient forest, listening to birds singing",
        "The old man sat alone on the park bench, watching children play",
        "Rain hammered against the windows as thunder rumbled in the distance",
        "The detective knelt beside the body, examining the scene for clues",
        "Their eyes met across the crowded room, and the world seemed to stop",
        "The ship creaked and groaned as massive waves crashed over the bow",
        "She opened the letter with trembling hands, afraid of what it might say",
    ],
    "literary_analysis": [
        "The novel explores themes of identity, belonging, and cultural displacement",
        "His eloquent speech captivated the audience with its vivid metaphors",
        "The unreliable narrator creates tension between what is told and what is true",
        "Foreshadowing in chapter three hints at the tragic ending to come",
        "The author employs stream of consciousness to reveal inner thoughts",
        "Symbolism of the green light represents Gatsby's unattainable dream",
        "The protagonist's journey follows the classic hero's quest archetype",
        "Irony pervades the text as characters remain unaware of their own contradictions",
    ],
    "poetry": [
        "Shall I compare thee to a summer's day? Thou art more lovely and more temperate",
        "Two roads diverged in a yellow wood, and I took the one less traveled by",
        "I wandered lonely as a cloud that floats on high o'er vales and hills",
        "Do not go gentle into that good night. Rage, rage against the dying of the light",
        "The fog comes on little cat feet. It sits looking over harbor and city",
        "Because I could not stop for Death, He kindly stopped for me",
        "Tyger Tyger, burning bright, in the forests of the night",
        "Hope is the thing with feathers that perches in the soul",
    ],
    "grammar": [
        "The subjunctive mood is used for hypothetical situations: If I were you",
        "A dangling modifier occurs when the subject of the modifier is unclear",
        "The Oxford comma separates the last item in a list: red, white, and blue",
        "Active voice is generally preferred: The dog bit the man, not The man was bitten",
        "Parallel structure requires matching grammatical forms in a series",
        "The semicolon connects two independent clauses that are closely related",
        "Pronoun-antecedent agreement: Everyone should bring their own lunch",
        "Split infinitives like 'to boldly go' are acceptable in modern usage",
    ],
    # ── KNOWLEDGE ──
    "history": [
        "World War II ended in 1945 with the surrender of Japan in September",
        "The French Revolution began in 1789 with the storming of the Bastille",
        "The Roman Empire fell in 476 AD when Romulus Augustulus was deposed",
        "The Industrial Revolution transformed manufacturing starting in the 1760s",
        "The printing press invented by Gutenberg around 1440 revolutionized communication",
        "The Cold War lasted from 1947 to 1991 between the USA and Soviet Union",
        "Ancient Egypt's pyramids at Giza were built around 2560 BC as royal tombs",
        "The Renaissance began in 14th century Italy, reviving classical art and learning",
    ],
    "geography": [
        "The Amazon River is the largest river by volume of water flow in the world",
        "Mount Everest stands at 8,849 meters above sea level in the Himalayas",
        "The Sahara Desert covers most of North Africa spanning 9.2 million square km",
        "The Mariana Trench is the deepest point in the ocean at 11,034 meters",
        "Australia is both a country and a continent surrounded by the Indian and Pacific oceans",
        "The Nile River flows north through eleven countries for 6,650 kilometers",
        "Iceland sits on the Mid-Atlantic Ridge where tectonic plates diverge",
        "The Great Barrier Reef off Australia's coast is the largest coral reef system",
    ],
    "law_politics": [
        "The Constitution establishes three branches of government: legislative, executive, judicial",
        "The First Amendment protects freedom of speech, religion, press, and assembly",
        "International law governs relations between sovereign nations and organizations",
        "The Supreme Court's power of judicial review was established in Marbury v. Madison",
        "Democracy requires free and fair elections with universal suffrage",
        "The Geneva Conventions establish rules for the humane treatment of prisoners of war",
        "Habeas corpus protects against unlawful detention by requiring a court hearing",
        "The European Union operates through a system of supranational governance",
    ],
    "economics": [
        "Supply and demand determine market prices in a competitive economy",
        "GDP measures the total value of goods and services produced in a country",
        "Inflation reduces purchasing power as the general price level increases over time",
        "The Federal Reserve controls monetary policy through interest rates and money supply",
        "Comparative advantage explains why countries benefit from trade specialization",
        "The Keynesian model suggests government spending can stimulate economic growth",
        "Stock markets allow companies to raise capital by selling shares to investors",
        "The Gini coefficient measures income inequality from 0 (equal) to 1 (unequal)",
    ],
    # ── SOCIAL ──
    "casual_chat": [
        "Hey, how are you doing today? I was thinking we could grab lunch",
        "Thanks for helping me with that! I really appreciate your time",
        "What do you think about the new restaurant downtown? I heard it's great",
        "Sorry I'm late, the traffic was terrible. Did I miss anything important?",
        "Happy birthday! I hope you have an amazing day with your family",
        "lol that's hilarious, I can't believe he actually said that to her",
        "omg did you see the game last night? What a comeback in the fourth quarter",
        "yeah I'm free this weekend, want to hang out? maybe grab some coffee",
    ],
    "formal_writing": [
        "Dear Sir/Madam, I am writing to express my interest in the position",
        "The quarterly report indicates a 15% increase in revenue year over year",
        "Please find attached the documents requested during our meeting on Tuesday",
        "We hereby notify you that the terms of the agreement have been modified",
        "The committee has reviewed the proposal and recommends the following amendments",
        "In accordance with company policy, all employees must complete annual training",
        "The purpose of this memorandum is to outline the strategic objectives for Q3",
        "Pursuant to our discussion, I have prepared the following action items",
    ],
    "instruction": [
        "First, preheat the oven to 350 degrees Fahrenheit. Then mix the dry ingredients",
        "Step 1: Remove the old filter. Step 2: Insert the new filter with arrows pointing up",
        "To reset your password, click the link in your email and enter a new password",
        "Begin by stretching for five minutes, then start with light cardio for warmup",
        "Install the package by running: pip install package-name in your terminal",
        "Warning: Do not mix bleach with ammonia. This produces toxic chloramine gas",
        "To assemble the shelf, align tabs A and B, then secure with the provided screws",
        "Hold the camera steady, focus on the subject, and press the shutter button halfway",
    ],
    "emotional": [
        "I'm so proud of you for finishing your degree despite all the challenges",
        "I feel devastated by the loss. The grief comes in waves that never seem to end",
        "Nothing makes me happier than seeing my children succeed and find their path",
        "I'm terrified of what the test results might show. The waiting is unbearable",
        "We are deeply sorry for your loss and extend our heartfelt condolences",
        "The joy of holding my newborn child for the first time was indescribable",
        "I'm furious that they lied to us. Trust is everything and they destroyed it",
        "After years of struggle, the relief of finally being free was overwhelming",
    ],
    # ── TECHNICAL ──
    "medical": [
        "The patient presents with acute onset chest pain radiating to the left arm",
        "Administer 325mg aspirin orally and obtain a 12-lead electrocardiogram stat",
        "Diagnosis: Type 2 diabetes mellitus with peripheral neuropathy complications",
        "The MRI reveals a 2cm lesion in the right temporal lobe consistent with glioma",
        "Prescribe metformin 500mg twice daily with meals, monitor HbA1c quarterly",
        "Differential diagnosis includes pulmonary embolism, pneumothorax, and GERD",
        "The patient's blood pressure is 140/90 mmHg indicating stage 1 hypertension",
        "Post-operative care includes wound monitoring, antibiotics, and physical therapy",
    ],
    "legal_text": [
        "The defendant is hereby charged with violation of Section 242 of Title 18",
        "Whereas the parties agree to the terms and conditions set forth herein",
        "The court finds that the plaintiff has established a prima facie case of negligence",
        "All intellectual property rights shall remain with the licensor unless otherwise stated",
        "The statute of limitations for filing a civil claim is three years from the date",
        "Under the terms of this agreement, neither party may assign rights without consent",
        "The arbitration clause requires disputes to be resolved through binding arbitration",
        "Force majeure events including natural disasters excuse performance obligations",
    ],
    "financial": [
        "The company reported earnings per share of $3.45, beating estimates by 12%",
        "Portfolio diversification reduces risk by spreading investments across asset classes",
        "The yield curve inverted today as 2-year Treasury rates exceeded 10-year rates",
        "Q3 revenue reached $4.2 billion, a 23% increase from the same quarter last year",
        "The price-to-earnings ratio of 25x suggests the stock is trading at a premium",
        "Compound interest formula: A = P(1 + r/n)^(nt) for principal P at rate r",
        "The options contract expires on the third Friday of the expiration month",
        "Free cash flow of $800M provides ample coverage for the $200M annual dividend",
    ],
    # ── STRUCTURE ──
    "punctuation": [
        "... , . ! ? ; : ' \" - ( ) [ ] { } / \\ @ # $ % & * + =",
        "Mr. Smith, Jr., arrived at 3:00 p.m. -- exactly on time.",
        "\"Hello,\" she said. \"How are you?\" He replied, \"Fine, thanks!\"",
        "The list includes: (a) first, (b) second, and (c) third.",
        "Section 3.2.1 -- Overview: The results (see Fig. 1) show that...",
        "Q&A: What is 2+2? A: 4. See also: FAQ (pp. 12-15).",
        "Email: john@example.com | Phone: +1 (555) 123-4567 | Fax: N/A",
        "[EDIT] The *original* text was {deleted} and replaced with [new content].",
    ],
    "numbers_data": [
        "The dataset contains 1,234,567 rows and 42 columns with 0.3% missing values",
        "Coordinates: 40.7128 N, 74.0060 W. Elevation: 10m. Temperature: 72.5F",
        "Results: 98.7% accuracy, 0.95 F1 score, 12.3ms latency, 1024 batch size",
        "Population: 8,336,817 (2020 census). Area: 302.6 sq mi. Density: 27,547/sq mi",
        "Version 3.14.159 released on 2024-03-14. SHA-256: a1b2c3d4e5f6...",
        "CPU: 3.8GHz, RAM: 64GB DDR5, Storage: 2TB NVMe, GPU: 16GB VRAM",
        "Flight AA123: Departs 14:30 UTC, arrives 22:15 UTC. Duration: 7h 45m",
        "Score: 3-2 (OT). Shots: 34-28. Saves: 26-31. Power play: 1/4 vs 2/5",
    ],
    "multilingual": [
        "Bonjour, comment allez-vous? Je suis tres content de vous voir aujourd'hui",
        "Hola, me llamo Carlos y vivo en Barcelona. El tiempo hoy esta soleado",
        "Guten Morgen! Wie geht es Ihnen? Ich spreche ein bisschen Deutsch",
        "Konnichiwa! Watashi wa nihongo o benkyou shite imasu. Muzukashii desu",
        "Ciao, come stai? La pizza margherita e il mio piatto preferito",
        "Privet! Kak dela? Ya uchusya programmirovaniju na Python",
        "Ni hao! Wo shi xuesheng. Wo xuexi zhongwen he shuxue",
        "Merhaba! Bugun hava cok guzel. Istanbul'da yasiyorum",
    ],
}

# ── Token type classification ──

FUNCTION_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
    'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'and', 'but', 'or', 'nor', 'not', 'so', 'yet',
    'both', 'either', 'neither', 'that', 'this', 'these', 'those', 'it',
    'its', 'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'our', 'their',
})


def classify_token(tok_str: str) -> str:
    """Classify a decoded token string into a syntactic type."""
    tok = tok_str.strip()
    if not tok:
        return "whitespace"
    if tok in '.,;:!?()[]{}"\'-/\\@#$%^&*+=<>~`|':
        return "punctuation"
    if tok.isdigit() or tok.replace('.', '').replace(',', '').isdigit():
        return "number"
    if tok.startswith('_') or any(c in tok for c in '{}()[];=<>'):
        return "code_syntax"
    if tok.lower() in FUNCTION_WORDS:
        return "function_word"
    if tok[0].isupper():
        return "capitalized"
    return "content_word"
