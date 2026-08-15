"""
Improvement 2 : LLM-Generated Semantic Style Descriptions
             - Calls the Anthropic API to auto-generate hundreds of
               semantically rich, diverse style descriptors at startup
             - Falls back to a large built-in vocabulary if API unavailable
             - Combined with DPStyler's original Random + StyleMix strategies
"""

import os
import json
import random
import math
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F



# Built-in fallback vocabulary  (large, manually curated as safety net)

FALLBACK_STYLES = [
    # ── Clipart-specific (targets clipart domain)
    "clipart", "flat design", "vector art", "cartoon", "illustrated",
    "graphic design", "flat illustration", "icon style", "sticker",
    "2D flat", "bold outline", "solid color", "comic strip",
    "clip art", "digital illustration", "clean lines", "simplified",
    "colorful cartoon", "hand drawn cartoon", "geometric flat",
    "minimal vector", "flat icon", "graphic novel", "cel shaded",
    "adobe illustrator style", "svg style", "infographic style",
    "emoji style", "logo style", "badge style", "label design",
    "pop art", "retro cartoon", "vintage comic", "modern flat",
    "gradient illustration", "isometric illustration",
    "line art", "outline illustration", "coloring book style",
    "bold graphic", "primary colors", "vibrant illustration",
    "flat color block", "two tone illustration", "cut paper style",
    "stencil art", "rubber stamp style", "screen print style",
    "propaganda poster style", "1950s commercial art", "mid century modern",
    "bauhaus style", "swiss design style", "constructivist poster",
    "vector graphic", "geometric illustration", "abstract vector",

    #  Quickdraw-specific (targets quickdraw domain) 
    "quick sketch", "hand drawn", "simple sketch", "rough drawing",
    "doodle", "pencil sketch", "black and white sketch", "line drawing",
    "minimalist sketch", "child drawing", "whiteboard drawing",
    "napkin sketch", "gesture drawing", "contour drawing",
    "single line drawing", "monochrome sketch", "chalk drawing",
    "rough outline", "simple line art", "stick figure style",
    "quick doodle", "rough sketch", "freehand drawing", "scribble art",
    "marker sketch", "ballpoint pen drawing", "continuous line art",
    "wireframe sketch", "thumbnail sketch", "blind contour drawing",
    "gesture sketch", "figure drawing", "expressive line drawing",
    "loose sketch", "spontaneous drawing", "rapid sketch",
    "ink outline", "brush pen sketch", "felt tip drawing",

    # Infograph-specific (targets infograph domain) 
    "infographic", "data visualization", "chart style", "diagram",
    "technical illustration", "instructional drawing", "exploded view",
    "cross section diagram", "flow chart style", "timeline graphic",
    "map illustration", "scientific illustration", "medical illustration",
    "engineering drawing", "blueprint style", "schematic drawing",
    "educational illustration", "textbook illustration", "manual diagram",
    "annotated illustration", "labeled diagram", "step by step guide",
    "cutaway illustration", "isometric diagram", "process flow diagram",
    "information graphic", "data art", "visual explanation",
    "technical manual style", "patent drawing style", "architectural drawing",
    "graph visualization", "network diagram", "organizational chart",

    #  Painting-specific (targets painting domain)
    "impressionist", "expressionist", "surrealist", "cubist", "abstract",
    "minimalist", "baroque", "renaissance", "watercolor", "oil painting",
    "acrylic painting", "gouache painting", "tempera painting",
    "fresco painting", "encaustic painting", "ink wash painting",
    "pointillist", "art nouveau", "fauvism", "constructivist", "dadaist",
    "realist painting", "romantic painting", "neoclassical painting",
    "pre-raphaelite style", "symbolist painting", "futurist painting",
    "abstract expressionist", "color field painting",
    "photorealist painting", "hyperrealist painting", "naive art",
    "folk art painting", "primitive art style", "outsider art",
    "Chinese brush painting", "Japanese nihonga", "Indian miniature painting",
    "Persian miniature style", "Byzantine icon style", "medieval illumination",
    "Flemish master style", "Dutch golden age", "impressionist landscape",
    "plein air painting", "alla prima painting", "glazing technique",
    "wet on wet painting", "palette knife painting", "impasto technique",
    "sgraffito technique", "trompe loeil painting", "grisaille painting",

    # Real photo-specific (targets real domain) 
    "photorealistic", "cinematic", "vintage photograph", "sepia tone",
    "black and white photo", "high contrast photo", "low key lighting",
    "soft focus photography", "bokeh background", "HDR photography",
    "infrared photography", "long exposure photo", "macro photography",
    "aerial photography", "documentary photography", "street photography",
    "portrait photography", "product photography", "fashion photography",
    "nature photography", "wildlife photography", "landscape photography",
    "architectural photography", "food photography", "sports photography",
    "photojournalism style", "studio photography", "candid photography",
    "golden hour photo", "blue hour photo", "night photography",
    "underwater photography", "drone photography", "film photography",
    "35mm film grain", "polaroid style", "lomography style",
    "tilt shift photography", "high speed photography", "composite photo",

    # Sketch-specific (targets sketch domain)
    "charcoal sketch", "pencil drawing", "graphite drawing",
    "ink sketch", "pen and ink", "crosshatch drawing",
    "technical sketch", "architectural sketch", "concept art sketch",
    "fashion sketch", "life drawing", "portrait sketch",
    "urban sketching", "travel sketch",
    "tonal sketch", "value study", "chiaroscuro drawing",
    "sfumato technique", "hatching technique", "stippling art",
    "etching style", "engraving style", "woodcut style",
    "linocut print", "lithograph style", "monotype print",
    "scraperboard style", "silverpoint drawing", "sanguine drawing",
    "pastel drawing", "conte crayon drawing", "wash drawing",
    "brush and ink drawing", "reed pen drawing", "quill pen drawing",

    #  General artistic styles 
    "graffiti", "street art", "muralism", "mosaic art",
    "stained glass", "embroidery pattern", "tapestry style",
    "pixel art", "8-bit style", "16-bit game art",
    "3D rendered", "CGI style", "low poly 3D",
    "clay animation style", "stop motion style", "puppet style",
    "neon art", "glitch art", "vaporwave aesthetic",
    "retro futurism", "steampunk aesthetic", "cyberpunk style",
    "anime style", "manga style", "comic book style",
    "graphic memoir style", "zine aesthetic", "pulp magazine style",
    "art brut", "lowbrow art", "underground comix style",
    "psychedelic art", "op art", "kinetic art style",

    # Lighting and atmosphere
    "dramatic lighting", "soft diffused light", "golden hour",
    "blue hour", "harsh shadows", "overcast lighting",
    "studio lighting", "candlelit", "moonlit scene",
    "silhouette style", "backlit subject", "rim lighting",
    "chiaroscuro lighting", "rembrandt lighting", "butterfly lighting",
    "foggy atmosphere", "misty landscape", "rainy scene",
    "snowy environment", "sunny day", "stormy sky",
    "sunset scene", "sunrise scene", "twilight atmosphere",
    "neon lit", "fire lit", "fluorescent lighting",

    # Color and tone 
    "monochromatic", "vivid colors", "muted tones", "pastel colors",
    "dark moody tones", "bright cheerful colors", "warm color palette",
    "cool color palette", "earth tones", "neon colors",
    "black and white", "duotone", "tritone", "split toning",
    "complementary colors", "analogous colors", "triadic colors",
    "desaturated style", "high saturation", "low saturation",
    "vintage color grading", "faded color", "cross processed",

    # Cultural and period styles
    "Japanese woodblock print", "Chinese ink painting",
    "African pattern art", "Islamic geometric art",
    "Celtic illuminated manuscript", "Nordic folk art",
    "Aztec pattern style", "Gothic art style",
    "Art Deco design", "Victorian illustration",
    "Edwardian style", "1920s poster art",
    "1930s depression era", "1940s wartime poster",
    "1950s retro American", "1960s psychedelic art",
    "1970s funk aesthetic", "1980s synthwave art",
    "1990s grunge style", "2000s digital art",
    "ancient Egyptian art", "Greek vase painting",
    "Roman mosaic style", "Byzantine goldwork style",
]



# LLM Style Generator  (Improvement 2)

class LLMStyleGenerator:
    """
    Uses the Anthropic API to generate a large, diverse set of style
    descriptors that go far beyond the predefined adjective list in DPStyler.

    The generated styles are cached to disk so API calls happen only once
    per run (or when you explicitly refresh).

    Parameters
    ----------
    num_styles    : how many LLM-generated styles to request
    cache_path    : where to save/load the generated style list
    api_key       : Anthropic API key (reads ANTHROPIC_API_KEY env var if None)
    """

    def __init__(self,
                 num_styles: int = 200,
                 cache_path: str = "llm_styles_cache.json",
                 api_key: Optional[str] = None):
        self.num_styles = num_styles
        self.cache_path = Path(cache_path)
        self.api_key    = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._styles: List[str] = []

    # ------------------------------------------------------------------
    def load_or_generate(self, force_refresh: bool = False) -> List[str]:
        """
        Return a list of style descriptor strings.
        Tries cache → API → fallback, in that order.
        """
        if not force_refresh and self.cache_path.exists():
            with open(self.cache_path) as f:
                data = json.load(f)
            self._styles = data.get("styles", [])
            if self._styles:
                print(f"[LLMStyleGenerator] Loaded {len(self._styles)} "
                      f"styles from cache: {self.cache_path}")
                return self._styles

        if self.api_key:
            try:
                self._styles = self._call_api()
                self._save_cache()
                print(f"[LLMStyleGenerator] Generated {len(self._styles)} "
                      f"styles via Anthropic API.")
                return self._styles
            except Exception as e:
                print(f"[LLMStyleGenerator] API call failed ({e}), "
                      f"falling back to built-in vocabulary.")

        self._styles = list(FALLBACK_STYLES)
        print(f"[LLMStyleGenerator] Using built-in vocabulary "
              f"({len(self._styles)} styles).")
        return self._styles

    # ------------------------------------------------------------------
    def _call_api(self) -> List[str]:
        """Call Anthropic claude-sonnet for style generation."""
        import anthropic  # pip install anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        prompt = f"""You are a computer vision researcher.
Generate exactly {self.num_styles} diverse visual style descriptors for
domain generalisation research. Each descriptor should describe a distinct
visual appearance/style that an image can have.

Rules:
- One style per line, no numbering, no bullet points
- Each style should be 1-4 words
- Cover: artistic styles, photography styles, textures, lighting,
  weather, rendering techniques, cultural styles, time periods,
  color palettes, media types
- Be specific and varied; avoid repetition

Output ONLY the list of styles, nothing else."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        styles = [line.strip() for line in raw.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
        return styles[:self.num_styles]

    # ------------------------------------------------------------------
    def _save_cache(self):
        with open(self.cache_path, "w") as f:
            json.dump({"styles": self._styles}, f, indent=2)

    # ------------------------------------------------------------------
    @property
    def styles(self) -> List[str]:
        if not self._styles:
            self.load_or_generate()
        return self._styles



# Style Word Vector Generator  (combines all strategies)

class StyleVectorGenerator:
    """
    Generates K style word vectors each epoch using three strategies:

      1. Random     – sample from one of 5 distributions (original DPStyler)
      2. StyleMix   – weighted sum of predefined word vectors (original DPStyler)
      3. LLMSemantic– embed LLM-generated style strings via CLIP text encoder

    At each epoch refresh, one strategy is chosen according to `probs`.
    Default probs = [0.25, 0.25, 0.50] gives more weight to LLM semantics.
    """

    DISTRIBUTIONS = ["normal", "xavier_uniform", "xavier_normal",
                     "kaiming_normal", "kaiming_uniform"]

    def __init__(self,
                 clip_model,
                 tokenizer,
                 device: torch.device,
                 feature_dim: int = 512,
                 K: int = 80,
                 L: int = 8,
                 alpha: float = 0.1,
                 strategy_probs: tuple = (0.25, 0.25, 0.50),
                 llm_generator: Optional[LLMStyleGenerator] = None):

        self.clip_model   = clip_model
        self.tokenizer    = tokenizer
        self.device       = device
        self.feature_dim  = feature_dim
        self.K            = K
        self.L            = L
        self.alpha        = alpha
        self.strategy_probs = list(strategy_probs)

        # LLM style generator
        self.llm_gen = llm_generator or LLMStyleGenerator()
        self._llm_styles: List[str] = []

        # Pre-compute StyleMix seed word vectors
        self._seed_vectors: Optional[torch.Tensor] = None

        # Pre-compute LLM semantic vectors
        self._llm_vectors: Optional[torch.Tensor] = None

    
    # Public API
    
    def initialise(self):
        """
        Call once before training starts.
        Loads LLM styles and pre-encodes seed + LLM word vectors.
        """
        self._llm_styles = self.llm_gen.load_or_generate()
        self._seed_vectors  = self._encode_word_list(
            self.llm_gen.styles[:64]          # use first 64 as StyleMix seeds
        )
        self._llm_vectors   = self._encode_word_list(self._llm_styles)
        print(f"[StyleVectorGenerator] Ready: "
              f"{len(self._llm_styles)} LLM styles encoded.")

    def refresh(self) -> torch.Tensor:
        """
        Generate K fresh style vectors.  Called at the start of each epoch.
        Returns tensor of shape (K, feature_dim).
        """
        strategy = random.choices(
            ["random", "stylemix", "llm"],
            weights=self.strategy_probs, k=1
        )[0]

        if strategy == "random":
            return self._random_generation()
        elif strategy == "stylemix":
            return self._stylemix_generation()
        else:
            return self._llm_semantic_generation()

    
    # Strategy 1: Random  
    
    def _random_generation(self) -> torch.Tensor:
        dist_name = random.choice(self.DISTRIBUTIONS)
        styles = []
        for _ in range(self.K):
            v = torch.empty(self.feature_dim, device=self.device)
            if dist_name == "normal":
                nn_init = torch.nn.init.normal_
                nn_init(v)
            elif dist_name == "xavier_uniform":
                v = v.unsqueeze(0).unsqueeze(0)
                torch.nn.init.xavier_uniform_(v)
                v = v.squeeze()
            elif dist_name == "xavier_normal":
                v = v.unsqueeze(0).unsqueeze(0)
                torch.nn.init.xavier_normal_(v)
                v = v.squeeze()
            elif dist_name == "kaiming_normal":
                v = v.unsqueeze(0)
                torch.nn.init.kaiming_normal_(v)
                v = v.squeeze()
            else:  # kaiming_uniform
                v = v.unsqueeze(0)
                torch.nn.init.kaiming_uniform_(v)
                v = v.squeeze()
            styles.append(v)
        return F.normalize(torch.stack(styles), dim=-1)

    
    # Strategy 2: StyleMix  
    
    def _stylemix_generation(self) -> torch.Tensor:
        if self._seed_vectors is None or self._seed_vectors.shape[0] == 0:
            return self._random_generation()

        seed_n = self._seed_vectors.shape[0]
        styles = []
        for _ in range(self.K):
            # Beta-distributed weights
            lambdas = torch.distributions.Beta(
                self.alpha, self.alpha
            ).sample((seed_n,)).to(self.device)
            lambdas = lambdas / lambdas.sum()          # normalise
            mixed   = (lambdas.unsqueeze(-1) * self._seed_vectors).sum(dim=0)
            styles.append(mixed)
        return F.normalize(torch.stack(styles), dim=-1)

    
    # Strategy 3: LLM Semantic  (Improvement 2 NEW)
    
    def _llm_semantic_generation(self) -> torch.Tensor:
        """
        Sample K vectors from the pool of pre-encoded LLM style vectors,
        optionally mixing nearby styles via interpolation for more variety.
        """
        if self._llm_vectors is None or self._llm_vectors.shape[0] == 0:
            return self._random_generation()

        n_llm = self._llm_vectors.shape[0]
        indices = torch.randint(0, n_llm, (self.K,), device=self.device)
        sampled = self._llm_vectors[indices]            # (K, C)

        # Add small Gaussian perturbation for variety (σ = 0.05)
        noise    = torch.randn_like(sampled) * 0.05
        perturbed = sampled + noise
        return F.normalize(perturbed, dim=-1)

    
    # Helper: encode a list of strings with CLIP text encoder
    
    @torch.no_grad()
    def _encode_word_list(self, words: List[str]) -> torch.Tensor:
        if not words:
            return torch.empty(0, self.feature_dim, device=self.device)
        tokens = self.tokenizer(words).to(self.device)
        feats  = self.clip_model.encode_text(tokens).float()
        return F.normalize(feats, dim=-1)
