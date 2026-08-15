

import multiprocessing
multiprocessing.freeze_support()

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os, sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
import clip
from torchvision import datasets
from torch.utils.data import DataLoader
from models.style_remover import EnhancedStyleRemover
from losses.losses import TotalLoss

# =============================================================================
# CONFIG — update DATA_ROOT and CLASS_NAMES after running check_domainnet.py
# =============================================================================
DATA_ROOT   = "D:\\LLM+SFDG\\DomainNet\\domain_net"
CHECKPOINT  = "checkpoints/dpstylerpp_final.pt"   # update if named differently
BACKBONE    = "ViT-B/16"
FEATURE_DIM = 512
BATCH_SIZE  = 64

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

# Placeholder — will be replaced after check_domainnet.py runs
# These are the standard 345 DomainNet classes (alphabetical)
CLASS_NAMES = ["The_Eiffel_Tower",
"The_Great_Wall_of_China",
"The_Mona_Lisa",
"aircraft_carrier",
"airplane",
"alarm_clock",
"ambulance",
"angel",
"animal_migration",
"ant",
"anvil",
"apple",
"arm",
"asparagus",
"axe",
"backpack",
"banana",
"bandage",
"barn",
"baseball",
"baseball_bat",
"basket",
"basketball",
"bat",
"bathtub",
"beach",
"bear",
"beard",
"bed",
"bee",
"belt",
"bench",
"bicycle",
"binoculars",
"bird",
"birthday_cake",
"blackberry",
"blueberry",
"book",
"boomerang",
"bottlecap",
"bowtie",
"bracelet",
"brain",
"bread",
"bridge",
"broccoli",
"broom",
"bucket",
"bulldozer",
"bus",
"bush",
"butterfly",
"cactus",
"cake",
"calculator",
"calendar",
"camel",
"camera",
"camouflage",
"campfire",
"candle",
"cannon",
"canoe",
"car",
"carrot",
"castle",
"cat",
"ceiling_fan",
"cell_phone",
"cello",
"chair",
"chandelier",
"church",
"circle",
"clarinet",
"clock",
"cloud",
"coffee_cup",
"compass",
"computer",
"cookie",
"cooler",
"couch",
"cow",
"crab",
"crayon",
"crocodile",
"crown",
"cruise_ship",
"cup",
"diamond",
"dishwasher",
"diving_board",
"dog",
"dolphin",
"donut",
"door",
"dragon",
"dresser",
"drill",
"drums",
"duck",
"dumbbell",
"ear",
"elbow",
"elephant",
"envelope",
"eraser",
"eye",
"eyeglasses",
"face",
"fan",
"feather",
"fence",
"finger",
"fire_hydrant",
"fireplace",
"firetruck",
"fish",
"flamingo",
"flashlight",
"flip_flops",
"floor_lamp",
"flower",
"flying_saucer",
"foot",
"fork",
"frog",
"frying_pan",
"garden",
"garden_hose",
"giraffe",
"goatee",
"golf_club",
"grapes",
"guitar",
"hamburger",
"hammer",
"hand",
"harp",
"hat",
"headphones",
"hedgehog",
"helicopter",
"helmet",
"hexagon",
"hockey_puck",
"hockey_stick",
"horse",
"hospital",
"hot_air_balloon",
"hot_dog",
"hot_tub",
"hourglass",
"house",
"house_plant",
"hurricane",
"ice_cream",
"jacket",
"jail",
"kangaroo",
"key",
"keyboard",
"knee",
"knife",
"ladder",
"lantern",
"laptop",
"leaf",
"leg",
"light_bulb",
"lighter",
"lighthouse",
"lightning",
"line",
"lion",
"lipstick",
"lobster",
"lollipop",
"mailbox",
"map",
"marker",
"matches",
"megaphone",
"mermaid",
"microphone",
"microwave",
"monkey",
"moon",
"mosquito",
"motorbike",
"mountain",
"mouse",
"moustache",
"mouth",
"mug",
"mushroom",
"nail",
"necklace",
"nose",
"ocean",
"octagon",
"octopus",
"onion",
"oven",
"owl",
"paint_can",
"paintbrush",
"palm_tree",
"panda",
"pants",
"paper_clip",
"parachute",
"parrot",
"passport",
"peanut",
"pear",
"peas",
"pencil",
"penguin",
"piano",
"pickup_truck",
"picture_frame",
"pig",
"pillow",
"pineapple",
"pizza",
"pliers",
"police_car",
"pond",
"pool",
"popsicle",
"postcard",
"potato",
"power_outlet",
"purse",
"rabbit",
"raccoon",
"radio",
"rain",
"rainbow",
"rake",
"remote_control",
"rhinoceros",
"rifle",
"river",
"roller_coaster",
"rollerskates",
"sailboat",
"sandwich",
"saw",
"saxophone",
"school_bus",
"scissors",
"scorpion",
"screwdriver",
"sea_turtle",
"see_saw",
"shark",
"sheep",
"shoe",
"shorts",
"shovel",
"sink",
"skateboard",
"skull",
"skyscraper",
"sleeping_bag",
"smiley_face",
"snail",
"snake",
"snorkel",
"snowflake",
"snowman",
"soccer_ball",
"sock",
"speedboat",
"spider",
"spoon",
"spreadsheet",
"square",
"squiggle",
"squirrel",
"stairs",
"star",
"steak",
"stereo",
"stethoscope",
"stitches",
"stop_sign",
"stove",
"strawberry",
"streetlight",
"string_bean",
"submarine",
"suitcase",
"sun",
"swan",
"sweater",
"swing_set",
"sword",
"syringe",
"t-shirt",
"table",
"teapot",
"teddy-bear",
"telephone",
"television",
"tennis_racquet",
"tent",
"tiger",
"toaster",
"toe",
"toilet",
"tooth",
"toothbrush",
"toothpaste",
"tornado",
"tractor",
"traffic_light",
"train",
"tree",
"triangle",
"trombone",
"truck",
"trumpet",
"umbrella",
"underwear",
"van",
"vase",
"violin",
"washing_machine",
"watermelon",
"waterslide",
"whale",
"wheel",
"windmill",
"wine_bottle",
"wine_glass",
"wristwatch",
"yoga",
"zebra",
"zigzag",]   # <-- populated below from file or check_domainnet.py

TEMPLATES = [
    "a {cls} in a {style} style",
    "a {style} style of a {cls}",
    "a photo of a {cls} with {style} like style",
]

# DPStyler paper Table V - ViT-B/16 CORRECT values
DPSTYLER_BASELINE = {
    "clipart":   71.5,
    "infograph": 49.1,
    "painting":  66.8,
    "quickdraw": 16.2,
    "real":      83.7,
    "sketch":    64.4,
    "average":   58.5,
}
# =============================================================================


def get_class_names(device):
    """Auto-detect class names from the dataset."""
    _, preprocess = clip.load(BACKBONE, device="cpu")
    for domain in DOMAINS:
        path = os.path.join(DATA_ROOT, domain)
        if not os.path.exists(path):
            continue
        try:
            ds = datasets.ImageFolder(path, transform=preprocess)
            print(f"[eval] Detected {len(ds.classes)} classes from '{domain}'")
            return ds.classes
        except Exception as e:
            continue
    raise RuntimeError("Could not detect class names from any domain folder.")


def load_models(device, num_classes):
    print(f"[eval] Loading CLIP {BACKBONE}...")
    clip_model, preprocess = clip.load(BACKBONE, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    num_templates = len(TEMPLATES)

    removers = nn.ModuleList([
        EnhancedStyleRemover(FEATURE_DIM, num_classes)
        for _ in range(num_templates)
    ])
    criterions = nn.ModuleList([
        TotalLoss(FEATURE_DIM, num_classes)
        for _ in range(num_templates)
    ])

    print(f"[eval] Loading checkpoint: {CHECKPOINT}")
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            f"Train first with: python run_train.py --dataset domainnet ..."
        )

    ckpt = torch.load(CHECKPOINT, map_location=device)
    for r, sd in zip(removers,   ckpt["removers"]):
        r.load_state_dict(sd)
    for c, sd in zip(criterions, ckpt["criterions"]):
        c.load_state_dict(sd)

    removers   = removers.to(device).eval()
    criterions = criterions.to(device).eval()
    print("[eval] Models loaded.\n")
    return clip_model, preprocess, removers, criterions


def predict_batch(img_feats, removers, criterions):
    img_feats  = F.normalize(img_feats, dim=-1)
    all_logits = []
    for remover, criterion in zip(removers, criterions):
        removed = F.normalize(remover(img_feats), dim=-1)
        logits  = criterion.cls_loss.get_logits(removed)
        all_logits.append(logits)
    stacked = torch.stack(all_logits, dim=0)
    max_logits, _ = stacked.max(dim=0)
    return max_logits.argmax(dim=-1)


def safe_loader(path):
    try:
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), (128, 128, 128))


def evaluate_domain(domain, clip_model, preprocess,
                    removers, criterions, class_names, device):
    path = os.path.join(DATA_ROOT, domain)
    if not os.path.exists(path):
        print(f"  [!] Not found: {path}")
        return None

    dataset = datasets.ImageFolder(
        root=path,
        transform=preprocess,
        loader=safe_loader
    )

    if dataset.classes != class_names:
        print(f"\n  [!] Class mismatch in {domain}!")
        print(f"      Run check_domainnet.py to verify class names.")

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False
    )

    correct = total = 0
    n_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            # Print progress every 50 batches so we know it's not frozen
            if batch_idx % 50 == 0:
                print(f"\r    batch {batch_idx}/{n_batches} ...",
                      end="", flush=True)
            images = images.to(device)
            labels = labels.to(device)
            feats  = clip_model.encode_image(images).float()
            preds  = predict_batch(feats, removers, criterions)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]
    print(f"\r    done ({total} images)        ", flush=True)

    return correct / total * 100.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}\n")

    # Auto-detect class names
    class_names = get_class_names(device)
    print(f"[eval] Classes: {len(class_names)}\n")

    # Check domain folders
    print("Checking domain folders...")
    found_domains = []
    for d in DOMAINS:
        path = os.path.join(DATA_ROOT, d)
        if os.path.exists(path):
            ds = datasets.ImageFolder(path)
            print(f"  {d}: OK  ({len(ds)} images)")
            found_domains.append(d)
        else:
            print(f"  {d}: NOT FOUND")
    print()

    if not found_domains:
        print(f"[!] No domains found. Check: {DATA_ROOT}")
        return

    clip_model, preprocess, removers, criterions = load_models(
        device, len(class_names)
    )

    print("=" * 62)
    print(f"  Evaluating DPStyler++ on DomainNet ({len(class_names)} classes)")
    print("=" * 62)

    results = {}
    for domain in found_domains:
        print(f"  {domain:<12}", end=" ", flush=True)
        acc = evaluate_domain(domain, clip_model, preprocess,
                              removers, criterions, class_names, device)
        if acc is not None:
            results[domain] = acc
            baseline = DPSTYLER_BASELINE.get(domain, 0)
            diff     = acc - baseline
            sign     = "+" if diff >= 0 else ""
            print(f"{acc:6.2f}%   "
                  f"(DPStyler: {baseline:.1f}%,  {sign}{diff:.2f}%)")
        else:
            print("SKIPPED")

    if not results:
        return

    avg          = sum(results.values()) / len(results)
    baseline_avg = DPSTYLER_BASELINE["average"]
    diff_avg     = avg - baseline_avg
    sign         = "+" if diff_avg >= 0 else ""

    print("-" * 62)
    print(f"  {'Average':<12}{avg:6.2f}%   "
          f"(DPStyler: {baseline_avg:.1f}%,  {sign}{diff_avg:.2f}%)")
    print("=" * 62)

    if avg > baseline_avg:
        print(f"\n  DPStyler++ BEATS DPStyler by {diff_avg:+.2f}% on DomainNet!")
    else:
        print(f"\n  Gap: {abs(diff_avg):.2f}% below DPStyler baseline.")


if __name__ == "__main__":
    main()
