"""Write alt-text for one or more clothing images with the fine-tuned model.

    python scripts/caption.py path/to/photo.jpg [more.jpg ...]
    python scripts/caption.py photo.jpg --system florence_zero_shot    # compare with a baseline

Runs on the GPU when there is one and on the CPU otherwise.
"""

import argparse

from utils.models import SYSTEMS, load_image, load_system


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--system", default="florence_finetuned", choices=sorted(SYSTEMS))
    args = parser.parse_args()

    system = load_system(args.system)
    for path in args.images:
        print(f"{path}\t{system.caption([load_image(path)])[0]}")


if __name__ == "__main__":
    main()
