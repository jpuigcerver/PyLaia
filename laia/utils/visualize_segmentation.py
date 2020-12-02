import argparse
from ast import literal_eval

import matplotlib.pyplot as plt


def visualize(args: argparse.Namespace):
    with open(args.segmentation_path) as f:
        lines = [l.split(args.separator, maxsplit=1) for l in f.readlines()]
        assert lines, "No segmentation data"
        for img_id, data in lines:
            if img_id == args.img_id:
                break
        else:
            raise ValueError(f'No segmentation data for img ID: "{args.img_id}"')

    img = plt.imread(args.img_path)
    fig, ax = plt.subplots()
    colors = plt.get_cmap(args.cmap).colors
    # img_id and data will be defined, otherwise an exception is raised
    fig.canvas.set_window_title(img_id)  # pylint: disable=undefined-loop-variable

    data = literal_eval(data)  # pylint: disable=undefined-loop-variable
    for i, (val, x1, y1, x2, y2) in enumerate(data):
        ax.axvspan(x1, x2 + 1, alpha=0.3, facecolor=colors[i % len(colors)])
        if val != args.space:
            ax.annotate(val, xy=(x1, 0), xytext=(0, 5), textcoords="offset points")

    ax.imshow(img, cmap="Greys_r")  # grayscale background
    plt.show()


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="Path to the image to visualize")
    parser.add_argument(
        "segmentation_path",
        type=str,
        help="Path to the segmentation outputted by pylaia-htr-decode-ctc",
    )
    parser.add_argument("img_id", type=str, help="Image ID to visualize")
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Use this string as the separator between the ids and the output",
    )
    parser.add_argument("--cmap", type=str, default="Set1")
    parser.add_argument(
        "--space",
        type=str,
        default="<space>",
        help="Input space symbol",
    )
    args = parser.parse_args()
    print(vars(args))
    return args


if __name__ == "__main__":
    visualize(args())
